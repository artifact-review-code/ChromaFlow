"""Competing-Sink separation losses for point-supervised nuclei.

compute_csbb_loss  (v3): training-side separation target on the equidistant
    competing-sink ridge ∩ local H-valley ∩ H>floor. Unconditional (all pixels
    BCE'd against the sparse target). Proof-of-mechanism: CoNIC +0.035 AJI but
    regime-mixed (MO/CPM slight cost).

compute_gcsbb_loss (v4): SAME target b*, but the loss is GATED so it only
    supervises where it should — regime-adaptive. Weight per pixel:
        w = G_patch * g_comp(x) * g_fg(x)
      G_patch : per-crop crowding scalar (fraction of confident-fg pixels whose
                2nd-nearest point is within crowd_r) -> sparse crops contribute
                little, dense crops dominate the gradient.
      g_comp  : exp(-(d2-d1)^2/2σ^2)*1[d2<Rb] -> only where two sinks compete.
      g_fg    : 1[sigmoid(fg_logits).detach() > fg_tau] -> only on confident
                nuclear material; "split, don't create new foreground" guard.
    decode-time barrier is inert (α0≈α2); this is a training auxiliary only.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _norm01(x: torch.Tensor) -> torch.Tensor:
    s_min = x.flatten(2).amin(dim=2).view(-1, 1, 1, 1)
    s_max = x.flatten(2).amax(dim=2).view(-1, 1, 1, 1)
    return ((x - s_min) / (s_max - s_min + 1e-6)).clamp(0.0, 1.0)


def _prep(boundary_logits, d_to_clu1, d_to_clu1_2nd, stain_h):
    bl = boundary_logits.unsqueeze(1) if boundary_logits.ndim == 3 else boundary_logits
    d1 = d_to_clu1.unsqueeze(1) if d_to_clu1.ndim == 3 else d_to_clu1
    d2 = d_to_clu1_2nd.unsqueeze(1) if d_to_clu1_2nd.ndim == 3 else d_to_clu1_2nd
    h = stain_h.unsqueeze(1) if stain_h.ndim == 3 else stain_h
    return bl, d1.to(bl.dtype), d2.to(bl.dtype), _norm01(h.to(bl.dtype))


def _csbb_target(d1, d2, H, sigma, rb, h_floor, valley_k, valley_margin):
    a3 = F.avg_pool2d(H, kernel_size=3, stride=1, padding=1)
    aK = F.avg_pool2d(H, kernel_size=valley_k, stride=1, padding=valley_k // 2)
    valley = (a3 < (aK - valley_margin)).to(H.dtype)
    ridge = torch.exp(-((d2 - d1) ** 2) / (2.0 * sigma * sigma))
    return (ridge * (d2 < rb).to(H.dtype) * valley * (H > h_floor).to(H.dtype)).clamp(0.0, 1.0)


def compute_csbb_loss(boundary_logits, clu, d_to_clu1, d_to_clu1_2nd, stain_h,
                      sigma: float = 3.0, rb: float = 18.0, h_floor: float = 0.4,
                      valley_k: int = 13, valley_margin: float = 0.02) -> torch.Tensor:
    bl, d1, d2, H = _prep(boundary_logits, d_to_clu1, d_to_clu1_2nd, stain_h)
    b_star = _csbb_target(d1, d2, H, sigma, rb, h_floor, valley_k, valley_margin)
    return F.binary_cross_entropy_with_logits(bl, b_star.detach(), reduction="mean")


def compute_gcsbb_loss(boundary_logits, fg_logits, clu, d_to_clu1, d_to_clu1_2nd, stain_h,
                       sigma: float = 3.0, rb: float = 18.0, h_floor: float = 0.4,
                       valley_k: int = 13, valley_margin: float = 0.02,
                       fg_tau: float = 0.3, crowd_r: float = 18.0) -> torch.Tensor:
    bl, d1, d2, H = _prep(boundary_logits, d_to_clu1, d_to_clu1_2nd, stain_h)
    fgl = fg_logits.unsqueeze(1) if fg_logits.ndim == 3 else fg_logits
    clu_ = clu.unsqueeze(1) if clu.ndim == 3 else clu

    b_star = _csbb_target(d1, d2, H, sigma, rb, h_floor, valley_k, valley_margin)

    g_comp = torch.exp(-((d2 - d1) ** 2) / (2.0 * sigma * sigma)) * (d2 < rb).to(bl.dtype)
    g_fg = (torch.sigmoid(fgl).detach() > fg_tau).to(bl.dtype)

    # per-crop crowding scalar G_patch in [0,1]: fraction of confident-fg (clu==1)
    # pixels whose 2nd-nearest point is within crowd_r.
    fgmask = (clu_ == 1).to(bl.dtype)
    near2 = ((d2 < crowd_r).to(bl.dtype) * fgmask).flatten(2).sum(2)
    denom = fgmask.flatten(2).sum(2).clamp(min=1.0)
    G_patch = (near2 / denom).view(-1, 1, 1, 1).clamp(0.0, 1.0)

    w = G_patch * g_comp * g_fg
    loss_px = F.binary_cross_entropy_with_logits(bl, b_star.detach(), reduction="none")
    return (loss_px * w).sum() / w.sum().clamp(min=1.0)


if __name__ == "__main__":
    torch.manual_seed(0)
    bl = torch.randn(2, 1, 48, 48, requires_grad=True)
    fg = torch.randn(2, 1, 48, 48)
    clu = torch.randint(0, 3, (2, 48, 48))
    d1 = torch.rand(2, 48, 48) * 25
    d2 = d1 + torch.rand(2, 48, 48) * 12
    h = torch.rand(2, 48, 48)
    for fn, args in [(compute_csbb_loss, (bl, clu, d1, d2, h)),
                     (compute_gcsbb_loss, (bl, fg, clu, d1, d2, h))]:
        l = fn(*args); l.backward(retain_graph=True)
        print(f"{fn.__name__}: loss={l.item():.4f} grad={bl.grad.norm().item():.5f}")
        bl.grad = None
