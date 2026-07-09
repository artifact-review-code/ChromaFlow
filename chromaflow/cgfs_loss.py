"""CGFS — Chromophore-Guided Flow Supervision.

Two losses, both supervising the BoNuS ignore zone (clu == 2) using the
oracle-verified hematoxylin field:

  DHS  (Dense H̃ Supervision)        — scalar supervision on fg head
       L_DHS  = mean BCE(σ(fg_logits), H̃_n)  over clu==2
       Replaces brittle (L_stain_soft + L_eosin_neg) dual-trigger scheme.

  HGFA (H-Gradient Flow Anchoring)   — vector supervision on flow head
       L_HGFA = Σ w(p)·(1 − cos(flow_pred, ∇H̃/|∇H̃|))  /  Σ w(p)
              over (clu==2 ∩ d_to_clu1 ≤ d_max)
       w(p) = |∇H̃(p)|.  Mask-free analog of HoVer-Net HV-map supervision.

Both losses derive from H̃ = Gaussian-smoothed, per-image-normalized H.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


EPS = 1e-6


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def per_image_minmax(x: torch.Tensor) -> torch.Tensor:
    """Per-image min-max normalize to [0, 1]. x: (B, H, W) or (B, 1, H, W)."""
    if x.ndim == 3:
        x = x.unsqueeze(1)
    flat = x.flatten(2)
    lo = flat.min(dim=2, keepdim=True).values.unsqueeze(-1)
    hi = flat.max(dim=2, keepdim=True).values.unsqueeze(-1)
    return (x - lo) / (hi - lo + EPS)


def _gaussian_kernel_2d(sigma: float, device, dtype) -> torch.Tensor:
    ksize = max(3, int(2 * round(3 * sigma) + 1))
    half = ksize // 2
    coords = torch.arange(-half, half + 1, device=device, dtype=dtype)
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    k2 = g[:, None] * g[None, :]
    return k2.view(1, 1, ksize, ksize)


def gaussian_smooth(x: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    """x: (B, 1, H, W). Returns same shape."""
    k = _gaussian_kernel_2d(sigma, x.device, x.dtype)
    pad = k.shape[-1] // 2
    return F.conv2d(F.pad(x, [pad] * 4, mode="reflect"), k)


# Sobel kernels matching scipy.ndimage.sobel (positive = increasing index)
_SOBEL_Y = torch.tensor([[-1., -2., -1.],
                           [0.,  0.,  0.],
                           [1.,  2.,  1.]]).view(1, 1, 3, 3)
_SOBEL_X = torch.tensor([[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]]).view(1, 1, 3, 3)


def sobel_xy(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x: (B, 1, H, W). Returns (gy, gx) each (B, 1, H, W)."""
    sy = _SOBEL_Y.to(x.device, dtype=x.dtype)
    sx = _SOBEL_X.to(x.device, dtype=x.dtype)
    xp = F.pad(x, [1] * 4, mode="reflect")
    gy = F.conv2d(xp, sy)
    gx = F.conv2d(xp, sx)
    return gy, gx


def compute_H_field(stain_h: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    """stain_h: (B, H, W) raw H from color deconvolution.
    Returns H̃: (B, 1, H, W) per-image-normalized + Gaussian-smoothed in [0, 1].
    """
    Hn = per_image_minmax(stain_h)                 # (B, 1, H, W) in [0, 1]
    if sigma > 0:
        Hn = gaussian_smooth(Hn, sigma=sigma)
        # re-clip to [0, 1] (numerical safety)
        Hn = Hn.clamp(0.0, 1.0)
    return Hn


# ---------------------------------------------------------------------------
# DHS — Dense H̃ Supervision (scalar)
# ---------------------------------------------------------------------------


def compute_dhs_loss(
    fg_logits: torch.Tensor,
    clu: torch.Tensor,
    stain_h: torch.Tensor,
    sigma: float = 1.5,
    target_low: float = 0.0,
    target_high: float = 1.0,
    return_aux: bool = False,
):
    """Dense H̃ soft-label BCE on clu == 2.

    Args:
        fg_logits: (B, 1, H, W) raw fg logits
        clu:        (B, H, W) long {0, 1, 2}
        stain_h:    (B, H, W) raw hematoxylin density (per color deconv)
        sigma:      Gaussian smoothing scale on H (pre-norm) (px)
        target_low, target_high: linearly map H̃ ∈ [0, 1] →
                                 [target_low, target_high] before BCE.
                                 Defaults preserve raw H̃ as probability target.

    Returns:
        scalar loss tensor (returns 0 if no clu==2 pixels).
    """
    Ht = compute_H_field(stain_h, sigma=sigma)            # (B, 1, H, W) ∈ [0, 1]
    target = target_low + (target_high - target_low) * Ht
    target = target.clamp(0.0, 1.0)

    if clu.ndim == 3:
        mask = (clu == 2).unsqueeze(1).to(fg_logits.dtype)
    else:
        mask = (clu == 2).to(fg_logits.dtype)
    if mask.sum() == 0:
        out = fg_logits.new_tensor(0.0)
        return (out, {"n": 0}) if return_aux else out

    bce = F.binary_cross_entropy_with_logits(fg_logits, target, reduction="none")
    loss = (bce * mask).sum() / mask.sum().clamp(min=EPS)
    if return_aux:
        return loss, {"n": int(mask.sum().item()),
                      "mean_target": float((target * mask).sum().item()
                                           / mask.sum().clamp(min=EPS).item())}
    return loss


# ---------------------------------------------------------------------------
# HGFA — H-Gradient Flow Anchoring (vector)
# ---------------------------------------------------------------------------


def compute_hgfa_loss(
    flow_pred: torch.Tensor,
    clu: torch.Tensor,
    d_to_clu1: torch.Tensor,
    stain_h: torch.Tensor,
    d_max: float = 5.0,
    sigma: float = 1.5,
    grad_mag_floor: float = 0.0,
    weight_pow: float = 1.0,
    return_aux: bool = False,
):
    """Distance-gated H-gradient flow direction supervision on clu == 2.

    flow_pred convention: (B, 2, H, W), channel 0 = row (y), channel 1 = col (x).
    target = ∇H̃ / |∇H̃|  (points toward H peak ≈ nucleus center).

    Args:
        flow_pred:  (B, 2, H, W) network flow output (tanh-bounded by model)
        clu:         (B, H, W) long
        d_to_clu1:   (B, H, W) distance to nearest clu==1 (px)
        stain_h:     (B, H, W) raw H
        d_max:       only supervise clu==2 ∩ d_to_clu1 ≤ d_max (boundary band)
        sigma:       Gaussian smoothing on H pre-gradient
        grad_mag_floor: pixels with |∇H̃| < this fraction of per-image max → 0 weight
        weight_pow:  apply |∇H̃| ** weight_pow as supervision weight (1 = linear)

    Returns:
        scalar loss tensor (returns 0 if trigger empty).
    """
    Ht = compute_H_field(stain_h, sigma=sigma)            # (B, 1, H, W)
    gy, gx = sobel_xy(Ht)                                  # each (B, 1, H, W)
    mag = torch.sqrt(gy * gy + gx * gx + EPS)              # (B, 1, H, W)

    # per-image max mag for weight normalization
    max_per = mag.flatten(2).max(dim=2, keepdim=True).values.unsqueeze(-1)
    weight = (mag / (max_per + EPS)).clamp(0.0, 1.0)
    if grad_mag_floor > 0.0:
        weight = torch.where(weight >= grad_mag_floor, weight,
                              torch.zeros_like(weight))
    if abs(weight_pow - 1.0) > EPS:
        weight = weight ** float(weight_pow)

    # unit-direction target (safe normalize)
    unit_y = gy / mag
    unit_x = gx / mag

    # cosine between predicted flow and target unit direction
    # NOTE: flow_pred is tanh-bounded but NOT unit; we normalize here so that
    # the cos is a true direction agreement in [-1, 1] (decoupling from
    # magnitude, which is separately supervised by L_flow on clu==1).
    fy = flow_pred[:, 0:1]
    fx = flow_pred[:, 1:2]
    fmag = torch.sqrt(fy * fy + fx * fx + EPS)
    fy_n = fy / fmag
    fx_n = fx / fmag
    cos = fy_n * unit_y + fx_n * unit_x                    # (B, 1, H, W) ∈ [-1, 1]

    # gating
    if clu.ndim == 3:
        gate = ((clu == 2) & (d_to_clu1 <= d_max)).unsqueeze(1).to(flow_pred.dtype)
    else:
        gate = ((clu == 2) & (d_to_clu1 <= d_max)).to(flow_pred.dtype)
    w = weight * gate

    denom = w.sum().clamp(min=EPS)
    if denom.item() == 0:
        out = flow_pred.new_tensor(0.0)
        return (out, {"n": 0}) if return_aux else out

    loss = (w * (1.0 - cos)).sum() / denom
    if return_aux:
        with torch.no_grad():
            mean_cos = (w * cos).sum() / denom
        return loss, {"n": int(gate.sum().item()), "mean_cos": float(mean_cos.item())}
    return loss


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def _smoke():
    torch.manual_seed(0)
    B, H, W = 2, 64, 64

    stain_h = torch.rand(B, H, W) * 0.8 + 0.1
    clu = torch.zeros(B, H, W, dtype=torch.long)
    clu[:, 20:40, 20:40] = 1            # cluster disks
    clu[:, 15:45, 15:45] = torch.where(clu[:, 15:45, 15:45] == 0,
                                         torch.full_like(clu[:, 15:45, 15:45], 2),
                                         clu[:, 15:45, 15:45])
    # d to nearest clu==1 (approximation via abs distance)
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    d = ((yy - 30).clamp(min=0) ** 2 + (xx - 30).clamp(min=0) ** 2).float().sqrt()
    d_to_clu1 = d.unsqueeze(0).expand(B, -1, -1)

    fg_logits = torch.randn(B, 1, H, W, requires_grad=True)
    flow_pred = torch.tanh(torch.randn(B, 2, H, W, requires_grad=True))

    l_dhs, aux1 = compute_dhs_loss(fg_logits, clu, stain_h, return_aux=True)
    l_hgfa, aux2 = compute_hgfa_loss(flow_pred, clu, d_to_clu1, stain_h,
                                       d_max=10.0, return_aux=True)
    print(f"DHS  loss={l_dhs.item():.4f}  n={aux1['n']}  mean_target={aux1['mean_target']:.3f}")
    print(f"HGFA loss={l_hgfa.item():.4f}  n={aux2['n']}  mean_cos={aux2['mean_cos']:+.3f}")
    # check backprop
    (l_dhs + l_hgfa).backward()
    print("Backprop OK.")


if __name__ == "__main__":
    _smoke()
