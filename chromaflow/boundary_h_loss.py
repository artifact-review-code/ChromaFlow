"""H-modulated boundary supervision for V4_BND.

Combines V4_BND's geometric boundary target with H-gradient gating:

    target(p) = exp(-d1(p)/σ_b) · exp(-d2(p)/σ_b) · |∇H̃(p)|_norm

The H-gradient term filters supervision to pixels where both:
  (a) the geometric inter-disk gap criterion fires, AND
  (b) there's a real chromatin density gradient (oracle AUC 0.885)

This suppresses false-boundary supervision on:
  - lymphocyte clumps (high H interior, no real inter-cell boundary)
  - stained debris (high d1·d2 but flat chromatin)

The H signal enters as a MODULATOR, not as primary supervision target,
avoiding the soft-target dilution that killed DHS/HGFA/CGFS.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


EPS = 1e-6


def _per_image_minmax(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(1)
    lo = x.flatten(2).min(dim=2, keepdim=True).values.unsqueeze(-1)
    hi = x.flatten(2).max(dim=2, keepdim=True).values.unsqueeze(-1)
    return (x - lo) / (hi - lo + EPS)


def _gaussian_kernel(sigma: float, device, dtype) -> torch.Tensor:
    ksize = max(3, int(2 * round(3 * sigma) + 1))
    half = ksize // 2
    coords = torch.arange(-half, half + 1, device=device, dtype=dtype)
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).view(1, 1, ksize, ksize)


def _gaussian_smooth(x: torch.Tensor, sigma: float) -> torch.Tensor:
    k = _gaussian_kernel(sigma, x.device, x.dtype)
    pad = k.shape[-1] // 2
    return F.conv2d(F.pad(x, [pad] * 4, mode="reflect"), k)


_SOBEL_Y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
_SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)


def compute_h_grad_norm(stain_h: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    """stain_h: (B, H, W) in [0, 1]. Returns |∇H̃|_norm: (B, 1, H, W) in [0, 1].
    Same convention as the SGBR/HGFA oracle diagnostics (Gaussian σ=1.5, Sobel)."""
    Hn = _per_image_minmax(stain_h)
    if sigma > 0:
        Hn = _gaussian_smooth(Hn, sigma=sigma).clamp(0.0, 1.0)
    sy = _SOBEL_Y.to(Hn.device, dtype=Hn.dtype)
    sx = _SOBEL_X.to(Hn.device, dtype=Hn.dtype)
    Hp = F.pad(Hn, [1] * 4, mode="reflect")
    gy = F.conv2d(Hp, sy)
    gx = F.conv2d(Hp, sx)
    mag = torch.sqrt(gy * gy + gx * gx + EPS)
    # per-image normalize magnitude to [0, 1]
    mag = _per_image_minmax(mag.squeeze(1))
    return mag                                              # (B, 1, H, W)


def compute_boundary_h_loss(
    boundary_logits: torch.Tensor,
    clu: torch.Tensor,
    d_to_clu1: torch.Tensor,
    d_to_clu1_2nd: torch.Tensor,
    stain_h: torch.Tensor,
    sigma: float = 3.0,
    h_sigma: float = 1.5,
    h_floor: float = 0.0,
) -> torch.Tensor:
    """Geometric inter-disk-gap target modulated by H-gradient norm.

    Args:
        boundary_logits: (B, 1, H, W) — raw logits from boundary head
        clu:             (B, H, W) — BoNuS cluster labels (unused, accepted
                         for API parity with V4_BND's compute_boundary_loss)
        d_to_clu1:       (B, H, W) — distance to nearest clu==1 (px)
        d_to_clu1_2nd:   (B, H, W) — distance to 2nd nearest clu==1 (px)
        stain_h:         (B, H, W) — raw H channel from color deconvolution
        sigma:           geometric distance scale (px)
        h_sigma:         Gaussian σ for H pre-gradient smoothing (px)
        h_floor:         pixels with |∇H̃|_norm < this → 0 modulation
                         (set 0 to allow soft modulation everywhere)
    """
    if boundary_logits.ndim == 3:
        bl = boundary_logits.unsqueeze(1)
    else:
        bl = boundary_logits
    d1 = d_to_clu1.unsqueeze(1) if d_to_clu1.ndim == 3 else d_to_clu1
    d2 = d_to_clu1_2nd.unsqueeze(1) if d_to_clu1_2nd.ndim == 3 else d_to_clu1_2nd
    d1 = d1.to(dtype=bl.dtype)
    d2 = d2.to(dtype=bl.dtype)

    geo = torch.exp(-d1 / float(sigma)) * torch.exp(-d2 / float(sigma))
    h_grad = compute_h_grad_norm(stain_h, sigma=float(h_sigma)).to(dtype=bl.dtype)
    if h_floor > 0.0:
        h_grad = torch.where(h_grad >= h_floor, h_grad, torch.zeros_like(h_grad))

    target = (geo * h_grad).clamp(0.0, 1.0)
    return F.binary_cross_entropy_with_logits(bl, target, reduction="mean")


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, W = 2, 64, 64
    bl = torch.randn(B, 1, H, W, requires_grad=True)
    clu = torch.randint(0, 3, (B, H, W))
    d1 = torch.rand(B, H, W) * 30
    d2 = d1 + torch.rand(B, H, W) * 10
    stain_h = torch.rand(B, H, W)
    loss = compute_boundary_h_loss(bl, clu, d1, d2, stain_h)
    loss.backward()
    print(f"loss={loss.item():.4f}  grad_norm={bl.grad.norm().item():.5f}")
    # also: print the modulation stats
    hg = compute_h_grad_norm(stain_h)
    print(f"h_grad stats: min={hg.min():.3f}  max={hg.max():.3f}  mean={hg.mean():.3f}")
