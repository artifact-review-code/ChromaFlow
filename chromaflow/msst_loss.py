"""MSST — Multi-Signal Sparse-Hard Trigger loss.

V3-style sparse-hard supervision on BoNuS clu==2, but with a JOINT trigger over
(d, smoothed_H) instead of chromophore alone:

  FG trigger: (clu==2) ∧ (d ≤ d_fg_max) ∧ (H_smooth ≥ h_fg_min)   → target = target_fg
  BG trigger: (clu==2) ∧ ((d > d_bg_min) ∨ (H_smooth < h_bg_max))  → target = target_bg
  Otherwise: ignored.

Where H_smooth = avg_pool(per-image-normalize(H), kernel=h_kernel).

Diagnostic on MO+CPM train showed:
  (d≤2 AND H_5≥q70) → coverage 9%, precision 0.73 (FG)
  (d>10 OR H_5<q20) → coverage 70%, precision_bg 0.995 (BG)
  Total 79% coverage with much higher precision than V3 baseline (~30%).

Defaults match the joint-trigger sweet spot from the diagnostic.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _per_image_norm01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-image min-max normalize to [0, 1]. x: (B, 1, H, W) or (B, H, W)."""
    if x.ndim == 3:
        x = x.unsqueeze(1)
    lo = x.flatten(2).amin(dim=2).view(-1, 1, 1, 1)
    hi = x.flatten(2).amax(dim=2).view(-1, 1, 1, 1)
    return (x - lo) / (hi - lo + eps)


def smooth_stain_h(stain_h: torch.Tensor, kernel: int = 5) -> torch.Tensor:
    """Per-image normalize H, then box-smooth (avg_pool with stride=1)."""
    Hn = _per_image_norm01(stain_h)
    if kernel <= 1:
        return Hn
    pad = kernel // 2
    return F.avg_pool2d(Hn, kernel_size=kernel, stride=1, padding=pad)


def compute_msst_loss(
    fg_logits: torch.Tensor,
    clu: torch.Tensor,
    d_to_clu1: torch.Tensor,
    stain_h: torch.Tensor,
    d_fg_max: float = 2.0,
    d_bg_min: float = 10.0,
    h_kernel: int = 5,
    h_fg_min: float = 0.5,
    h_bg_max: float = 0.3,
    target_fg: float = 0.7,
    target_bg: float = 0.0,
    return_aux: bool = False,
):
    """Joint (d, smoothed-H) sparse-hard supervision on clu==2.

    Args:
        fg_logits: (B, 1, H, W)
        clu:        (B, H, W)
        d_to_clu1:  (B, H, W) or (B, 1, H, W) — distance to nearest clu==1 pixel
        stain_h:    (B, H, W) or (B, 1, H, W) — raw hematoxylin density
        d_fg_max:   FG trigger requires d ≤ d_fg_max (px)
        d_bg_min:   BG trigger satisfies d > d_bg_min (px)
        h_kernel:   box-smoothing kernel for H (px); diagnostic sweet spot = 5
        h_fg_min:   FG trigger requires H_smooth ≥ h_fg_min (after per-image normalize)
        h_bg_max:   BG trigger satisfies H_smooth < h_bg_max
        target_fg:  supervision target for FG-triggered pixels (default 0.7, V3 L_stain_soft style)
        target_bg:  supervision target for BG-triggered pixels (default 0.0)
        return_aux: if True, also return masks dict for debug

    Returns:
        scalar loss (or (loss, aux))
    """
    # Normalize tensor shapes
    d = d_to_clu1 if d_to_clu1.ndim == 4 else d_to_clu1.unsqueeze(1)
    d = d.to(dtype=fg_logits.dtype)
    H_smooth = smooth_stain_h(stain_h, kernel=h_kernel).to(dtype=fg_logits.dtype)

    clu_mask = (clu == 2).unsqueeze(1)

    mask_fg = clu_mask & (d <= float(d_fg_max)) & (H_smooth >= float(h_fg_min))
    mask_bg = clu_mask & ((d > float(d_bg_min)) | (H_smooth < float(h_bg_max)))

    # BG trigger should win in any conflict (FG ∩ BG should be empty by construction but be safe).
    # If both fire (shouldn't happen with d_fg_max < d_bg_min), prefer FG.
    target_fg_t = torch.full_like(d, float(target_fg))
    target_bg_t = torch.full_like(d, float(target_bg))
    target = torch.where(mask_fg, target_fg_t, torch.where(mask_bg, target_bg_t, target_fg_t))
    eff_mask = (mask_fg | mask_bg).to(dtype=fg_logits.dtype)

    bce = F.binary_cross_entropy_with_logits(fg_logits, target, reduction="none")
    num = (bce * eff_mask).sum()
    den = eff_mask.sum().clamp(min=1.0)
    loss = num / den

    if return_aux:
        return loss, {
            "mask_fg": mask_fg.detach(),
            "mask_bg": mask_bg.detach(),
            "target": target.detach(),
            "H_smooth": H_smooth.detach(),
        }
    return loss


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def _test_basic():
    torch.manual_seed(0)
    fg_logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    clu = torch.randint(0, 3, (2, 32, 32))
    d = torch.rand(2, 32, 32) * 30.0
    H = torch.rand(2, 32, 32) * 0.8 + 0.1
    loss = compute_msst_loss(fg_logits, clu, d, H)
    loss.backward()
    g = fg_logits.grad.norm().item()
    print(f"  [basic] loss={loss.item():.4f}, grad_norm={g:.5f}")
    assert g > 1e-7, f"gradient too small: {g}"


def _test_triggers_exclusive():
    """With d_fg_max=2, d_bg_min=10, no pixel should be both FG and BG."""
    H, W = 16, 16
    fg_logits = torch.zeros(1, 1, H, W)
    clu = torch.full((1, H, W), 2, dtype=torch.long)
    d = torch.linspace(0, 15, H * W).view(1, H, W)
    stain_h = torch.full((1, H, W), 0.5)
    loss, aux = compute_msst_loss(
        fg_logits, clu, d, stain_h,
        d_fg_max=2.0, d_bg_min=10.0, h_fg_min=0.5, h_bg_max=0.3,
        return_aux=True,
    )
    overlap = (aux["mask_fg"] & aux["mask_bg"]).sum().item()
    n_fg = aux["mask_fg"].sum().item()
    n_bg = aux["mask_bg"].sum().item()
    print(f"  [exclusive] n_fg={n_fg} n_bg={n_bg} overlap={overlap}")
    assert overlap == 0, f"FG and BG masks overlap: {overlap}"


def _test_no_clu2():
    """If no clu==2 pixels, loss is 0."""
    fg = torch.zeros(1, 1, 8, 8)
    clu = torch.zeros(1, 8, 8, dtype=torch.long)
    d = torch.ones(1, 8, 8) * 5
    H = torch.ones(1, 8, 8) * 0.5
    loss = compute_msst_loss(fg, clu, d, H)
    print(f"  [no clu2] loss={loss.item():.4f}")
    assert loss.item() < 0.05


def _test_smoothing_kernel():
    H_raw = torch.zeros(1, 1, 16, 16)
    H_raw[0, 0, 8, 8] = 1.0
    H5 = smooth_stain_h(H_raw, kernel=5)
    # After per-image normalize, the single peak becomes 1.0 then smoothed
    # by 5x5 avg = peak value 1/(5*5) = 0.04, with high-value blob around it
    print(f"  [smooth] kernel=5  H5.max={H5.max().item():.4f}, center={H5[0,0,8,8].item():.4f}")
    assert H5[0, 0, 8, 8] > 0  # smoothed value at peak
    assert H5.min() >= 0
    assert H5.max() <= 1.0 + 1e-4


if __name__ == "__main__":
    print("=== MSST loss unit tests ===")
    _test_basic()
    _test_triggers_exclusive()
    _test_no_clu2()
    _test_smoothing_kernel()
    print("\nAll tests passed.")
