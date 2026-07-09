"""L_NICHE — Geometric supervision on chromophore-neutral niche of BoNuS clu==2.

ADDITIVE module on top of V3 chromophore stack (L_stain_soft + L_eosin_neg +
L_PCGS additive). Specifically fills the unsupervised gap where chromophore
signal is weak but geometric distance is informative.

Trigger logic:
    chromophore-neutral = (clu==2) ∧ (|ρ| < rho_max)
    FG-target zone:  chromophore-neutral ∩ (d ≤ d_fg_max)  → target = target_fg (default 0.7)
    BG-target zone:  chromophore-neutral ∩ (d > d_bg_min)  → target = target_bg (default 0.0)
    else: ignored

Diagnostic (MO train, A4_R34 forward) confirmed:
  - chromophore-neutral ∩ d≤2: A4_R34 accuracy 55%, mean fg_prob 0.55 on true-fg (uncertain)
  - chromophore-neutral ∩ d>12: A4_R34 accuracy 87%, but recall on true-fg only 47%
  - These zones are explicitly ignored by L_stain_soft (H>0.5) and L_eosin_neg (E>0.5∧H<0.5).

Defaults from diagnostic-confirmed sweet spot:
  rho_max=0.2, d_fg_max=2.0, d_bg_min=12.0, target_fg=0.7, target_bg=0.0
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _per_image_norm01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(1)
    lo = x.flatten(2).amin(dim=2).view(-1, 1, 1, 1)
    hi = x.flatten(2).amax(dim=2).view(-1, 1, 1, 1)
    return (x - lo) / (hi - lo + eps)


def compute_niche_loss(
    fg_logits: torch.Tensor,
    clu: torch.Tensor,
    d_to_clu1: torch.Tensor,
    stain_h: torch.Tensor,
    stain_e: torch.Tensor,
    rho_max: float = 0.2,
    d_fg_max: float = 2.0,
    d_bg_min: float = 12.0,
    target_fg: float = 0.7,
    target_bg: float = 0.0,
    eps: float = 1e-6,
    return_aux: bool = False,
):
    """L_NICHE: hard {target_fg, target_bg} BCE on chromophore-neutral niche of clu==2.

    Args:
        fg_logits:  (B, 1, H, W) raw foreground logits
        clu:         (B, H, W) long {0, 1, 2}
        d_to_clu1:   (B, H, W) distance to nearest clu==1 (px)
        stain_h:     (B, H, W) raw hematoxylin density
        stain_e:     (B, H, W) raw eosin density
        rho_max:     |ρ| upper bound for chromophore-neutral
        d_fg_max:    FG-target trigger: d ≤ d_fg_max
        d_bg_min:    BG-target trigger: d > d_bg_min
        target_fg:   FG supervision target (default 0.7, matches L_stain_soft)
        target_bg:   BG supervision target (default 0.0, matches L_eosin_neg)

    Returns:
        scalar loss (or (loss, aux))
    """
    # Compute polarization ρ = (Hn - En) / (Hn + En)
    Hn = _per_image_norm01(stain_h).to(dtype=fg_logits.dtype)
    En = _per_image_norm01(stain_e).to(dtype=fg_logits.dtype)
    rho = (Hn - En) / (Hn + En + eps)

    d = d_to_clu1 if d_to_clu1.ndim == 4 else d_to_clu1.unsqueeze(1)
    d = d.to(dtype=fg_logits.dtype)

    clu_mask = (clu == 2).unsqueeze(1)
    neutral = clu_mask & (rho.abs() < float(rho_max))

    mask_fg = neutral & (d <= float(d_fg_max))
    mask_bg = neutral & (d > float(d_bg_min))

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
            "neutral": neutral.detach(),
            "rho": rho.detach(),
        }
    return loss


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def _test_basic():
    torch.manual_seed(0)
    fg = torch.randn(2, 1, 32, 32, requires_grad=True)
    clu = torch.randint(0, 3, (2, 32, 32))
    d = torch.rand(2, 32, 32) * 30
    H = torch.rand(2, 32, 32) * 0.8 + 0.1
    E = torch.rand(2, 32, 32) * 0.8 + 0.1
    loss = compute_niche_loss(fg, clu, d, H, E)
    loss.backward()
    g = fg.grad.norm().item()
    print(f"  [basic] loss={loss.item():.4f}  grad_norm={g:.5f}")
    assert g > 1e-7


def _test_exclusive_triggers():
    """FG and BG masks should not overlap with d_fg_max < d_bg_min."""
    H, W = 16, 16
    fg = torch.zeros(1, 1, H, W)
    clu = torch.full((1, H, W), 2, dtype=torch.long)
    d = torch.linspace(0, 20, H * W).view(1, H, W)
    Hh = torch.full((1, H, W), 0.5)
    Ee = torch.full((1, H, W), 0.4)  # ρ ≈ 0.11 → neutral
    loss, aux = compute_niche_loss(fg, clu, d, Hh, Ee, return_aux=True)
    overlap = (aux["mask_fg"] & aux["mask_bg"]).sum().item()
    n_fg = aux["mask_fg"].sum().item()
    n_bg = aux["mask_bg"].sum().item()
    n_neu = aux["neutral"].sum().item()
    print(f"  [exclusive] neutral={n_neu}  n_fg={n_fg}  n_bg={n_bg}  overlap={overlap}")
    assert overlap == 0


def _test_neutral_filter():
    """When ρ is far from 0, NICHE should not fire."""
    H, W = 8, 8
    fg = torch.zeros(1, 1, H, W)
    clu = torch.full((1, H, W), 2, dtype=torch.long)
    d = torch.ones(1, H, W) * 1.0
    Hh = torch.ones(1, H, W) * 0.9   # ρ = (0.9-0.1)/(1.0) = 0.8 → not neutral
    Ee = torch.ones(1, H, W) * 0.1
    loss, aux = compute_niche_loss(fg, clu, d, Hh, Ee, rho_max=0.2, return_aux=True)
    n_neu = aux["neutral"].sum().item()
    print(f"  [neutral filter] high-ρ scene: neutral={n_neu} (should be 0)")
    assert n_neu == 0


def _test_no_clu2():
    fg = torch.zeros(1, 1, 8, 8)
    clu = torch.zeros(1, 8, 8, dtype=torch.long)
    d = torch.ones(1, 8, 8) * 5
    Hh = torch.ones(1, 8, 8) * 0.5
    Ee = torch.ones(1, 8, 8) * 0.5
    loss = compute_niche_loss(fg, clu, d, Hh, Ee)
    print(f"  [no clu2] loss={loss.item():.4f}")
    assert loss.item() < 0.05


if __name__ == "__main__":
    print("=== L_NICHE unit tests ===")
    _test_basic()
    _test_exclusive_triggers()
    _test_neutral_filter()
    _test_no_clu2()
    print("\nAll tests passed.")
