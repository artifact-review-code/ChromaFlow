"""CPR — Cross-Pixel Polarity Ranking loss (Module A of DSC paper).

Mechanism (single sentence):
  Within the BoNuS clu==2 ignore zone, sample pixel pairs with large ρ = H-E
  polarity difference and enforce σ(fg) to respect the ρ-ordering via a margin
  ranking loss. Avoids the absolute-soft-target dilution that killed SPP/GAP/
  CGFS by using *relative* (per-pair) supervision instead of pointwise target.

Pre-flight evidence (cpr_preflight.py on MO train, 80 patches):
  - Kendall τ(ρ, true_fg) = 0.701 (concordance 85% — STRONG signal)
  - Baseline σ(fg) ranking acc on ρ-ordered pairs = 0.768 (headroom)

Hyperparameters:
  K = n_pairs_per_image (default 64)
  rho_diff_min = 0.3 (filter for ρ-separated pairs)
  margin: curriculum 0.05 → 0.25 over `margin_warmup_epochs` epochs

Inputs:
  fg_logits   (B, 1, H, W) raw foreground logits
  clu          (B, H, W) long {0, 1, 2}
  stain_h      (B, H, W) raw H density (rgb2hed channel 0, post-per-image-norm
                          robust 1-99 percentile; same as dataset.stain_h_channel)
  stain_e      (B, H, W) raw E density (same prep as H)
  epoch        int (current epoch, for margin curriculum)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


EPS = 1e-6


def _per_image_norm01(x: torch.Tensor) -> torch.Tensor:
    """Per-image min-max normalize to [0, 1].
    x: (B, H, W) or (B, 1, H, W) → (B, 1, H, W)."""
    if x.ndim == 3:
        x = x.unsqueeze(1)
    lo = x.flatten(2).amin(dim=2).view(-1, 1, 1, 1)
    hi = x.flatten(2).amax(dim=2).view(-1, 1, 1, 1)
    return (x - lo) / (hi - lo + EPS)


def _cpr_margin(epoch: int, margin_init: float, margin_final: float,
                margin_warmup_epochs: int) -> float:
    if margin_warmup_epochs <= 0:
        return margin_final
    frac = min(1.0, max(0.0, epoch / float(margin_warmup_epochs)))
    return margin_init + (margin_final - margin_init) * frac


@torch.no_grad()
def _sample_pair_indices(rho_flat: torch.Tensor,
                         clu2_idx: torch.Tensor,
                         n_pairs: int,
                         rho_diff_min: float,
                         oversample: int = 4):
    """Sample (i_hi, i_lo) indices into rho_flat with ρ(hi) - ρ(lo) >= rho_diff_min.
    Returns (ia, ib) where rho_flat[ia] > rho_flat[ib]. If insufficient valid pairs,
    returns whatever's available (may be < n_pairs).
    """
    if clu2_idx.numel() < 20:
        return None, None
    K = n_pairs * oversample
    rand_a = torch.randint(0, clu2_idx.numel(), (K,), device=clu2_idx.device)
    rand_b = torch.randint(0, clu2_idx.numel(), (K,), device=clu2_idx.device)
    ia = clu2_idx[rand_a]
    ib = clu2_idx[rand_b]
    ra = rho_flat[ia]
    rb = rho_flat[ib]
    # Swap so ia is the higher-ρ index
    swap = ra < rb
    ia, ib = torch.where(swap, ib, ia), torch.where(swap, ia, ib)
    ra = rho_flat[ia]; rb = rho_flat[ib]
    valid = (ra - rb) >= rho_diff_min
    ia = ia[valid]; ib = ib[valid]
    if ia.numel() == 0:
        return None, None
    if ia.numel() > n_pairs:
        ia = ia[:n_pairs]
        ib = ib[:n_pairs]
    return ia, ib


def compute_cpr_loss(
    fg_logits: torch.Tensor,
    clu: torch.Tensor,
    stain_h: torch.Tensor,
    stain_e: torch.Tensor,
    epoch: int,
    n_pairs: int = 64,
    rho_diff_min: float = 0.3,
    margin_init: float = 0.05,
    margin_final: float = 0.25,
    margin_warmup_epochs: int = 40,
    return_aux: bool = False,
):
    """Cross-Pixel Polarity Ranking loss on clu==2.

    Returns (loss_scalar, aux_dict).
    aux_dict contains 'cpr_pair_acc' (fraction of pairs where σ(fg)_hi > σ(fg)_lo)
    and 'cpr_n_pairs_used' (total pairs across batch).
    """
    B, _, H, W = fg_logits.shape
    device = fg_logits.device
    dtype = fg_logits.dtype

    # Per-image rho
    h_n = _per_image_norm01(stain_h.to(dtype))  # (B, 1, H, W)
    e_n = _per_image_norm01(stain_e.to(dtype))
    rho = (h_n - e_n) / (h_n + e_n + EPS)  # in [-1, 1]

    fg_flat_all = fg_logits.view(B, -1)
    rho_flat_all = rho.view(B, -1)
    clu_flat_all = clu.view(B, -1)

    margin = _cpr_margin(epoch, margin_init, margin_final, margin_warmup_epochs)
    per_image_losses = []
    pair_acc_sum = 0
    pair_count = 0
    for b in range(B):
        clu2_idx = (clu_flat_all[b] == 2).nonzero(as_tuple=True)[0]
        ia, ib = _sample_pair_indices(rho_flat_all[b], clu2_idx, n_pairs, rho_diff_min)
        if ia is None:
            continue
        sig_hi = torch.sigmoid(fg_flat_all[b].index_select(0, ia))
        sig_lo = torch.sigmoid(fg_flat_all[b].index_select(0, ib))
        per_pair = F.relu(margin - (sig_hi - sig_lo))
        per_image_losses.append(per_pair.mean())
        with torch.no_grad():
            pair_acc_sum += int((sig_hi > sig_lo).sum().item())
            pair_count += int(ia.numel())

    if not per_image_losses:
        loss = fg_logits.new_tensor(0.0)
    else:
        loss = torch.stack(per_image_losses).mean()

    if return_aux:
        aux = {
            "cpr_margin": float(margin),
            "cpr_pair_acc": float(pair_acc_sum / max(1, pair_count)),
            "cpr_n_pairs_used": int(pair_count),
        }
        return loss, aux
    return loss
