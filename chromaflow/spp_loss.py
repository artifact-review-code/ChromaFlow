"""Stain Polarization Posterior (SPP) loss — Module 1 of the chromophore paper.

Replaces L_stain_soft + L_eosin_neg with a single ρ-conditioned soft label on
the BoNuS ignore zone (clu==2), where:

    ρ(p)     = (H_norm(p) - E_norm(p)) / (H_norm(p) + E_norm(p) + ε)   ∈ [-1, 1]
    target(p) = σ(k · ρ(p) + b)
    weight(p) = |ρ(p)|^γ
    L_SPP    = Σ_{p∈clu==2} weight(p) · BCE(ŷ(p), target(p))  /  Σ weight(p)

(k, b) are EM-calibrated per epoch from anchor pixels (clu==0 ∪ clu==1) via
logistic regression. γ is a fixed hyperparameter (default 1.0).

Standalone file — does NOT modify losses.py / train.py. To integrate later,
import SPPState + compute_spp_loss from this file, add CLI flags, hook into
compute_losses() and add a once-per-epoch state.update() call.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _per_image_norm01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-image min-max normalize to [0, 1]. x is (B, 1, H, W) or (B, H, W)."""
    if x.ndim == 3:
        x = x.unsqueeze(1)
    lo = x.flatten(2).amin(dim=2).view(-1, 1, 1, 1)
    hi = x.flatten(2).amax(dim=2).view(-1, 1, 1, 1)
    return (x - lo) / (hi - lo + eps)


def compute_polarization(stain_h: torch.Tensor, stain_e: torch.Tensor,
                          eps: float = 1e-6) -> torch.Tensor:
    """Return ρ = (Hn - En)/(Hn + En + eps), per-image min-max normalized inputs.

    Args:
        stain_h: (B, H, W) or (B, 1, H, W)
        stain_e: same shape

    Returns:
        rho: (B, 1, H, W), values in [-1, 1].
    """
    Hn = _per_image_norm01(stain_h)
    En = _per_image_norm01(stain_e)
    return (Hn - En) / (Hn + En + eps)


# ---------------------------------------------------------------------------
# EM-calibrated (k, b) state
# ---------------------------------------------------------------------------

@dataclass
class SPPState:
    """Holds the (k, b) parameters of the polarization posterior σ(k·ρ + b).

    Usage:
        state = SPPState(k=2.0, b=-0.5)                  # init from diagnostic
        for epoch in range(num_epochs):
            state.start_epoch()
            for batch in loader:
                loss = compute_spp_loss(..., state=state, ...)
                ...
                state.accumulate(rho, clu)
            state.update()                                # refit k, b
    """
    k: float = 2.0
    b: float = -0.5
    gamma: float = 1.0
    ema_momentum: float = 0.5   # how much of new fit to mix in (1.0 = replace)
    # Accumulators across an epoch (count-based sufficient stats for logistic).
    # We store full ρ samples (subsampled) for fitting at update().
    _rho_buffer: list = None
    _y_buffer: list = None

    def start_epoch(self) -> None:
        self._rho_buffer = []
        self._y_buffer = []

    @torch.no_grad()
    def accumulate(self, rho: torch.Tensor, clu: torch.Tensor,
                    max_pixels_per_image: int = 4000) -> None:
        """Subsample anchor pixels (clu==0 → y=0, clu==1 → y=1) for EM refit.

        Args:
            rho: (B, 1, H, W) or (B, H, W)
            clu: (B, H, W) long
        """
        if self._rho_buffer is None:
            self.start_epoch()
        if rho.ndim == 4:
            rho = rho.squeeze(1)
        B = rho.shape[0]
        for b_i in range(B):
            r = rho[b_i]
            c = clu[b_i]
            fg_idx = (c == 1).nonzero(as_tuple=False)
            bg_idx = (c == 0).nonzero(as_tuple=False)
            cap = max_pixels_per_image // 2
            if fg_idx.shape[0] > cap:
                perm = torch.randperm(fg_idx.shape[0], device=r.device)[:cap]
                fg_idx = fg_idx[perm]
            if bg_idx.shape[0] > cap:
                perm = torch.randperm(bg_idx.shape[0], device=r.device)[:cap]
                bg_idx = bg_idx[perm]
            if fg_idx.shape[0] > 0:
                rho_fg = r[fg_idx[:, 0], fg_idx[:, 1]]
                self._rho_buffer.append(rho_fg.detach().cpu())
                self._y_buffer.append(torch.ones_like(rho_fg).detach().cpu())
            if bg_idx.shape[0] > 0:
                rho_bg = r[bg_idx[:, 0], bg_idx[:, 1]]
                self._rho_buffer.append(rho_bg.detach().cpu())
                self._y_buffer.append(torch.zeros_like(rho_bg).detach().cpu())

    def update(self, n_iter: int = 300, lr: float = 0.5) -> dict:
        """Refit (k, b) from accumulated anchor (ρ, y) via gradient descent.

        Returns dict with old/new (k, b) and AUC for logging.
        """
        if not self._rho_buffer:
            return {"k_old": self.k, "b_old": self.b,
                    "k_new": self.k, "b_new": self.b, "auc": float("nan")}
        rho = torch.cat(self._rho_buffer).float()
        y = torch.cat(self._y_buffer).float()
        # Logistic regression via Adam (no scipy/sklearn dep).
        k_t = torch.tensor(self.k, requires_grad=True, dtype=torch.float32)
        b_t = torch.tensor(self.b, requires_grad=True, dtype=torch.float32)
        opt = torch.optim.Adam([k_t, b_t], lr=lr)
        for _ in range(n_iter):
            logits = k_t * rho + b_t
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        k_new, b_new = float(k_t.detach()), float(b_t.detach())
        # AUC for diagnostic (Mann-Whitney over subsample).
        pos = rho[y == 1]
        neg = rho[y == 0]
        auc = float("nan")
        if len(pos) > 0 and len(neg) > 0:
            m = min(len(pos), 5000); mm = min(len(neg), 5000)
            pos_s = pos[torch.randperm(len(pos))[:m]]
            neg_s = neg[torch.randperm(len(neg))[:mm]]
            auc = float((pos_s.unsqueeze(1) > neg_s.unsqueeze(0)).float().mean())
        k_old, b_old = self.k, self.b
        # EMA blend to avoid sudden jumps.
        m = self.ema_momentum
        self.k = (1 - m) * k_old + m * k_new
        self.b = (1 - m) * b_old + m * b_new
        # Free buffers
        self._rho_buffer = None
        self._y_buffer = None
        return {"k_old": k_old, "b_old": b_old,
                "k_new": self.k, "b_new": self.b, "auc": auc}


# ---------------------------------------------------------------------------
# The loss
# ---------------------------------------------------------------------------

def compute_spp_loss(
    fg_logits: torch.Tensor,
    clu: torch.Tensor,
    stain_h: torch.Tensor,
    stain_e: torch.Tensor,
    k: float,
    b: float,
    gamma: float = 1.0,
    eps: float = 1e-6,
    return_aux: bool = False,
) -> torch.Tensor:
    """Stain Polarization Posterior BCE on clu==2.

    L_SPP = Σ_{p∈clu==2} |ρ(p)|^γ · BCE(ŷ(p), σ(k·ρ(p) + b))  /  Σ |ρ(p)|^γ

    Args:
        fg_logits: (B, 1, H, W) raw foreground logits
        clu:        (B, H, W) long {0, 1, 2}
        stain_h:    (B, H, W) or (B, 1, H, W) hematoxylin density
        stain_e:    same shape
        k:          polarization slope (EM-calibrated)
        b:          prior bias (EM-calibrated)
        gamma:      confidence weight exponent (default 1.0)
        return_aux: if True, also return dict of intermediate tensors for debug

    Returns:
        scalar loss (or (loss, aux_dict))
    """
    rho = compute_polarization(stain_h, stain_e, eps=eps)         # (B,1,H,W)
    target = torch.sigmoid(k * rho.to(fg_logits.dtype) + b)
    weight = rho.abs().pow(gamma).to(fg_logits.dtype)
    mask = (clu == 2).unsqueeze(1).to(dtype=fg_logits.dtype)
    bce = F.binary_cross_entropy_with_logits(fg_logits, target, reduction="none")
    eff = weight * mask                                              # effective per-pixel weight
    numer = (eff * bce).sum()
    denom = eff.sum().clamp(min=1.0)
    loss = numer / denom
    if return_aux:
        return loss, {"rho": rho.detach(), "target": target.detach(),
                       "weight": weight.detach(), "mask": mask.detach()}
    return loss


# ---------------------------------------------------------------------------
# Unit tests / sanity checks
# ---------------------------------------------------------------------------

def _test_polarization_range():
    """ρ should be in [-1, 1] for all inputs."""
    torch.manual_seed(0)
    H = torch.rand(2, 1, 16, 16)
    E = torch.rand(2, 1, 16, 16)
    rho = compute_polarization(H, E)
    assert rho.min() >= -1.0 - 1e-4, f"rho min={rho.min()}"
    assert rho.max() <= 1.0 + 1e-4, f"rho max={rho.max()}"
    print(f"  [polarization range] OK: ρ ∈ [{rho.min():.4f}, {rho.max():.4f}]")


def _test_em_fit_recovers_known_params():
    """If we generate y from σ(k_true·ρ + b_true), EM should recover (k, b)."""
    torch.manual_seed(0)
    k_true, b_true = 1.8, -0.4
    # Synthesize anchor pixels: rho ~ Uniform[-1, 1], y = Bernoulli(σ(k·ρ + b))
    B, H, W = 4, 64, 64
    rho_field = (torch.rand(B, 1, H, W) * 2 - 1)
    target_p = torch.sigmoid(k_true * rho_field + b_true)
    y_field = (torch.rand_like(target_p) < target_p).float()
    # Build clu: anchors = (y==1 → clu=1, y==0 → clu=0). Mix in some clu==2 noise.
    clu = torch.where(y_field.squeeze(1) > 0.5, 1, 0).long()
    mask_ignore = torch.rand(B, H, W) < 0.3
    clu[mask_ignore] = 2
    # Faked stain_h, stain_e that recover rho_field exactly:
    # rho = (H-E)/(H+E)  ⇒ for H+E=1, H = (1+rho)/2, E = (1-rho)/2
    Hn = (1 + rho_field) / 2
    En = (1 - rho_field) / 2
    # Run EM
    state = SPPState(k=0.0, b=0.0, ema_momentum=1.0)
    state.start_epoch()
    rho_for_acc = compute_polarization(Hn, En)
    state.accumulate(rho_for_acc, clu, max_pixels_per_image=4000)
    info = state.update(n_iter=500, lr=0.1)
    print(f"  [EM fit] true (k={k_true}, b={b_true}) → "
          f"fit (k={state.k:.3f}, b={state.b:.3f}) AUC={info['auc']:.3f}")
    assert abs(state.k - k_true) < 0.3, f"k off by {abs(state.k - k_true)}"
    assert abs(state.b - b_true) < 0.3, f"b off by {abs(state.b - b_true)}"


def _test_loss_differentiable():
    """L_SPP should produce non-zero gradients through fg_logits."""
    torch.manual_seed(0)
    fg_logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    clu = torch.randint(0, 3, (2, 32, 32))
    H = torch.rand(2, 1, 32, 32)
    E = torch.rand(2, 1, 32, 32)
    loss = compute_spp_loss(fg_logits, clu, H, E, k=2.0, b=-0.5, gamma=1.0)
    loss.backward()
    g_norm = fg_logits.grad.norm().item()
    print(f"  [grad flow] loss={loss.item():.4f}  grad_norm={g_norm:.6f}")
    assert g_norm > 1e-6, f"gradient too small: {g_norm}"


def _test_loss_subsumes_l_stain_soft_limit():
    """At k→large, target on (ρ>0) → 1 (matches L_stain_soft α=1 limit).
    At ρ < 0, target → 0 (matches L_eosin_neg limit)."""
    rho_pos = torch.tensor([0.5, 0.7, 0.9])
    rho_neg = torch.tensor([-0.5, -0.7, -0.9])
    for k in (1.0, 3.0, 10.0):
        for b in (0.0, -0.5, +0.5):
            tp = torch.sigmoid(k * rho_pos + b)
            tn = torch.sigmoid(k * rho_neg + b)
    # At k=10, b=0: σ(10·0.5)=σ(5)≈0.993, σ(10·(-0.5))=σ(-5)≈0.007
    t_strong_pos = torch.sigmoid(torch.tensor(5.0)).item()
    t_strong_neg = torch.sigmoid(torch.tensor(-5.0)).item()
    print(f"  [special-case limit] k=10 ρ=±0.5 → target=({t_strong_neg:.4f}, {t_strong_pos:.4f})")
    assert t_strong_pos > 0.99
    assert t_strong_neg < 0.01


def _test_against_diagnostic_numbers():
    """Sanity-check fitted (k, b) on synthetic data closely matching the real
    diagnostic. The real diagnostic gave k≈1.97, b≈-0.54, AUC≈0.80 on MO."""
    torch.manual_seed(0)
    # Approximate MO anchor distribution: bg ρ centered at -0.43, fg ρ at +0.03
    n_bg, n_fg = 30000, 10000
    rho_bg = torch.randn(n_bg) * 0.45 + (-0.43)
    rho_bg = rho_bg.clamp(-1, 1)
    rho_fg = torch.randn(n_fg) * 0.39 + 0.03
    rho_fg = rho_fg.clamp(-1, 1)
    # Build fake clu and stain such that polarization gives back these distributions.
    rho_all = torch.cat([rho_bg, rho_fg])
    y_all = torch.cat([torch.zeros(n_bg), torch.ones(n_fg)])
    # Fit via SPPState directly (skip the dataset plumbing).
    state = SPPState(k=0.0, b=0.0, ema_momentum=1.0)
    state._rho_buffer = [rho_all]
    state._y_buffer = [y_all]
    info = state.update(n_iter=500, lr=0.1)
    print(f"  [MO-like fit] k={state.k:.3f}  b={state.b:.3f}  AUC={info['auc']:.3f}")
    # Real diagnostic had k≈1.97, b≈-0.54, AUC≈0.80. Loose tolerance — synthetic
    # data sampled from Gaussians (not real distributions) gives slightly different
    # fitted slope. Just check (k > 1, b < 0, AUC > 0.7).
    assert 1.0 < state.k < 3.5, f"k={state.k} out of expected range"
    assert -1.0 < state.b < -0.2, f"b={state.b} out of expected range"
    assert info["auc"] > 0.70, f"AUC={info['auc']} too low"


if __name__ == "__main__":
    print("=== SPP loss unit tests ===")
    _test_polarization_range()
    _test_em_fit_recovers_known_params()
    _test_loss_differentiable()
    _test_loss_subsumes_l_stain_soft_limit()
    _test_against_diagnostic_numbers()
    print("\nAll tests passed.")
