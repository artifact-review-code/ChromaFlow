"""Exponential Moving Average of model parameters.

Used to smooth optimizer-step-level oscillation in dense-prediction tasks where
the test set is small (high inference variance). Standard SOTA recipe — Cellpose,
CellViT, HoVer-Net, all major nuclei segmentation works use EMA at inference.

Typical decay = 0.999 for ~150-epoch training runs with bs=32. For smaller
schedules (e.g. ep ≤ 50) use decay = 0.99; for very long runs (ep ≥ 500) use
decay = 0.9999.

Minimal-invasive: only trainable params are tracked. Frozen UNI base weights
(requires_grad=False) are NOT shadowed (saves ~600MB GPU memory).

Use pattern in train.py:
    ema = EMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    # ... after each optimizer.step() ...
    if ema is not None:
        ema.update(model)
    # ... before eval / ckpt save ...
    if ema is not None:
        backup = ema.apply_to(model)
    # ... eval and ckpt save use EMA weights ...
    if ema is not None:
        ema.restore(model, backup)
"""
from __future__ import annotations

import copy
from typing import Dict

import torch
import torch.nn as nn


class EMA:
    """Maintain an exponential moving average of a model's trainable params."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()
        n = sum(p.numel() for p in self.shadow.values())
        print(
            f"[EMA] tracking {len(self.shadow)} trainable params "
            f"({n:,} floats), decay={decay}",
            flush=True,
        )

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Call this AFTER optimizer.step()."""
        d = self.decay
        one_minus_d = 1.0 - d
        for name, param in model.named_parameters():
            if name in self.shadow:
                # In-place: shadow = d*shadow + (1-d)*param
                self.shadow[name].mul_(d).add_(param.detach(), alpha=one_minus_d)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Swap model's trainable weights with EMA shadow weights.

        Returns a backup dict so the caller can restore() later.
        """
        backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name])
        return backup

    @torch.no_grad()
    def restore(self, model: nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """For ckpt save."""
        return {k: v.cpu().clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        for k, v in state.items():
            if k in self.shadow:
                self.shadow[k].copy_(v.to(self.shadow[k].device))


__all__ = ["EMA"]
