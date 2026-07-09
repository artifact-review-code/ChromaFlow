"""Test-time augmentation wrapper for nuclei segmentation models.

Wraps a base model that outputs a dict with keys {fg_logits, centerness_logits,
flow, boundary_logits, feat, ...}. For each TTA mode, applies a spatial flip to
the input, runs the base model, then inverts the flip on the output (including
sign-flipping the flow dx/dy components as appropriate). Averages over all
modes.

Modes supported: "hflip", "vflip", "hvflip" (180° rotation). Identity is always
included. Total of N+1 forward passes per input where N = number of modes.

Use pattern in eval.py / decode_variants.py:
    if args.tta:
        from chromaflow.tta import TTAModelWrapper
        model = TTAModelWrapper(model, modes=args.tta.split(","))

Expected lift: +0.5 to +1 pp AJI on small test sets (≤ 20 imgs) with high
per-image variance. Standard recipe — Cellpose, HoVer-Net, CellViT all use TTA
at inference.

Channel convention for flow: model output (B, 2, H, W) with channel 0 = dx,
channel 1 = dy. This matches model.py / uni_encoder_hybrid.py.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

import torch
import torch.nn as nn


_VALID_MODES = {"none", "hflip", "vflip", "hvflip"}


def _flip_tensor(t: torch.Tensor, mode: str) -> torch.Tensor:
    """Apply 2D spatial flip to a (..., H, W) tensor."""
    if mode == "none":
        return t
    if mode == "hflip":
        return torch.flip(t, dims=[-1])
    if mode == "vflip":
        return torch.flip(t, dims=[-2])
    if mode == "hvflip":
        return torch.flip(t, dims=[-2, -1])
    raise ValueError(f"unknown TTA mode: {mode}")


def _apply_input(x: Any, mode: str) -> Any:
    """Flip an input which may be a Tensor or a dict of Tensors."""
    if isinstance(x, dict):
        return {k: _flip_tensor(v, mode) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return _flip_tensor(x, mode)
    raise TypeError(f"unexpected input type for TTA: {type(x)}")


def _invert_output_dict(out: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Map a model output dict back to the original-image coordinate frame.

    For 4D spatial tensors (B, C, H, W): unflip; if the tensor is flow
    (key == 'flow'), also reverse the sign of dx (channel 0) when hflip
    happened and dy (channel 1) when vflip happened.
    """
    if mode == "none":
        return out
    new: Dict[str, Any] = {}
    for k, v in out.items():
        if not isinstance(v, torch.Tensor):
            new[k] = v
            continue
        if v.ndim < 4:
            # scalar diagnostic (e.g. b2_adapter_gamma): keep as-is
            new[k] = v
            continue
        v_back = _flip_tensor(v, mode)
        if k == "flow":
            v_back = v_back.clone()
            if mode in ("hflip", "hvflip"):
                v_back[:, 0] = -v_back[:, 0]
            if mode in ("vflip", "hvflip"):
                v_back[:, 1] = -v_back[:, 1]
        new[k] = v_back
    return new


def _merge_outputs(outputs_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Element-wise mean over a list of output dicts.

    Non-spatial entries (scalars, None, strings) are taken from the first dict.
    """
    keys = outputs_list[0].keys()
    merged: Dict[str, Any] = {}
    for k in keys:
        vs = [o[k] for o in outputs_list if k in o]
        v0 = vs[0]
        if not isinstance(v0, torch.Tensor):
            merged[k] = v0
            continue
        if v0.ndim < 4:
            merged[k] = v0
            continue
        merged[k] = torch.stack(vs, dim=0).mean(dim=0)
    return merged


def parse_modes(spec: str) -> List[str]:
    """Parse a comma-separated list like 'hflip,vflip,hvflip'. Identity always added."""
    if not spec:
        return ["none"]
    modes = [m.strip().lower() for m in spec.split(",") if m.strip()]
    out = ["none"]
    for m in modes:
        if m in _VALID_MODES and m != "none" and m not in out:
            out.append(m)
        elif m not in _VALID_MODES:
            raise ValueError(f"invalid TTA mode '{m}'; valid: {sorted(_VALID_MODES)}")
    return out


class TTAModelWrapper(nn.Module):
    """Wrap a model so that forward(x) runs TTA-averaged inference."""

    def __init__(self, base: nn.Module, modes: Iterable[str] = ("hflip", "vflip", "hvflip")) -> None:
        super().__init__()
        self.base = base
        modes_list = list(modes)
        if "none" not in modes_list:
            modes_list = ["none"] + [m for m in modes_list if m != "none"]
        for m in modes_list:
            if m not in _VALID_MODES:
                raise ValueError(f"invalid TTA mode '{m}'")
        self.modes: List[str] = modes_list
        print(f"[TTA] wrapping model with modes={self.modes}", flush=True)

    def forward(self, x: Any) -> Dict[str, Any]:
        outputs_list: List[Dict[str, Any]] = []
        for mode in self.modes:
            x_t = _apply_input(x, mode)
            out = self.base(x_t)
            out_back = _invert_output_dict(out, mode)
            outputs_list.append(out_back)
        return _merge_outputs(outputs_list)


__all__ = ["TTAModelWrapper", "parse_modes"]
