"""D8 encoder: D4 backbone + Evidential 2-class fg head.

Replaces the 1-channel fg_head logit with a 2-channel evidential head that
outputs evidence for fg/bg classes. Following Sensoy et al. (NeurIPS 2018):
    e_k = softplus(logit_k)   ∈ [0, ∞)
    α_k = e_k + 1             Dirichlet concentration
    S   = sum_k α_k           Dirichlet strength
    p_k = α_k / S             expected class probability
    u   = K / S = 2 / S       epistemic uncertainty in [0, 1]

For backward-compat (decoders/eval expect 1-ch fg_logits), we expose:
    fg_logits = logit(p_fg)   — the standard logit form of fg probability
    prob_fg, uncertainty, alpha — added to outputs dict for the new losses.

Architecture-wise the only change is fg_head: Conv2d(64→1) → Conv2d(64→2).
Everything else is identical to D4 (NISFlowNetUNIHybridHLoRA).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from chromaflow.uni_encoder_b1 import NISFlowNetUNIHybridHLoRA


class NISFlowNetUNIHybridHLoRAEvUC(NISFlowNetUNIHybridHLoRA):
    """D8 = D4 backbone with 2-channel evidential fg_head."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace fg_head (1 ch) with 2-channel evidential head.
        in_ch = self.fg_head.in_channels
        self.fg_head = nn.Conv2d(in_ch, 2, kernel_size=1)
        # Init biases so initial p_fg ≈ 0.5 and S ≈ 2 (high initial uncertainty).
        nn.init.zeros_(self.fg_head.bias)
        nn.init.zeros_(self.fg_head.weight)
        # small std to break symmetry
        nn.init.normal_(self.fg_head.weight, std=0.01)
        print(f"[D8 LoRA+EvUC] evidential fg_head: 2 channels (e_fg, e_bg)", flush=True)

    def forward(self, x):
        # Parent's forward sets fg_logits = self.fg_head(feat). With our new
        # 2-channel head, that produces (B, 2, H, W). We post-process here.
        outputs = super().forward(x)
        fg_2ch = outputs["fg_logits"]  # (B, 2, H, W)
        # Evidence
        evidence = F.softplus(fg_2ch)
        alpha = evidence + 1.0  # (B, 2, H, W), α ≥ 1
        S = alpha.sum(dim=1, keepdim=True)  # (B, 1, H, W), Dirichlet strength
        alpha_fg = alpha[:, 0:1]
        alpha_bg = alpha[:, 1:2]
        prob_fg = alpha_fg / S
        # Uncertainty: K / S = 2 / S ∈ (0, 1]
        uncertainty = 2.0 / S
        # Backward-compat fg_logits = logit(p_fg)
        eps = 1e-6
        p_clamped = prob_fg.clamp(eps, 1.0 - eps)
        fg_logits_back = torch.log(p_clamped) - torch.log(1.0 - p_clamped)
        outputs["fg_logits"] = fg_logits_back   # (B, 1, H, W)
        outputs["prob_fg"] = prob_fg
        outputs["uncertainty"] = uncertainty
        outputs["alpha"] = alpha                # (B, 2, H, W) for evidential loss
        return outputs


__all__ = ["NISFlowNetUNIHybridHLoRAEvUC"]
