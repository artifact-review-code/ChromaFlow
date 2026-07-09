"""B5 encoder: HCA + Hematoxylin-aware Stain-weighted Distance Regression (HSDR).

Same encoder as B2.5 (LoRA on UNI + Hematoxylin-Conditioned Attention) but the
output representation is changed from Cellpose flow to HoVer-Net-style hv-map:
two channels of per-instance normalized signed distance from center.

Why: AJI is bounded by SQ (segmentation quality, mean IoU of matched
instances), and our SQ lags BoNuS by ~1.7 pp on MoNuSeg. The hv-map output
gives sharper boundary localization via Sobel + watershed than Cellpose flow
integration. Combined with our HCA encoder, we expect to close the MO AJI gap
to BoNuS.

The "hv_map" output key is *additive* — we keep the parent's "flow" output
unchanged so existing losses and Cellpose dynamics continue to work as
auxiliary supervision and as a fallback inference path. Downstream code can
opt into hv_map by name.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from chromaflow.uni_encoder_b25 import NISFlowNetUNIHybridHLoRAHCA


class NISFlowNetUNIHybridHLoRAHCAHV(NISFlowNetUNIHybridHLoRAHCA):
    """B5 = B2.5 HCA encoder + hv-map output head (2ch tanh).

    Note: we keep the parent's flow_head as auxiliary supervision (low weight)
    so the model is not destabilized by replacing the main output. Practically
    the hv_head dominates and is used for inference.
    """

    def __init__(
        self,
        uni_path: str,
        flow_param: str = "direct",
        img_size: int = 224,
        pretrained_r34: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        hca_hidden_dim: int = 256,
        hca_num_heads: int = 8,
        hca_dropout: float = 0.1,
        hca_gamma_init: float = 0.1,
        hca_h_bias_init: float = 1.0,
    ) -> None:
        super().__init__(
            uni_path=uni_path,
            flow_param=flow_param,
            img_size=img_size,
            pretrained_r34=pretrained_r34,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            hca_hidden_dim=hca_hidden_dim,
            hca_num_heads=hca_num_heads,
            hca_dropout=hca_dropout,
            hca_gamma_init=hca_gamma_init,
            hca_h_bias_init=hca_h_bias_init,
        )
        # New hv-map output head: same shape (2ch) but interpreted as
        # (h_dist, v_dist) ∈ [-1, 1] instead of Cellpose flow (dx, dy).
        self.hv_head = nn.Conv2d(64, 2, kernel_size=1)
        print(f"[B5 HSDR] added hv_head: 64ch -> 2ch tanh (h_dist, v_dist).", flush=True)

    def forward(self, x):
        out = super().forward(x)
        # New: hv_map output. Same final feat map as flow_head.
        out["hv_map"] = torch.tanh(self.hv_head(out["feat"]))
        return out


__all__ = ["NISFlowNetUNIHybridHLoRAHCAHV"]
