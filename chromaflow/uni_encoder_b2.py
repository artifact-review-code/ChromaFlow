"""B2 encoder: B1 (UNI+LoRA) + ViT-Adapter style cross-attn fusion.

Architecture vs B1:
    B1:  RGB → UNI(LoRA) → tokens at /16 → proj 1024→256 → decoder
              H → R34 shallow → s0/s1/s2 → decoder skip
         (UNI and R34 only meet at decoder skip concat)

    B2:  RGB → UNI(LoRA) → tokens at /16
                                ↓
                       CrossAttnBlock(q=UNI tokens, kv=R34 s2 @ /8)
                                ↓
              enriched UNI tokens at /16 → proj 1024→256 → decoder
              H → R34 shallow → s0/s1/s2 → decoder skip (unchanged)

The cross-attention block is a single block with:
  - q dim 1024 (UNI), kv dim 128 (R34 s2), hidden 256, 8 heads
  - gamma=0.1 init scalar so block is mostly identity at start
  - out_proj init std=0.02 (NOT zero — avoid dead-gradient trap)

Hypothesis (Plan B2): if simple /16 concat fusion is the bottleneck (R34
boundary info never reaches the /16 UNI token representation), enriching UNI
tokens via cross-attention should let UNI tokens absorb high-frequency H
channel info before the decoder gets them. Expected: MO test AJI ≥ 0.600
(B1 was 0.5789 under S4_drop decode).

Trainable additions vs B1: ~1.3M params (the CrossAttnBlock).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from chromaflow.uni_encoder_hybrid import NISFlowNetUNIHybridH, potential_to_flow
from chromaflow.lora import add_lora_to_uni, count_lora_params
from chromaflow.vit_adapter import CrossAttnBlock


class NISFlowNetUNIHybridHLoRAAdapter(NISFlowNetUNIHybridH):
    """B2 = B1 LoRA + one cross-attn block fusing UNI tokens with R34 s2.

    Inherits R34 stem/layer1/layer2, UNI encoder, proj_uni, decoder, and heads
    from NISFlowNetUNIHybridH. We add LoRA to UNI's qkv (same as B1) and a
    single CrossAttnBlock applied to UNI tokens before they enter proj_uni.
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
        adapter_hidden_dim: int = 256,
        adapter_num_heads: int = 8,
        adapter_dropout: float = 0.1,
        adapter_gamma_init: float = 0.1,
    ) -> None:
        super().__init__(
            uni_path=uni_path,
            flow_param=flow_param,
            img_size=img_size,
            pretrained_r34=pretrained_r34,
        )

        # === LoRA on UNI qkv (identical to B1) ===
        n_wrapped = add_lora_to_uni(
            self.uni.backbone,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
        if n_wrapped == 0:
            raise RuntimeError("LoRA wrapping found no qkv Linears in UNI backbone")
        self.uni.frozen = False  # allow grad flow through LoRA path

        # === New: ViT-Adapter style cross-attn between UNI tokens (q) and R34 s2 (kv) ===
        # UNI embed_dim = 1024; R34 layer2 output channels = 128
        self.lora_adapter = CrossAttnBlock(
            dim_q=1024,
            dim_kv=128,
            hidden_dim=adapter_hidden_dim,
            num_heads=adapter_num_heads,
            dropout=adapter_dropout,
            gamma_init=adapter_gamma_init,
        )

        n_lora = count_lora_params(self.uni.backbone)
        n_adapter = sum(p.numel() for p in self.lora_adapter.parameters() if p.requires_grad)
        print(
            f"[B2 LoRA+Adapter] wrapped {n_wrapped} UNI qkv layers; "
            f"LoRA params={n_lora:,}; CrossAttn adapter params={n_adapter:,}; "
            f"adapter cfg: hidden={adapter_hidden_dim} heads={adapter_num_heads} "
            f"gamma_init={adapter_gamma_init}",
            flush=True,
        )

    def train(self, mode: bool = True):
        return nn.Module.train(self, mode)

    def forward(self, x):
        # Dual-input dispatch (identical to NISFlowNetUNIHybridH)
        if isinstance(x, dict):
            x_h = x["image"]
            x_rgb = x.get("image_rgb", x["image"])
        else:
            x_h = x
            x_rgb = x

        H, W = x_h.shape[-2:]
        s0 = self.r34_stem(x_h)
        s1 = self.r34_layer1(self.r34_maxpool(s0))
        s2 = self.r34_layer2(s1)  # (B, 128, H/8, W/8)

        # UNI forward; shape (B, 1024, H/16, W/16)
        s3 = self.uni(x_rgb)

        # === B2-specific: cross-attn enrichment of UNI tokens by R34 s2 ===
        B, C_u, Hp, Wp = s3.shape
        s3_tokens = s3.flatten(2).transpose(1, 2).contiguous()  # (B, Hp*Wp, 1024)
        s3_enriched_tokens = self.lora_adapter(s3_tokens, s2)   # (B, Hp*Wp, 1024)
        s3 = s3_enriched_tokens.transpose(1, 2).contiguous().view(B, C_u, Hp, Wp)
        # =============================================================

        feat = self.proj_uni(s3)
        feat = self.up3(feat, s2)
        feat = self.up2(feat, s1)
        feat = self.up1(feat, s0)
        feat = self.final_up(feat)
        if feat.shape[-2:] != (H, W):
            feat = nn.functional.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)

        fg_logits = self.fg_head(feat)
        centerness_logits = self.centerness_head(feat)
        boundary_logits = self.boundary_head(feat)
        outputs = {
            "fg_logits": fg_logits,
            "centerness_logits": centerness_logits,
            "boundary_logits": boundary_logits,
            "feat": feat,
            "b2_adapter_gamma": self.lora_adapter.gamma.detach(),  # diagnostic
        }
        if self.flow_param == "direct":
            outputs["flow"] = torch.tanh(self.flow_head(feat))
        else:
            phi = self.potential_head(feat)
            outputs["phi"] = phi
            outputs["flow"] = potential_to_flow(phi)
        return outputs


__all__ = ["NISFlowNetUNIHybridHLoRAAdapter"]
