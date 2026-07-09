"""B2.5 encoder: LoRA-on-UNI + Hematoxylin-Conditioned Attention fusion.

Replaces B2's generic CrossAttnBlock with HCABlock — a cross-attention whose
attention scores are biased by hematoxylin density. This grounds the foundation
model's attention pathway in the canonical nucleus marker (H stain → DNA),
rather than a generic ViT-Adapter mechanism.

Architecture (diff vs B2 generic):
    Same encoder topology.
    Same dual-input H + RGB.
    Same LoRA-on-UNI-qkv.
    Same downstream decoder + flow/fg/center heads.
    ONLY DIFFERENCE: the cross-attn block between UNI tokens and R34 s2 is
                     replaced with an HCABlock that consumes an additional
                     hematoxylin-density map computed from the input H channel.

The H density map is derived in the forward pass as follows:
  1) Input x["image"] is H-replicated 3-channel, ImageNet-normalized.
  2) De-normalize: H_raw = x_h * imagenet_std + imagenet_mean, then mean over channels.
     This recovers the original H stain intensity ∈ [~0, ~1] (per the
     stain_h_channel() routine in dataset.py).
  3) Downsample to /8 via avg_pool2d to match R34 s2 spatial grid.
  4) Pass to HCABlock as `h_at_kv_scale`.

Trainable additions vs B1: HCABlock has the same param count as CrossAttnBlock
plus one extra scalar (h_bias_scale). ~854K params + 1 scalar.

Hypothesis: hematoxylin density is the strongest single-channel nucleus prior
in H&E; biasing the FM attention with it should outperform the generic
ViT-Adapter fusion (B2) and provide a clean nuclei-specific contribution
story for the paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from chromaflow.uni_encoder_hybrid import NISFlowNetUNIHybridH, potential_to_flow
from chromaflow.lora import add_lora_to_uni, count_lora_params
from chromaflow.vit_adapter import HCABlock


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class NISFlowNetUNIHybridHLoRAHCA(NISFlowNetUNIHybridH):
    """B2.5 = B1 LoRA + HCA cross-attn fusion (UNI tokens enriched by R34 s2,
    biased by hematoxylin density)."""

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
        )

        # === LoRA on UNI qkv ===
        n_wrapped = add_lora_to_uni(
            self.uni.backbone,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
        if n_wrapped == 0:
            raise RuntimeError("LoRA wrapping found no qkv Linears in UNI backbone")
        self.uni.frozen = False

        # === Hematoxylin-Conditioned Attention ===
        self.hca = HCABlock(
            dim_q=1024,
            dim_kv=128,
            hidden_dim=hca_hidden_dim,
            num_heads=hca_num_heads,
            dropout=hca_dropout,
            gamma_init=hca_gamma_init,
            h_bias_init=hca_h_bias_init,
        )

        # ImageNet de-normalization buffers (for recovering raw H density from x_h).
        self.register_buffer(
            "imagenet_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "imagenet_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

        n_lora = count_lora_params(self.uni.backbone)
        n_hca = sum(p.numel() for p in self.hca.parameters() if p.requires_grad)
        print(
            f"[B2.5 LoRA+HCA] wrapped {n_wrapped} UNI qkv layers; "
            f"LoRA params={n_lora:,}; HCA params={n_hca:,}; "
            f"HCA cfg: hidden={hca_hidden_dim} heads={hca_num_heads} "
            f"gamma_init={hca_gamma_init} h_bias_init={hca_h_bias_init}",
            flush=True,
        )

    def train(self, mode: bool = True):
        return nn.Module.train(self, mode)

    def _compute_h_density_at_8(self, x_h: torch.Tensor) -> torch.Tensor:
        """Recover raw hematoxylin density from x_h, downsample to /8 grid.

        x_h is H-repeated 3-channel ImageNet-normalized (shape B,3,H,W).
        Returns (B, 1, H/8, W/8) in [~0, ~1] (approximately — per-image
        normalization is applied inside HCABlock).
        """
        # De-ImageNet-normalize and average over 3 (replicated) channels.
        h_raw = (x_h * self.imagenet_std + self.imagenet_mean).mean(dim=1, keepdim=True)
        # Downsample to /8 via avg_pool.
        return F.avg_pool2d(h_raw, kernel_size=8, stride=8)

    def forward(self, x):
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

        # Hematoxylin density at /8 (matches s2's spatial grid).
        h_at_8 = self._compute_h_density_at_8(x_h)  # (B, 1, H/8, W/8)

        # UNI forward → tokens at /16.
        s3 = self.uni(x_rgb)  # (B, 1024, H/16, W/16)

        # HCA: UNI tokens (q) attend R34 s2 (k,v) with hematoxylin density bias.
        B, C_u, Hp, Wp = s3.shape
        s3_tokens = s3.flatten(2).transpose(1, 2).contiguous()  # (B, Hp*Wp, 1024)
        s3_enriched_tokens = self.hca(s3_tokens, s2, h_at_8)
        s3 = s3_enriched_tokens.transpose(1, 2).contiguous().view(B, C_u, Hp, Wp)

        # Standard decoder path (unchanged from B1/B2).
        feat = self.proj_uni(s3)
        feat = self.up3(feat, s2)
        feat = self.up2(feat, s1)
        feat = self.up1(feat, s0)
        feat = self.final_up(feat)
        if feat.shape[-2:] != (H, W):
            feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)

        fg_logits = self.fg_head(feat)
        centerness_logits = self.centerness_head(feat)
        boundary_logits = self.boundary_head(feat)
        outputs = {
            "fg_logits": fg_logits,
            "centerness_logits": centerness_logits,
            "boundary_logits": boundary_logits,
            "feat": feat,
            "b25_hca_gamma": self.hca.gamma.detach(),
            "b25_hca_h_bias_scale": self.hca.h_bias_scale.detach(),
        }
        if self.flow_param == "direct":
            outputs["flow"] = torch.tanh(self.flow_head(feat))
        else:
            phi = self.potential_head(feat)
            outputs["phi"] = phi
            outputs["flow"] = potential_to_flow(phi)
        return outputs


__all__ = ["NISFlowNetUNIHybridHLoRAHCA"]
