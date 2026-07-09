"""B2.6 encoder: LoRA on UNI + SRMA (Stain-Routed Mutual Attention).

Bidirectional cross-modal fusion between the R34 H-branch and the UNI
RGB-branch, routed by complementary stain channels:
  * Forward UNI ← R34, biased by hematoxylin (H) density.
  * Backward R34 ← UNI, biased by eosin (E) density.

Both branches are enriched and fed into the decoder. The decoder topology
stays identical to NISFlowNetUNIHybridH: only s2 (R34 layer-2 output) is
replaced by the SRMA-enriched s2'; s0, s1 stay raw R34; the proj_uni input is
the SRMA-enriched UNI token map.

H and E density maps are computed on-the-fly inside forward() from x_rgb via
the Ruifrok-Johnston deconvolution matrix (same as HCAv2; see
uni_encoder_b25v2.rgb_to_hed_torch).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from chromaflow.uni_encoder_hybrid import NISFlowNetUNIHybridH, potential_to_flow
from chromaflow.lora import add_lora_to_uni, count_lora_params
from chromaflow.srma_block import SRMABlock
from chromaflow.uni_encoder_b25v2 import rgb_to_hed_torch, _IMAGENET_MEAN, _IMAGENET_STD, _HED_FROM_RGB


class NISFlowNetUNIHybridHLoRASRMA(NISFlowNetUNIHybridH):
    """LoRA on UNI + SRMA bidirectional cross-modal fusion."""

    def __init__(
        self,
        uni_path: str,
        flow_param: str = "direct",
        img_size: int = 224,
        pretrained_r34: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        srma_hidden_dim: int = 256,
        srma_num_heads: int = 8,
        srma_dropout: float = 0.1,
        srma_gamma_init: float = 0.1,
        srma_h_bias_init: float = 1.0,
        srma_e_bias_init: float = 1.0,
    ) -> None:
        super().__init__(
            uni_path=uni_path,
            flow_param=flow_param,
            img_size=img_size,
            pretrained_r34=pretrained_r34,
        )

        # LoRA on UNI qkv (same as B1/B2.5).
        n_wrapped = add_lora_to_uni(
            self.uni.backbone, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout
        )
        if n_wrapped == 0:
            raise RuntimeError("LoRA wrapping found no qkv Linears in UNI backbone")
        self.uni.frozen = False

        # SRMA bidirectional cross-attn block.
        self.srma = SRMABlock(
            uni_dim=1024,
            cnn_dim=128,
            hidden_dim=srma_hidden_dim,
            num_heads=srma_num_heads,
            dropout=srma_dropout,
            gamma_init=srma_gamma_init,
            h_bias_init=srma_h_bias_init,
            e_bias_init=srma_e_bias_init,
        )

        self.register_buffer(
            "imagenet_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "imagenet_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "hed_matrix", torch.tensor(_HED_FROM_RGB), persistent=False
        )

        n_lora = count_lora_params(self.uni.backbone)
        n_srma = sum(p.numel() for p in self.srma.parameters() if p.requires_grad)
        print(
            f"[SRMA] wrapped {n_wrapped} UNI qkv; LoRA={n_lora:,}; SRMA params={n_srma:,}; "
            f"H_init={srma_h_bias_init} E_init={srma_e_bias_init}",
            flush=True,
        )

    def train(self, mode: bool = True):
        return nn.Module.train(self, mode)

    def _compute_he_at_8(self, x_rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hed = rgb_to_hed_torch(x_rgb, self.imagenet_mean, self.imagenet_std, self.hed_matrix)
        h_full = hed[:, 0:1]
        e_full = hed[:, 1:2]
        h_at_8 = F.avg_pool2d(h_full, kernel_size=8, stride=8)
        e_at_8 = F.avg_pool2d(e_full, kernel_size=8, stride=8)
        return h_at_8, e_at_8

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

        h_at_8, e_at_8 = self._compute_he_at_8(x_rgb)

        s3 = self.uni(x_rgb)  # (B, 1024, H/16, W/16)
        B, C_u, Hp, Wp = s3.shape

        # SRMA bidirectional cross-attn: both UNI tokens AND R34 s2 get enriched.
        uni_tokens = s3.flatten(2).transpose(1, 2).contiguous()  # (B, Hp*Wp, 1024)
        uni_enriched, s2_enriched = self.srma(uni_tokens, s2, h_at_8, e_at_8)
        s3 = uni_enriched.transpose(1, 2).contiguous().view(B, C_u, Hp, Wp)
        s2 = s2_enriched  # enriched R34 feature flows into decoder

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
            "srma_h_bias": self.srma.forward_xa.bias_scale.detach(),
            "srma_e_bias": self.srma.backward_xa.bias_scale.detach(),
            "srma_gamma_f": self.srma.forward_xa.gamma.detach(),
            "srma_gamma_b": self.srma.backward_xa.gamma.detach(),
        }
        if self.flow_param == "direct":
            outputs["flow"] = torch.tanh(self.flow_head(feat))
        else:
            phi = self.potential_head(feat)
            outputs["phi"] = phi
            outputs["flow"] = potential_to_flow(phi)
        return outputs


__all__ = ["NISFlowNetUNIHybridHLoRASRMA"]
