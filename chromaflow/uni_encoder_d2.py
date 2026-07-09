"""D2 encoder: HCA + R34 reads RGB (NOT H).

Identical to B2.5 (LoRA + HCA) except the ResNet-34 spatial branch consumes
RGB instead of H-replicated input. This makes the H signal flow into the
fused representation **only** through HCA's attention bias.

Tests hypothesis H1: HCA underperformed because R34 already had H in the input,
making HCA's H-bias redundant. By removing H from R34, HCA's H-bias becomes
the *only* H pathway. If H1 is true, D2 should beat noHCA-RGB-only baseline
by a large margin.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from chromaflow.uni_encoder_b25 import NISFlowNetUNIHybridHLoRAHCA


class NISFlowNetUNIHybridHLoRAHCA_R34RGB(NISFlowNetUNIHybridHLoRAHCA):
    """D2: B2.5 HCA but R34 takes RGB."""

    def forward(self, x):
        if isinstance(x, dict):
            x_h = x["image"]
            x_rgb = x.get("image_rgb", x["image"])
        else:
            x_h = x
            x_rgb = x

        H, W = x_h.shape[-2:]
        # D2 change: feed RGB (not H) to the R34 branch.
        s0 = self.r34_stem(x_rgb)
        s1 = self.r34_layer1(self.r34_maxpool(s0))
        s2 = self.r34_layer2(s1)  # (B, 128, H/8, W/8)

        # H density at /8 still recovered from x_h (UNI's H-replicated input).
        h_at_8 = self._compute_h_density_at_8(x_h)

        s3 = self.uni(x_rgb)  # (B, 1024, H/16, W/16)
        B, C_u, Hp, Wp = s3.shape
        s3_tokens = s3.flatten(2).transpose(1, 2).contiguous()
        s3_enriched_tokens = self.hca(s3_tokens, s2, h_at_8)
        s3 = s3_enriched_tokens.transpose(1, 2).contiguous().view(B, C_u, Hp, Wp)

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
            "d2_hca_gamma": self.hca.gamma.detach(),
            "d2_hca_h_bias_scale": self.hca.h_bias_scale.detach(),
        }
        if self.flow_param == "direct":
            outputs["flow"] = torch.tanh(self.flow_head(feat))
        else:
            from chromaflow.uni_encoder_hybrid import potential_to_flow
            phi = self.potential_head(feat)
            outputs["phi"] = phi
            outputs["flow"] = potential_to_flow(phi)
        return outputs


__all__ = ["NISFlowNetUNIHybridHLoRAHCA_R34RGB"]
