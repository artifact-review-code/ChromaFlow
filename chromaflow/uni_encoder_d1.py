"""D1 encoder: LoRA-on-UNI + Stain-Differential Cross-Attention (SDCA) fusion.

Difference vs B2.5 (HCA):
    HCA uses log(H_norm + eps) as attention bias — ALWAYS non-positive, biased
    toward high-H regions. But H bleed in eosin-heavy regions still produces
    moderate H_norm > 0, so HCA can attend there too (failure mode of HCA).

    SDCA uses the SIGNED H-E differential D = H_norm - E_norm in [-1, 1]:
      D > 0  (high H, low E)  -> nucleus prior, attend MORE
      D < 0  (high E, low H)  -> stroma prior, attend LESS
      D ~ 0                   -> boundary/ambiguous, neutral

    This explicitly punishes attention to stroma, fixing HCA's main failure.
    Mirrors the supervision-side dual-stain logic (DS-MPS) at the architecture
    level — "Dual-Stain throughout".

Computation:
    1) E channel derived inside encoder via skimage-style rgb2hed on x_rgb.
    2) H channel derived from x_h (already H-replicated, ImageNet-normalized).
    3) Both per-image min-max normalized.
    4) D = H_norm - E_norm, downsampled to /8 via avg_pool.
    5) Passed to SDCABlock as `d_at_kv_scale`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from chromaflow.uni_encoder_hybrid import NISFlowNetUNIHybridH, potential_to_flow
from chromaflow.lora import add_lora_to_uni, count_lora_params
from chromaflow.vit_adapter import SDCABlock


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# skimage 0.22 hed_from_rgb matrix (rgb -> H,E,D stains).
# Apply: stains = log_rgb @ HED_FROM_RGB
_HED_FROM_RGB = (
    ( 1.87798274, -0.06980835, -0.06980835),
    (-1.00767639,  1.13660261, -0.4609135 ),
    (-0.55611412, -0.13551866,  1.57007844),
)
_LOG10_255 = 2.4082399653118496  # math.log10(255.0)


class NISFlowNetUNIHybridHLoRASDCA(NISFlowNetUNIHybridH):
    """D1 = B1 LoRA + SDCA (signed H-E differential) cross-attn fusion."""

    def __init__(
        self,
        uni_path: str,
        flow_param: str = "direct",
        img_size: int = 224,
        pretrained_r34: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        sdca_hidden_dim: int = 256,
        sdca_num_heads: int = 8,
        sdca_dropout: float = 0.1,
        sdca_gamma_init: float = 0.1,
        sdca_bias_init: float = 2.0,
    ) -> None:
        super().__init__(
            uni_path=uni_path,
            flow_param=flow_param,
            img_size=img_size,
            pretrained_r34=pretrained_r34,
        )

        n_wrapped = add_lora_to_uni(
            self.uni.backbone,
            rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout,
        )
        if n_wrapped == 0:
            raise RuntimeError("LoRA wrapping found no qkv Linears")
        self.uni.frozen = False

        self.sdca = SDCABlock(
            dim_q=1024, dim_kv=128,
            hidden_dim=sdca_hidden_dim, num_heads=sdca_num_heads,
            dropout=sdca_dropout,
            gamma_init=sdca_gamma_init, bias_init=sdca_bias_init,
        )

        self.register_buffer(
            "imagenet_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "imagenet_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "hed_from_rgb",
            torch.tensor(_HED_FROM_RGB, dtype=torch.float32),
            persistent=False,
        )

        n_lora = count_lora_params(self.uni.backbone)
        n_sdca = sum(p.numel() for p in self.sdca.parameters() if p.requires_grad)
        print(
            f"[D1 LoRA+SDCA] wrapped {n_wrapped} UNI qkv layers; "
            f"LoRA params={n_lora:,}; SDCA params={n_sdca:,}; "
            f"SDCA cfg: hidden={sdca_hidden_dim} heads={sdca_num_heads} "
            f"gamma_init={sdca_gamma_init} bias_init={sdca_bias_init}",
            flush=True,
        )

    def train(self, mode: bool = True):
        return nn.Module.train(self, mode)

    def _rgb_to_he(self, x_rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Recover H and E channels at full resolution from ImageNet-norm RGB."""
        rgb = (x_rgb * self.imagenet_std + self.imagenet_mean).clamp(min=1e-6)  # (B,3,H,W) in [0,1]
        # skimage's separate_stains: log10(rgb*255+1) / log10(255), negate.
        log_rgb = -torch.log10(rgb * 255.0 + 1.0) / _LOG10_255  # (B,3,H,W) negative log absorbance
        # (B,H,W,3) for matmul
        log_rgb_hw3 = log_rgb.permute(0, 2, 3, 1).contiguous()
        # Cast hed_from_rgb to log_rgb's dtype (avoids AMP fp16 mismatch).
        hed_mat = self.hed_from_rgb.to(dtype=log_rgb_hw3.dtype)
        hed = log_rgb_hw3 @ hed_mat  # (B,H,W,3) — last dim is [h, e, d]
        h = hed[..., 0:1].permute(0, 3, 1, 2)  # (B,1,H,W)
        e = hed[..., 1:2].permute(0, 3, 1, 2)  # (B,1,H,W)
        return h, e

    @staticmethod
    def _norm01(x: torch.Tensor) -> torch.Tensor:
        """Per-image min-max normalize, clamp to [0, 1]."""
        x_flat = x.flatten(2)  # (B,1,N)
        lo = x_flat.amin(dim=2, keepdim=True).unsqueeze(-1)
        hi = x_flat.amax(dim=2, keepdim=True).unsqueeze(-1)
        return ((x - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)

    def _compute_stain_diff_at_8(self, x_rgb: torch.Tensor) -> torch.Tensor:
        """Compute signed D = H_norm - E_norm at /8."""
        h, e = self._rgb_to_he(x_rgb)
        h_norm = self._norm01(h)
        e_norm = self._norm01(e)
        d = h_norm - e_norm   # (B, 1, H, W) in [-1, 1]
        return F.avg_pool2d(d, kernel_size=8, stride=8)

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

        # Signed stain differential at /8 from RGB.
        d_at_8 = self._compute_stain_diff_at_8(x_rgb)  # (B, 1, H/8, W/8)

        s3 = self.uni(x_rgb)  # (B, 1024, H/16, W/16)
        B, C_u, Hp, Wp = s3.shape
        s3_tokens = s3.flatten(2).transpose(1, 2).contiguous()  # (B, Hp*Wp, 1024)
        s3_enriched_tokens = self.sdca(s3_tokens, s2, d_at_8)
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
            "d1_sdca_gamma": self.sdca.gamma.detach(),
            "d1_sdca_d_bias_scale": self.sdca.d_bias_scale.detach(),
        }
        if self.flow_param == "direct":
            outputs["flow"] = torch.tanh(self.flow_head(feat))
        else:
            phi = self.potential_head(feat)
            outputs["phi"] = phi
            outputs["flow"] = potential_to_flow(phi)
        return outputs


__all__ = ["NISFlowNetUNIHybridHLoRASDCA"]
