"""B2.5 v2: HCA with H+E dual-channel stain decomposition.

HCA v1 used only the H (hematoxylin) density as attention bias. HCA v2 uses
BOTH H and E (eosin) channels, derived from the full RGB image via the standard
Ruifrok-Johnston color deconvolution matrix.

Why both channels?
  * H (hematoxylin) binds DNA → marks NUCLEI. High-H regions should attract
    more attention (positive bias).
  * E (eosin) binds protein in cytoplasm → marks NON-NUCLEUS tissue. High-E
    regions should be suppressed during nucleus attention (negative bias).
  * Together they form the full stain decomposition; using only H ignores
    half the chemical information.

attn_logits(i, k) = (q_i · k_k) / sqrt(d)
                  + h_bias_scale * log(H_density(k) + eps)
                  - e_bias_scale * log(E_density(k) + eps)

Two learnable scalars (h_bias_scale, e_bias_scale) control the strength of
each stain prior; the sign of E bias is HARDCODED negative since E marks
non-nucleus tissue (the model can still drive |e_bias_scale|→0 if E is
unhelpful).

Implementation note: x_h passed to the model has already lost the E channel
(dataset only kept H). To recover E we de-normalize x_rgb and apply the
Ruifrok-Johnston deconvolution matrix on the fly. Cost: one einsum per forward.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from chromaflow.uni_encoder_hybrid import NISFlowNetUNIHybridH, potential_to_flow
from chromaflow.lora import add_lora_to_uni, count_lora_params
from chromaflow.vit_adapter import HCABlock  # we'll extend it externally below


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
# Ruifrok-Johnston H&E deconvolution matrix (skimage.color hed_from_rgb).
_HED_FROM_RGB = (
    (1.87798274, -1.00767869, -0.55611582),
    (-0.06590806, 1.13473037, -0.13552180),
    (-0.60190736, -0.48041419, 1.57358807),
)


class HCAv2Block(nn.Module):
    """Cross-attn with H+E dual-channel stain bias.

    Identical to HCABlock (vit_adapter.py) but takes TWO density maps and
    combines them as additive attention biases with separate learnable scalars.
    """

    def __init__(
        self,
        dim_q: int = 1024,
        dim_kv: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        gamma_init: float = 0.1,
        h_bias_init: float = 1.0,
        e_bias_init: float = 0.5,
        h_eps: float = 0.01,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} % num_heads={num_heads} != 0")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.norm_q = nn.LayerNorm(dim_q)
        self.norm_kv = nn.LayerNorm(dim_kv)
        self.q_proj = nn.Linear(dim_q, hidden_dim, bias=False)
        self.k_proj = nn.Linear(dim_kv, hidden_dim, bias=False)
        self.v_proj = nn.Linear(dim_kv, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, dim_q, bias=False)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        self.gamma = nn.Parameter(torch.full((1,), gamma_init))
        # H pulls attention toward nuclei (positive); E pushes away (negative).
        # We keep both as raw scalars and apply the sign at forward time so the
        # model can also flip signs if needed (rarely happens but harmless).
        self.h_bias_scale = nn.Parameter(torch.full((1,), h_bias_init))
        self.e_bias_scale = nn.Parameter(torch.full((1,), e_bias_init))
        self.dropout = dropout
        self.h_eps = h_eps

    def forward(
        self,
        q_tokens: torch.Tensor,
        kv_feat: torch.Tensor,
        h_at_kv_scale: torch.Tensor,
        e_at_kv_scale: torch.Tensor,
    ) -> torch.Tensor:
        import math
        B = kv_feat.shape[0]
        Hp, Wp = kv_feat.shape[-2], kv_feat.shape[-1]
        M = Hp * Wp
        N_q = q_tokens.shape[1]

        kv_flat = kv_feat.flatten(2).transpose(1, 2)
        h_flat = h_at_kv_scale.flatten(2).squeeze(1)  # (B, M)
        e_flat = e_at_kv_scale.flatten(2).squeeze(1)  # (B, M)

        # Per-image min-max normalize each stain map to [0, 1].
        def _norm(t: torch.Tensor) -> torch.Tensor:
            t = t - t.amin(dim=-1, keepdim=True)
            t = t / t.amax(dim=-1, keepdim=True).clamp(min=1e-6)
            return t.clamp(min=0.0)

        h_norm = _norm(h_flat)
        e_norm = _norm(e_flat)
        log_h = torch.log(h_norm + self.h_eps)
        log_e = torch.log(e_norm + self.h_eps)
        # Combine: pull toward H, push from E.
        stain_bias = self.h_bias_scale * log_h - self.e_bias_scale * log_e

        q = self.q_proj(self.norm_q(q_tokens))
        kv_n = self.norm_kv(kv_flat)
        k = self.k_proj(kv_n)
        v = self.v_proj(kv_n)

        q = q.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_logits = (q @ k.transpose(-2, -1)) * scale
        attn_logits = attn_logits + stain_bias[:, None, None, :]
        attn = torch.softmax(attn_logits, dim=-1)
        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, N_q, self.hidden_dim)
        out = self.out_proj(out)
        return q_tokens + self.gamma * out

    def extra_repr(self) -> str:
        return (
            f"hidden={self.hidden_dim}, heads={self.num_heads}, "
            f"h_init={float(self.h_bias_scale.detach()):.3g}, "
            f"e_init={float(self.e_bias_scale.detach()):.3g}"
        )


def rgb_to_hed_torch(x_rgb: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
                     hed_matrix: torch.Tensor) -> torch.Tensor:
    """Compute H, E (and residual) density maps from ImageNet-normalized RGB.

    Mirrors skimage.color.rgb2hed: -log(RGB) @ HED_FROM_RGB.

    Args:
        x_rgb: (B, 3, H, W) ImageNet-normalized.
        mean / std: (1, 3, 1, 1) ImageNet stats.
        hed_matrix: (3, 3) Ruifrok-Johnston matrix.
    Returns:
        (B, 3, H, W) — channel 0 = H, 1 = E, 2 = residual.
    """
    rgb = x_rgb * std + mean        # ~ [0, 1]
    rgb = rgb.clamp(min=1e-6, max=1.0)
    # Match skimage.color.separate_stains: log(rgb) / log(1e-6) @ M, where
    # log(1e-6) is the reference "saturation" constant used to normalize log-rgb
    # to [0, 1] before deconvolution.
    log_adjust = float(torch.log(torch.tensor(1e-6)))   # ~ -13.8155
    log_norm = torch.log(rgb) / log_adjust              # (B, 3, H, W)
    return torch.einsum("bchw,cd->bdhw", log_norm, hed_matrix)


class NISFlowNetUNIHybridHLoRAHCAv2(NISFlowNetUNIHybridH):
    """B2.5 v2 = LoRA on UNI + HCA-v2 (H+E dual-stain bias)."""

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
        hca_e_bias_init: float = 0.5,
    ) -> None:
        super().__init__(
            uni_path=uni_path,
            flow_param=flow_param,
            img_size=img_size,
            pretrained_r34=pretrained_r34,
        )

        n_wrapped = add_lora_to_uni(
            self.uni.backbone, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout
        )
        if n_wrapped == 0:
            raise RuntimeError("LoRA wrapping found no qkv Linears in UNI backbone")
        self.uni.frozen = False

        self.hca = HCAv2Block(
            dim_q=1024,
            dim_kv=128,
            hidden_dim=hca_hidden_dim,
            num_heads=hca_num_heads,
            dropout=hca_dropout,
            gamma_init=hca_gamma_init,
            h_bias_init=hca_h_bias_init,
            e_bias_init=hca_e_bias_init,
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
        n_hca = sum(p.numel() for p in self.hca.parameters() if p.requires_grad)
        print(
            f"[B2.5v2 LoRA+HCAv2] wrapped {n_wrapped} UNI qkv; LoRA={n_lora:,}; "
            f"HCAv2 params={n_hca:,}; H_init={hca_h_bias_init} E_init={hca_e_bias_init}",
            flush=True,
        )

    def train(self, mode: bool = True):
        return nn.Module.train(self, mode)

    def _compute_he_at_8(self, x_rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hed = rgb_to_hed_torch(x_rgb, self.imagenet_mean, self.imagenet_std, self.hed_matrix)
        h_full = hed[:, 0:1]  # (B, 1, H, W)
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
        s2 = self.r34_layer2(s1)

        h_at_8, e_at_8 = self._compute_he_at_8(x_rgb)

        s3 = self.uni(x_rgb)
        B, C_u, Hp, Wp = s3.shape
        s3_tokens = s3.flatten(2).transpose(1, 2).contiguous()
        s3_enriched_tokens = self.hca(s3_tokens, s2, h_at_8, e_at_8)
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
            "hca_gamma": self.hca.gamma.detach(),
            "hca_h_bias": self.hca.h_bias_scale.detach(),
            "hca_e_bias": self.hca.e_bias_scale.detach(),
        }
        if self.flow_param == "direct":
            outputs["flow"] = torch.tanh(self.flow_head(feat))
        else:
            phi = self.potential_head(feat)
            outputs["phi"] = phi
            outputs["flow"] = potential_to_flow(phi)
        return outputs


__all__ = ["NISFlowNetUNIHybridHLoRAHCAv2", "HCAv2Block", "rgb_to_hed_torch"]
