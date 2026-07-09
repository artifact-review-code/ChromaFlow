"""D3 encoder: AdaSCA — Adaptive Stain Cross-Attention with image-conditioned γ.

Replaces HCA's single learned scalar γ with a per-image gate:

    γ(x) = sigmoid(MLP(GAP(h_at_kv_scale)))

Hypothesis: cross-attention with a fixed γ undertrades the stain signal across
images of varying stain quality. AdaSCA lets the model dial up H-bias on
strongly-stained images and dial it down on weakly-stained ones.

Otherwise identical to B2.5 HCA. Same attention bias formula
(log(H_norm + eps) at K positions), same hidden_dim/heads.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from chromaflow.uni_encoder_hybrid import NISFlowNetUNIHybridH, potential_to_flow
from chromaflow.lora import add_lora_to_uni, count_lora_params


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class AdaSCABlock(nn.Module):
    """HCABlock with image-conditioned γ (per-image gate)."""

    def __init__(
        self,
        dim_q: int = 1024,
        dim_kv: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        gate_hidden: int = 64,
        h_bias_init: float = 1.0,
        h_eps: float = 0.01,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} not divisible by num_heads={num_heads}")
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
        # Per-image γ predictor: 5 stats of H at /8 → MLP → γ ∈ [0, 1]
        # stats: mean, std, min, max, frac-above-0.5
        self.gate_mlp = nn.Sequential(
            nn.Linear(5, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )
        # Initialize gate so its sigmoid output starts near 0.1 (same as HCA's
        # gamma_init) — sigmoid(-2.2) ≈ 0.1.
        nn.init.constant_(self.gate_mlp[-1].bias, -2.2)
        self.h_bias_scale = nn.Parameter(torch.full((1,), h_bias_init))
        self.dropout = dropout
        self.h_eps = h_eps

    def _gate(self, h_at_kv_scale: torch.Tensor) -> torch.Tensor:
        """Compute per-image γ ∈ (0, 1)."""
        h_flat = h_at_kv_scale.flatten(2)  # (B, 1, M)
        h_mean = h_flat.mean(dim=2).squeeze(1)
        h_std = h_flat.std(dim=2).squeeze(1)
        h_min = h_flat.amin(dim=2).squeeze(1)
        h_max = h_flat.amax(dim=2).squeeze(1)
        h_frac = (h_flat > 0.5).float().mean(dim=2).squeeze(1)
        stats = torch.stack([h_mean, h_std, h_min, h_max, h_frac], dim=-1)  # (B, 5)
        return torch.sigmoid(self.gate_mlp(stats))  # (B, 1)

    def forward(self, q_tokens: torch.Tensor, kv_feat: torch.Tensor,
                h_at_kv_scale: torch.Tensor) -> torch.Tensor:
        B = kv_feat.shape[0]
        Hp, Wp = kv_feat.shape[-2], kv_feat.shape[-1]
        M = Hp * Wp
        N_q = q_tokens.shape[1]

        kv_flat = kv_feat.flatten(2).transpose(1, 2)
        h_flat = h_at_kv_scale.flatten(2).squeeze(1)  # (B, M)

        h_amax = h_flat.amax(dim=-1, keepdim=True).clamp(min=1e-6)
        h_norm = (h_flat / h_amax).clamp(min=0.0)
        h_bias = torch.log(h_norm + self.h_eps)

        q = self.q_proj(self.norm_q(q_tokens))
        kv_n = self.norm_kv(kv_flat)
        k = self.k_proj(kv_n)
        v = self.v_proj(kv_n)

        q = q.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_logits = (q @ k.transpose(-2, -1)) * scale
        attn_logits = attn_logits + self.h_bias_scale * h_bias[:, None, None, :]
        attn = torch.softmax(attn_logits, dim=-1)
        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, N_q, self.hidden_dim)
        out = self.out_proj(out)

        gamma = self._gate(h_at_kv_scale)  # (B, 1)
        return q_tokens + gamma[:, :, None] * out  # broadcast → (B, N_q, dim_q)


class NISFlowNetUNIHybridHLoRAAdaSCA(NISFlowNetUNIHybridH):
    """D3: LoRA + AdaSCA (image-conditioned γ)."""

    def __init__(
        self,
        uni_path: str,
        flow_param: str = "direct",
        img_size: int = 224,
        pretrained_r34: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        adasca_hidden_dim: int = 256,
        adasca_num_heads: int = 8,
        adasca_dropout: float = 0.1,
        adasca_h_bias_init: float = 1.0,
    ) -> None:
        super().__init__(
            uni_path=uni_path, flow_param=flow_param,
            img_size=img_size, pretrained_r34=pretrained_r34,
        )
        n_wrapped = add_lora_to_uni(
            self.uni.backbone,
            rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout,
        )
        if n_wrapped == 0:
            raise RuntimeError("LoRA wrapping found no qkv Linears")
        self.uni.frozen = False
        self.adasca = AdaSCABlock(
            dim_q=1024, dim_kv=128,
            hidden_dim=adasca_hidden_dim, num_heads=adasca_num_heads,
            dropout=adasca_dropout, h_bias_init=adasca_h_bias_init,
        )
        self.register_buffer(
            "imagenet_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "imagenet_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )
        n_lora = count_lora_params(self.uni.backbone)
        n_adasca = sum(p.numel() for p in self.adasca.parameters() if p.requires_grad)
        print(
            f"[D3 LoRA+AdaSCA] wrapped {n_wrapped} UNI qkv; "
            f"LoRA params={n_lora:,}; AdaSCA params={n_adasca:,}",
            flush=True,
        )

    def train(self, mode: bool = True):
        return nn.Module.train(self, mode)

    def _compute_h_density_at_8(self, x_h: torch.Tensor) -> torch.Tensor:
        h_raw = (x_h * self.imagenet_std + self.imagenet_mean).mean(dim=1, keepdim=True)
        return F.avg_pool2d(h_raw, kernel_size=8, stride=8)

    def forward(self, x):
        if isinstance(x, dict):
            x_h = x["image"]
            x_rgb = x.get("image_rgb", x["image"])
        else:
            x_h = x; x_rgb = x

        H, W = x_h.shape[-2:]
        s0 = self.r34_stem(x_h)
        s1 = self.r34_layer1(self.r34_maxpool(s0))
        s2 = self.r34_layer2(s1)
        h_at_8 = self._compute_h_density_at_8(x_h)
        s3 = self.uni(x_rgb)
        B, C_u, Hp, Wp = s3.shape
        s3_tokens = s3.flatten(2).transpose(1, 2).contiguous()
        s3_enriched = self.adasca(s3_tokens, s2, h_at_8)
        s3 = s3_enriched.transpose(1, 2).contiguous().view(B, C_u, Hp, Wp)
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
            "d3_adasca_h_bias_scale": self.adasca.h_bias_scale.detach(),
        }
        if self.flow_param == "direct":
            outputs["flow"] = torch.tanh(self.flow_head(feat))
        else:
            phi = self.potential_head(feat)
            outputs["phi"] = phi
            outputs["flow"] = potential_to_flow(phi)
        return outputs


__all__ = ["NISFlowNetUNIHybridHLoRAAdaSCA"]
