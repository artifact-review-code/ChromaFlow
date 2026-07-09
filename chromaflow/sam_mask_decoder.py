"""Lightweight point-prompted instance mask decoder (B3 head).

Given:
  - A pixel-level feature map (B, C, H, W) at the output resolution
  - N point prompts (B, N, 2) — normalized to [0, 1] x,y coords
  - A valid mask (B, N) — bool, marks which queries are real (vs padding)

For each prompt, produce:
  - An instance mask (B, N, H, W) — logits
  - An IoU prediction (B, N) — scalar

Design (simplified MaskFormer / SAM mask decoder):
  1. Embed each point coord into a query vector via sin/cos positional encoding + MLP.
  2. Project the feature map to a common embedding dim (k/v for cross-attn).
  3. Run K cross-attn blocks: each query attends to all pixel positions in the
     projected feature map. Standard pre-norm transformer block layout.
  4. After refinement, produce mask = einsum(query, projected_feat) → (B, N, H, W).
     This is the "dot-product mask" trick from MaskFormer.
  5. IoU prediction: small MLP on refined queries.

Pad queries are masked in cross-attn via key_padding_mask=False (no need since they
contribute zero to loss anyway). Output masks for invalid queries are garbage and
filtered downstream by the valid mask.

This is the simplest point-conditional instance decoder that's principled and
well-validated. Not Hungarian — each centroid has a deterministic mask target
in training, so 1-to-1 alignment is automatic.

Reference: Cheng et al., "Per-Pixel Classification is Not All You Need for
Semantic Segmentation" (MaskFormer, NeurIPS 2021).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_position_encoding(points: torch.Tensor, num_bands: int = 16) -> torch.Tensor:
    """Convert (B, N, 2) normalized [0,1] coords into (B, N, 4*num_bands) PE.

    Standard sin/cos positional encoding applied independently to x and y, with
    frequencies in [1, 2^(num_bands-1)] (per-dim).
    """
    B, N, _ = points.shape
    device = points.device
    # frequencies: (num_bands,) log-spaced
    freqs = 2.0 ** torch.arange(num_bands, device=device, dtype=points.dtype)  # (num_bands,)
    freqs = freqs * math.pi  # multiply by π so sin/cos go through full cycles
    # points: (B, N, 2) → expand to (B, N, 2, num_bands)
    angle = points.unsqueeze(-1) * freqs.view(1, 1, 1, num_bands)
    sin = torch.sin(angle)
    cos = torch.cos(angle)
    out = torch.stack([sin, cos], dim=-1)  # (B, N, 2, num_bands, 2)
    return out.flatten(-3)  # (B, N, 4*num_bands)


class CrossAttnDecoderBlock(nn.Module):
    """Pre-norm transformer cross-attn block: Q attends to KV.

    Forward: q' = q + dropout(MHA(LN(q), LN(kv), LN(kv)))
             q'' = q' + dropout(MLP(LN(q')))
    """
    def __init__(self, dim_q: int, dim_kv: int, hidden_dim: int, num_heads: int = 4,
                 ffn_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm_q1 = nn.LayerNorm(dim_q)
        self.norm_kv = nn.LayerNorm(dim_kv)
        self.q_proj = nn.Linear(dim_q, hidden_dim, bias=False)
        self.k_proj = nn.Linear(dim_kv, hidden_dim, bias=False)
        self.v_proj = nn.Linear(dim_kv, hidden_dim, bias=False)
        self.attn_out = nn.Linear(hidden_dim, dim_q, bias=False)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        nn.init.normal_(self.attn_out.weight, std=0.02)

        self.norm_q2 = nn.LayerNorm(dim_q)
        self.ffn = nn.Sequential(
            nn.Linear(dim_q, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim_q),
        )
        nn.init.normal_(self.ffn[-1].weight, std=0.02)
        self.dropout = dropout

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        q:  (B, N, dim_q)
        kv: (B, M, dim_kv)
        """
        B, N, _ = q.shape
        M = kv.shape[1]
        # Cross-attn
        q_n = self.norm_q1(q)
        kv_n = self.norm_kv(kv)
        qh = self.q_proj(q_n).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        kh = self.k_proj(kv_n).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        vh = self.v_proj(kv_n).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (qh @ kh.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)
        out = attn @ vh  # (B, h, N, d)
        out = out.transpose(1, 2).contiguous().view(B, N, self.hidden_dim)
        out = self.attn_out(out)
        if self.training and self.dropout > 0:
            out = F.dropout(out, p=self.dropout)
        q = q + out

        # FFN
        ffn_in = self.norm_q2(q)
        ffn_out = self.ffn(ffn_in)
        if self.training and self.dropout > 0:
            ffn_out = F.dropout(ffn_out, p=self.dropout)
        return q + ffn_out


class SamMaskDecoder(nn.Module):
    """Point-prompted instance mask decoder.

    Args:
        feat_dim: channels of the input feature map (C in (B, C, H, W))
        query_dim: internal query embedding dim
        num_blocks: number of cross-attn refinement blocks
        num_heads: attention heads
        num_pe_bands: frequency bands for the sinusoidal PE
    """
    def __init__(self, feat_dim: int = 64, query_dim: int = 128, num_blocks: int = 2,
                 num_heads: int = 4, num_pe_bands: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.query_dim = query_dim
        self.num_pe_bands = num_pe_bands

        # Point → query embedding
        pe_dim = 4 * num_pe_bands
        self.point_mlp = nn.Sequential(
            nn.Linear(pe_dim, query_dim * 2),
            nn.GELU(),
            nn.Linear(query_dim * 2, query_dim),
            nn.LayerNorm(query_dim),
        )

        # Feature projection: feat_dim → query_dim
        self.feat_proj = nn.Conv2d(feat_dim, query_dim, kernel_size=1, bias=False)

        # Cross-attn refinement blocks
        self.blocks = nn.ModuleList([
            CrossAttnDecoderBlock(
                dim_q=query_dim,
                dim_kv=query_dim,
                hidden_dim=query_dim,
                num_heads=num_heads,
                ffn_dim=query_dim * 2,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

        # IoU prediction head
        self.iou_mlp = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, 1),
        )

    def forward(self, feat: torch.Tensor, points: torch.Tensor) -> dict:
        """
        Args:
            feat:   (B, C, H, W) feature map at output resolution
            points: (B, N, 2) normalized [0, 1] x,y coords

        Returns:
            'mask_logits': (B, N, H, W)
            'iou_pred':    (B, N)
        """
        B, C, H, W = feat.shape
        N = points.shape[1]

        # Project feat → (B, query_dim, H, W) → (B, HW, query_dim)
        f_proj = self.feat_proj(feat)  # (B, qd, H, W)
        f_flat = f_proj.flatten(2).transpose(1, 2)  # (B, HW, qd)

        # Embed points
        pe = sinusoidal_position_encoding(points, num_bands=self.num_pe_bands)
        q = self.point_mlp(pe)  # (B, N, qd)

        # Refine queries via cross-attn to feat
        for block in self.blocks:
            q = block(q, f_flat)

        # Mask via dot-product (MaskFormer trick)
        # q: (B, N, qd), f_proj: (B, qd, H, W) → (B, N, H, W)
        mask_logits = torch.einsum('bnd,bdhw->bnhw', q, f_proj)

        # IoU prediction (scalar per query)
        iou_pred = self.iou_mlp(q).squeeze(-1)  # (B, N)

        return {"mask_logits": mask_logits, "iou_pred": iou_pred}


__all__ = ["SamMaskDecoder", "sinusoidal_position_encoding"]
