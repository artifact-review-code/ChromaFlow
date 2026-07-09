"""Stain-Routed Mutual Attention (SRMA) block.

Bidirectional cross-attention between two encoder branches with complementary
stain routing:
  * Forward (UNI ← R34): UNI tokens attend to R34 features, biased by log(H).
    "Take edges FROM CNN where NUCLEI are dense."
  * Backward (R34 ← UNI): R34 features attend to UNI tokens, biased by log(E).
    "Look INTO FM where CYTOPLASM/TISSUE context is informative."

Both branches are enriched. Decoder receives both enriched outputs. The two
stain channels are computed once externally (Ruifrok-Johnston deconvolution)
and reused for both directions.

Paper claim:
    "Each direction of cross-modal attention is routed by a *complementary*
    stain channel. Hematoxylin (DNA marker) drives the foundation-model→CNN
    direction; eosin (cytoplasm marker) drives the CNN→foundation-model
    direction. This embeds H&E chemistry into the bidirectional information
    flow between encoders, generalizing the prior single-direction HCA."
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _StainBiasedCrossAttn(nn.Module):
    """One direction of cross-attn with an additive log-density bias.

    bias_mode:
        "per_key"   — bias is a function of the K position (each k draws more
                       attention if bias_k is high). Shape: (B, M).
        "per_query" — bias is a function of the Q position (each q sends more
                       attention to all keys if bias_q is high). Shape: (B, N).
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        hidden_dim: int,
        num_heads: int,
        bias_mode: str,
        dropout: float = 0.1,
        gamma_init: float = 0.1,
        bias_scale_init: float = 1.0,
        bias_eps: float = 0.01,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if bias_mode not in ("per_key", "per_query"):
            raise ValueError(f"unknown bias_mode={bias_mode}")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.dim_q = dim_q
        self.bias_mode = bias_mode

        self.norm_q = nn.LayerNorm(dim_q)
        self.norm_kv = nn.LayerNorm(dim_kv)
        self.q_proj = nn.Linear(dim_q, hidden_dim, bias=False)
        self.k_proj = nn.Linear(dim_kv, hidden_dim, bias=False)
        self.v_proj = nn.Linear(dim_kv, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, dim_q, bias=False)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        self.gamma = nn.Parameter(torch.full((1,), gamma_init))
        self.bias_scale = nn.Parameter(torch.full((1,), bias_scale_init))
        self.bias_eps = bias_eps
        self.dropout = dropout

    @staticmethod
    def _log_density_bias(density_flat: torch.Tensor, eps: float) -> torch.Tensor:
        """Per-image min-max normalize then log. density_flat: (B, K)."""
        t = density_flat - density_flat.amin(dim=-1, keepdim=True)
        t = t / t.amax(dim=-1, keepdim=True).clamp(min=1e-6)
        t = t.clamp(min=0.0)
        return torch.log(t + eps)

    def forward(
        self,
        q_in: torch.Tensor,
        kv_in: torch.Tensor,
        density_at_bias_side: torch.Tensor,
    ) -> torch.Tensor:
        """
        q_in:  (B, N_q, dim_q) — flattened query tokens.
        kv_in: (B, N_kv, dim_kv) — flattened key/value tokens.
        density_at_bias_side: (B, K) — log-density bias array; K equals N_kv if
            bias_mode="per_key", N_q if "per_query".
        """
        B, N_q, _ = q_in.shape
        N_kv = kv_in.shape[1]

        q = self.q_proj(self.norm_q(q_in))
        kv_n = self.norm_kv(kv_in)
        k = self.k_proj(kv_n)
        v = self.v_proj(kv_n)

        q = q.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_logits = (q @ k.transpose(-2, -1)) * scale          # (B, h, N_q, N_kv)

        log_density = self._log_density_bias(density_at_bias_side, self.bias_eps)
        if self.bias_mode == "per_key":
            attn_logits = attn_logits + self.bias_scale * log_density[:, None, None, :]
        else:  # per_query
            attn_logits = attn_logits + self.bias_scale * log_density[:, None, :, None]

        attn = torch.softmax(attn_logits, dim=-1)
        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, N_q, self.hidden_dim)
        out = self.out_proj(out)
        return q_in + self.gamma * out


class SRMABlock(nn.Module):
    """Stain-Routed Mutual Attention: bidirectional FM↔CNN cross-attn with
    complementary stain biases.

    Args:
        uni_dim: UNI token feature dim (typically 1024).
        cnn_dim: R34 feature dim (typically 128 at /8).
        hidden_dim: shared inner dim for both directions (typically 256).
        num_heads: number of attention heads per direction.
        h_bias_init: initial scalar for H bias (forward direction).
        e_bias_init: initial scalar for E bias (backward direction).

    Forward inputs:
        uni_tokens: (B, N_u=Hu*Wu, uni_dim) — UNI tokens at /16, flattened.
        cnn_feat:   (B, cnn_dim, Hc, Wc)     — R34 s2 at /8.
        h_at_cnn:   (B, 1, Hc, Wc)            — hematoxylin density at /8 grid.
        e_at_cnn:   (B, 1, Hc, Wc)            — eosin density at /8 grid.

    NOTE: the H density is at /8 (CNN grid). For the backward direction
    (R34 ← UNI), per-query bias is on R34 side so we just use the same H/E maps.
    """

    def __init__(
        self,
        uni_dim: int = 1024,
        cnn_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        gamma_init: float = 0.1,
        h_bias_init: float = 1.0,
        e_bias_init: float = 1.0,
    ) -> None:
        super().__init__()
        # Forward: UNI ← R34, bias = H on KV side (R34 side); bias mode = per_key.
        self.forward_xa = _StainBiasedCrossAttn(
            dim_q=uni_dim, dim_kv=cnn_dim,
            hidden_dim=hidden_dim, num_heads=num_heads,
            bias_mode="per_key", dropout=dropout,
            gamma_init=gamma_init, bias_scale_init=h_bias_init,
        )
        # Backward: R34 ← UNI, bias = E on Q side (R34 side); bias mode = per_query.
        self.backward_xa = _StainBiasedCrossAttn(
            dim_q=cnn_dim, dim_kv=uni_dim,
            hidden_dim=hidden_dim, num_heads=num_heads,
            bias_mode="per_query", dropout=dropout,
            gamma_init=gamma_init, bias_scale_init=e_bias_init,
        )

    def forward(
        self,
        uni_tokens: torch.Tensor,
        cnn_feat: torch.Tensor,
        h_at_cnn: torch.Tensor,
        e_at_cnn: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, C_c, Hc, Wc = cnn_feat.shape
        M = Hc * Wc

        # Flatten R34 spatial dims to token sequence.
        cnn_flat = cnn_feat.flatten(2).transpose(1, 2)  # (B, M, cnn_dim)
        h_flat = h_at_cnn.flatten(2).squeeze(1)         # (B, M)
        e_flat = e_at_cnn.flatten(2).squeeze(1)         # (B, M)

        # Forward direction: UNI ← R34, bias by H on R34 side (per-key, since R34 = KV).
        uni_enriched = self.forward_xa(uni_tokens, cnn_flat, h_flat)

        # Backward direction: R34 ← UNI(enriched), bias by E on R34 side (per-query).
        # (We use the enriched UNI tokens — gives the backward direction richer K/V.)
        cnn_enriched_flat = self.backward_xa(cnn_flat, uni_enriched, e_flat)
        cnn_enriched = cnn_enriched_flat.transpose(1, 2).contiguous().view(B, C_c, Hc, Wc)

        return uni_enriched, cnn_enriched

    def extra_repr(self) -> str:
        f = float(self.forward_xa.bias_scale.detach())
        b = float(self.backward_xa.bias_scale.detach())
        return f"H_init={f:.3g}, E_init={b:.3g}, hidden={self.forward_xa.hidden_dim}"


__all__ = ["SRMABlock", "_StainBiasedCrossAttn"]
