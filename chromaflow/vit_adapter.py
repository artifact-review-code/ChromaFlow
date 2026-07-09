"""ViT-Adapter style cross-attention between ViT tokens and CNN feature map.

B2 minimal building block. One cross-attention block that lets a frozen
ViT-L/16 token grid (at /16) absorb high-frequency information from a CNN
feature map (typically R34 s2 at /8). Output preserves token shape.

Key design choices vs the failed huni modules (URCF/UDAC/UBIM/UCAF/UDAR/MTA):
  - We modify the *encoder output before the decoder*, not gating decoder
    features. This is a different lever.
  - Cross-attention is dynamic per-token (UNI token at any (i,j) can query R34
    at any spatial position), unlike FiLM/Conv1x1 which are pointwise.
  - Output projection uses small Kaiming init (std=0.02), not zero-init, to
    avoid the dead-gradient trap that crippled MTA-v1 / UBIM-v1 / UDAC-v1.
  - A learnable scalar gate gamma (init=0.1) scales the residual contribution.
    Standard ViT-Adapter recipe ([arXiv 2205.08534]).

References:
  - Chen et al., "Vision Transformer Adapter for Dense Predictions", ICLR 2023.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttnBlock(nn.Module):
    """One cross-attn block: q from ViT tokens, k/v from a CNN feature map.

    Output shape == q-token shape, computed as:
        q' = q + gamma * out_proj( MHA(LN(q), LN(kv_flat), LN(kv_flat)) )

    where MHA's hidden dim is `hidden_dim` and the projections map q and kv to
    a common subspace before attention. gamma is a learnable scalar starting at
    0.1 so the block is mostly identity at init but receives non-trivial gradient.
    """

    def __init__(
        self,
        dim_q: int = 1024,
        dim_kv: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        gamma_init: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.norm_q = nn.LayerNorm(dim_q)
        self.norm_kv = nn.LayerNorm(dim_kv)
        self.q_proj = nn.Linear(dim_q, hidden_dim, bias=False)
        self.k_proj = nn.Linear(dim_kv, hidden_dim, bias=False)
        self.v_proj = nn.Linear(dim_kv, hidden_dim, bias=False)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True, bias=False
        )
        self.out_proj = nn.Linear(hidden_dim, dim_q, bias=False)
        nn.init.normal_(self.out_proj.weight, std=0.02)  # small init, NOT zero
        self.gamma = nn.Parameter(torch.full((1,), gamma_init))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, q_tokens: torch.Tensor, kv_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_tokens: (B, N_q, dim_q). Typically UNI tokens at /16 flattened.
            kv_feat:  (B, dim_kv, H_kv, W_kv). Typically R34 s2 features at /8.
        Returns:
            (B, N_q, dim_q). Same shape as q_tokens, with residual update.
        """
        B = kv_feat.shape[0]
        kv_flat = kv_feat.flatten(2).transpose(1, 2)  # (B, H*W, dim_kv)

        q_n = self.norm_q(q_tokens)
        kv_n = self.norm_kv(kv_flat)

        q = self.q_proj(q_n)
        k = self.k_proj(kv_n)
        v = self.v_proj(kv_n)
        out, _ = self.attn(q, k, v, need_weights=False)
        out = self.dropout(self.out_proj(out))
        return q_tokens + self.gamma * out


__all__ = ["CrossAttnBlock", "HCABlock"]



class HCABlock(nn.Module):
    """Hematoxylin-Conditioned Attention block.

    Cross-attn from ViT tokens (q) to CNN feature map (k, v), with attention
    scores **biased by hematoxylin density at the K position**. Lets the foundation
    model attend more aggressively to CNN spatial features in nucleus-rich regions.

    Mechanism:
        attn_logits(i, k) = (q_i · k_k) / sqrt(d) + h_bias_scale * log(H_norm(k) + eps)

    where H_norm(k) is the per-image-normalized hematoxylin density at K position
    k (downsampled from input H channel), and h_bias_scale is a learned scalar.

    Pitch (paper):
      * Hematoxylin is the canonical nucleus marker in H&E pathology (H stain binds
        DNA). Most existing nuclei FMs ignore this domain knowledge inside their
        attention pathway.
      * Our HCA injects H-density as an additive attention bias — interpretable
        ("attend more where H is concentrated"), 0 extra params except 1 scalar,
        0 inference overhead.
      * Unlike ViT-Adapter's generic SPM, HCA is **specifically grounded in H&E
        chemistry** and explicitly designed for nuclei segmentation.

    Backward-compat with CrossAttnBlock interface: forward takes the same q_tokens
    and kv_feat, plus a new positional arg `h_at_kv_scale` for the H density map.
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
        h_eps: float = 0.01,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
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
        self.h_bias_scale = nn.Parameter(torch.full((1,), h_bias_init))
        self.dropout = dropout
        self.h_eps = h_eps

    def forward(self, q_tokens: torch.Tensor, kv_feat: torch.Tensor,
                h_at_kv_scale: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_tokens: (B, N_q, dim_q). UNI tokens flattened at /16.
            kv_feat:  (B, dim_kv, H_kv, W_kv). R34 s2 at /8.
            h_at_kv_scale: (B, 1, H_kv, W_kv). Hematoxylin density at same scale as kv_feat.
                            Should be in [0, 1] range; per-image normalization
                            happens inside this block.
        """
        import math
        B = kv_feat.shape[0]
        Hp, Wp = kv_feat.shape[-2], kv_feat.shape[-1]
        M = Hp * Wp
        N_q = q_tokens.shape[1]

        # Flatten kv to (B, M, dim_kv) and h to (B, M)
        kv_flat = kv_feat.flatten(2).transpose(1, 2)
        h_flat = h_at_kv_scale.flatten(2).squeeze(1)  # (B, M)

        # Per-image normalize H to [0, ~1]; log + epsilon to get bias.
        h_amax = h_flat.amax(dim=-1, keepdim=True).clamp(min=1e-6)
        h_norm = (h_flat / h_amax).clamp(min=0.0)
        h_bias = torch.log(h_norm + self.h_eps)  # (B, M), values in [log(eps), log(1+eps)]

        # Project Q / K / V
        q = self.q_proj(self.norm_q(q_tokens))
        kv_n = self.norm_kv(kv_flat)
        k = self.k_proj(kv_n)
        v = self.v_proj(kv_n)

        # Multi-head split
        q = q.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, N_q, d)
        k = k.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention with H bias
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_logits = (q @ k.transpose(-2, -1)) * scale  # (B, h, N_q, M)
        attn_logits = attn_logits + self.h_bias_scale * h_bias[:, None, None, :]
        attn = torch.softmax(attn_logits, dim=-1)
        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)
        out = attn @ v  # (B, h, N_q, d)
        out = out.transpose(1, 2).contiguous().view(B, N_q, self.hidden_dim)
        out = self.out_proj(out)
        return q_tokens + self.gamma * out

    def extra_repr(self) -> str:
        return (f"hidden={self.hidden_dim}, heads={self.num_heads}, "
                f"h_bias_scale_init={self.h_bias_scale.detach().item():.3g}")


class SDCABlock(nn.Module):
    """Stain-Differential Cross-Attention (D1).

    Same architecture as HCABlock but the attention bias is the SIGNED H-E
    stain differential D = H_norm - E_norm in [-1, 1]:

        attn_logits(i, k) = (q_i . k_k) / sqrt(d) + bias_scale * D(k)

    - D > 0  (high H, low E) -> nucleus prior  -> ATTEND MORE
    - D < 0  (low H, high E) -> stroma prior   -> ATTEND LESS
    - D ~ 0  -> boundary/ambiguous             -> neutral

    Resolves HCA's failure mode: in eosin-heavy regions H bleed is non-zero,
    so log(H) bias is still mildly positive there. Signed differential
    explicitly punishes attention to stroma.
    """

    def __init__(
        self,
        dim_q: int = 1024,
        dim_kv: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        gamma_init: float = 0.1,
        bias_init: float = 2.0,
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
        self.gamma = nn.Parameter(torch.full((1,), gamma_init))
        self.d_bias_scale = nn.Parameter(torch.full((1,), bias_init))
        self.dropout = dropout

    def forward(self, q_tokens: torch.Tensor, kv_feat: torch.Tensor,
                d_at_kv_scale: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_tokens: (B, N_q, dim_q)
            kv_feat:  (B, dim_kv, H_kv, W_kv)
            d_at_kv_scale: (B, 1, H_kv, W_kv) signed differential D in [-1, 1].
        """
        import math
        B = kv_feat.shape[0]
        Hp, Wp = kv_feat.shape[-2], kv_feat.shape[-1]
        M = Hp * Wp
        N_q = q_tokens.shape[1]

        kv_flat = kv_feat.flatten(2).transpose(1, 2)
        d_flat = d_at_kv_scale.flatten(2).squeeze(1)  # (B, M)
        # No log/normalize; D is already signed and bounded.

        q = self.q_proj(self.norm_q(q_tokens))
        kv_n = self.norm_kv(kv_flat)
        k = self.k_proj(kv_n)
        v = self.v_proj(kv_n)

        q = q.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_logits = (q @ k.transpose(-2, -1)) * scale
        attn_logits = attn_logits + self.d_bias_scale * d_flat[:, None, None, :]
        attn = torch.softmax(attn_logits, dim=-1)
        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, N_q, self.hidden_dim)
        out = self.out_proj(out)
        return q_tokens + self.gamma * out

    def extra_repr(self) -> str:
        return (f"hidden={self.hidden_dim}, heads={self.num_heads}, "
                f"bias_init={self.d_bias_scale.detach().item():.3g}")
