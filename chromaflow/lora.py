"""LoRA adapter for foundation-model attention layers.

Used by B1 (uni_hybrid_h_lora): wraps the fused qkv Linear in each ViT block
of the UNI encoder. Base weights stay frozen; only lora_A / lora_B are trainable.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wrap an existing nn.Linear with a rank-r residual.

    output = base(x) + scaling * B @ A @ dropout(x)

    Base weights are frozen. lora_A is Kaiming-initialised, lora_B is zero so
    the LoRA branch outputs zero at initialisation (identity start).
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 16,
        alpha: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be > 0")
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)

        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B intentionally left at zero so identity init.
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = F.linear(self.lora_dropout(x), self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B)
        return base_out + lora_out * self.scaling

    def extra_repr(self) -> str:
        return f"rank={self.rank}, scaling={self.scaling}"


def _iter_qkv_linears(module: nn.Module) -> Iterable[tuple[nn.Module, str, nn.Linear]]:
    """Yield (parent, attr_name, qkv_linear) for every fused qkv Linear in the tree.

    Convention used by timm ViT: each transformer block has block.attn.qkv as a
    single Linear with out_features = 3 * dim. We match by attribute name 'qkv'.
    """
    for parent_name, parent in module.named_modules():
        for attr in ("qkv",):
            child = getattr(parent, attr, None)
            if isinstance(child, nn.Linear):
                yield parent, attr, child


def add_lora_to_uni(
    uni_backbone: nn.Module,
    rank: int = 16,
    alpha: int = 16,
    dropout: float = 0.1,
) -> int:
    """Replace every block.attn.qkv Linear in `uni_backbone` with a LoRALinear.

    Returns the number of layers wrapped. Caller is responsible for ensuring
    `uni_backbone` corresponds to the *frozen* foundation backbone whose
    requires_grad is already False.
    """
    n_wrapped = 0
    for parent, attr, qkv in list(_iter_qkv_linears(uni_backbone)):
        wrapped = LoRALinear(qkv, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, attr, wrapped)
        n_wrapped += 1
    return n_wrapped


def count_lora_params(module: nn.Module) -> int:
    """Return total trainable parameter count whose name contains 'lora_'."""
    return sum(p.numel() for n, p in module.named_parameters() if "lora_" in n and p.requires_grad)
