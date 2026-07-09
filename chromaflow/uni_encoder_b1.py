"""B1 encoder: H+UNI hybrid with LoRA-adapted UNI attention.

This is an additive extension of NISFlowNetUNIHybridH (uni_hybrid_h).
Architecture is identical to the baseline:
    H 3-ch → R34 shallow (s0,s1,s2)
    RGB 3-ch → frozen UNI ViT-L/16 → /16 feature
    proj → U-Net decoder with skip concats → fg + flow + center heads

The only change vs uni_hybrid_h is that every block.attn.qkv Linear in the UNI
backbone is wrapped with LoRA (rank=16, alpha=16, dropout=0.1). Base weights of
the UNI backbone stay frozen; only lora_A / lora_B are trainable
(~1.6M params on ViT-L/16 with 24 blocks).

Hypothesis (Plan B1): if encoder representation quality is the bottleneck
(rather than fusion topology or output representation), enabling small-budget
adaptation of UNI's attention should lift MO test AJI from 0.5732 to ≥ 0.580
without any other change.

Author: B1 implementation pass, 2026-05-19.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from chromaflow.uni_encoder_hybrid import NISFlowNetUNIHybridH
from chromaflow.lora import add_lora_to_uni, count_lora_params


class NISFlowNetUNIHybridHLoRA(NISFlowNetUNIHybridH):
    """H+UNI hybrid identical to NISFlowNetUNIHybridH plus LoRA on UNI qkv.

    Trainable params come from:
      - All R34 shallow params (same as parent)
      - All decoder/head params (same as parent)
      - Plus lora_A and lora_B on every block.attn.qkv in self.uni.backbone

    Frozen params:
      - All UNI backbone base weights (untouched by this subclass)
    """

    def __init__(
        self,
        uni_path: str,
        flow_param: str = "direct",
        img_size: int = 224,
        pretrained_r34: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ) -> None:
        super().__init__(
            uni_path=uni_path,
            flow_param=flow_param,
            img_size=img_size,
            pretrained_r34=pretrained_r34,
        )

        # Wrap UNI's fused qkv attention Linears with LoRA. add_lora_to_uni walks
        # the backbone tree and replaces each block.attn.qkv with LoRALinear,
        # which freezes the underlying weight/bias and registers trainable
        # lora_A / lora_B parameters.
        n_wrapped = add_lora_to_uni(
            self.uni.backbone,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
        if n_wrapped == 0:
            raise RuntimeError(
                "LoRA wrapping found no qkv Linears in UNI backbone — verify timm ViT structure."
            )

        # UNIEncoder.__init__ set self.uni.frozen = True, which makes
        # UNIEncoder.forward route through an @torch.no_grad branch and also
        # forces backbone.eval() inside UNIEncoder.train(). Both block LoRA
        # gradients. Flip frozen=False here; base backbone params remain at
        # requires_grad=False (set by parent), so only LoRA params receive grad.
        self.uni.frozen = False

        n_lora = count_lora_params(self.uni.backbone)
        print(
            f"[B1 LoRA] wrapped {n_wrapped} UNI qkv layers; "
            f"rank={lora_rank} alpha={lora_alpha} dropout={lora_dropout}; "
            f"trainable LoRA params={n_lora:,}",
            flush=True,
        )

    def train(self, mode: bool = True):
        """Keep parent training-mode semantics; do NOT force backbone into eval.

        The base UNI parameters are requires_grad=False, so they receive no grad
        updates. Putting the backbone in train(mode) does NOT update them. For
        timm vit_large_patch16_224 with default drop_path_rate=0 / drop_rate=0,
        train mode and eval mode produce identical forward outputs (no BN, no
        active dropout in the backbone). LoRA's own dropout sits in lora_dropout
        and follows the LoRALinear sub-module's mode, which is correct under
        the standard nn.Module.train(mode) recursion.
        """
        return nn.Module.train(self, mode)


__all__ = ["NISFlowNetUNIHybridHLoRA"]
