"""D7 encoder: D4 backbone + Stain Reconstruction Auxiliary head.

= NISFlowNetUNIHybridHLoRA (= D4 backbone) + an additional 1×1 conv that
predicts the H channel from the final decoder feature.

Aux loss (added in compute_losses):
    L_stain_recon = L1(h_pred, H_norm_target)

Where H_norm_target is the per-image normalized hematoxylin density derived
from the input H channel (same as what HCA used as bias).

Rationale: forces the decoder representation to explicitly encode stain
information without adding any cross-attention or fancy attention bias.
Cleaner than HCA (which failed) and orthogonal to DS-MPS (which uses stain
in supervision, not in feature representation).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from chromaflow.uni_encoder_b1 import NISFlowNetUNIHybridHLoRA


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class NISFlowNetUNIHybridHLoRAStainRecon(NISFlowNetUNIHybridHLoRA):
    """D7 = D4 backbone + h_recon_head."""

    def __init__(self, *args, h_recon_hidden: int = 32, **kwargs):
        super().__init__(*args, **kwargs)
        # Final decoder feature is 64-d (NISFlowNetUNIHybridH).
        self.h_recon_head = nn.Sequential(
            nn.Conv2d(64, h_recon_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(h_recon_hidden, 1, kernel_size=1),
        )
        self.register_buffer(
            "imagenet_mean",
            torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )
        n_params = sum(p.numel() for p in self.h_recon_head.parameters() if p.requires_grad)
        print(f"[D7 LoRA+StainRecon] aux h_recon head params={n_params:,}", flush=True)

    def forward(self, x):
        outputs = super().forward(x)
        # Add reconstruction prediction (continuous; sigmoid in [0, 1] to match
        # normalized H target).
        h_recon_logits = self.h_recon_head(outputs["feat"])
        outputs["h_recon"] = torch.sigmoid(h_recon_logits)
        # Provide H target derived from input x for the loss.
        if isinstance(x, dict):
            x_h = x["image"]
        else:
            x_h = x
        # De-normalize and per-image min-max → [0, 1]
        with torch.no_grad():
            h_raw = (x_h * self.imagenet_std + self.imagenet_mean).mean(dim=1, keepdim=True)
            h_flat = h_raw.flatten(2)
            lo = h_flat.amin(dim=2, keepdim=True).unsqueeze(-1)
            hi = h_flat.amax(dim=2, keepdim=True).unsqueeze(-1)
            h_target = ((h_raw - lo) / (hi - lo + 1e-6)).clamp(0, 1)
        outputs["h_target"] = h_target
        return outputs


__all__ = ["NISFlowNetUNIHybridHLoRAStainRecon"]
