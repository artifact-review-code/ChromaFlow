"""B3 encoder: B2.5 (LoRA + HCA) + point-prompted instance mask decoder.

Adds an instance-aware decoding path on top of B2.5. Each GT centroid (training)
or each detected center peak (inference) is fed as a point prompt to a small
mask decoder; the decoder produces a per-instance mask logit map and an IoU
prediction.

Training mode (when `instance_id` is in the input dict):
  1. Standard B2.5 forward → feat (B, 64, H, W) + flow + fg + centerness + boundary
  2. Extract centroid coords from instance_id → (B, N_max, 2) padded to N_max
  3. Build per-instance binary target masks from instance_id → (B, N_max, H, W)
  4. Forward mask decoder with centroids → mask_logits + iou_pred
  5. Return all in the output dict; losses computed externally in losses.py

Inference mode (no instance_id in input):
  1. Standard B2.5 forward
  2. Mask decoder is NOT invoked here; the inference loop in decode_variants.py
     should peak-detect from centerness, build points, and call .forward_masks(...)
     explicitly. This keeps the train-time forward signature simple.

Hyperparameters:
  - n_queries: max instances per image. Train images typically <= 100 nuclei
    per 224×224 crop; CPM has up to ~150. We pad to 128 to cover most cases.
  - mask_query_dim: 128
  - num_mask_blocks: 2 (matches SAM original)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from chromaflow.uni_encoder_b25 import NISFlowNetUNIHybridHLoRAHCA
from chromaflow.sam_mask_decoder import SamMaskDecoder


def _extract_centroids_and_targets(
    instance_id: torch.Tensor,
    n_queries: int,
    use_random_subsample: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For each image in the batch, find unique instance IDs, compute centroids
    and binary target masks. Pad to (B, n_queries, ...).

    Args:
        instance_id: (B, H, W) long tensor, 0 = background, >0 = instance label.
        n_queries: number of slots to pad to.
        use_random_subsample: if True and # instances > n_queries, randomly
            subsample n_queries of them; otherwise take the first n_queries
            in sorted order.

    Returns:
        centroids: (B, n_queries, 2) — normalized [0, 1] (x, y).
        targets:   (B, n_queries, H, W) — binary masks (float, 0 or 1).
        valid:     (B, n_queries) — bool, True for real instances.
    """
    B, H, W = instance_id.shape
    device = instance_id.device
    centroids = torch.zeros(B, n_queries, 2, device=device, dtype=torch.float32)
    targets = torch.zeros(B, n_queries, H, W, device=device, dtype=torch.float32)
    valid = torch.zeros(B, n_queries, device=device, dtype=torch.bool)

    for b in range(B):
        ids_b = torch.unique(instance_id[b])
        ids_b = ids_b[ids_b > 0]
        if ids_b.numel() == 0:
            continue
        if ids_b.numel() > n_queries and use_random_subsample:
            perm = torch.randperm(ids_b.numel(), device=device)[:n_queries]
            ids_b = ids_b[perm]
        elif ids_b.numel() > n_queries:
            ids_b = ids_b[:n_queries]

        for i, uid in enumerate(ids_b):
            mask = (instance_id[b] == uid).float()
            ys, xs = torch.where(mask > 0)
            if ys.numel() == 0:
                continue
            cy = ys.float().mean() / float(H - 1)
            cx = xs.float().mean() / float(W - 1)
            centroids[b, i, 0] = cx
            centroids[b, i, 1] = cy
            targets[b, i] = mask
            valid[b, i] = True

    return centroids, targets, valid


class NISFlowNetUNIHybridHLoRAHCAMask(NISFlowNetUNIHybridHLoRAHCA):
    """B3 = B2.5 + point-prompted mask decoder."""

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
        mask_n_queries: int = 128,
        mask_query_dim: int = 128,
        mask_num_blocks: int = 2,
        mask_num_heads: int = 4,
        mask_dropout: float = 0.1,
    ) -> None:
        super().__init__(
            uni_path=uni_path,
            flow_param=flow_param,
            img_size=img_size,
            pretrained_r34=pretrained_r34,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            hca_hidden_dim=hca_hidden_dim,
            hca_num_heads=hca_num_heads,
            hca_dropout=hca_dropout,
            hca_gamma_init=hca_gamma_init,
            hca_h_bias_init=hca_h_bias_init,
        )
        self.mask_n_queries = mask_n_queries
        self.mask_decoder = SamMaskDecoder(
            feat_dim=64,  # final upsample feat channels in NISFlowNetUNIHybridH
            query_dim=mask_query_dim,
            num_blocks=mask_num_blocks,
            num_heads=mask_num_heads,
            dropout=mask_dropout,
        )
        n_mask = sum(p.numel() for p in self.mask_decoder.parameters() if p.requires_grad)
        print(
            f"[B3 mask decoder] params={n_mask:,}; n_queries={mask_n_queries}; "
            f"q_dim={mask_query_dim} blocks={mask_num_blocks} heads={mask_num_heads}",
            flush=True,
        )

    def forward(self, x):
        # Standard B2.5 forward → feat, flow, fg, centerness, boundary.
        out = super().forward(x)

        # Training-time mask path: only if dict input contains instance_id.
        # Inference path is separate (called via forward_masks_from_points).
        if isinstance(x, dict) and ("instance_id" in x) and self.training:
            instance_id = x["instance_id"]
            centroids, targets, valid = _extract_centroids_and_targets(
                instance_id, n_queries=self.mask_n_queries
            )
            mask_out = self.mask_decoder(out["feat"], centroids)
            out["mask_logits"] = mask_out["mask_logits"]
            out["mask_iou_pred"] = mask_out["iou_pred"]
            out["mask_targets"] = targets
            out["mask_valid"] = valid
            out["mask_centroids"] = centroids
        return out

    def forward_masks_from_points(self, feat: torch.Tensor, points: torch.Tensor) -> dict:
        """Inference helper: given a feature map + N point prompts (normalized
        [0, 1] coords), return per-point mask logits and IoU."""
        return self.mask_decoder(feat, points)


__all__ = ["NISFlowNetUNIHybridHLoRAHCAMask", "_extract_centroids_and_targets"]
