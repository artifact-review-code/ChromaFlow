"""Pairwise boundary affinity loss adapted from BoNuS train_fine.

Strategy:
- Precompute PathIndex once per (crop_size, radius) at startup.
- During forward: take boundary_logits (B,1,H,W) → sigmoid → "edge map".
- For each pixel (src), look up paths to nearby destinations (dst).
- aff[src,dst] = 1 - max(edge along path) ∈ [0, 1]
- Target: if inst_id[src] == inst_id[dst] and both fg → pos label (aff should be ~1)
         if inst_id differ and both valid → neg label (aff should be ~0)
- Loss: BCE-style:
    pos_loss = -log(aff + eps) on pos pairs
    neg_loss = -log(1 - aff + eps) on neg pairs
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class PathIndex:
    """Adapted from BoNuS external/bonus/segmentation/indexing.py."""

    def __init__(self, radius: float, default_size: tuple[int, int]):
        self.radius = radius
        self.radius_floor = int(np.ceil(radius) - 1)
        self.search_paths, self.search_dst = self._get_search_paths_dst(int(radius))
        self.path_indices, self.src_indices, self.dst_indices = self._get_path_indices(default_size)

    def _get_search_paths_dst(self, max_radius: int):
        coord_indices_by_length = [[] for _ in range(max_radius * 4)]
        search_dirs = []
        for x in range(1, max_radius):
            search_dirs.append((0, x))
        for y in range(1, max_radius):
            for x in range(-max_radius + 1, max_radius):
                if x * x + y * y < max_radius ** 2:
                    search_dirs.append((y, x))
        for dir in search_dirs:
            length_sq = dir[0] ** 2 + dir[1] ** 2
            path_coords = []
            min_y, max_y = sorted((0, dir[0]))
            min_x, max_x = sorted((0, dir[1]))
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    dist_sq = (dir[0] * x - dir[1] * y) ** 2 / length_sq
                    if dist_sq < 1:
                        path_coords.append([y, x])
            path_coords.sort(key=lambda v: -abs(v[0]) - abs(v[1]))
            path_length = len(path_coords)
            coord_indices_by_length[path_length].append(path_coords)
        path_list_by_length = [np.asarray(v) for v in coord_indices_by_length if v]
        path_destinations = np.concatenate([p[:, 0] for p in path_list_by_length], axis=0)
        return path_list_by_length, path_destinations

    def _get_path_indices(self, size):
        full_indices = np.reshape(np.arange(0, size[0] * size[1], dtype=np.int64), (size[0], size[1]))
        cropped_height = size[0] - self.radius_floor
        cropped_width = size[1] - 2 * self.radius_floor
        path_indices = []
        for paths in self.search_paths:
            path_indices_list = []
            for p in paths:
                coord_indices_list = []
                for dy, dx in p:
                    coord_indices = full_indices[
                        dy:dy + cropped_height,
                        self.radius_floor + dx:self.radius_floor + dx + cropped_width
                    ]
                    coord_indices = np.reshape(coord_indices, [-1])
                    coord_indices_list.append(coord_indices)
                path_indices_list.append(coord_indices_list)
            path_indices.append(np.array(path_indices_list))
        src_indices = np.reshape(
            full_indices[:cropped_height, self.radius_floor:self.radius_floor + cropped_width],
            -1
        )
        dst_indices = np.concatenate([p[:, 0] for p in path_indices], axis=0)
        return path_indices, src_indices, dst_indices


def boundary_to_affinity(edge: torch.Tensor, path_index: PathIndex,
                          path_buffers: list[torch.Tensor]) -> torch.Tensor:
    """edge: (B, 1, H, W) sigmoid'd boundary map.
    path_buffers: list of tensors registered as buffers, one per path length.
    Returns: aff (B, n_paths, n_src) ∈ [0, 1]
    """
    edge_flat = edge.view(edge.size(0), -1)  # (B, H*W)
    aff_list = []
    for pi_buf in path_buffers:
        # pi_buf: (n_paths_of_this_length, path_length, n_src)
        ind_flat = pi_buf.view(-1)
        dist = torch.index_select(edge_flat, dim=-1, index=ind_flat)  # (B, n_paths * path_length * n_src)
        dist = dist.view(dist.size(0), pi_buf.size(0), pi_buf.size(1), pi_buf.size(2))
        # max pool over path_length, then aff = 1 - max
        aff = (1.0 - F.max_pool2d(dist, (dist.size(2), 1))).squeeze(2)
        aff_list.append(aff)
    return torch.cat(aff_list, dim=1)  # (B, n_total_paths, n_src)


def derive_affinity_labels(instance_id: torch.Tensor, path_index: PathIndex,
                             src_idx_buf: torch.Tensor, dst_idx_buf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """instance_id: (B, H, W) long, with 0 = bg/unknown, >0 = instance id.
    Returns:
        pos_label: (B, n_paths, n_src) float, 1 where src and dst share same valid instance
        neg_label: (B, n_paths, n_src) float, 1 where src and dst have different valid labels
    """
    B = instance_id.size(0)
    flat = instance_id.view(B, -1)
    label_from = flat[:, src_idx_buf]  # (B, n_src)
    # dst_idx_buf: (n_paths, n_src)
    label_to = flat[:, dst_idx_buf.view(-1)].view(B, dst_idx_buf.size(0), dst_idx_buf.size(1))  # (B, n_paths, n_src)
    label_from_e = label_from.unsqueeze(1).expand_as(label_to)  # (B, n_paths, n_src)

    valid = (label_from_e >= 0) & (label_to >= 0)
    same = (label_from_e == label_to)
    # Pos: same instance AND both > 0 (fg)
    pos = (same & (label_from_e > 0) & valid).float()
    # Neg: differ AND both valid
    neg = (~same & valid).float()
    return pos, neg


def affinity_loss(boundary_logits: torch.Tensor, instance_id: torch.Tensor,
                   path_index: PathIndex, path_buffers: list[torch.Tensor],
                   src_idx_buf: torch.Tensor, dst_idx_buf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (pos_loss, neg_loss) — average BCE on positive/negative pairs."""
    edge = torch.sigmoid(boundary_logits)
    aff = boundary_to_affinity(edge, path_index, path_buffers)
    pos_label, neg_label = derive_affinity_labels(instance_id, path_index, src_idx_buf, dst_idx_buf)

    eps = 1e-5
    pos_aff_loss = -torch.log(aff + eps)
    neg_aff_loss = -torch.log(1.0 + eps - aff)

    pos_denom = pos_label.sum().clamp(min=1.0)
    neg_denom = neg_label.sum().clamp(min=1.0)
    pos_l = (pos_label * pos_aff_loss).sum() / pos_denom
    neg_l = (neg_label * neg_aff_loss).sum() / neg_denom
    return pos_l, neg_l


if __name__ == "__main__":
    # Smoke
    pi = PathIndex(radius=5, default_size=(224, 224))
    print("n_path_lengths:", len(pi.path_indices))
    for i, p in enumerate(pi.path_indices):
        print(f"  length-{i}: shape={p.shape}")
    print("src_indices:", pi.src_indices.shape)
    print("dst_indices:", pi.dst_indices.shape)
