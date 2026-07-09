"""HoVer-Net-style horizontal/vertical distance maps for nuclei segmentation,
with stain-aware extensions.

Two pieces:
  1. compute_hv_target_from_instances: build (h_dist, v_dist) per-instance
     normalized signed distance maps from an instance label image.
  2. hv_watershed_decode: HoVer-Net-style post-processing from (hv_map, fg_prob),
     with optional H-channel-modulated marker selection.

Notation matches HoVer-Net:
  h_dist  ∈ [-1, 1]: per-instance horizontal (x) signed distance, normalized
                     so that the leftmost pixel of the instance is at -1 and
                     the rightmost at +1.
  v_dist  ∈ [-1, 1]: same for vertical (y).
  Outside instances: 0.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def compute_hv_target_torch(instance_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute hv-map target + fg mask from instance_id.

    Args:
        instance_id: (B, H, W) long tensor; 0 = background.

    Returns:
        hv_target: (B, 2, H, W) float — channel 0 = h_dist (x), channel 1 = v_dist (y).
        fg_mask:   (B, 1, H, W) float — 1 inside instances, 0 outside.
    """
    B, H, W = instance_id.shape
    device = instance_id.device
    hv = torch.zeros(B, 2, H, W, device=device, dtype=torch.float32)
    fg = (instance_id > 0).float().unsqueeze(1)

    for b in range(B):
        ids = torch.unique(instance_id[b])
        ids = ids[ids > 0]
        if ids.numel() == 0:
            continue
        for uid in ids:
            mask = instance_id[b] == uid
            ys, xs = torch.where(mask)
            if ys.numel() == 0:
                continue
            cy = ys.float().mean()
            cx = xs.float().mean()
            dy = ys.float() - cy
            dx = xs.float() - cx
            max_dy = dy.abs().max().clamp(min=1.0)
            max_dx = dx.abs().max().clamp(min=1.0)
            hv[b, 0, ys, xs] = (dx / max_dx).clamp(-1.0, 1.0)
            hv[b, 1, ys, xs] = (dy / max_dy).clamp(-1.0, 1.0)

    return hv, fg


def _sobel_gradient_np(img: np.ndarray) -> np.ndarray:
    """5×5 Sobel gradient magnitude. Used internally by hv_watershed_decode."""
    from scipy.ndimage import sobel
    gx = sobel(img.astype(np.float32), axis=1)
    gy = sobel(img.astype(np.float32), axis=0)
    return np.sqrt(gx * gx + gy * gy)


def hv_watershed_decode(
    hv_map: np.ndarray,
    fg_prob: np.ndarray,
    stain_h: np.ndarray | None = None,
    fg_thresh: float = 0.5,
    energy_thresh: float = 0.4,
    marker_h_thresh: float = 0.3,
    min_distance: int = 4,
    min_area: int = 15,
    max_area: int = 5000,
) -> np.ndarray:
    """HoVer-Net-style watershed on hv-map, with optional H-channel modulation.

    Args:
        hv_map:   (2, H, W) float, channel 0 = h_dist, channel 1 = v_dist.
        fg_prob:  (H, W) float in [0, 1] or logits (will be auto-detected).
        stain_h:  (H, W) float in [0, 1] (optional). If given, used to gate
                  marker selection: only seeds where stain_h > marker_h_thresh
                  are kept as watershed markers. This is the **H-modulated
                  marker selection** novelty.
        fg_thresh: threshold for the foreground mask.
        energy_thresh: threshold for the boundary energy (Sobel of hv-map).
        marker_h_thresh: H channel value above which a region can be a marker.
        min_distance: minimum pixel separation between markers.
        min_area, max_area: per-instance area filtering.

    Returns:
        instance_id: (H, W) uint16, 0 = background, 1..N = instance labels.
    """
    from scipy import ndimage as ndi
    from skimage.morphology import remove_small_objects
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max

    # Foreground mask (auto-detect logits vs prob).
    fg_arr = fg_prob.astype(np.float32)
    if fg_arr.ndim == 3:
        fg_arr = fg_arr[0]
    if fg_arr.min() < 0:
        fg_arr = 1.0 / (1.0 + np.exp(-fg_arr))
    fg_mask = fg_arr > fg_thresh
    if not fg_mask.any():
        return np.zeros_like(fg_arr, dtype=np.uint16)

    # Boundary energy: max(|d(h)/dx|, |d(v)/dy|) — gradient peaks at instance edges.
    h_map = hv_map[0] if hv_map.ndim == 3 else hv_map[:, :, 0]
    v_map = hv_map[1] if hv_map.ndim == 3 else hv_map[:, :, 1]
    grad_h = np.abs(_sobel_gradient_np(h_map))
    grad_v = np.abs(_sobel_gradient_np(v_map))
    grad_h = grad_h / max(grad_h.max(), 1e-6)
    grad_v = grad_v / max(grad_v.max(), 1e-6)
    energy = np.maximum(grad_h, grad_v)

    # Interior of instances (away from boundaries) = low-energy fg regions.
    interior = fg_mask & (energy < energy_thresh)

    # Marker seeds: local maxima of (fg_prob * (1 - energy)).
    inv_energy = (1.0 - energy) * fg_arr
    if stain_h is not None:
        stain_arr = np.asarray(stain_h, dtype=np.float32)
        if stain_arr.ndim == 3:
            stain_arr = stain_arr[0]
        if stain_arr.max() > 1.0:
            stain_arr = stain_arr / max(stain_arr.max(), 1.0)
        # H-modulated marker selection: scale inv_energy by H so seeds prefer
        # hematoxylin-rich regions (nuclei).
        h_gate = (stain_arr > marker_h_thresh).astype(np.float32)
        inv_energy = inv_energy * (0.3 + 0.7 * stain_arr) * (interior.astype(np.float32)) * h_gate
    else:
        inv_energy = inv_energy * interior.astype(np.float32)

    seed_coords = peak_local_max(
        inv_energy, min_distance=min_distance, threshold_abs=0.05, labels=fg_mask.astype(np.int32)
    )
    if seed_coords.size == 0:
        return np.zeros_like(fg_arr, dtype=np.uint16)
    seed_mask = np.zeros_like(fg_arr, dtype=np.int32)
    seed_mask[seed_coords[:, 0], seed_coords[:, 1]] = np.arange(1, seed_coords.shape[0] + 1)

    # Watershed elevation: gradient energy (high at boundaries).
    labels = watershed(energy, markers=seed_mask, mask=fg_mask)

    # Area filter.
    out = np.zeros_like(labels, dtype=np.uint16)
    next_id = 1
    for uid in range(1, labels.max() + 1):
        m = labels == uid
        area = int(m.sum())
        if min_area <= area <= max_area:
            out[m] = next_id
            next_id += 1
    return out


__all__ = ["compute_hv_target_torch", "hv_watershed_decode"]
