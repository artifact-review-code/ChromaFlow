"""Seed-anchored decoding variants for R1.1 inference-only evaluation.
Runs ONE forward pass per image, then evaluates multiple decoding strategies (S0..S4)
that share the same fg / flow / centerness outputs. Variant selection should use VAL only.

S0: current Cellpose dynamics (free attractor) — reference
S1: NMS peaks as seeds; for each fg pixel x, endpoint e(x) = x + tau * flow(x); assign to nearest seed
S2: multi-step flow integration (T iterations) then nearest seed (uses cellpose dynamics internally for trajectory)
S3: S2 + reject pixels whose endpoint is farther than r_max from any seed
S4: Cellpose basin + seed-Voronoi within basin (current reconcile, but configurable drop_zero_peak)

Center P/R/F1, zero/multi-peak basin ratios, n_pred/n_gt all reported per variant.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from skimage.feature import peak_local_max
from skimage.segmentation import expand_labels
from skimage.morphology import binary_closing, binary_opening, disk
from scipy import ndimage as _ndi


def _smooth_instances(inst: np.ndarray, radius: int = 1, opening_radius: int = 0) -> np.ndarray:
    """Per-instance fill-holes + binary-closing with disk(radius).

    Mirrors BoNuS post-process pattern that produces smooth, near-elliptical
    masks. Operates per-instance to avoid merging neighbors. New pixels added
    by closing only fill into background (label==0), never overwrite another
    instance.

    Args:
        inst: (H, W) uint16/int32 instance label map.
        radius: morphological closing footprint radius (1 = light, 2 = stronger).

    Returns:
        Smoothed instance map (same dtype).
    """
    out = inst.copy()
    footprint = disk(radius)
    # Sort by area to fill bigger instances first so smaller ones can re-occupy
    # boundary pixels if there's contention.
    uids = [u for u in np.unique(inst) if u != 0]
    for uid in uids:
        m = (inst == uid)
        m = _ndi.binary_fill_holes(m)
        if opening_radius > 0:
            m = binary_opening(m, footprint=disk(opening_radius))
        m = binary_closing(m, footprint=footprint)
        # Only write into background or this instance's original pixels.
        write_mask = m & ((out == 0) | (out == uid))
        out[write_mask] = uid
    return out

from chromaflow.dataset import normalize_image, build_input_tensor
from chromaflow.eval import (
    DEFAULT_INFERENCE_CONFIG,
    METRIC_NAMES,
    _predict_full_outputs,
    _predict_tiled_outputs_blend,
    compute_metrics,
    load_eval_items,
    load_checkpoint,
    summarize,
)
from chromaflow.model import NISFlowNet
from chromaflow.uni_encoder import NISFlowNetUNI
from chromaflow.uni_encoder_hybrid import (
    NISFlowNetUNIHybrid, NISFlowNetUNIHybridIGA,
    NISFlowNetUNIHybridH, NISFlowNetUNIHybridHPSC2F, NISFlowNetUNIHybridHPSIC,
    NISFlowNetUNIHybridHMTA, NISFlowNetUNIHybridHU2HSRG,
    NISFlowNetUNIHybridHUBIM, NISFlowNetUNIHybridHUCAF, NISFlowNetUNIHybridHUDAC,
    NISFlowNetUNIHybridHUDAR, NISFlowNetUNIHybridHURCF, NISFlowNetUNIHybridHURCFV2,
)


# -- centerness peaks -------------------------------------------------------
def nms_peaks(center_logits: np.ndarray, threshold: float, min_distance: int) -> np.ndarray:
    center_prob = 1.0 / (1.0 + np.exp(-center_logits))
    return peak_local_max(
        center_prob, min_distance=min_distance, threshold_abs=threshold, exclude_border=False
    )  # (K, 2) (y, x)


# -- decoding variants ------------------------------------------------------
def _pcgs_refine_cellprob(cellprob_np: np.ndarray, cfg: dict) -> np.ndarray:
    """PCGS test-time refinement (Module 1.6).

    Per-image: use the model's own confident predictions as fg/bg anchor regions,
    fit Gaussian likelihoods on H_norm and (1-E_norm)/E_norm, compute Bayesian
    posterior on all pixels, temperature-sharpen, and blend with the original
    cellprob. Closes the gap between training-time PCGS (uses BoNuS oracle anchors)
    and test-time inference (no clu labels available).

    Args:
        cellprob_np: (H, W) raw logits (cellprob_mode=logits) or sigmoid (prob).
        cfg: decode config; expects polarize_h / polarize_e arrays, plus
             pcgs_refine_T, pcgs_refine_blend, pcgs_fg_thresh, pcgs_bg_thresh.
    Returns:
        refined cellprob in same scale as input.
    """
    h = cfg.get("polarize_h")
    e = cfg.get("polarize_e")
    if h is None or e is None:
        return cellprob_np

    # Original prob (for anchor selection + blending).
    if cfg["cellprob_mode"] == "logits":
        cellprob_prob = 1.0 / (1.0 + np.exp(-cellprob_np))
    else:
        cellprob_prob = cellprob_np

    def _n01(x):
        lo, hi = float(x.min()), float(x.max())
        return ((x - lo) / max(hi - lo, 1e-6)).astype(np.float32)
    h_n = _n01(h)
    e_n = _n01(e)
    e_comp = 1.0 - e_n

    fg_thresh = float(cfg.get("pcgs_fg_thresh", 0.7))
    bg_thresh = float(cfg.get("pcgs_bg_thresh", 0.3))
    fg_mask = cellprob_prob > fg_thresh
    bg_mask = cellprob_prob < bg_thresh

    def _gfit(vals, mask, mu0, sg0):
        v = vals[mask]
        if v.size < 64:
            return mu0, sg0
        return float(v.mean()), float(np.clip(v.std(), 0.05, 0.40))

    mu_Hf, sg_Hf = _gfit(h_n,    fg_mask, 0.65, 0.15)
    mu_Ef, sg_Ef = _gfit(e_comp, fg_mask, 0.60, 0.18)
    mu_Hb, sg_Hb = _gfit(h_n,    bg_mask, 0.30, 0.18)
    mu_Eb, sg_Eb = _gfit(e_n,    bg_mask, 0.55, 0.20)

    def _gpdf(x, mu, sg):
        return np.exp(-0.5 * ((x - mu) / sg) ** 2) / (sg * np.sqrt(2.0 * np.pi))

    lik_fg = _gpdf(h_n, mu_Hf, sg_Hf) * _gpdf(e_comp, mu_Ef, sg_Ef)
    lik_bg = _gpdf(h_n, mu_Hb, sg_Hb) * _gpdf(e_n,    mu_Eb, sg_Eb)
    posterior = 0.5 * lik_fg / (0.5 * lik_fg + 0.5 * lik_bg + 1e-8)

    T = float(cfg.get("pcgs_refine_T", 2.0))
    if T != 1.0 and T > 0:
        pc = np.clip(posterior, 1e-3, 1.0 - 1e-3)
        posterior = 1.0 / (1.0 + np.exp(-T * np.log(pc / (1.0 - pc))))
    posterior = np.clip(posterior, 0.01, 0.99).astype(cellprob_prob.dtype)

    blend = float(cfg.get("pcgs_refine_blend", 0.5))
    refined_prob = blend * cellprob_prob + (1.0 - blend) * posterior

    if cfg["cellprob_mode"] == "logits":
        return np.log(refined_prob / (1.0 - refined_prob)).astype(cellprob_np.dtype)
    return refined_prob.astype(cellprob_np.dtype)


def decode_S0(fg_logits, flow, center_logits, cfg):
    """Cellpose dynamics (free attractor, no reconcile)."""
    from cellpose import dynamics

    d_p = flow[[1, 0]] * float(cfg["flow_scale"])
    cellprob = fg_logits if cfg["cellprob_mode"] == "logits" else 1.0 / (1.0 + np.exp(-fg_logits))
    # PCGS test-time refinement (Module 1.6).
    if cfg.get("pcgs_refine", False):
        cellprob = _pcgs_refine_cellprob(cellprob, cfg)
    # Polarization-guided cellprob (test-time, optional, default no-op).
    # Multiplies cellprob by (H_norm + eps) ** beta, where H/E are stain channels
    # passed via cfg["polarize_h"] / cfg["polarize_e"]. cfg["polarize_beta"]=0 disables.
    pol_beta = float(cfg.get("polarize_beta", 0.0) or 0.0)
    if pol_beta != 0.0 and cfg.get("polarize_h") is not None:
        pol_mode = cfg.get("polarize_mode", "log")
        h = cfg["polarize_h"]; e = cfg.get("polarize_e")
        def _n01(x):
            lo, hi = float(x.min()), float(x.max())
            return ((x - lo) / max(hi - lo, 1e-6)).astype(cellprob.dtype)
        h_n = _n01(h)
        e_n = _n01(e) if e is not None else None
        if pol_mode == "log":
            factor = (h_n + 0.01) ** pol_beta
        elif pol_mode == "linear":
            if e_n is None:
                factor = 1.0 + pol_beta * h_n
            else:
                factor = 1.0 + pol_beta * (h_n - e_n)
            factor = np.clip(factor, 0.01, None)
        elif pol_mode == "diff_log":
            diff = (h_n - (e_n if e_n is not None else 0.0)) + 0.5
            factor = np.clip(diff, 0.01, None) ** pol_beta
        else:
            raise ValueError(f"Unknown polarize_mode: {pol_mode}")
        cellprob = cellprob * factor
    # EWB — E-channel Watershed Barrier (Module C, decode-only).
    # Pre-flight v3 falsified raw E gating (ratio 0.70). The signal that
    # discriminates touching-ridge vs inside-FG is the H-valley (1 - H_smooth),
    # pooled ratio 6.42 (median 6.99). Cellprob is reduced by beta * gate(H_valley)
    # before cellpose dynamics, so touching basins separate without damaging
    # in-instance FG (which has gate ~ 0.03).
    ewb_beta = float(cfg.get("ewb_beta", 0.0) or 0.0)
    if ewb_beta != 0.0 and cfg.get("polarize_h") is not None:
        from scipy.ndimage import uniform_filter as _uf
        h_raw = cfg["polarize_h"]
        lo, hi = float(h_raw.min()), float(h_raw.max())
        h_n = ((h_raw - lo) / max(hi - lo, 1e-6)).astype(cellprob.dtype)
        h_smooth = _uf(h_n, size=3, mode="reflect")
        hv = 1.0 - h_smooth  # H-valley in [0,1]
        ewb_q = float(cfg.get("ewb_quantile", 0.70))
        ewb_gamma = float(cfg.get("ewb_gamma", 8.0))
        thresh = float(np.quantile(hv, ewb_q))
        gate = 1.0 / (1.0 + np.exp(-ewb_gamma * (hv - thresh)))
        gate = gate.astype(cellprob.dtype)
        cellprob = cellprob - ewb_beta * gate
    result = dynamics.compute_masks(
        dP=d_p,
        cellprob=cellprob,
        niter=int(cfg["niter"]),
        cellprob_threshold=float(cfg["cellprob_threshold"]),
        flow_threshold=float(cfg["flow_threshold"]),
        min_size=int(cfg["min_size"]),
        device=cfg["device"],
    )
    masks = result[0] if isinstance(result, tuple) else result
    return np.asarray(masks, dtype=np.uint16)


def decode_S4(fg_logits, flow, center_logits, cfg, drop_zero_peak: bool = False):
    """Cellpose basin + seed-Voronoi within basin. drop_zero_peak=False keeps singleton basins."""
    masks_S0 = decode_S0(fg_logits, flow, center_logits, cfg)
    peaks = nms_peaks(center_logits, cfg["seed_threshold"], cfg["seed_min_distance"])
    if len(peaks) == 0:
        return np.zeros_like(masks_S0) if drop_zero_peak else masks_S0
    labels = masks_S0.astype(np.int64)
    peak_cc = labels[peaks[:, 0], peaks[:, 1]]
    cc_peak_idx: dict[int, list[int]] = {}
    for i, cc in enumerate(peak_cc.tolist()):
        if cc == 0:
            continue
        cc_peak_idx.setdefault(int(cc), []).append(i)
    refined = np.zeros_like(labels)
    next_id = 1
    for cc in np.unique(labels):
        cc = int(cc)
        if cc == 0:
            continue
        cc_mask = labels == cc
        idxs = cc_peak_idx.get(cc, [])
        if len(idxs) == 0:
            if drop_zero_peak:
                continue
            refined[cc_mask] = next_id
            next_id += 1
            continue
        if len(idxs) == 1:
            refined[cc_mask] = next_id
            next_id += 1
            continue
        yy, xx = np.where(cc_mask)
        cc_peaks_arr = peaks[idxs]
        dist = (yy[:, None] - cc_peaks_arr[None, :, 0]) ** 2 + (xx[:, None] - cc_peaks_arr[None, :, 1]) ** 2
        nearest = dist.argmin(axis=1)
        for k in range(len(idxs)):
            sel = nearest == k
            refined[yy[sel], xx[sel]] = next_id
            next_id += 1
    return refined.astype(np.uint16)


def decode_S1(fg_logits, flow, center_logits, cfg):
    """One-step endpoint: e(x) = x + tau * flow(x); nearest NMS seed; bg if no fg or no seed."""
    fg_prob = 1.0 / (1.0 + np.exp(-fg_logits))
    fg_mask = fg_prob > cfg.get("fg_thresh", 0.5)
    peaks = nms_peaks(center_logits, cfg["seed_threshold"], cfg["seed_min_distance"])
    H, W = fg_logits.shape
    out = np.zeros((H, W), dtype=np.uint16)
    if len(peaks) == 0 or fg_mask.sum() == 0:
        return out
    tau = float(cfg.get("endpoint_tau", cfg["flow_scale"]))
    ys, xs = np.where(fg_mask)
    # flow channels: 0=dx, 1=dy
    dx = flow[0]
    dy = flow[1]
    ex = xs + tau * dx[ys, xs]
    ey = ys + tau * dy[ys, xs]
    # nearest seed via KDTree
    from scipy.spatial import cKDTree

    tree = cKDTree(peaks)  # peaks: (K, 2) [y, x]
    pts = np.stack([ey, ex], axis=1)
    dist, idx = tree.query(pts, k=1)
    out[ys, xs] = (idx + 1).astype(np.uint16)
    return out


def decode_S2(fg_logits, flow, center_logits, cfg):
    """Multi-step integration with cellpose dynamics, then nearest seed of final coords.
    Uses cellpose.dynamics.steps2D under the hood to get trajectories.
    """
    fg_prob = 1.0 / (1.0 + np.exp(-fg_logits))
    fg_mask = fg_prob > cfg.get("fg_thresh", 0.5)
    peaks = nms_peaks(center_logits, cfg["seed_threshold"], cfg["seed_min_distance"])
    H, W = fg_logits.shape
    out = np.zeros((H, W), dtype=np.uint16)
    if len(peaks) == 0 or fg_mask.sum() == 0:
        return out
    # Use cellpose's GPU step solver — but we want endpoints not labels.
    # We can mimic via simple loop on GPU with grid_sample.
    device = cfg.get("device", torch.device("cpu"))
    dx_t = torch.from_numpy(flow[0]).to(device).float()
    dy_t = torch.from_numpy(flow[1]).to(device).float()
    ys, xs = np.where(fg_mask)
    n = ys.shape[0]
    ex = torch.from_numpy(xs.astype(np.float32)).to(device)
    ey = torch.from_numpy(ys.astype(np.float32)).to(device)
    T = int(cfg.get("endpoint_steps", 30))
    eta = float(cfg.get("endpoint_eta", 1.0))
    scale = float(cfg.get("flow_scale", 70.0))
    # Use scale per cellpose convention: each step adds flow * (scale / T_norm)
    # We follow cellpose default of unit step in pixels using the (scaled) flow
    step = eta * scale / T
    for _ in range(T):
        # bilinear sample at (ey, ex)
        gx = ex.clamp(0, W - 1)
        gy = ey.clamp(0, H - 1)
        # use grid_sample
        norm_x = (gx / (W - 1)) * 2 - 1
        norm_y = (gy / (H - 1)) * 2 - 1
        grid = torch.stack([norm_x, norm_y], dim=-1).view(1, 1, n, 2)
        dxs = torch.nn.functional.grid_sample(dx_t.view(1, 1, H, W), grid, mode="bilinear", align_corners=True).view(n)
        dys = torch.nn.functional.grid_sample(dy_t.view(1, 1, H, W), grid, mode="bilinear", align_corners=True).view(n)
        ex = ex + step * dxs
        ey = ey + step * dys
    ex_np = ex.cpu().numpy()
    ey_np = ey.cpu().numpy()
    from scipy.spatial import cKDTree

    tree = cKDTree(peaks)
    pts = np.stack([ey_np, ex_np], axis=1)
    dist, idx = tree.query(pts, k=1)
    labels = (idx + 1).astype(np.uint16)
    if cfg.get("endpoint_reject_radius", 0) > 0:
        r_max = float(cfg["endpoint_reject_radius"])
        labels[dist > r_max] = 0
    out[ys, xs] = labels
    return out


def decode_S3(fg_logits, flow, center_logits, cfg):
    cfg2 = {**cfg, "endpoint_reject_radius": cfg.get("endpoint_reject_radius", 12.0)}
    return decode_S2(fg_logits, flow, center_logits, cfg2)


VARIANTS = {
    "S0": lambda fg, fl, ce, cfg: decode_S0(fg, fl, ce, cfg),
    "S1": lambda fg, fl, ce, cfg: decode_S1(fg, fl, ce, cfg),
    "S2": lambda fg, fl, ce, cfg: decode_S2(fg, fl, ce, cfg),
    "S3": lambda fg, fl, ce, cfg: decode_S3(fg, fl, ce, cfg),
    "S4_drop": lambda fg, fl, ce, cfg: decode_S4(fg, fl, ce, cfg, drop_zero_peak=True),
    "S4_keep": lambda fg, fl, ce, cfg: decode_S4(fg, fl, ce, cfg, drop_zero_peak=False),
}


# -- diagnostics ------------------------------------------------------------
def basin_peak_stats(masks_S0: np.ndarray, peaks: np.ndarray) -> dict[str, float]:
    """zero/single/multi-peak basin ratios from S0 cellpose basins + NMS peaks."""
    n_basins = int(masks_S0.max())
    if n_basins == 0:
        return {"n_basins": 0, "zero_peak": 0.0, "single_peak": 0.0, "multi_peak": 0.0}
    if len(peaks) == 0:
        return {"n_basins": n_basins, "zero_peak": 1.0, "single_peak": 0.0, "multi_peak": 0.0}
    pcc = masks_S0[peaks[:, 0], peaks[:, 1]]
    cc_count: dict[int, int] = {}
    for c in pcc:
        if c == 0:
            continue
        cc_count[int(c)] = cc_count.get(int(c), 0) + 1
    n0 = sum(1 for c in range(1, n_basins + 1) if cc_count.get(c, 0) == 0)
    n1 = sum(1 for c in range(1, n_basins + 1) if cc_count.get(c, 0) == 1)
    nm = sum(1 for c in range(1, n_basins + 1) if cc_count.get(c, 0) >= 2)
    return {
        "n_basins": n_basins,
        "zero_peak": n0 / n_basins,
        "single_peak": n1 / n_basins,
        "multi_peak": nm / n_basins,
    }


def center_pr_f1(peaks: np.ndarray, gt_points: np.ndarray, match_radius: float = 8.0) -> dict[str, float]:
    """Greedy 1-1 match of NMS peaks to GT points within match_radius.
    Returns precision/recall/F1.
    """
    n_pred = len(peaks)
    n_gt = len(gt_points)
    if n_pred == 0 and n_gt == 0:
        return {"center_P": 1.0, "center_R": 1.0, "center_F1": 1.0, "n_seeds": 0, "n_gt_pts": 0}
    if n_pred == 0:
        return {"center_P": 0.0, "center_R": 0.0, "center_F1": 0.0, "n_seeds": 0, "n_gt_pts": n_gt}
    if n_gt == 0:
        return {"center_P": 0.0, "center_R": 0.0, "center_F1": 0.0, "n_seeds": n_pred, "n_gt_pts": 0}
    from scipy.spatial import cKDTree

    tree_gt = cKDTree(gt_points)
    dist, idx = tree_gt.query(peaks, k=1)
    matched_gt: set[int] = set()
    tp = 0
    order = np.argsort(dist)
    for k in order:
        if dist[k] <= match_radius and int(idx[k]) not in matched_gt:
            matched_gt.add(int(idx[k]))
            tp += 1
    fp = n_pred - tp
    fn = n_gt - tp
    P = tp / max(1, n_pred)
    R = tp / max(1, n_gt)
    F1 = 2 * P * R / max(1e-6, P + R)
    return {"center_P": P, "center_R": R, "center_F1": F1, "n_seeds": n_pred, "n_gt_pts": n_gt}


# -- main entrypoint --------------------------------------------------------
def predict_outputs(model, image_path: Path, cfg: dict[str, Any], device: torch.device):
    image = Image.open(image_path).convert("RGB")
    # Optional Macenko stain normalization to align target-domain to source-domain.
    if cfg.get("stain_norm") is not None:
        _arr = np.array(image)
        try:
            _normalized, _, _ = cfg["stain_norm"].normalize(_arr, stains=False)
            image = Image.fromarray(np.asarray(_normalized).astype(np.uint8))
        except Exception as _e:
            # Some patches lack enough tissue (e.g. blank CoNIC patches); skip norm.
            pass
    _iscale = float(cfg.get("input_scale", 1.0))
    if _iscale != 1.0:
        _w, _h = image.size
        image = image.resize((max(1, round(_w * _iscale)), max(1, round(_h * _iscale))), Image.BILINEAR)
    tile_size = int(cfg.get("tile_size", 0))
    input_mode = str(cfg.get("input_mode", "rgb"))
    if tile_size > 0:
        fg, fl, ce = _predict_tiled_outputs_blend(
            model,
            image,
            device,
            tile_size,
            int(cfg.get("tile_overlap", 64)),
            int(cfg.get("tile_batch_size", 16)),
            input_mode,
        )
    else:
        fg, fl, ce = _predict_full_outputs(model, image, device, input_mode)
    return fg, fl, ce


def load_gt_points(gt_instance: np.ndarray) -> np.ndarray:
    # Per-instance centroid from a GT instance label map.
    ids = [int(v) for v in np.unique(gt_instance) if int(v) != 0]
    if not ids:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.zeros((len(ids), 2), dtype=np.float32)
    for i, iid in enumerate(ids):
        ys, xs = np.where(gt_instance == iid)
        pts[i, 0] = float(ys.mean())
        pts[i, 1] = float(xs.mean())
    return pts


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--dataset", choices=["MO", "CPM", "CONIC", "NUINSSEG"], required=True)
    p.add_argument("--split", nargs="+", default=["val"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tta", type=str, default="hflip,vflip",
                   help="TTA modes, comma-separated: hflip,vflip,hvflip. Empty disables.")
    p.add_argument("--variants", nargs="+", default=["S4_drop"])
    p.add_argument("--seed-threshold", type=float, default=0.4)
    p.add_argument("--seed-min-distance", type=int, default=8)
    p.add_argument("--center-match-radius", type=float, default=8.0)
    p.add_argument("--endpoint-tau", type=float, default=70.0)
    p.add_argument("--endpoint-steps", type=int, default=30)
    p.add_argument("--endpoint-reject-radius", type=float, default=12.0)
    p.add_argument("--fg-thresh", type=float, default=0.5)
    # Polarization-guided cellprob (test-time SCS extension).
    p.add_argument("--polarize-beta", type=float, default=0.0,
                   help="Multiply cellprob by (H_norm+eps)**beta (log mode) before dynamics. 0=disabled.")
    p.add_argument("--ewb-beta", type=float, default=0.0,
                   help="EWB: cellprob -= beta * gate(1 - H_smooth). 0=disabled.")
    p.add_argument("--ewb-gamma", type=float, default=8.0,
                   help="EWB: sigmoid sharpness for gate; ignored if --ewb-beta=0.")
    p.add_argument("--ewb-quantile", type=float, default=0.70,
                   help="EWB: per-image quantile threshold on H-valley; ignored if --ewb-beta=0.")
    p.add_argument("--sip-tau", type=float, default=0.0,
                   help="SIP: drop predicted instances with mean interior H̃ < tau. 0=disabled.")
    p.add_argument("--sip-min-area", type=int, default=0,
                   help="SIP: keep instance regardless of H̃ if area >= this (size override). 0=off.")
    p.add_argument("--polarize-mode", default="log", choices=["log", "linear", "diff_log"],
                   help="Polarization weighting formula.")
    # PCGS test-time refinement (Module 1.6) — Bayesian posterior calibration of
    # cellprob using the model's own confident predictions as fg/bg anchor regions.
    p.add_argument("--pcgs-refine", action="store_true",
                   help="Enable PCGS test-time refinement: blend cellprob with Bayesian posterior derived from per-image H/E likelihood anchored on model's own confident fg/bg pixels.")
    p.add_argument("--pcgs-refine-T", type=float, default=2.0,
                   help="Temperature for posterior sharpening at refinement time. T=1 raw posterior, T>1 sharpens toward binary.")
    p.add_argument("--pcgs-refine-blend", type=float, default=0.5,
                   help="Blend weight in [0,1]: refined = blend·cellprob + (1-blend)·posterior_T. 0 = posterior-only, 1 = no refine.")
    p.add_argument("--pcgs-fg-thresh", type=float, default=0.7,
                   help="Sigmoid threshold above which model output is treated as fg anchor.")
    p.add_argument("--pcgs-bg-thresh", type=float, default=0.3,
                   help="Sigmoid threshold below which model output is treated as bg anchor.")
    p.add_argument("--flow-scale", type=float, default=70.0)
    p.add_argument("--flow-from", default="model",
                   choices=["model", "poisson_gt", "poisson_center", "poisson_gtflow", "gtflow_direct", "poisson_pointstain", "pointstain_direct"],
                   help="Stage-0 probe: override model flow with a Poisson-derived flow.")
    p.add_argument("--poisson-sigma", type=float, default=3.0)
    p.add_argument("--poisson-grad", default="central", choices=["central", "forward"])
    p.add_argument("--poisson-pad", type=int, default=0)
    p.add_argument("--h-thresh", type=float, default=0.0)
    p.add_argument("--h-power", type=float, default=1.0)
    p.add_argument("--points", default="gt", choices=["gt", "pred"])
    p.add_argument("--niter", type=int, default=30)
    p.add_argument("--boundary-decode-alpha", type=float, default=0.0,
                   help="V4_BND: cellprob -= alpha * sigmoid(boundary_logits) before dynamics.")
    p.add_argument("--min-size", type=int, default=15)
    p.add_argument("--cellprob-threshold", type=float, default=0.0)
    p.add_argument("--flow-threshold", type=float, default=0.0)
    p.add_argument("--cellprob-mode", default="logits")
    p.add_argument("--tile-size", type=int, default=224)
    p.add_argument("--tile-overlap", type=int, default=64)
    p.add_argument("--tile-batch-size", type=int, default=16)
    p.add_argument("--input-scale", type=float, default=1.0,
                   help="Test-time input upsampling factor (2.0 = 2x). Pred resized back to GT res (nearest) before metrics; AJI is scale-invariant.")
    # Post-process: expand each instance label by N pixels (distance-bounded,
    # safe against merging with neighbor instances). Targets SQ gap.
    p.add_argument("--mask-dilation", type=int, default=1,
                   help="Per-instance label expansion radius in pixels. 0=off.")
    p.add_argument("--mask-smooth", type=int, default=0,
                   help="Per-instance fill_holes + binary_closing(disk(N)) radius. 0=off.")
    p.add_argument("--mask-opening", type=int, default=0,
                   help="Per-instance binary_opening(disk(N)) before closing — removes small protrusions.")
    p.add_argument("--stain-template", type=str, default="",
                   help="Path to .npz with HE+maxC for Macenko normalization (test-time stain "
                        "alignment for cross-domain inference).")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_checkpoint(Path(args.checkpoint), device)
    ckpt_cfg = loaded["checkpoint"].get("config", {}) if isinstance(loaded["checkpoint"], dict) else {}
    flow_param = ckpt_cfg.get("flow_param", "direct")
    enc = ckpt_cfg.get("encoder", "resnet34")
    if enc == "sacaf":
        from chromaflow.model_sacaf import NISFlowNetSACAF
        model = NISFlowNetSACAF(pretrained=False, flow_param=flow_param)
    elif enc == "r34dual":
        from chromaflow.model_r34dual import NISFlowNetR34Dual
        model = NISFlowNetR34Dual(pretrained=False, flow_param=flow_param)
    elif enc == "latef":
        from chromaflow.model_r34latef import NISFlowNetLateFusionR34
        model = NISFlowNetLateFusionR34(pretrained=False, flow_param=flow_param)
    elif enc == "uni":
        uni_path = ckpt_cfg.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_p = uni_path if Path(uni_path).is_absolute() else str(root / uni_path)
        model = NISFlowNetUNI(uni_path=uni_p, flow_param=flow_param)
    elif enc == "uni_hybrid":
        uni_path = ckpt_cfg.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_p = uni_path if Path(uni_path).is_absolute() else str(root / uni_path)
        model = NISFlowNetUNIHybrid(uni_path=uni_p, flow_param=flow_param, pretrained_r34=False)
    elif enc == "uni_hybrid_iga":
        uni_path = ckpt_cfg.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_p = uni_path if Path(uni_path).is_absolute() else str(root / uni_path)
        model = NISFlowNetUNIHybridIGA(uni_path=uni_p, flow_param=flow_param, pretrained_r34=False)
    elif enc == "uni_hybrid_h":
        uni_path = ckpt_cfg.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_p = uni_path if Path(uni_path).is_absolute() else str(root / uni_path)
        model = NISFlowNetUNIHybridH(uni_path=uni_p, flow_param=flow_param, pretrained_r34=False)
    elif enc == "uni_hybrid_h_psc2f":
        uni_path = ckpt_cfg.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_p = uni_path if Path(uni_path).is_absolute() else str(root / uni_path)
        model = NISFlowNetUNIHybridHPSC2F(uni_path=uni_p, flow_param=flow_param, pretrained_r34=False)
    elif enc == "uni_hybrid_h_psic":
        uni_path = ckpt_cfg.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_p = uni_path if Path(uni_path).is_absolute() else str(root / uni_path)
        model = NISFlowNetUNIHybridHPSIC(uni_path=uni_p, flow_param=flow_param, pretrained_r34=False)
    elif enc == "uni_hybrid_h_mta":
        uni_path = ckpt_cfg.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_p = uni_path if Path(uni_path).is_absolute() else str(root / uni_path)
        model = NISFlowNetUNIHybridHMTA(uni_path=uni_p, flow_param=flow_param, pretrained_r34=False)
    elif enc == "uni_hybrid_h_u2h_srg":
        uni_path = ckpt_cfg.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_p = uni_path if Path(uni_path).is_absolute() else str(root / uni_path)
        model = NISFlowNetUNIHybridHU2HSRG(uni_path=uni_p, flow_param=flow_param, pretrained_r34=False)
    elif enc == "uni_hybrid_h_ubim":
        uni_path = ckpt_cfg.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_p = uni_path if Path(uni_path).is_absolute() else str(root / uni_path)
        model = NISFlowNetUNIHybridHUBIM(uni_path=uni_p, flow_param=flow_param, pretrained_r34=False)
    elif enc == "uni_hybrid_h_ucaf":
        uni_path = ckpt_cfg.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_p = uni_path if Path(uni_path).is_absolute() else str(root / uni_path)
        model = NISFlowNetUNIHybridHUCAF(uni_path=uni_p, flow_param=flow_param, pretrained_r34=False)
    elif enc == "uni_hybrid_h_udac":
        uni_path = ckpt_cfg.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_p = uni_path if Path(uni_path).is_absolute() else str(root / uni_path)
        model = NISFlowNetUNIHybridHUDAC(uni_path=uni_p, flow_param=flow_param, pretrained_r34=False)
    elif enc == "uni_hybrid_h_udar":
        uni_path = ckpt_cfg.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_p = uni_path if Path(uni_path).is_absolute() else str(root / uni_path)
        model = NISFlowNetUNIHybridHUDAR(uni_path=uni_p, flow_param=flow_param, pretrained_r34=False)
    elif enc == "uni_hybrid_h_urcf":
        uni_path = ckpt_cfg.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_p = uni_path if Path(uni_path).is_absolute() else str(root / uni_path)
        model = NISFlowNetUNIHybridHURCF(uni_path=uni_p, flow_param=flow_param, pretrained_r34=False)
    elif enc == "uni_hybrid_h_urcf_v2":
        uni_path = ckpt_cfg.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_p = uni_path if Path(uni_path).is_absolute() else str(root / uni_path)
        model = NISFlowNetUNIHybridHURCFV2(uni_path=uni_p, flow_param=flow_param, pretrained_r34=False)
    elif enc == "uni_hybrid_h_lora":
        from chromaflow.uni_encoder_b1 import NISFlowNetUNIHybridHLoRA
        uni_path = ckpt_cfg.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_p = uni_path if Path(uni_path).is_absolute() else str(root / uni_path)
        model = NISFlowNetUNIHybridHLoRA(uni_path=uni_p, flow_param=flow_param, pretrained_r34=False,
                                          lora_rank=16, lora_alpha=16, lora_dropout=0.1)
    elif enc == "uni_hybrid_h_lora_adapter":
        from chromaflow.uni_encoder_b2 import NISFlowNetUNIHybridHLoRAAdapter
        uni_path = ckpt_cfg.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_p = uni_path if Path(uni_path).is_absolute() else str(root / uni_path)
        model = NISFlowNetUNIHybridHLoRAAdapter(uni_path=uni_p, flow_param=flow_param, pretrained_r34=False,
                                                 lora_rank=16, lora_alpha=16, lora_dropout=0.1,
                                                 adapter_hidden_dim=256, adapter_num_heads=8,
                                                 adapter_dropout=0.1, adapter_gamma_init=0.1)
    elif enc == "uni_hybrid_h_lora_hca":
        from chromaflow.uni_encoder_b25 import NISFlowNetUNIHybridHLoRAHCA
        uni_path = ckpt_cfg.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_p = uni_path if Path(uni_path).is_absolute() else str(root / uni_path)
        model = NISFlowNetUNIHybridHLoRAHCA(uni_path=uni_p, flow_param=flow_param, pretrained_r34=False,
                                              lora_rank=16, lora_alpha=16, lora_dropout=0.1,
                                              hca_hidden_dim=256, hca_num_heads=8,
                                              hca_dropout=0.1, hca_gamma_init=0.1, hca_h_bias_init=1.0)
    elif enc == "uni_hybrid_h_lora_sdca":
        from chromaflow.uni_encoder_d1 import NISFlowNetUNIHybridHLoRASDCA
        uni_path = ckpt_cfg.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_p = uni_path if Path(uni_path).is_absolute() else str(root / uni_path)
        model = NISFlowNetUNIHybridHLoRASDCA(uni_path=uni_p, flow_param=flow_param, pretrained_r34=False,
                                              lora_rank=16, lora_alpha=16, lora_dropout=0.1,
                                              sdca_hidden_dim=256, sdca_num_heads=8,
                                              sdca_dropout=0.1, sdca_gamma_init=0.1, sdca_bias_init=2.0)
    elif enc == "uni_hybrid_h_lora_hca_r34rgb":
        from chromaflow.uni_encoder_d2 import NISFlowNetUNIHybridHLoRAHCA_R34RGB
        uni_path = ckpt_cfg.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_p = uni_path if Path(uni_path).is_absolute() else str(root / uni_path)
        model = NISFlowNetUNIHybridHLoRAHCA_R34RGB(uni_path=uni_p, flow_param=flow_param, pretrained_r34=False,
                                                   lora_rank=16, lora_alpha=16, lora_dropout=0.1,
                                                   hca_hidden_dim=256, hca_num_heads=8,
                                                   hca_dropout=0.1, hca_gamma_init=0.1, hca_h_bias_init=1.0)
    elif enc == "uni_hybrid_h_lora_adasca":
        from chromaflow.uni_encoder_d3 import NISFlowNetUNIHybridHLoRAAdaSCA
        uni_path = ckpt_cfg.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_p = uni_path if Path(uni_path).is_absolute() else str(root / uni_path)
        model = NISFlowNetUNIHybridHLoRAAdaSCA(uni_path=uni_p, flow_param=flow_param, pretrained_r34=False,
                                                lora_rank=16, lora_alpha=16, lora_dropout=0.1,
                                                adasca_hidden_dim=256, adasca_num_heads=8,
                                                adasca_dropout=0.1, adasca_h_bias_init=1.0)
    elif enc == "uni_hybrid_h_lora_stainrecon":
        from chromaflow.uni_encoder_d7 import NISFlowNetUNIHybridHLoRAStainRecon
        uni_path = ckpt_cfg.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_p = uni_path if Path(uni_path).is_absolute() else str(root / uni_path)
        model = NISFlowNetUNIHybridHLoRAStainRecon(uni_path=uni_p, flow_param=flow_param, pretrained_r34=False,
                                                    lora_rank=16, lora_alpha=16, lora_dropout=0.1)
    elif enc == "uni_hybrid_h_lora_evuc":
        from chromaflow.uni_encoder_d8 import NISFlowNetUNIHybridHLoRAEvUC
        uni_path = ckpt_cfg.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_p = uni_path if Path(uni_path).is_absolute() else str(root / uni_path)
        model = NISFlowNetUNIHybridHLoRAEvUC(uni_path=uni_p, flow_param=flow_param, pretrained_r34=False,
                                              lora_rank=16, lora_alpha=16, lora_dropout=0.1)
    elif enc == "uni_hybrid_h_lora_hca_hv":
        from chromaflow.uni_encoder_b5 import NISFlowNetUNIHybridHLoRAHCAHV
        uni_path = ckpt_cfg.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_p = uni_path if Path(uni_path).is_absolute() else str(root / uni_path)
        model = NISFlowNetUNIHybridHLoRAHCAHV(uni_path=uni_p, flow_param=flow_param, pretrained_r34=False,
                                                lora_rank=16, lora_alpha=16, lora_dropout=0.1,
                                                hca_hidden_dim=256, hca_num_heads=8,
                                                hca_dropout=0.1, hca_gamma_init=0.1, hca_h_bias_init=1.0)
    elif enc == "uni_hybrid_h_lora_srma":
        from chromaflow.uni_encoder_srma import NISFlowNetUNIHybridHLoRASRMA
        uni_path = ckpt_cfg.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_p = uni_path if Path(uni_path).is_absolute() else str(root / uni_path)
        model = NISFlowNetUNIHybridHLoRASRMA(uni_path=uni_p, flow_param=flow_param, pretrained_r34=False,
                                              lora_rank=16, lora_alpha=16, lora_dropout=0.1,
                                              srma_hidden_dim=256, srma_num_heads=8, srma_dropout=0.1,
                                              srma_gamma_init=0.1, srma_h_bias_init=1.0, srma_e_bias_init=1.0)
    elif enc == "uni_hybrid_h_lora_hca_v2":
        from chromaflow.uni_encoder_b25v2 import NISFlowNetUNIHybridHLoRAHCAv2
        uni_path = ckpt_cfg.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_p = uni_path if Path(uni_path).is_absolute() else str(root / uni_path)
        model = NISFlowNetUNIHybridHLoRAHCAv2(uni_path=uni_p, flow_param=flow_param, pretrained_r34=False,
                                                lora_rank=16, lora_alpha=16, lora_dropout=0.1,
                                                hca_hidden_dim=256, hca_num_heads=8,
                                                hca_dropout=0.1, hca_gamma_init=0.1,
                                                hca_h_bias_init=1.0, hca_e_bias_init=0.5)
    elif enc == "uni_hybrid_h_lora_hca_mask":
        from chromaflow.uni_encoder_b3 import NISFlowNetUNIHybridHLoRAHCAMask
        uni_path = ckpt_cfg.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_p = uni_path if Path(uni_path).is_absolute() else str(root / uni_path)
        model = NISFlowNetUNIHybridHLoRAHCAMask(uni_path=uni_p, flow_param=flow_param, pretrained_r34=False,
                                                  lora_rank=16, lora_alpha=16, lora_dropout=0.1,
                                                  hca_hidden_dim=256, hca_num_heads=8,
                                                  hca_dropout=0.1, hca_gamma_init=0.1, hca_h_bias_init=1.0,
                                                  mask_n_queries=128, mask_query_dim=128, mask_num_blocks=2,
                                                  mask_num_heads=4, mask_dropout=0.1)
    elif enc == "uni_hybrid_h_lora_film":
        from chromaflow.uni_encoder_e1film import NISFlowNetUNIHybridHLoRAFiLM
        uni_path = ckpt_cfg.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_p = uni_path if Path(uni_path).is_absolute() else str(root / uni_path)
        model = NISFlowNetUNIHybridHLoRAFiLM(uni_path=uni_p, flow_param=flow_param, pretrained_r34=False,
                                              lora_rank=16, lora_alpha=16, lora_dropout=0.1,
                                              film_hidden=64, bg_anchor_min_dist=40.0)
    elif enc == "uni_hybrid_h_lora_paa":
        from chromaflow.uni_encoder_e1paa import NISFlowNetUNIHybridHLoRAPaA
        uni_path = ckpt_cfg.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_p = uni_path if Path(uni_path).is_absolute() else str(root / uni_path)
        model = NISFlowNetUNIHybridHLoRAPaA(uni_path=uni_p, flow_param=flow_param, pretrained_r34=False,
                                              lora_rank=16, lora_alpha=16, lora_dropout=0.1,
                                              bg_anchor_min_dist=40.0)
    elif enc == "uni_hybrid_h_lora_msg":
        from chromaflow.uni_encoder_e1msg import NISFlowNetUNIHybridHLoRAMSG
        uni_path = ckpt_cfg.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_p = uni_path if Path(uni_path).is_absolute() else str(root / uni_path)
        model = NISFlowNetUNIHybridHLoRAMSG(uni_path=uni_p, flow_param=flow_param, pretrained_r34=False,
                                              lora_rank=16, lora_alpha=16, lora_dropout=0.1,
                                              bg_anchor_min_dist=40.0)
    elif enc == "mssp":
        from chromaflow.mssp_model import NISFlowNetMSSP
        model = NISFlowNetMSSP(pretrained_r34=False, flow_param=flow_param)
    elif enc == "scde":
        from chromaflow.model_scde import NISFlowNetSCDE
        model = NISFlowNetSCDE(pretrained=False, flow_param=flow_param)
    else:
        _ckpt_im = ckpt_cfg.get("input_mode", "rgb")
        _in_ch = 5 if _ckpt_im == "he_rgb" else 3
        _has_bnd = any(k.startswith("boundary_head.") for k in loaded["state"].keys()) or bool(ckpt_cfg.get("use_boundary", False)) or bool(ckpt_cfg.get("use_boundary_head", False))
        _bb = enc if enc in ("resnet18", "resnet34", "resnet50", "resnet101") else "resnet34"
        model = NISFlowNet(pretrained=False, flow_param=flow_param, in_channels=_in_ch, with_boundary=_has_bnd, backbone=_bb)
    model.load_state_dict(loaded["state"], strict=False)
    if hasattr(model, "decode_alpha"):
        model.decode_alpha = float(args.boundary_decode_alpha)
    model.to(device)
    model.eval()
    if args.tta:
        from chromaflow.tta import TTAModelWrapper, parse_modes
        model = TTAModelWrapper(model, modes=parse_modes(args.tta))
        model.to(device)
        model.eval()

    cfg = {
        "flow_scale": args.flow_scale,
        "niter": args.niter,
        "cellprob_threshold": args.cellprob_threshold,
        "flow_threshold": args.flow_threshold,
        "min_size": args.min_size,
        "cellprob_mode": args.cellprob_mode,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "tile_batch_size": args.tile_batch_size,
        "input_scale": args.input_scale,
        "seed_threshold": args.seed_threshold,
        "seed_min_distance": args.seed_min_distance,
        "endpoint_tau": args.endpoint_tau,
        "endpoint_steps": args.endpoint_steps,
        "endpoint_reject_radius": args.endpoint_reject_radius,
        "fg_thresh": args.fg_thresh,
        "device": device,
        # input_mode read from ckpt config (default rgb for legacy ckpts)
        "input_mode": ckpt_cfg.get("input_mode", "rgb"),
        "polarize_beta": args.polarize_beta,
        "polarize_mode": args.polarize_mode,
        "ewb_beta": args.ewb_beta,
        "ewb_gamma": args.ewb_gamma,
        "ewb_quantile": args.ewb_quantile,
        "pcgs_refine": args.pcgs_refine,
        "pcgs_refine_T": args.pcgs_refine_T,
        "pcgs_refine_blend": args.pcgs_refine_blend,
        "pcgs_fg_thresh": args.pcgs_fg_thresh,
        "pcgs_bg_thresh": args.pcgs_bg_thresh,
    }

    # Optional Macenko stain template (cross-domain alignment).
    cfg["stain_norm"] = None
    if args.stain_template:
        import torchstain as _ts
        _tpl = np.load(args.stain_template)
        _norm = _ts.normalizers.MacenkoNormalizer(backend="numpy")
        _norm.HERef = _tpl["HE"].astype(np.float32)
        _norm.maxCRef = _tpl["maxC"].astype(np.float32)
        cfg["stain_norm"] = _norm
        print(f"[stain-norm] loaded Macenko template from {args.stain_template} "
              f"HE col1={_norm.HERef[:,0]}", flush=True)

    rows: list[dict[str, Any]] = []
    for split in args.split:
        items = load_eval_items(root, args.dataset, split)
        for idx, item in enumerate(items, 1):
            print(f"[{args.dataset} {split}] {idx}/{len(items)} {item.name}", flush=True)
            fg, fl, ce = predict_outputs(model, item.image_path, cfg, device)
            # Inject per-image H/E for polarization-guided cellprob OR PCGS refinement OR EWB OR SIP.
            if args.polarize_beta != 0.0 or args.pcgs_refine or args.ewb_beta != 0.0 or float(getattr(args, "sip_tau", 0.0)) > 0.0:
                from chromaflow.dataset import stain_h_channel, stain_e_channel
                _img = np.asarray(Image.open(item.image_path).convert("RGB"))
                cfg["polarize_h"] = stain_h_channel(_img)
                cfg["polarize_e"] = stain_e_channel(_img)
            peaks = nms_peaks(ce, args.seed_threshold, args.seed_min_distance)
            gt = np.load(item.label_path)
            gt_pts = load_gt_points(gt)
            if args.flow_from != "model":
                from chromaflow.poisson import poisson_override_flow
                _pcfg = {"flow_from": args.flow_from, "poisson_sigma": args.poisson_sigma,
                    "poisson_grad": args.poisson_grad, "poisson_pad": args.poisson_pad,
                    "h_thresh": args.h_thresh, "h_power": args.h_power}
                if args.flow_from in ("poisson_pointstain", "pointstain_direct"):
                    from chromaflow.dataset import stain_h_channel
                    _pcfg["stain_h"] = stain_h_channel(np.asarray(Image.open(item.image_path).convert("RGB")))
                _pts = peaks if getattr(args, "points", "gt") == "pred" else gt_pts
                fl = poisson_override_flow(fl, ce, gt, _pts, _pcfg)
            # S0 once (also used by S4)
            masks_S0 = decode_S0(fg, fl, ce, cfg)
            stats = basin_peak_stats(masks_S0, peaks)
            center_stats = center_pr_f1(peaks, gt_pts if gt_pts is not None else np.zeros((0, 2)),
                                        match_radius=args.center_match_radius)
            for v in args.variants:
                if v == "S0":
                    pred = masks_S0
                else:
                    pred = VARIANTS[v](fg, fl, ce, cfg)
                # SIP — Stain Instance Pruning (Module A', decode-only).
                # Drop predicted instances whose mean interior H̃ < sip_tau.
                # Pre-flight AUC(mean-H̃|TP,FP)=0.775. Targets stain-poor texture FPs
                # that pixel-level ρ ranking could not remove.
                if float(getattr(args, "sip_tau", 0.0)) > 0.0 and pred.max() > 0 and cfg.get("polarize_h") is not None:
                    h_raw = cfg["polarize_h"]
                    lo, hi = float(h_raw.min()), float(h_raw.max())
                    h_n = (h_raw - lo) / max(hi - lo, 1e-6)
                    tau = float(args.sip_tau)
                    min_keep_area = int(getattr(args, "sip_min_area", 0))
                    relabeled = np.zeros_like(pred)
                    next_id = 1
                    for iid in np.unique(pred):
                        if iid == 0:
                            continue
                        m_inst = pred == iid
                        mean_h = float(h_n[m_inst].mean())
                        area = int(m_inst.sum())
                        # keep if H̃ above tau OR instance is large (size override)
                        if mean_h >= tau or (min_keep_area > 0 and area >= min_keep_area):
                            relabeled[m_inst] = next_id
                            next_id += 1
                    pred = relabeled.astype(pred.dtype)
                if (args.mask_smooth > 0 or args.mask_opening > 0) and pred.max() > 0:
                    pred = _smooth_instances(pred, radius=max(1, int(args.mask_smooth)), opening_radius=int(args.mask_opening))
                if args.mask_dilation > 0 and pred.max() > 0:
                    pred = expand_labels(pred.astype(np.int32), distance=int(args.mask_dilation)).astype(pred.dtype)
                if float(getattr(args, "input_scale", 1.0)) != 1.0 and pred.shape[:2] != gt.shape[:2]:
                    from skimage.transform import resize as _sk_resize
                    pred = _sk_resize(pred.astype(np.int32), gt.shape[:2], order=0,
                                      preserve_range=True, anti_aliasing=False).astype(np.int32)
                m = compute_metrics(pred, gt)
                row = {
                    "split": split,
                    "variant": v,
                    "image": item.name,
                    "n_pred": int(pred.max()),
                    "n_gt": int(len(np.unique(gt)) - 1),
                    **m,
                    **stats,
                    **center_stats,
                }
                rows.append(row)

    # write per-image
    fieldnames = ["split", "variant", "image", "n_pred", "n_gt"] + METRIC_NAMES + [
        "n_basins", "zero_peak", "single_peak", "multi_peak",
        "center_P", "center_R", "center_F1", "n_seeds", "n_gt_pts",
    ]
    with (out_dir / "per_image.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # summary per (split, variant)
    summary_rows: list[dict[str, Any]] = []
    for split in args.split:
        for v in args.variants:
            sub = [r for r in rows if r["split"] == split and r["variant"] == v]
            if not sub:
                continue
            agg = {"split": split, "variant": v, "n_images": len(sub)}
            for k in METRIC_NAMES:
                vals = [r[k] for r in sub if r[k] is not None and r[k] >= -0.5]
                agg[k] = float(np.mean(vals)) if vals else None
            for k in ["n_pred", "n_gt"]:
                agg[k] = float(np.mean([r[k] for r in sub]))
            agg["ratio_pred_gt"] = agg["n_pred"] / max(1.0, agg["n_gt"])
            for k in ["zero_peak", "single_peak", "multi_peak", "center_P", "center_R", "center_F1"]:
                agg[k] = float(np.mean([r[k] for r in sub]))
            summary_rows.append(agg)
    with (out_dir / "summary.csv").open("w", newline="") as f:
        fns = ["split", "variant", "n_images"] + METRIC_NAMES + [
            "n_pred", "n_gt", "ratio_pred_gt", "zero_peak", "single_peak", "multi_peak",
            "center_P", "center_R", "center_F1",
        ]
        writer = csv.DictWriter(f, fieldnames=fns)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(json.dumps(summary_rows, indent=2, default=float))


if __name__ == "__main__":
    main()
