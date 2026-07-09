from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from scipy.optimize import linear_sum_assignment

from chromaflow.dataset import normalize_image, build_input_tensor, build_dual_input_tensor
from chromaflow.model import NISFlowNet
from chromaflow.uni_encoder import NISFlowNetUNI
from chromaflow.uni_encoder_hybrid import (
    NISFlowNetUNIHybrid, NISFlowNetUNIHybridIGA,
    NISFlowNetUNIHybridH, NISFlowNetUNIHybridHPSC2F, NISFlowNetUNIHybridHPSIC,
    NISFlowNetUNIHybridHMTA, NISFlowNetUNIHybridHU2HSRG,
    NISFlowNetUNIHybridHUBIM, NISFlowNetUNIHybridHUCAF, NISFlowNetUNIHybridHUDAC,
    NISFlowNetUNIHybridHUDAR, NISFlowNetUNIHybridHURCF, NISFlowNetUNIHybridHURCFV2,
)


METRIC_NAMES = ["AJI", "Dice", "DQ", "SQ", "PQ", "F1", "Acc", "Precision", "Recall", "mean_iou_matched"]


@dataclass(frozen=True)
class EvalItem:
    name: str
    image_path: Path
    label_path: Path


DEFAULT_INFERENCE_CONFIG = {
    "cellprob_mode": "logits",
    "flow_scale": 70.0,
    "niter": 30,
    "cellprob_threshold": 0.0,
    "flow_threshold": 0.0,
    "min_size": 15,
    "tile_size": 224,
    "tile_overlap": 64,
    "tile_batch_size": 16,
    "stitch_mode": "blend",
}


def remap_contiguous(label_map: np.ndarray) -> np.ndarray:
    ids = [int(v) for v in np.unique(label_map) if int(v) != 0]
    out = np.zeros(label_map.shape, dtype=np.int32)
    for new_id, old_id in enumerate(ids, 1):
        out[label_map == old_id] = new_id
    return out


def aji_fast(gt: np.ndarray, pred: np.ndarray) -> float:
    true_ids = [int(v) for v in np.unique(gt) if int(v) != 0]
    pred_ids = [int(v) for v in np.unique(pred) if int(v) != 0]
    if not true_ids:
        return -1.0
    if not pred_ids:
        return 0.0
    true_area = np.bincount(gt.reshape(-1), minlength=max(true_ids) + 1)
    pred_area = np.bincount(pred.reshape(-1), minlength=max(pred_ids) + 1)
    true_index = {inst_id: i for i, inst_id in enumerate(true_ids)}
    pred_index = {inst_id: i for i, inst_id in enumerate(pred_ids)}
    inter = np.zeros((len(true_ids), len(pred_ids)), dtype=np.int64)
    pair_ids, pair_counts = np.unique(
        np.stack([gt.reshape(-1), pred.reshape(-1)], axis=1),
        axis=0,
        return_counts=True,
    )
    for (true_id, pred_id), count in zip(pair_ids, pair_counts):
        true_id = int(true_id)
        pred_id = int(pred_id)
        if true_id == 0 or pred_id == 0:
            continue
        inter[true_index[true_id], pred_index[pred_id]] = int(count)
    true_areas = np.asarray([true_area[inst_id] for inst_id in true_ids], dtype=np.int64)
    pred_areas = np.asarray([pred_area[inst_id] for inst_id in pred_ids], dtype=np.int64)
    union = true_areas[:, None] + pred_areas[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter, dtype=np.float64), where=union > 0)
    paired_pred = np.argmax(iou, axis=1)
    numerator = int(inter[np.arange(len(true_ids)), paired_pred].sum())
    denominator = int(union[np.arange(len(true_ids)), paired_pred].sum())
    used_pred = np.zeros(len(pred_ids), dtype=bool)
    used_pred[paired_pred] = True
    denominator += int(pred_areas[~used_pred].sum())
    return float(numerator / denominator) if denominator > 0 else 0.0


def object_level_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    true_ids = [int(v) for v in np.unique(gt) if int(v) != 0]
    pred_ids = [int(v) for v in np.unique(pred) if int(v) != 0]
    true_area_total = int((gt > 0).sum())
    pred_area_total = int((pred > 0).sum())
    if true_area_total == 0 and pred_area_total == 0:
        return 1.0
    if true_area_total == 0 or pred_area_total == 0:
        return 0.0
    dice_g = 0.0
    for true_id in true_ids:
        true_mask = gt == true_id
        overlap_ids, overlap_counts = np.unique(pred[true_mask], return_counts=True)
        valid = overlap_ids != 0
        gamma = float(true_mask.sum()) / true_area_total
        if not np.any(valid):
            continue
        best_pred = int(overlap_ids[valid][np.argmax(overlap_counts[valid])])
        inter = int(overlap_counts[valid].max())
        denom = int(true_mask.sum() + (pred == best_pred).sum())
        dice_g += gamma * (2.0 * inter / denom if denom > 0 else 0.0)
    dice_s = 0.0
    for pred_id in pred_ids:
        pred_mask = pred == pred_id
        overlap_ids, overlap_counts = np.unique(gt[pred_mask], return_counts=True)
        valid = overlap_ids != 0
        sigma = float(pred_mask.sum()) / pred_area_total
        if not np.any(valid):
            continue
        best_true = int(overlap_ids[valid][np.argmax(overlap_counts[valid])])
        inter = int(overlap_counts[valid].max())
        denom = int(pred_mask.sum() + (gt == best_true).sum())
        dice_s += sigma * (2.0 * inter / denom if denom > 0 else 0.0)
    return float((dice_g + dice_s) / 2.0)


def pairwise_iou(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    true_ids = [int(v) for v in np.unique(gt) if int(v) != 0]
    pred_ids = [int(v) for v in np.unique(pred) if int(v) != 0]
    if not true_ids or not pred_ids:
        return np.zeros((len(true_ids), len(pred_ids)), dtype=np.float64)
    true_area = np.bincount(gt.reshape(-1), minlength=max(true_ids) + 1)
    pred_area = np.bincount(pred.reshape(-1), minlength=max(pred_ids) + 1)
    true_index = {inst_id: i for i, inst_id in enumerate(true_ids)}
    pred_index = {inst_id: i for i, inst_id in enumerate(pred_ids)}
    inter = np.zeros((len(true_ids), len(pred_ids)), dtype=np.float64)
    pair_ids, pair_counts = np.unique(
        np.stack([gt.reshape(-1), pred.reshape(-1)], axis=1),
        axis=0,
        return_counts=True,
    )
    for (true_id, pred_id), count in zip(pair_ids, pair_counts):
        true_id = int(true_id)
        pred_id = int(pred_id)
        if true_id == 0 or pred_id == 0:
            continue
        inter[true_index[true_id], pred_index[pred_id]] = float(count)
    true_areas = np.asarray([true_area[inst_id] for inst_id in true_ids], dtype=np.float64)
    pred_areas = np.asarray([pred_area[inst_id] for inst_id in pred_ids], dtype=np.float64)
    union = true_areas[:, None] + pred_areas[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def panoptic_quality(gt: np.ndarray, pred: np.ndarray, threshold: float = 0.5) -> tuple[float, float, float, float, float, float]:
    """Returns (DQ, SQ, PQ, F1, Precision, Recall) at instance-level IoU >= threshold.

    Precision = TP / (TP + FP)  — fraction of predicted instances that match GT
    Recall    = TP / (TP + FN)  — fraction of GT instances that are detected
    F1        = 2*P*R / (P+R)   — note F1 == DQ algebraically
    """
    iou = pairwise_iou(gt, pred)
    n_true, n_pred = iou.shape
    if n_true == 0 and n_pred == 0:
        return 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    if n_true == 0 or n_pred == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    true_ind, pred_ind = linear_sum_assignment(-iou)
    matched = [(t, p) for t, p in zip(true_ind, pred_ind) if iou[t, p] > threshold]
    tp = len(matched)
    fp = n_pred - tp
    fn = n_true - tp
    dq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
    sq = float(np.mean([iou[t, p] for t, p in matched])) if matched else 0.0
    pq = dq * sq
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(dq), float(sq), float(pq), float(f1), float(precision), float(recall)


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    gt = remap_contiguous(gt.astype(np.int32))
    pred = remap_contiguous(pred.astype(np.int32))
    dq, sq, pq, f1, precision, recall = panoptic_quality(gt, pred)
    # Pixel-level fg/bg accuracy
    pred_fg = (pred > 0)
    gt_fg = (gt > 0)
    acc = float((pred_fg == gt_fg).mean())
    return {
        "AJI": float(aji_fast(gt, pred)),
        "Dice": float(object_level_dice(pred, gt)),
        "DQ": dq,
        "SQ": sq,
        "PQ": pq,
        "F1": f1,
        "Acc": acc,
        "Precision": precision,
        "Recall": recall,
        "mean_iou_matched": sq,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    summary: dict[str, float | int] = {"num_images": len(rows)}
    for name in METRIC_NAMES:
        vals = [float(row[name]) for row in rows]
        summary[name] = float(np.mean(vals)) if vals else float("nan")
    return summary


def load_eval_items(root: Path, dataset: str, split: str) -> list[EvalItem]:
    dataset = dataset.upper()
    if dataset == "MO":
        split_json = root / "data/official_clean/core/monuseg/splits/bonus_official_ids.json"
        names = json.loads(split_json.read_text())[split]
        return [
            EvalItem(
                name=name,
                image_path=root / "data/official_clean/core/monuseg/images" / f"{name}.png",
                label_path=root / "data/official_clean/core/monuseg/labels_instance" / f"{name}.npy",
            )
            for name in names
        ]
    if dataset == "CPM":
        if split == "test":
            split_json = root / "data/official_clean/core/cpm17/splits/bonus_official_arranged_ids.json"
            mapping = json.loads((root / "data/official_clean/core/cpm17/mappings/bonus_arranged_to_official_train.json").read_text())
            names = json.loads(split_json.read_text())["test"]
            official = [(name, mapping[name]) for name in names]
        elif split == "val":
            official = [(f"test_image_{i:02d}", f"test_image_{i:02d}") for i in range(8)]
        elif split == "extra_test":
            official = [(f"test_image_{i:02d}", f"test_image_{i:02d}") for i in range(8, 32)]
        else:
            split_json = root / "data/official_clean/core/cpm17/splits/bonus_official_mapped_to_official_train.json"
            names = json.loads(split_json.read_text())[split]
            official = [(name, name) for name in names]
        return [
            EvalItem(
                name=name,
                image_path=root / "data/official_clean/core/cpm17/images" / f"{official_name}.png",
                label_path=root / "data/official_clean/core/cpm17/labels_instance" / f"{official_name}.npy",
            )
            for name, official_name in official
        ]
    if dataset == "CONIC":
        # CoNIC stratified subset: 5 sources × 50 patches (matches bonus_full_eval.py).
        import random as _rnd
        img_dir = root / "data/CoNIC/raw/images"
        lab_dir = root / "data/CoNIC/raw/labels"
        rng = _rnd.Random(0)
        by_source: dict[str, list[Path]] = {}
        for p in sorted(img_dir.glob("*.png")):
            src = p.stem.split("_")[0]
            by_source.setdefault(src, []).append(p)
        items: list[EvalItem] = []
        for src, paths in sorted(by_source.items()):
            rng.shuffle(paths)
            for p in paths[:50]:
                lp = lab_dir / f"{p.stem}.npy"
                if lp.exists():
                    items.append(EvalItem(name=f"{src}/{p.stem}", image_path=p, label_path=lp))
        return items
    if dataset == "NUINSSEG":
        base = root / "data/official_clean/nuinsseg_prepared"
        names = json.loads((base / "splits.json").read_text())["test"]
        return [
            EvalItem(name=n, image_path=base / "images" / (n + ".png"), label_path=base / "labels_instance" / (n + ".npy"))
            for n in names
        ]
    raise ValueError(f"Unsupported dataset: {dataset}")


def crop_or_pad_to_shape(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    out = arr[..., :h, :w]
    if out.shape[-2:] == shape:
        return out
    padded = np.zeros(arr.shape[:-2] + shape, dtype=arr.dtype)
    padded[..., : out.shape[-2], : out.shape[-1]] = out
    return padded


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _predict_full_outputs(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    input_mode: str = "rgb",
) -> tuple[np.ndarray, np.ndarray]:
    width, height = image.size
    if input_mode == "h_and_rgb":
        h_t, rgb_t = build_dual_input_tensor(image)
        model_in = {"image": h_t.unsqueeze(0).to(device),
                     "image_rgb": rgb_t.unsqueeze(0).to(device)}
    else:
        model_in = build_input_tensor(image, input_mode).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(model_in)
    fg_logits = outputs["fg_logits"][0, 0].detach().cpu().numpy()
    flow = outputs["flow"][0].detach().cpu().numpy()
    if "centerness_logits" in outputs:
        center = outputs["centerness_logits"][0, 0].detach().cpu().numpy()
    else:
        center = np.zeros_like(fg_logits)
    return (crop_or_pad_to_shape(fg_logits, (height, width)),
            crop_or_pad_to_shape(flow, (height, width)),
            crop_or_pad_to_shape(center, (height, width)))


def _predict_tiled_outputs(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    tile_size: int,
    tile_overlap: int,
    tile_batch_size: int,
    input_mode: str = "rgb",
) -> tuple[np.ndarray, np.ndarray]:
    input_tensor = build_input_tensor(image, input_mode).unsqueeze(0)
    _, _, height, width = input_tensor.shape
    if height <= tile_size and width <= tile_size:
        return _predict_full_outputs(model, image, device, input_mode)
    if tile_overlap < 0 or tile_overlap >= tile_size:
        raise ValueError(f"tile_overlap must be in [0, tile_size), got {tile_overlap}")
    stride = tile_size - tile_overlap

    pad_h = 0
    if height - tile_size > 0 and (height - tile_size) % stride > 0:
        pad_h = stride - (height - tile_size) % stride
    pad_w = 0
    if width - tile_size > 0 and (width - tile_size) % stride > 0:
        pad_w = stride - (width - tile_size) % stride
    if pad_h or pad_w:
        input_tensor = torch.nn.functional.pad(input_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    _, _, padded_h, padded_w = input_tensor.shape

    fg_out = torch.zeros((1, 1, padded_h, padded_w), dtype=torch.float32, device=device)
    flow_out = torch.zeros((1, 2, padded_h, padded_w), dtype=torch.float32, device=device)
    center_out = torch.zeros((1, 1, padded_h, padded_w), dtype=torch.float32, device=device)
    batch_tiles: list[torch.Tensor] = []
    batch_coords: list[tuple[int, int, int, int, int, int, int, int]] = []

    def flush_batch() -> None:
        nonlocal batch_tiles, batch_coords
        if not batch_tiles:
            return
        batch = torch.cat(batch_tiles, dim=0).to(device)
        with torch.no_grad():
            outputs = model(batch)
        for idx, (y0, x0, y1, y2, x1, x2, ly1, ly2) in enumerate(batch_coords):
            lx1 = x1 - x0
            lx2 = x2 - x0
            fg_out[:, :, y1:y2, x1:x2] = outputs["fg_logits"][idx : idx + 1, :, ly1:ly2, lx1:lx2]
            flow_out[:, :, y1:y2, x1:x2] = outputs["flow"][idx : idx + 1, :, ly1:ly2, lx1:lx2]
            if "centerness_logits" in outputs:
                center_out[:, :, y1:y2, x1:x2] = outputs["centerness_logits"][idx : idx + 1, :, ly1:ly2, lx1:lx2]
        batch_tiles = []
        batch_coords = []

    for y0 in range(0, padded_h - tile_overlap, stride):
        y_end = y0 + tile_size if y0 + tile_size < padded_h else padded_h
        y1 = y0 + tile_overlap // 2 if y0 > 0 else 0
        y2 = y0 + tile_size - tile_overlap // 2 if y0 + tile_size < padded_h else padded_h
        ly1 = y1 - y0
        ly2 = y2 - y0
        for x0 in range(0, padded_w - tile_overlap, stride):
            x_end = x0 + tile_size if x0 + tile_size < padded_w else padded_w
            x1 = x0 + tile_overlap // 2 if x0 > 0 else 0
            x2 = x0 + tile_size - tile_overlap // 2 if x0 + tile_size < padded_w else padded_w
            batch_tiles.append(input_tensor[:, :, y0:y_end, x0:x_end])
            batch_coords.append((y0, x0, y1, y2, x1, x2, ly1, ly2))
            if len(batch_tiles) >= tile_batch_size:
                flush_batch()
    flush_batch()
    fg_np = fg_out[0, 0, :height, :width].detach().cpu().numpy()
    flow_np = flow_out[0, :, :height, :width].detach().cpu().numpy()
    center_np = center_out[0, 0, :height, :width].detach().cpu().numpy()
    return fg_np, flow_np, center_np


def _make_tile_weight(
    tile_h: int,
    tile_w: int,
    overlap: int,
    is_top: bool,
    is_left: bool,
    is_bottom: bool,
    is_right: bool,
) -> np.ndarray:
    # Cosine-half ramp on tile sides that border another tile; weight=1 on image-boundary sides.
    # Strictly positive everywhere (values in (0, 1]) to avoid divide-by-zero at corners.
    def axis(length: int, ramp: int, ramp_start: bool, ramp_end: bool) -> np.ndarray:
        w = np.ones(length, dtype=np.float32)
        if ramp > 0 and ramp < length:
            r = 0.5 * (1.0 - np.cos(np.pi * (np.arange(ramp) + 1) / (ramp + 1))).astype(np.float32)
            if ramp_start:
                w[:ramp] = r
            if ramp_end:
                w[-ramp:] = r[::-1]
        return w

    wy = axis(tile_h, overlap, ramp_start=not is_top, ramp_end=not is_bottom)
    wx = axis(tile_w, overlap, ramp_start=not is_left, ramp_end=not is_right)
    return (wy[:, None] * wx[None, :]).astype(np.float32)


def _predict_tiled_outputs_blend(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    tile_size: int,
    tile_overlap: int,
    tile_batch_size: int,
    input_mode: str = "rgb",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Weighted-blend variant of tile stitching: each tile contributes its full output, weighted
    # by a 2D cosine-half ramp. The original hard-replace version creates flow direction
    # discontinuities at every seam, which Cellpose dynamics turns into spurious basins.
    dual_input = (input_mode == "h_and_rgb")
    if dual_input:
        h_t, rgb_t = build_dual_input_tensor(image)
        input_tensor = h_t.unsqueeze(0)
        input_tensor_rgb = rgb_t.unsqueeze(0)
    else:
        input_tensor = build_input_tensor(image, input_mode).unsqueeze(0)
        input_tensor_rgb = None
    _, _, height, width = input_tensor.shape
    if height <= tile_size and width <= tile_size:
        return _predict_full_outputs(model, image, device, input_mode)
    if tile_overlap < 0 or tile_overlap >= tile_size:
        raise ValueError(f"tile_overlap must be in [0, tile_size), got {tile_overlap}")
    stride = tile_size - tile_overlap

    pad_h = 0
    if height - tile_size > 0 and (height - tile_size) % stride > 0:
        pad_h = stride - (height - tile_size) % stride
    pad_w = 0
    if width - tile_size > 0 and (width - tile_size) % stride > 0:
        pad_w = stride - (width - tile_size) % stride
    if pad_h or pad_w:
        input_tensor = torch.nn.functional.pad(input_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        if dual_input:
            input_tensor_rgb = torch.nn.functional.pad(input_tensor_rgb, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    _, _, padded_h, padded_w = input_tensor.shape

    y_starts = _tile_starts(padded_h, tile_size, stride)
    x_starts = _tile_starts(padded_w, tile_size, stride)

    fg_sum = torch.zeros((1, 1, padded_h, padded_w), dtype=torch.float32, device=device)
    flow_sum = torch.zeros((1, 2, padded_h, padded_w), dtype=torch.float32, device=device)
    center_sum = torch.zeros((1, 1, padded_h, padded_w), dtype=torch.float32, device=device)
    weight_sum = torch.zeros((1, 1, padded_h, padded_w), dtype=torch.float32, device=device)

    weight_cache: dict[tuple[bool, bool, bool, bool], torch.Tensor] = {}

    def get_weight(is_top: bool, is_left: bool, is_bot: bool, is_right: bool) -> torch.Tensor:
        key = (is_top, is_left, is_bot, is_right)
        cached = weight_cache.get(key)
        if cached is None:
            w_np = _make_tile_weight(tile_size, tile_size, tile_overlap, is_top, is_left, is_bot, is_right)
            cached = torch.from_numpy(w_np).to(device).unsqueeze(0).unsqueeze(0)
            weight_cache[key] = cached
        return cached

    batch_tiles: list[torch.Tensor] = []
    batch_tiles_rgb: list[torch.Tensor] = []
    batch_coords: list[tuple[int, int, bool, bool, bool, bool]] = []

    def flush_batch() -> None:
        nonlocal batch_tiles, batch_tiles_rgb, batch_coords
        if not batch_tiles:
            return
        batch = torch.cat(batch_tiles, dim=0).to(device)
        with torch.no_grad():
            if dual_input:
                batch_rgb = torch.cat(batch_tiles_rgb, dim=0).to(device)
                outputs = model({"image": batch, "image_rgb": batch_rgb})
            else:
                outputs = model(batch)
        has_center = "centerness_logits" in outputs
        for idx, (y0, x0, is_top, is_left, is_bot, is_right) in enumerate(batch_coords):
            w = get_weight(is_top, is_left, is_bot, is_right)
            fg_sum[:, :, y0:y0 + tile_size, x0:x0 + tile_size] += (
                outputs["fg_logits"][idx:idx + 1] * w
            )
            flow_sum[:, :, y0:y0 + tile_size, x0:x0 + tile_size] += (
                outputs["flow"][idx:idx + 1] * w
            )
            if has_center:
                center_sum[:, :, y0:y0 + tile_size, x0:x0 + tile_size] += (
                    outputs["centerness_logits"][idx:idx + 1] * w
                )
            weight_sum[:, :, y0:y0 + tile_size, x0:x0 + tile_size] += w
        batch_tiles = []
        batch_tiles_rgb = []
        batch_coords = []

    n_y = len(y_starts)
    n_x = len(x_starts)
    for yi, y0 in enumerate(y_starts):
        is_top = yi == 0
        is_bot = yi == n_y - 1
        for xi, x0 in enumerate(x_starts):
            is_left = xi == 0
            is_right = xi == n_x - 1
            batch_tiles.append(input_tensor[:, :, y0:y0 + tile_size, x0:x0 + tile_size])
            if dual_input:
                batch_tiles_rgb.append(input_tensor_rgb[:, :, y0:y0 + tile_size, x0:x0 + tile_size])
            batch_coords.append((y0, x0, is_top, is_left, is_bot, is_right))
            if len(batch_tiles) >= tile_batch_size:
                flush_batch()
    flush_batch()

    eps = 1e-6
    fg_avg = fg_sum / (weight_sum + eps)
    flow_avg = flow_sum / (weight_sum + eps)
    center_avg = center_sum / (weight_sum + eps)
    fg_np = fg_avg[0, 0, :height, :width].detach().cpu().numpy()
    flow_np = flow_avg[0, :, :height, :width].detach().cpu().numpy()
    center_np = center_avg[0, 0, :height, :width].detach().cpu().numpy()
    return fg_np, flow_np, center_np


def reconcile_with_centerness_seeds(labels: np.ndarray, center_logits: np.ndarray,
                                      seed_threshold: float = 0.4, seed_min_distance: int = 8) -> np.ndarray:
    """Reconcile cellpose basin labels with NMS peaks from centerness:
      - For each CC label in cellpose output, count NMS peaks inside.
      - 0 peaks → drop CC (likely false-positive basin).
      - 1 peak → keep CC as one instance.
      - 2+ peaks → split CC by voronoi (each pixel to nearest peak).
    Peaks outside any CC are not recovered (conservative pilot).
    """
    from skimage.feature import peak_local_max
    import numpy as _np
    center_prob = 1.0 / (1.0 + _np.exp(-center_logits))
    peaks = peak_local_max(center_prob, min_distance=seed_min_distance,
                              threshold_abs=seed_threshold, exclude_border=False)
    if len(peaks) == 0:
        return labels.astype(_np.uint16)
    labels = labels.astype(_np.int64)
    peak_cc = labels[peaks[:, 0], peaks[:, 1]]
    cc_peak_idx = {}
    for i, cc in enumerate(peak_cc.tolist()):
        if cc == 0:
            continue
        cc_peak_idx.setdefault(int(cc), []).append(i)
    refined = _np.zeros_like(labels, dtype=_np.int64)
    next_id = 1
    for cc in _np.unique(labels):
        cc = int(cc)
        if cc == 0:
            continue
        cc_mask = labels == cc
        idxs = cc_peak_idx.get(cc, [])
        if len(idxs) == 0:
            continue  # drop CC: no peak inside
        if len(idxs) == 1:
            refined[cc_mask] = next_id
            next_id += 1
        else:
            yy, xx = _np.where(cc_mask)
            cc_peaks_arr = peaks[idxs]
            dist = (yy[:, None] - cc_peaks_arr[None, :, 0]) ** 2 + (xx[:, None] - cc_peaks_arr[None, :, 1]) ** 2
            nearest = dist.argmin(axis=1)
            for k in range(len(idxs)):
                sel = nearest == k
                refined[yy[sel], xx[sel]] = next_id
                next_id += 1
    return refined.astype(_np.uint16)


def predict_instance_map(
    model: torch.nn.Module,
    image_path: Path,
    inference_config: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    from cellpose import dynamics

    image = Image.open(image_path).convert("RGB")
    tile_size = int(inference_config.get("tile_size", 0))
    tile_overlap = int(inference_config.get("tile_overlap", 0))
    tile_batch_size = int(inference_config.get("tile_batch_size", 16))
    stitch_mode = str(inference_config.get("stitch_mode", "hard_replace")).lower()
    input_mode = str(inference_config.get("input_mode", "rgb")).lower()
    if tile_size > 0:
        if stitch_mode == "blend":
            fg_logits, flow, center_logits = _predict_tiled_outputs_blend(
                model, image, device, tile_size, tile_overlap, tile_batch_size, input_mode
            )
        elif stitch_mode == "hard_replace":
            fg_logits, flow, center_logits = _predict_tiled_outputs(
                model, image, device, tile_size, tile_overlap, tile_batch_size, input_mode
            )
        else:
            raise ValueError(f"Unsupported stitch_mode: {stitch_mode}")
    else:
        fg_logits, flow, center_logits = _predict_full_outputs(model, image, device, input_mode)
    cellprob_mode = str(inference_config.get("cellprob_mode", "logits")).lower()
    center_alpha = float(inference_config.get("center_alpha", 0.0))
    if cellprob_mode == "logits":
        cellprob = fg_logits
        if center_alpha > 0.0:
            cellprob = (1.0 - center_alpha) * fg_logits + center_alpha * center_logits
    elif cellprob_mode == "prob":
        fg_p = 1.0 / (1.0 + np.exp(-fg_logits))
        if center_alpha > 0.0:
            center_p = 1.0 / (1.0 + np.exp(-center_logits))
            cellprob = (1.0 - center_alpha) * fg_p + center_alpha * center_p
        else:
            cellprob = fg_p
    else:
        raise ValueError(f"Unsupported cellprob_mode: {cellprob_mode}")
    use_seeds = bool(inference_config.get("use_centerness_seeds", False))
    seed_thr = float(inference_config.get("seed_threshold", 0.4))
    seed_md = int(inference_config.get("seed_min_distance", 8))
    postprocess = str(inference_config.get("postprocess", "cellpose"))
    if postprocess == "cellpose":
        d_p = flow[[1, 0]] * float(inference_config["flow_scale"])
        result = dynamics.compute_masks(
            dP=d_p,
            cellprob=cellprob,
            niter=int(inference_config["niter"]),
            cellprob_threshold=float(inference_config["cellprob_threshold"]),
            flow_threshold=float(inference_config["flow_threshold"]),
            min_size=int(inference_config["min_size"]),
            device=device,
        )
        masks = result[0] if isinstance(result, tuple) else result
        masks_np = np.asarray(masks, dtype=np.uint16)
        if use_seeds:
            masks_np = reconcile_with_centerness_seeds(masks_np, center_logits, seed_thr, seed_md)
        return masks_np
    elif postprocess == "cc":
        # BoNuS-style: connected components on (cellprob > threshold)
        from skimage import measure, morphology
        thr = float(inference_config["cellprob_threshold"])
        # If cellprob is logits (default), threshold=0 means prob>0.5
        fg_bin = cellprob > thr
        fg_bin = morphology.remove_small_objects(fg_bin, min_size=int(inference_config.get("cc_min_size", 15)))
        labeled = measure.label(fg_bin, connectivity=1)
        return labeled.astype(np.uint16)
    elif postprocess == "hybrid":
        # Step 1: BoNuS-style CC on fg
        from skimage import measure, morphology
        thr = float(inference_config["cellprob_threshold"])
        fg_bin = cellprob > thr
        fg_bin = morphology.remove_small_objects(fg_bin, min_size=int(inference_config.get("cc_min_size", 15)))
        labeled = measure.label(fg_bin, connectivity=1).astype(np.int32)
        # Step 2: for large CCs (likely touching), run cellpose flow split locally
        touching_thr = int(inference_config.get("cc_touching_size", 400))
        d_p_full = flow[[1, 0]] * float(inference_config["flow_scale"])
        max_label = labeled.max()
        next_id = max_label + 1
        from scipy import ndimage as ndi
        regions = ndi.find_objects(labeled)
        for cc_id_minus_1, sl in enumerate(regions):
            if sl is None:
                continue
            cc_id = cc_id_minus_1 + 1
            mask_local = labeled[sl] == cc_id
            if mask_local.sum() < touching_thr:
                continue
            # Run cellpose dynamics restricted to this CC
            cellprob_local = cellprob[sl].copy().astype(np.float32)
            cellprob_local[~mask_local] = thr - 10.0
            d_p_local = d_p_full[:, sl[0], sl[1]].astype(np.float32)
            try:
                result = dynamics.compute_masks(
                    dP=d_p_local,
                    cellprob=cellprob_local,
                    niter=int(inference_config["niter"]),
                    cellprob_threshold=thr,
                    flow_threshold=float(inference_config["flow_threshold"]),
                    min_size=int(inference_config["min_size"]),
                    device=device,
                )
                local_masks = result[0] if isinstance(result, tuple) else result
                local_masks = np.asarray(local_masks)
                if local_masks.max() <= 1:
                    continue
                # Remap: replace CC with multiple new labels
                labeled[sl][mask_local] = 0
                for new_lab in range(1, int(local_masks.max()) + 1):
                    sub = local_masks == new_lab
                    if sub.sum() == 0:
                        continue
                    labeled[sl][sub] = next_id
                    next_id += 1
            except Exception:
                pass
        # Re-label to compact ids
        from skimage import measure as msr
        labeled = msr.label(labeled > 0, connectivity=1) if False else labeled
        return labeled.astype(np.uint16)
    else:
        raise ValueError(f"Unsupported postprocess: {postprocess}")


def evaluate_model(
    model: torch.nn.Module,
    dataset: str,
    split: str,
    root: str | Path,
    inference_config: dict[str, Any] | None = None,
    out_dir: str | Path | None = None,
    save_predictions: bool = False,
    limit: int = 0,
    device: str | torch.device | None = None,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    root = Path(root)
    inference_config = {**DEFAULT_INFERENCE_CONFIG, **(inference_config or {})}
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.eval()
    items = load_eval_items(root, dataset, split)
    if limit:
        items = items[:limit]

    rows: list[dict[str, Any]] = []
    pred_dir = None
    if out_dir is not None and save_predictions:
        pred_dir = Path(out_dir) / "predictions" / split
        pred_dir.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(items, 1):
        print(f"[eval {dataset} {split}] {idx}/{len(items)} {item.name}", flush=True)
        pred = predict_instance_map(model, item.image_path, inference_config, device)
        gt = np.load(item.label_path)
        metrics = compute_metrics(pred, gt)
        if pred_dir is not None:
            np.save(pred_dir / f"{item.name}.npy", pred)
        rows.append(
            {
                "split": split,
                "image": item.name,
                "image_path": str(item.image_path),
                "label_path": str(item.label_path),
                "height": int(gt.shape[0]),
                "width": int(gt.shape[1]),
                "num_instances_pred": int(pred.max()),
                "num_instances_gt": int(len(np.unique(gt)) - 1),
                **metrics,
            }
        )
    summary = summarize(rows)
    if out_dir is not None:
        write_split_outputs(Path(out_dir), split, rows, summary, inference_config)
    return summary, rows


def write_split_outputs(
    out_dir: Path,
    split: str,
    rows: list[dict[str, Any]],
    summary: dict[str, float | int],
    inference_config: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "inference_config": inference_config,
        "rows": rows,
    }
    (out_dir / f"metrics_{split}.json").write_text(json.dumps(payload, indent=2))
    with (out_dir / f"per_image_metrics_{split}.csv").open("w", newline="") as f:
        fieldnames = [
            "split",
            "image",
            "image_path",
            "label_path",
            "height",
            "width",
            "num_instances_pred",
            "num_instances_gt",
        ] + METRIC_NAMES
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_combined_per_image(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    with (out_dir / "per_image_metrics.csv").open("w", newline="") as f:
        fieldnames = [
            "split",
            "image",
            "image_path",
            "label_path",
            "height",
            "width",
            "num_instances_pred",
            "num_instances_gt",
        ] + METRIC_NAMES
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        state = checkpoint.get("model_state") or checkpoint.get("state_dict") or checkpoint
    else:
        state = checkpoint
        checkpoint = {}
    cleaned = {k.replace("module.", "", 1): v for k, v in state.items()}
    return {"checkpoint": checkpoint, "state": cleaned}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--dataset", choices=["MO", "CPM"], required=True)
    parser.add_argument("--split", nargs="+", default=["val", "test"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--flow-param", choices=["direct", "potential"], default="")
    parser.add_argument("--flow-scale", type=float, default=70.0)
    parser.add_argument("--niter", type=int, default=30)
    parser.add_argument("--cellprob-mode", choices=["logits", "prob"], default="logits")
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
    parser.add_argument("--flow-threshold", type=float, default=0.0)
    parser.add_argument("--min-size", type=int, default=15)
    parser.add_argument("--center-alpha", type=float, default=0.0)
    parser.add_argument("--postprocess", choices=["cellpose", "cc", "hybrid"], default="cellpose",
                        help="cc=connected components on fg only; hybrid=cc + cellpose flow split for big CCs")
    parser.add_argument("--cc-min-size", type=int, default=15)
    parser.add_argument("--cc-touching-size", type=int, default=400, help="CC larger than this is considered touching, run flow split")
    parser.add_argument("--use-centerness-seeds", action="store_true",
                        help="Post-hoc reconcile cellpose output with NMS peaks from centerness head")
    parser.add_argument("--seed-threshold", type=float, default=0.4)
    parser.add_argument("--seed-min-distance", type=int, default=8)
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--tile-batch-size", type=int, default=16)
    parser.add_argument("--tta", type=str, default="",
                        help="TTA modes, comma-separated: hflip,vflip,hvflip. Empty disables.")
    parser.add_argument("--stitch-mode", choices=["hard_replace", "blend"], default="blend",
                        help="blend=cosine-half ramp blending (default, recommended); hard_replace=legacy seam (kept for reproducing pre-2026-05-16 numbers)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_checkpoint(Path(args.checkpoint), device)
    checkpoint_config = loaded["checkpoint"].get("config", {}) if isinstance(loaded["checkpoint"], dict) else {}
    flow_param = args.flow_param or checkpoint_config.get("flow_param", "direct")
    ckpt_encoder = checkpoint_config.get("encoder", "resnet34")
    if ckpt_encoder == "uni":
        ckpt_uni_path = checkpoint_config.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_path = ckpt_uni_path if Path(ckpt_uni_path).is_absolute() else str(root / ckpt_uni_path)
        model = NISFlowNetUNI(uni_path=uni_path, flow_param=flow_param)
    elif ckpt_encoder == "uni_hybrid":
        ckpt_uni_path = checkpoint_config.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_path = ckpt_uni_path if Path(ckpt_uni_path).is_absolute() else str(root / ckpt_uni_path)
        model = NISFlowNetUNIHybrid(uni_path=uni_path, flow_param=flow_param, pretrained_r34=False)
    elif ckpt_encoder == "uni_hybrid_iga":
        ckpt_uni_path = checkpoint_config.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_path = ckpt_uni_path if Path(ckpt_uni_path).is_absolute() else str(root / ckpt_uni_path)
        model = NISFlowNetUNIHybridIGA(uni_path=uni_path, flow_param=flow_param, pretrained_r34=False)
    elif ckpt_encoder == "uni_hybrid_h":
        ckpt_uni_path = checkpoint_config.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_path = ckpt_uni_path if Path(ckpt_uni_path).is_absolute() else str(root / ckpt_uni_path)
        model = NISFlowNetUNIHybridH(uni_path=uni_path, flow_param=flow_param, pretrained_r34=False)
    elif ckpt_encoder == "uni_hybrid_h_psc2f":
        ckpt_uni_path = checkpoint_config.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_path = ckpt_uni_path if Path(ckpt_uni_path).is_absolute() else str(root / ckpt_uni_path)
        model = NISFlowNetUNIHybridHPSC2F(uni_path=uni_path, flow_param=flow_param, pretrained_r34=False)
    elif ckpt_encoder == "uni_hybrid_h_psic":
        ckpt_uni_path = checkpoint_config.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_path = ckpt_uni_path if Path(ckpt_uni_path).is_absolute() else str(root / ckpt_uni_path)
        model = NISFlowNetUNIHybridHPSIC(uni_path=uni_path, flow_param=flow_param, pretrained_r34=False)
    elif ckpt_encoder == "uni_hybrid_h_mta":
        ckpt_uni_path = checkpoint_config.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_path = ckpt_uni_path if Path(ckpt_uni_path).is_absolute() else str(root / ckpt_uni_path)
        model = NISFlowNetUNIHybridHMTA(uni_path=uni_path, flow_param=flow_param, pretrained_r34=False)
    elif ckpt_encoder == "uni_hybrid_h_u2h_srg":
        ckpt_uni_path = checkpoint_config.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_path = ckpt_uni_path if Path(ckpt_uni_path).is_absolute() else str(root / ckpt_uni_path)
        model = NISFlowNetUNIHybridHU2HSRG(uni_path=uni_path, flow_param=flow_param, pretrained_r34=False)
    elif ckpt_encoder == "uni_hybrid_h_ubim":
        ckpt_uni_path = checkpoint_config.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_path = ckpt_uni_path if Path(ckpt_uni_path).is_absolute() else str(root / ckpt_uni_path)
        model = NISFlowNetUNIHybridHUBIM(uni_path=uni_path, flow_param=flow_param, pretrained_r34=False)
    elif ckpt_encoder == "uni_hybrid_h_ucaf":
        ckpt_uni_path = checkpoint_config.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_path = ckpt_uni_path if Path(ckpt_uni_path).is_absolute() else str(root / ckpt_uni_path)
        model = NISFlowNetUNIHybridHUCAF(uni_path=uni_path, flow_param=flow_param, pretrained_r34=False)
    elif ckpt_encoder == "uni_hybrid_h_udac":
        ckpt_uni_path = checkpoint_config.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_path = ckpt_uni_path if Path(ckpt_uni_path).is_absolute() else str(root / ckpt_uni_path)
        model = NISFlowNetUNIHybridHUDAC(uni_path=uni_path, flow_param=flow_param, pretrained_r34=False)
    elif ckpt_encoder == "uni_hybrid_h_udar":
        ckpt_uni_path = checkpoint_config.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_path = ckpt_uni_path if Path(ckpt_uni_path).is_absolute() else str(root / ckpt_uni_path)
        model = NISFlowNetUNIHybridHUDAR(uni_path=uni_path, flow_param=flow_param, pretrained_r34=False)
    elif ckpt_encoder == "uni_hybrid_h_urcf":
        ckpt_uni_path = checkpoint_config.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_path = ckpt_uni_path if Path(ckpt_uni_path).is_absolute() else str(root / ckpt_uni_path)
        model = NISFlowNetUNIHybridHURCF(uni_path=uni_path, flow_param=flow_param, pretrained_r34=False)
    elif ckpt_encoder == "uni_hybrid_h_urcf_v2":
        ckpt_uni_path = checkpoint_config.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_path = ckpt_uni_path if Path(ckpt_uni_path).is_absolute() else str(root / ckpt_uni_path)
        model = NISFlowNetUNIHybridHURCFV2(uni_path=uni_path, flow_param=flow_param, pretrained_r34=False)
    elif ckpt_encoder == "uni_hybrid_h_lora":
        from chromaflow.uni_encoder_b1 import NISFlowNetUNIHybridHLoRA
        ckpt_uni_path = checkpoint_config.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_path = ckpt_uni_path if Path(ckpt_uni_path).is_absolute() else str(root / ckpt_uni_path)
        model = NISFlowNetUNIHybridHLoRA(uni_path=uni_path, flow_param=flow_param, pretrained_r34=False,
                                          lora_rank=16, lora_alpha=16, lora_dropout=0.1)
    elif ckpt_encoder == "uni_hybrid_h_lora_adapter":
        from chromaflow.uni_encoder_b2 import NISFlowNetUNIHybridHLoRAAdapter
        ckpt_uni_path = checkpoint_config.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_path = ckpt_uni_path if Path(ckpt_uni_path).is_absolute() else str(root / ckpt_uni_path)
        model = NISFlowNetUNIHybridHLoRAAdapter(uni_path=uni_path, flow_param=flow_param, pretrained_r34=False,
                                                 lora_rank=16, lora_alpha=16, lora_dropout=0.1,
                                                 adapter_hidden_dim=256, adapter_num_heads=8,
                                                 adapter_dropout=0.1, adapter_gamma_init=0.1)
    elif ckpt_encoder == "uni_hybrid_h_lora_hca":
        from chromaflow.uni_encoder_b25 import NISFlowNetUNIHybridHLoRAHCA
        ckpt_uni_path = checkpoint_config.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_path = ckpt_uni_path if Path(ckpt_uni_path).is_absolute() else str(root / ckpt_uni_path)
        model = NISFlowNetUNIHybridHLoRAHCA(uni_path=uni_path, flow_param=flow_param, pretrained_r34=False,
                                              lora_rank=16, lora_alpha=16, lora_dropout=0.1,
                                              hca_hidden_dim=256, hca_num_heads=8,
                                              hca_dropout=0.1, hca_gamma_init=0.1, hca_h_bias_init=1.0)
    elif ckpt_encoder == "uni_hybrid_h_lora_hca_hv":
        from chromaflow.uni_encoder_b5 import NISFlowNetUNIHybridHLoRAHCAHV
        ckpt_uni_path = checkpoint_config.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_path = ckpt_uni_path if Path(ckpt_uni_path).is_absolute() else str(root / ckpt_uni_path)
        model = NISFlowNetUNIHybridHLoRAHCAHV(uni_path=uni_path, flow_param=flow_param, pretrained_r34=False,
                                                lora_rank=16, lora_alpha=16, lora_dropout=0.1,
                                                hca_hidden_dim=256, hca_num_heads=8,
                                                hca_dropout=0.1, hca_gamma_init=0.1, hca_h_bias_init=1.0)
    elif ckpt_encoder == "uni_hybrid_h_lora_srma":
        from chromaflow.uni_encoder_srma import NISFlowNetUNIHybridHLoRASRMA
        ckpt_uni_path = checkpoint_config.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_path = ckpt_uni_path if Path(ckpt_uni_path).is_absolute() else str(root / ckpt_uni_path)
        model = NISFlowNetUNIHybridHLoRASRMA(uni_path=uni_path, flow_param=flow_param, pretrained_r34=False,
                                              lora_rank=16, lora_alpha=16, lora_dropout=0.1,
                                              srma_hidden_dim=256, srma_num_heads=8, srma_dropout=0.1,
                                              srma_gamma_init=0.1, srma_h_bias_init=1.0, srma_e_bias_init=1.0)
    elif ckpt_encoder == "uni_hybrid_h_lora_hca_v2":
        from chromaflow.uni_encoder_b25v2 import NISFlowNetUNIHybridHLoRAHCAv2
        ckpt_uni_path = checkpoint_config.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_path = ckpt_uni_path if Path(ckpt_uni_path).is_absolute() else str(root / ckpt_uni_path)
        model = NISFlowNetUNIHybridHLoRAHCAv2(uni_path=uni_path, flow_param=flow_param, pretrained_r34=False,
                                                lora_rank=16, lora_alpha=16, lora_dropout=0.1,
                                                hca_hidden_dim=256, hca_num_heads=8,
                                                hca_dropout=0.1, hca_gamma_init=0.1,
                                                hca_h_bias_init=1.0, hca_e_bias_init=0.5)
    elif ckpt_encoder == "uni_hybrid_h_lora_hca_mask":
        from chromaflow.uni_encoder_b3 import NISFlowNetUNIHybridHLoRAHCAMask
        ckpt_uni_path = checkpoint_config.get("uni_path", "weights/uni/UNI/pytorch_model.bin")
        uni_path = ckpt_uni_path if Path(ckpt_uni_path).is_absolute() else str(root / ckpt_uni_path)
        model = NISFlowNetUNIHybridHLoRAHCAMask(uni_path=uni_path, flow_param=flow_param, pretrained_r34=False,
                                                  lora_rank=16, lora_alpha=16, lora_dropout=0.1,
                                                  hca_hidden_dim=256, hca_num_heads=8,
                                                  hca_dropout=0.1, hca_gamma_init=0.1, hca_h_bias_init=1.0,
                                                  mask_n_queries=128, mask_query_dim=128, mask_num_blocks=2,
                                                  mask_num_heads=4, mask_dropout=0.1)
    else:
        model = NISFlowNet(pretrained=False, flow_param=flow_param)
    model.load_state_dict(loaded["state"], strict=False)
    if args.tta:
        from chromaflow.tta import TTAModelWrapper, parse_modes
        model = TTAModelWrapper(model, modes=parse_modes(args.tta))
    inference_config = {
        "cellprob_mode": args.cellprob_mode,
        "flow_scale": args.flow_scale,
        "niter": args.niter,
        "cellprob_threshold": args.cellprob_threshold,
        "flow_threshold": args.flow_threshold,
        "min_size": args.min_size,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "tile_batch_size": args.tile_batch_size,
        "center_alpha": args.center_alpha,
        "postprocess": args.postprocess,
        "cc_min_size": args.cc_min_size,
        "cc_touching_size": args.cc_touching_size,
        "use_centerness_seeds": args.use_centerness_seeds,
        "seed_threshold": args.seed_threshold,
        "seed_min_distance": args.seed_min_distance,
        "stitch_mode": args.stitch_mode,
        # Pull input_mode from ckpt config (rgb for legacy R34/Hybrid/IGA; h_repeat for H-R34 ckpts).
        "input_mode": checkpoint_config.get("input_mode", "rgb"),
    }
    all_rows: list[dict[str, Any]] = []
    summaries = {}
    for split in args.split:
        summary, rows = evaluate_model(
            model,
            args.dataset,
            split,
            root,
            inference_config,
            out_dir=out_dir,
            save_predictions=args.save_predictions,
            limit=args.limit,
            device=device,
        )
        summaries[split] = summary
        all_rows.extend(rows)
    write_combined_per_image(out_dir, all_rows)
    (out_dir / "eval_config.yaml").write_text(
        yaml.safe_dump(
            {
                "dataset": args.dataset,
                "splits": args.split,
                "checkpoint": args.checkpoint,
                "flow_param": flow_param,
                "inference": inference_config,
                "summaries": summaries,
            },
            sort_keys=False,
        )
    )


if __name__ == "__main__":
    main()
