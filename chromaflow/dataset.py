from __future__ import annotations

import json
import random
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt as _edt_for_gap, label as _cc_label_for_idg
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_STAIN_ALPHA = 0.1


@dataclass(frozen=True)
class PatchRecord:
    tile: str
    image_path: Path
    clu_path: Path
    pob_path: Path
    sam_path: Path | None


def decode_clu(path: Path, image: np.ndarray) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB"))
    label = np.ones(arr.shape[:2], dtype=np.uint8) * 2
    label[arr[..., 0] > 255 * 0.3] = 0
    label[arr[..., 1] > 255 * 0.5] = 1
    black = (image[..., 0] == 0) & (image[..., 1] == 0) & (image[..., 2] == 0)
    label[black] = 0
    return label


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    image = image.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(image.transpose(2, 0, 1).copy()).float()


def normalize_image(image: Image.Image | np.ndarray) -> torch.Tensor:
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    return image_to_tensor(image)


def build_input_tensor(image: Image.Image | np.ndarray, input_mode: str = "rgb") -> torch.Tensor:
    """Dispatcher used by eval / decode_variants / ecau_lite at inference time.
    Returns a SINGLE tensor — for dual-input ('h_and_rgb'), use build_dual_input_tensor.
    """
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    if input_mode == "rgb":
        return image_to_tensor(image)
    if input_mode == "h_repeat":
        return h_repeat_to_tensor(image)
    if input_mode == "h_and_rgb":
        # By convention, primary "image" is H-3ch (R34 path); RGB is fetched separately.
        return h_repeat_to_tensor(image)
    if input_mode == "he_rgb":
        import torch as _torch
        rgb_t = image_to_tensor(image)  # (3, H, W) ImageNet-norm
        H = _torch.from_numpy(stain_h_channel(image)).float().unsqueeze(0)
        E = _torch.from_numpy(stain_e_channel(image)).float().unsqueeze(0)
        return _torch.cat([rgb_t, H, E], dim=0)
    raise ValueError(f"Unknown input_mode: {input_mode}")


def build_dual_input_tensor(image: Image.Image | np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """For H-PS-C2F-Net: returns (h_tensor, rgb_tensor), both (3, H, W) ImageNet-normalized."""
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    return h_repeat_to_tensor(image), image_to_tensor(image)


def h_repeat_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Build an H-channel-replicated 3-channel tensor for the H-R34 branch.
    Uses the same `stain_h_channel` extraction as the rgb pipeline. The H channel
    is in [0,1]; we apply ImageNet normalization per-channel after replication so
    the trainable R34 stem (pretrained on ImageNet RGB) can consume the input.
    """
    H = stain_h_channel(image)  # (H, W) float32 in [0,1]
    H3 = np.stack([H, H, H], axis=-1)  # (H, W, 3) float32
    # ImageNet normalize per-channel; same scheme as RGB path keeps the head stable.
    H3 = (H3 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(H3.transpose(2, 0, 1).copy()).float()


def stain_h_channel(image: np.ndarray) -> np.ndarray:
    """Return robustly normalized hematoxylin channel in [0, 1]."""
    from skimage.color import rgb2hed

    image_float = image.astype(np.float32) / 255.0
    h_channel = rgb2hed(image_float)[..., 0].astype(np.float32)
    lo, hi = np.percentile(h_channel, [1.0, 99.0])
    if hi <= lo + 1e-6:
        return np.zeros(h_channel.shape, dtype=np.float32)
    h_channel = (h_channel - lo) / (hi - lo)
    return np.clip(h_channel, 0.0, 1.0).astype(np.float32)


def stain_e_channel(image: np.ndarray) -> np.ndarray:
    """Return robustly normalized eosin channel in [0, 1].

    Symmetric to stain_h_channel but extracts the E (cytoplasm marker) channel
    via Ruifrok-Johnston color deconvolution (skimage rgb2hed index 1). Used by
    L_eosin_neg supervision: BoNuS-ignore-zone pixels with high E + low H are
    biologically near-certain bg (cytoplasm), so we anchor hard bg supervision
    on them — complementary to L_stain_soft which anchors soft fg via H.
    """
    from skimage.color import rgb2hed

    image_float = image.astype(np.float32) / 255.0
    e_channel = rgb2hed(image_float)[..., 1].astype(np.float32)
    lo, hi = np.percentile(e_channel, [1.0, 99.0])
    if hi <= lo + 1e-6:
        return np.zeros(e_channel.shape, dtype=np.float32)
    e_channel = (e_channel - lo) / (hi - lo)
    return np.clip(e_channel, 0.0, 1.0).astype(np.float32)


def _crop_hw(arr: np.ndarray, top: int, left: int, size: int) -> np.ndarray:
    return arr[top : top + size, left : left + size]


def _crop_chw(arr: np.ndarray, top: int, left: int, size: int) -> np.ndarray:
    return arr[:, top : top + size, left : left + size]


def _rotate_flow(flow: np.ndarray, k: int) -> np.ndarray:
    flow = np.rot90(flow, k, axes=(1, 2)).copy()
    if k == 0:
        return flow
    dx = flow[0].copy()
    dy = flow[1].copy()
    if k == 1:
        flow[0] = dy
        flow[1] = -dx
    elif k == 2:
        flow[0] = -dx
        flow[1] = -dy
    elif k == 3:
        flow[0] = -dy
        flow[1] = dx
    return flow


def _load_pseudo(path: Path | None, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    if path is None or not path.exists():
        h, w = shape
        return {
            "sam_fg_mask": np.zeros((h, w), dtype=np.uint8),
            "sam_flow": np.zeros((2, h, w), dtype=np.float32),
            "sam_flow_mask": np.zeros((h, w), dtype=np.uint8),
        }
    data = np.load(path)
    return {
        "sam_fg_mask": data["sam_fg_mask"].astype(np.uint8),
        "sam_flow": data["sam_flow"].astype(np.float32),
        "sam_flow_mask": data["sam_flow_mask"].astype(np.uint8),
    }



def _hed_color_jitter(image_rgb_u8: np.ndarray, sigma: float = 0.05,
                       bias: float = 0.05, rng=None) -> np.ndarray:
    """HED-space color augmentation (Tellez et al. MIA 2019).

    Decompose to Haematoxylin / Eosin / Background channels via Ruifrok-Johnston
    color deconvolution, multiply each channel by 1+U(-sigma,sigma), add
    U(-bias,bias), then convert back. Widens training stain distribution to
    improve cross-stain generalization.
    """
    from skimage.color import rgb2hed, hed2rgb
    if rng is None:
        rng = np.random
    img_f = image_rgb_u8.astype(np.float32) / 255.0
    hed = rgb2hed(img_f)
    for c in range(3):
        alpha = 1.0 + rng.uniform(-sigma, sigma)
        beta = rng.uniform(-bias, bias)
        hed[..., c] = hed[..., c] * alpha + beta
    out = hed2rgb(hed)
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)



def _apply_point_ratio(point_map: np.ndarray, clu: np.ndarray,
                        ratio: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Keep `ratio` fraction of points; mark dropped points' disks (clu==1) as ignore (clu==2).

    BoNuS-style annotation efficiency test. Voronoi gap (clu==2) unchanged.
    """
    from scipy.ndimage import label as sclabel
    from skimage.draw import disk as skdisk
    pm = (point_map > 0).astype(np.uint8)
    labeled, n = sclabel(pm)
    if n == 0:
        return point_map, clu
    keep = rng.random(n) < ratio
    new_pm = np.zeros_like(point_map)
    new_clu = clu.copy()
    h, w = clu.shape
    for i in range(1, n + 1):
        mask = (labeled == i)
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue
        if keep[i - 1]:
            new_pm[mask] = 1
        else:
            cy, cx = int(ys.mean()), int(xs.mean())
            rr, cc = skdisk((cy, cx), 15, shape=(h, w))
            sel = (new_clu[rr, cc] == 1)
            rr_sel, cc_sel = rr[sel], cc[sel]
            new_clu[rr_sel, cc_sel] = 2
    return new_pm, new_clu


def _apply_point_shift(point_map: np.ndarray, clu: np.ndarray,
                        max_shift: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Shift each point by uniform random pixels in [-max_shift, max_shift].

    Tests robustness to noisy click positions. Cluster disks (clu==1) get
    redrawn at shifted positions; original disks become ignore.
    """
    from scipy.ndimage import label as sclabel
    from skimage.draw import disk as skdisk
    pm = (point_map > 0).astype(np.uint8)
    labeled, n = sclabel(pm)
    if n == 0 or max_shift <= 0:
        return point_map, clu
    new_pm = np.zeros_like(point_map)
    new_clu = clu.copy()
    h, w = clu.shape
    # Erase all original fg disks first (mark as 2).
    new_clu[new_clu == 1] = 2
    for i in range(1, n + 1):
        mask = (labeled == i)
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue
        cy, cx = int(ys.mean()), int(xs.mean())
        dy = int(rng.integers(-max_shift, max_shift + 1))
        dx = int(rng.integers(-max_shift, max_shift + 1))
        ny, nx = int(np.clip(cy + dy, 0, h - 1)), int(np.clip(cx + dx, 0, w - 1))
        new_pm[ny, nx] = 1
        rr, cc = skdisk((ny, nx), 15, shape=(h, w))
        # Only set clu==1 where it was previously ignore (clu==2) — preserve bg (clu==0).
        sel = (new_clu[rr, cc] == 2)
        rr_sel, cc_sel = rr[sel], cc[sel]
        new_clu[rr_sel, cc_sel] = 1
    return new_pm, new_clu


def _scale_jitter(image, stain_h, stain_e, clu, offset, basin_valid, point_map,
                  instance_id, s: float, crop_size: int):
    """Synthesize varied nucleus sizes by rescaling crop content by factor s, output
    kept at crop_size. s<1 -> smaller nuclei (mean-padded); s>1 -> larger (center-crop).
    offset is a centroid-radial direction field -> bilinear resize WITHOUT renorm
    (directions preserved under uniform scaling; magnitudes kept in-distribution, NOT
    forced to unit). point_map points re-placed at scaled coords (not interpolated).
    clu/instance_id use nearest. Pad fill is bg-consistent."""
    import cv2
    S = int(crop_size)
    inner = max(8, int(round(S * float(s))))

    def _lin(a):
        return cv2.resize(a, (inner, inner), interpolation=cv2.INTER_LINEAR)

    def _near_int(a):
        r = cv2.resize(a.astype(np.float32), (inner, inner), interpolation=cv2.INTER_NEAREST)
        return r.astype(a.dtype)

    img_r = _lin(image)
    sh_r = _lin(stain_h.astype(np.float32))
    se_r = _lin(stain_e.astype(np.float32))
    clu_r = _near_int(clu)
    bv_r = _lin(basin_valid.astype(np.float32))
    iid_r = _near_int(instance_id)
    off_r = cv2.resize(np.transpose(offset, (1, 2, 0)), (inner, inner),
                       interpolation=cv2.INTER_LINEAR)
    off_r = np.ascontiguousarray(np.transpose(off_r, (2, 0, 1)).astype(np.float32))

    ys, xs = np.where(point_map > 0)
    pm_r = np.zeros((inner, inner), dtype=point_map.dtype)
    if len(ys):
        nys = np.clip(np.round(ys * float(s)).astype(int), 0, inner - 1)
        nxs = np.clip(np.round(xs * float(s)).astype(int), 0, inner - 1)
        pm_r[nys, nxs] = 1

    if inner <= S:
        o = (S - inner) // 2
        sl = slice(o, o + inner)
        mean_col = image.reshape(-1, image.shape[-1]).mean(0)
        out_img = np.empty((S, S, image.shape[-1]), dtype=image.dtype)
        out_img[:] = mean_col.astype(image.dtype)
        out_img[sl, sl] = img_r
        out_sh = np.full((S, S), float(stain_h.mean()), np.float32); out_sh[sl, sl] = sh_r
        out_se = np.full((S, S), float(stain_e.mean()), np.float32); out_se[sl, sl] = se_r
        out_clu = np.zeros((S, S), clu.dtype); out_clu[sl, sl] = clu_r
        out_bv = np.zeros((S, S), np.float32); out_bv[sl, sl] = bv_r
        out_iid = np.zeros((S, S), instance_id.dtype); out_iid[sl, sl] = iid_r
        out_off = np.zeros((2, S, S), np.float32); out_off[:, sl, sl] = off_r
        out_pm = np.zeros((S, S), point_map.dtype); out_pm[sl, sl] = pm_r
    else:
        o = (inner - S) // 2
        sl = slice(o, o + S)
        out_img = np.ascontiguousarray(img_r[sl, sl])
        out_sh = np.ascontiguousarray(sh_r[sl, sl])
        out_se = np.ascontiguousarray(se_r[sl, sl])
        out_clu = np.ascontiguousarray(clu_r[sl, sl])
        out_bv = np.ascontiguousarray(bv_r[sl, sl])
        out_iid = np.ascontiguousarray(iid_r[sl, sl])
        out_off = np.ascontiguousarray(off_r[:, sl, sl])
        out_pm = np.ascontiguousarray(pm_r[sl, sl])
    return out_img, out_sh, out_se, out_clu, out_off, out_bv, out_pm, out_iid


class NucleiV3PatchDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        dataset: str,
        subset: str,
        clu_subdir: str = "labels_clu",
        sags_snap_window: int = 0,
        crop_size: int = 224,
        augment: bool = False,
        use_sam_pseudo: bool = False,
        sam_pseudo_path: str | Path | None = None,
        stage2_pseudo_dir: str | Path | None = None,
        input_mode: str = "rgb",
        stain_aug: bool = False,
        stain_aug_sigma: float = 0.05,
        stain_aug_bias: float = 0.05,
        point_ratio: float = 1.0,
        point_shift: int = 0,
        point_subset_seed: int = 0,
        scale_jitter: bool = False,
        scale_jitter_min: float = 0.5,
        scale_jitter_max: float = 1.3,
    ) -> None:
        self.root = Path(root)
        self.dataset = dataset.upper()
        self.subset = subset
        self.clu_subdir = clu_subdir
        self.sags_snap_window = sags_snap_window
        self.crop_size = crop_size
        self.augment = augment
        self.use_sam_pseudo = use_sam_pseudo
        self.sam_pseudo_path = Path(sam_pseudo_path) if sam_pseudo_path else None
        # When set, overrides `instance_id` with stage-1 model's pseudo predictions
        # for Stage-2 self-train (BoNuS-style L_aff on predicted instances).
        self.stage2_pseudo_dir = Path(stage2_pseudo_dir) if stage2_pseudo_dir else None
        if input_mode not in ("rgb", "h_repeat", "h_and_rgb", "he_rgb"):
            raise ValueError(f"input_mode must be 'rgb', 'h_repeat', 'h_and_rgb', or 'he_rgb', got {input_mode!r}")
        self.input_mode = input_mode
        self.stain_aug = stain_aug
        self.stain_aug_sigma = float(stain_aug_sigma)
        self.stain_aug_bias = float(stain_aug_bias)
        self.point_ratio = float(point_ratio)
        self.point_shift = int(point_shift)
        self.point_subset_seed = int(point_subset_seed)
        self.scale_jitter = bool(scale_jitter)
        self.scale_jitter_min = float(scale_jitter_min)
        self.scale_jitter_max = float(scale_jitter_max)
        self.records = self._build_records()
        if not self.records:
            raise RuntimeError(f"No v3 patch records found for {self.dataset}/{subset}")

    def _build_records(self) -> list[PatchRecord]:
        bonus_root = self.root / "data/official_clean/bonus_prepared" / self.dataset
        ours_root = self.root / "data/official_clean/ours_prepared" / self.dataset
        image_dir = bonus_root / "images" / self.subset
        clu_dir = bonus_root / f"ratio2.0/{self.clu_subdir}" / self.subset
        pob_dir = ours_root / "ratio2.0/pob_targets" / self.subset
        records: list[PatchRecord] = []
        for image_path in sorted(image_dir.glob("*.png")):
            tile = image_path.stem
            clu_path = clu_dir / f"{tile}_label_clu.png"
            pob_path = pob_dir / f"{tile}.npz"
            if not clu_path.exists() or not pob_path.exists():
                continue
            sam_path = None
            if self.sam_pseudo_path is not None:
                candidate = self.sam_pseudo_path / self.subset / f"{tile}.npz"
                sam_path = candidate
            records.append(PatchRecord(tile, image_path, clu_path, pob_path, sam_path))
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        image = np.asarray(Image.open(rec.image_path).convert("RGB"))
        if self.augment and self.stain_aug and random.random() < 0.8:
            image = _hed_color_jitter(image, sigma=self.stain_aug_sigma,
                                       bias=self.stain_aug_bias)
        stain_h = stain_h_channel(image)
        stain_e = stain_e_channel(image)
        clu = decode_clu(rec.clu_path, image)
        pob = np.load(rec.pob_path)
        offset = pob["offset"].astype(np.float32)
        basin_valid = pob["basin_valid"].astype(np.float32)
        point_map = pob["point_map"].astype(np.uint8)
        # Apply training-time point ratio / shift perturbations (BoNuS-style analyses).
        if self.augment and (self.point_ratio < 1.0 or self.point_shift > 0):
            if self.point_ratio < 1.0:
                # FIXED sparse annotation: deterministic per-image subset seeded by
                # the tile id (crc32), so the SAME points are kept every epoch and
                # subsets are nested across ratios. This measures true fixed-budget
                # annotation efficiency, NOT per-epoch resampling (which unions to
                # ~full supervision over 150 epochs and flattens the curve).
                _pr_rng = np.random.default_rng(
                    (zlib.crc32(rec.tile.encode()) + self.point_subset_seed) & 0x7fffffff)
                point_map, clu = _apply_point_ratio(point_map, clu, self.point_ratio, _pr_rng)
            if self.point_shift > 0:
                # FIXED noisy clicks: deterministic per-image displacement (same every
                # epoch), seeded by tile+seed (+777 to decorrelate from the ratio rng).
                # Measures robustness to a fixed set of imprecise annotations, NOT
                # per-epoch jitter augmentation (which would average noise to ~clean).
                _ps_rng = np.random.default_rng(
                    (zlib.crc32(rec.tile.encode()) + self.point_subset_seed + 777) & 0x7fffffff)
                point_map, clu = _apply_point_shift(point_map, clu, self.point_shift, _ps_rng)
        instance_id = pob["instance_id_soft"].astype(np.int32)

        # D9 SAGS — snap GT clicks to local H-peak (per-tile, before crop).
        if self.sags_snap_window > 0:
            half = int(self.sags_snap_window) // 2
            ys, xs = np.where(point_map > 0)
            new_pm = np.zeros_like(point_map)
            H_full, W_full = point_map.shape
            for cy, cx in zip(ys, xs):
                y0 = max(0, cy - half); y1 = min(H_full, cy + half + 1)
                x0 = max(0, cx - half); x1 = min(W_full, cx + half + 1)
                local = stain_h[y0:y1, x0:x1]
                if local.size == 0: continue
                py, px = np.unravel_index(np.argmax(local), local.shape)
                new_pm[y0 + py, x0 + px] = 1
            point_map = new_pm
        # Stage-2: replace voronoi instance_id with stage-1 pseudo predictions when available.
        if self.stage2_pseudo_dir is not None:
            stage2_path = self.stage2_pseudo_dir / self.subset / f"{rec.tile}.npz"
            if stage2_path.exists():
                instance_id = np.load(stage2_path)["instance_id"].astype(np.int32)
        pseudo = _load_pseudo(rec.sam_path if self.use_sam_pseudo else None, clu.shape)

        h, w = clu.shape
        if self.crop_size > 0 and (h != self.crop_size or w != self.crop_size):
            if self.augment:
                top = random.randint(0, h - self.crop_size)
                left = random.randint(0, w - self.crop_size)
            else:
                top = max(0, (h - self.crop_size) // 2)
                left = max(0, (w - self.crop_size) // 2)
            image = _crop_hw(image, top, left, self.crop_size)
            stain_h = _crop_hw(stain_h, top, left, self.crop_size)
            stain_e = _crop_hw(stain_e, top, left, self.crop_size)
            clu = _crop_hw(clu, top, left, self.crop_size)
            offset = _crop_chw(offset, top, left, self.crop_size)
            basin_valid = _crop_hw(basin_valid, top, left, self.crop_size)
            point_map = _crop_hw(point_map, top, left, self.crop_size)
            instance_id = _crop_hw(instance_id, top, left, self.crop_size)
            pseudo["sam_fg_mask"] = _crop_hw(pseudo["sam_fg_mask"], top, left, self.crop_size)
            pseudo["sam_flow"] = _crop_chw(pseudo["sam_flow"], top, left, self.crop_size)
            pseudo["sam_flow_mask"] = _crop_hw(pseudo["sam_flow_mask"], top, left, self.crop_size)

        if self.augment:
            if random.random() < 0.5:
                image = image[:, ::-1].copy()
                stain_h = stain_h[:, ::-1].copy()
                stain_e = stain_e[:, ::-1].copy()
                clu = clu[:, ::-1].copy()
                offset = offset[:, :, ::-1].copy()
                offset[0] *= -1.0
                basin_valid = basin_valid[:, ::-1].copy()
                point_map = point_map[:, ::-1].copy()
                instance_id = instance_id[:, ::-1].copy()
                pseudo["sam_fg_mask"] = pseudo["sam_fg_mask"][:, ::-1].copy()
                pseudo["sam_flow"] = pseudo["sam_flow"][:, :, ::-1].copy()
                pseudo["sam_flow"][0] *= -1.0
                pseudo["sam_flow_mask"] = pseudo["sam_flow_mask"][:, ::-1].copy()
            if random.random() < 0.5:
                image = image[::-1, :].copy()
                stain_h = stain_h[::-1, :].copy()
                stain_e = stain_e[::-1, :].copy()
                clu = clu[::-1, :].copy()
                offset = offset[:, ::-1, :].copy()
                offset[1] *= -1.0
                basin_valid = basin_valid[::-1, :].copy()
                point_map = point_map[::-1, :].copy()
                instance_id = instance_id[::-1, :].copy()
                pseudo["sam_fg_mask"] = pseudo["sam_fg_mask"][::-1, :].copy()
                pseudo["sam_flow"] = pseudo["sam_flow"][:, ::-1, :].copy()
                pseudo["sam_flow"][1] *= -1.0
                pseudo["sam_flow_mask"] = pseudo["sam_flow_mask"][::-1, :].copy()
            k = random.randint(0, 3)
            if k:
                image = np.rot90(image, k, axes=(0, 1)).copy()
                stain_h = np.rot90(stain_h, k).copy()
                stain_e = np.rot90(stain_e, k).copy()
                clu = np.rot90(clu, k).copy()
                offset = _rotate_flow(offset, k)
                basin_valid = np.rot90(basin_valid, k).copy()
                point_map = np.rot90(point_map, k).copy()
                instance_id = np.rot90(instance_id, k).copy()
                pseudo["sam_fg_mask"] = np.rot90(pseudo["sam_fg_mask"], k).copy()
                pseudo["sam_flow"] = _rotate_flow(pseudo["sam_flow"], k)
                pseudo["sam_flow_mask"] = np.rot90(pseudo["sam_flow_mask"], k).copy()
            if (self.scale_jitter and not self.use_sam_pseudo
                    and random.random() < 0.5):
                _sj = random.uniform(self.scale_jitter_min, self.scale_jitter_max)
                (image, stain_h, stain_e, clu, offset, basin_valid,
                 point_map, instance_id) = _scale_jitter(
                    image, stain_h, stain_e, clu, offset, basin_valid,
                    point_map, instance_id, _sj, self.crop_size)

        # GAP: distance from each pixel to nearest clu==1 pixel (post-augmentation).
        # Used by L_GAP loss for geometric anchor-proximity supervision on clu==2.
        try:
            d_to_clu1 = _edt_for_gap((clu != 1).astype(np.uint8)).astype(np.float32)
            # IDG: distance to 2nd-nearest clu==1 connected component (for touching detection)
            _clu1_mask_idg = (clu == 1).astype(np.uint8)
            _cc_arr, _n_cc = _cc_label_for_idg(_clu1_mask_idg)
            if _n_cc >= 2:
                _dists = np.stack([_edt_for_gap((_cc_arr != _i+1).astype(np.uint8)).astype(np.float32)
                                   for _i in range(_n_cc)], axis=0)
                _sorted = np.sort(_dists, axis=0)
                d_to_clu1_2nd = _sorted[1]
            else:
                d_to_clu1_2nd = np.full_like(d_to_clu1, 1000.0)
        except Exception:
            d_to_clu1 = np.zeros_like(clu, dtype=np.float32)
            d_to_clu1_2nd = np.full_like(d_to_clu1, 1000.0)
        stain_sigma = DEFAULT_STAIN_ALPHA + (1.0 - DEFAULT_STAIN_ALPHA) * stain_h
        # Input mode dispatch.
        # - "rgb": legacy ImageNet-normalized RGB to `image`.
        # - "h_repeat": Hematoxylin channel replicated to 3ch (H-R34 branch).
        # - "h_and_rgb": H-PS-C2F-Net dual-input — both H-3ch and RGB-3ch in batch.
        #                R34 path consumes H-3ch (local nuclear morphology),
        #                UNI ViT path consumes RGB-3ch (semantic / contextual prior).
        rgb_tensor = image_to_tensor(image)
        if self.input_mode == "h_repeat":
            H3 = np.stack([stain_h, stain_h, stain_h], axis=-1)
            H3 = (H3 - IMAGENET_MEAN) / IMAGENET_STD
            image_tensor = torch.from_numpy(H3.transpose(2, 0, 1).copy()).float()
            image_rgb_tensor = rgb_tensor  # also expose RGB so models that need it can read
        elif self.input_mode == "h_and_rgb":
            H3 = np.stack([stain_h, stain_h, stain_h], axis=-1)
            H3 = (H3 - IMAGENET_MEAN) / IMAGENET_STD
            image_tensor = torch.from_numpy(H3.transpose(2, 0, 1).copy()).float()
            image_rgb_tensor = rgb_tensor
        elif self.input_mode == "he_rgb":
            # 5-ch: [R,G,B (ImageNet-norm), H, E (raw [0,1] from stain_*_channel)]
            H1 = torch.from_numpy(stain_h.copy()).float().unsqueeze(0)
            E1 = torch.from_numpy(stain_e.copy()).float().unsqueeze(0)
            image_tensor = torch.cat([rgb_tensor, H1, E1], dim=0)
            image_rgb_tensor = rgb_tensor
        else:
            image_tensor = rgb_tensor
            image_rgb_tensor = rgb_tensor
        return {
            "image": image_tensor,
            "image_rgb": image_rgb_tensor,
            "stain_h": torch.from_numpy(stain_h.copy()).float(),
            "stain_e": torch.from_numpy(stain_e.copy()).float(),
            "stain_sigma": torch.from_numpy(stain_sigma.copy()).float(),
            "clu": torch.from_numpy(clu.copy()).long(),
            "d_to_clu1": torch.from_numpy(d_to_clu1.copy()).float(),
            "d_to_clu1_2nd": torch.from_numpy(d_to_clu1_2nd.copy()).float(),
            "offset": torch.from_numpy(offset.copy()).float(),
            "basin_valid": torch.from_numpy(basin_valid.copy()).float(),
            "point_map": torch.from_numpy(point_map.copy()).float(),
            "instance_id": torch.from_numpy(instance_id.copy()).long(),
            "sam_fg_mask": torch.from_numpy(pseudo["sam_fg_mask"].copy()).float(),
            "sam_flow": torch.from_numpy(pseudo["sam_flow"].copy()).float(),
            "sam_flow_mask": torch.from_numpy(pseudo["sam_flow_mask"].copy()).float(),
            "tile": rec.tile,
        }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())
