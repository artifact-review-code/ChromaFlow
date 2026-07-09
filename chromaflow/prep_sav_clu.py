"""Generate Stain-Aware Voronoi (SAV) cluster labels for D6.

For each tile:
  1. Read image RGB.
  2. Compute H_norm via skimage.rgb2hed → per-tile min-max normalize.
  3. Read point_map (GT centers) from pob_targets/{tile}.npz.
  4. Build cost map: cost(p) = 1 / (H_norm(p) + 0.1).
       → low cost in nucleus-rich pixels, high cost in stroma/background.
  5. Run multi-source geodesic (skimage.graph.MCP_Geometric) from GT points.
  6. Threshold geodesic distance:
       d <= hard_thresh  → clu=1 (stain-aware hard fg)
       d <= voronoi_thresh → clu=2 (ignore zone)
       otherwise → clu=0 (bg)
  7. Encode to RGB matching decode_clu's scheme:
       clu==0 → (255, 0, 0) red
       clu==1 → (0, 255, 0) green
       clu==2 → (0, 0, 255) blue

Output: bonus_prepared/{dataset}/ratio2.0/labels_clu_sav/{subset}/{tile}_label_clu.png

Run: python prep_sav_clu.py --dataset MO
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2hed
from skimage.graph import MCP_Geometric

DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def make_sav_clu(image: np.ndarray, point_map: np.ndarray,
                 hard_thresh: float = 10.0,
                 voronoi_thresh: float = 70.0,
                 h_eps: float = 0.1) -> np.ndarray:
    """Return clu in {0=bg, 1=hard-fg, 2=ignore} via stain-aware geodesic voronoi."""
    H, W = image.shape[:2]
    # Hematoxylin channel + per-image normalize to [0, 1]
    hed = rgb2hed(image.astype(np.float32) / 255.0)
    h = hed[..., 0]
    lo, hi = np.percentile(h, 2.0), np.percentile(h, 98.0)
    h_norm = np.clip((h - lo) / (hi - lo + 1e-8), 0, 1)
    # Cost = 1 / (H + ε): high H → low cost, low H → high cost
    cost = 1.0 / (h_norm + h_eps)  # ∈ [~0.9, 10]
    # Sources = GT points
    sources = [(int(y), int(x)) for y, x in np.argwhere(point_map > 0)]
    if not sources:
        return np.zeros((H, W), dtype=np.uint8)
    mcp = MCP_Geometric(cost.astype(np.float64), fully_connected=True)
    dist, _ = mcp.find_costs(sources)
    dist = np.asarray(dist)
    dist[~np.isfinite(dist)] = 1e9
    clu = np.zeros((H, W), dtype=np.uint8)
    clu[dist <= voronoi_thresh] = 2
    clu[dist <= hard_thresh] = 1
    # Black pixels → bg
    black = (image[..., 0] == 0) & (image[..., 1] == 0) & (image[..., 2] == 0)
    clu[black] = 0
    return clu


def encode_clu_rgb(clu: np.ndarray) -> np.ndarray:
    """Encode {0, 1, 2} → RGB matching decode_clu (R>76, G>127, else)."""
    H, W = clu.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[clu == 0] = (255, 0, 0)      # red → R>76, G≤127 → bg
    rgb[clu == 1] = (0, 255, 0)      # green → R≤76, G>127 → hard fg
    rgb[clu == 2] = (0, 0, 255)      # blue → R≤76, G≤127 → ignore
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="Repository or data root containing data/official_clean")
    ap.add_argument("--dataset", default="MO", choices=["MO", "CPM"])
    ap.add_argument("--subsets", nargs="+", default=["train", "val"])
    ap.add_argument("--hard-thresh", type=float, default=10.0)
    ap.add_argument("--voronoi-thresh", type=float, default=70.0)
    ap.add_argument("--h-eps", type=float, default=0.1)
    args = ap.parse_args()
    root = Path(args.root).resolve()

    bonus_root = root / "data/official_clean/bonus_prepared" / args.dataset
    ours_root = root / "data/official_clean/ours_prepared" / args.dataset
    out_root = bonus_root / "ratio2.0/labels_clu_sav"

    for subset in args.subsets:
        img_dir = bonus_root / "images" / subset
        pob_dir = ours_root / "ratio2.0/pob_targets" / subset
        out_dir = out_root / subset
        out_dir.mkdir(parents=True, exist_ok=True)
        tiles = sorted(img_dir.glob("*.png"))
        print(f"[{subset}] {len(tiles)} tiles → {out_dir}")
        for i, img_path in enumerate(tiles):
            tile = img_path.stem
            pob_path = pob_dir / f"{tile}.npz"
            if not pob_path.exists():
                print(f"  skip {tile} (no pob)"); continue
            out_path = out_dir / f"{tile}_label_clu.png"
            if out_path.exists():
                continue
            image = np.asarray(Image.open(img_path).convert("RGB"))
            point_map = np.load(pob_path)["point_map"].astype(np.uint8)
            clu = make_sav_clu(image, point_map,
                               hard_thresh=args.hard_thresh,
                               voronoi_thresh=args.voronoi_thresh,
                               h_eps=args.h_eps)
            rgb = encode_clu_rgb(clu)
            Image.fromarray(rgb).save(out_path)
            if (i + 1) % 50 == 0:
                print(f"  [{subset}] {i+1}/{len(tiles)}")
    print("[done]")


if __name__ == "__main__":
    main()
