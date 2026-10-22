# Data format and preparation

`--root` is the data root, with full-resolution original images (not training crops):

```text
splits/{monuseg,cpm17,conic,nuinsseg}.json
data/official_clean/core/{monuseg,cpm17,conic,nuinsseg}/
  images/{original_image_id}.png
  points/{original_image_id}.npy
  labels_instance/{original_image_id}.npy
```

`points` is an HxW binary map containing single-pixel point positions.
Instance masks are HxW integer maps (0 background); they are used for evaluation.
With `--generate-points-from-masks`, preprocessing extracts one maximum-EDT
pixel per training instance, then discards the mask from target construction.
All downstream supervision uses RGB and the retained points only.

```bash
# From point annotations; prepare full budget (default) or 5% before tiling.
python -m chromaflow.prepare --root DATA_ROOT --dataset MO
python -m chromaflow.prepare --root SPARSE_DATA_ROOT --dataset CPM --point-ratio 0.05 --point-subset-seed 0
# Optional reproduction of the paper's mask-to-point extraction:
python -m chromaflow.prepare --root DATA_ROOT --dataset MO --generate-points-from-masks
```

Use separate prepared data roots for different budgets/selections. Existing
prepared targets are never silently reused or overwritten. The default tiling
stride is 250; `--stride` is recorded in the preparation manifest. Images
smaller than 250 are padded with white RGB and ignored labels.

Outputs use the existing folder layout for compatibility:

```text
data/official_clean/bonus_prepared/{MO,CPM,CONIC,NUINSSEG}/
  images/train/{tile}.png
  ratio2.0/labels_clu/train/{tile}_label_clu.png
data/official_clean/ours_prepared/{MO,CPM,CONIC,NUINSSEG}/
  preparation.json
  ratio2.0/pob_targets/train/{tile}.npz
```

Each prepared patch is 250x250. NPZ fields:

- `offset`: (2,H,W), `(point_x-x, point_y-y)/250`, channel order dx,dy.
- `basin_valid`: (H,W), weak foreground support.
- `point_map`: retained point locations within the patch.
- `instance_id_soft`: point-derived Voronoi IDs, not ground-truth instance IDs.
- `rho`, `center_target`: max Gaussian maps, sigma=3, constructed before cropping.

`preparation.json` records selected original-image points, point budget and seed,
patch origins, protocol version and split hash. Training rejects mismatches and
missing targets. Random 224x224 crops and geometric transforms are applied
jointly to RGB and targets; flow normalization remains 250.

The cluster map uses RGB/255 plus capped point distance/250 in three-class
K-means within each point Voronoi cell. Foreground is the cluster most represented
within a radius-2 point disk; background is the remaining cluster with greatest
mean distance; the remaining cluster is ignored. Disk(2)-dilated Voronoi
boundaries are ignored, and point disks are foreground. `random_state` and
`n_init=10` make clustering reproducible. These choices make the implementation
explicit where the manuscript gives only the high-level clustering rule.
An image without retained points has ignored weak labels and zero point/flow
source targets. The discarded nuclei's positions or masks never define a loss mask.

The old `prep_sav_clu.py` is a legacy stain-geodesic experiment and is not used
by the manuscript preparation or training commands.

## Point displacement robustness

Use `--point-shift R` during preparation to sample a fixed integer offset within
Euclidean radius R for each retained original-image point. Out-of-image or
colliding destinations are rejected. Every target is rebuilt from shifted
positions before tiling. Pass the same `--point-shift R` to `chromaflow.train`;
validation checks that it matches the manifest. The selection seed determines
these positions once; epochs do not resample them.
