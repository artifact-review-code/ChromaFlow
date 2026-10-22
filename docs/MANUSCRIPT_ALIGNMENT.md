# Manuscript alignment record — 2026-10-21

The local release was aligned against the 10-page ChromaFlow manuscript with
46 references. Existing uncommitted changes were preserved. A complete backup
(including Git state) was made outside the repository before editing.

| Area | Change |
|---|---|
| CAR | Chromatic Anchor Regularization -> Chromophore-Anchored Recovery |
| Loss weights | fg/flow/point/div/center/H/E = 1/2/2/0.5/0.75/1/0.5 |
| Optimizer | Adam defaults beta=(0.9,0.999), lr=1e-4, weight decay=3e-4 |
| Training | 150 epochs; all four datasets supported by the common wrapper |
| Coarse labels | Canonical RGB+distance per-Voronoi K-means preparation added; SAV is legacy |
| Points | Uniform fixed-size sampling without replacement per original image before tiling |
| Sparse supervision | Rebuild every weak target from retained points; no discarded-point exclusion disks |
| Crop | 250x250 prepared patches jointly cropped to 224x224, displacement denominator fixed at 250 |
| Source/center targets | Max-Gaussian maps prepared before cropping, sigma=3; center MSE gamma=2 |
| CAR normalization | 1/99 percentiles with epsilon=1e-6, then per-crop min-max |
| Augmentation | Flips, 90-degree rotations and scale jitter for all datasets; HED only an explicit control |
| Decoder | Shared S4_drop for validation/final evaluation; zero-seed images yield no instances; expansion=1 |
| Splits | Old CPM/CoNIC/NuInsSeg manifests removed on request; external final patient/WSI manifests required |
| Evaluation | Removed CPM hard-coded val/test IDs and CoNIC random 250-patch sampling; honors requested split |
| AJI | Corrected zero-overlap unmatched-object union accounting |
| Results | Both CSV tables and Markdown tables synchronized to the current manuscript |
| Figures | Updated nine figure assets and numbering to the manuscript |

## Final data required

CPM-17, CoNIC and NuInsSeg final membership and image-to-patient/WSI records
must be provided before running their training/evaluation. See [Datasets](DATASETS.md).
The program validates provenance rather than inferring patient IDs from patch
numbers. The retained MoNuSeg manifest is checked using TCGA participant IDs.

## Explicit implementation conventions

The manuscript does not specify all K-means feature scaling, cluster assignment,
patch stride, integer rounding and random-seed details. These are now explicit
in [Data format](DATA_FORMAT.md) and [Training](TRAINING.md). The preprocessing
implementation is the current documented protocol; matching historical runs
requires their original data preparation records and final manifests.

The bundled CSVs record manuscript results. Tests exercise equations, sampling,
provenance, augmentation, decoder edge cases and synthetic pipeline execution.

## Verification performed

- 15 pytest checks passed on CPU (Python 3.11, PyTorch, Cellpose 3.1.1.2).
- A synthetic-data CLI run completed preparation, one training batch, validation
  selection, checkpoint saving/loading and final validation/test evaluation.
- The independent evaluation CLI successfully loaded that checkpoint.
- Model parameter count: 24,954,539 (24.95M rounded, including the retained
  ResNet classifier parameters in the existing model definition).
- Both manuscript result tables match their CSV transcriptions.
- Bundled MoNuSeg counts and patient/WSI disjointness pass; other manifests are
  required external inputs and are reported as such by release validation.
- Python compilation, shell syntax and Git whitespace checks passed.

The synthetic smoke run used randomly initialized weights and synthetic images;
it verifies execution and checkpoint handling. Reported paper scores remain in
`results/` as manuscript transcriptions.
