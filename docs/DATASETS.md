# Datasets, sources, and splits

## Dataset summary

| Dataset | Content | Image size | Magnification | Train / val / test used here | Source |
|---|---|---:|---:|---:|---|
| MoNuSeg | 44 multi-organ H&E images | 1000x1000 | 40x | 24 / 6 / 14 images | [Grand Challenge](https://monuseg.grand-challenge.org/Data/) |
| CPM-17 | H&E images from four cancer groups | 500x500 or 600x600 | 20x/40x | 20 / 4 / 8 images | [Dataset repository](https://github.com/nauyan/NucleiSegmentation) |
| CoNIC | Multi-source colorectal H&E patches | 256x256 | 20x | 3372 / 516 / 1093 patches | [CoNIC Challenge](https://conic-challenge.grand-challenge.org/) |
| NuInsSeg | 665 patches from 31 human/mouse organs | 512x512 | dataset-dependent | 472 / 65 / 128 patches | [Kaggle](https://www.kaggle.com/datasets/ipateam/nuinsseg) |

## Split principles

- Splits are fixed and published in `splits/`; they are not regenerated during training.
- MoNuSeg uses 24 development images for training, six for validation, and all 14 official test images.
- CPM-17 is balanced across four cancer groups: each group contributes five training, one validation, and two test images.
- CoNIC uses a fixed source-aware split of 3372/516/1093 patches.
- NuInsSeg uses tissue-stratified sampling so that organ composition is represented across the 472/65/128 split.
- In-domain models are selected only on the validation partition of the same dataset.
- Cross-domain models are trained and selected entirely in the source domain. Post-processing is selected on source validation and frozen before the target test is evaluated.
- Every transfer direction sharing a target uses the same target test membership.

## Machine-readable manifests

| Dataset | Manifest |
|---|---|
| MoNuSeg | [`splits/monuseg.json`](../splits/monuseg.json) |
| CPM-17 | [`splits/cpm17.json`](../splits/cpm17.json) |
| CoNIC | [`splits/conic.json`](../splits/conic.json) |
| NuInsSeg | [`splits/nuinsseg.json`](../splits/nuinsseg.json) |

Run `python tools/validate_release.py` to verify counts and pairwise disjointness.

## Prepared training data

Training uses 1176 prepared 224x224 patches per source. The exact directory layout and NPZ fields are described in [Data format](DATA_FORMAT.md). Full datasets are not redistributed by this repository. Download each dataset from its source, accept its terms, and convert it to the documented layout.

## Citations and licenses

Please cite the original MoNuSeg, CPM-17, CoNIC, and NuInsSeg publications when using their data. Dataset licenses remain controlled by their providers. NuInsSeg is distributed as CC BY 4.0 on its source page; users should verify the current terms of every dataset before redistribution.
