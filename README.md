# ChromaFlow

**Anonymous artifact for point-supervised nuclei instance segmentation in H&E images.**

ChromaFlow learns nuclear extent and instance separation from one point per nucleus. It augments fixed point-derived cluster supervision with two analytic priors: **Divergence-Sink Flow (DSF)** for point-conditioned instance assignment and **Chromophore-Anchored Recovery (CAR)** for stain-supported nuclear extent.

## Network architecture

![ChromaFlow network architecture](assets/paper/figure2_architecture.png)

[High-resolution architecture PDF](assets/paper/figure2_architecture.pdf)

The shared ResNet-34 U-Net predicts foreground, centerness, and scalar potential. DSF turns the potential into sink-directed flow basins, while CAR constrains nuclear support using hematoxylin and eosin evidence. Centerness-guided basin refinement produces the final instance masks.

The canonical path is deliberately small: `dataset.py` prepares the training tensors, `model.py` defines the three-head network, `losses.py` implements the training objective, `train.py` runs optimization, and `eval.py`/`decode_variants.py` perform tiled inference and instance decoding. See [Code structure](docs/CODE_STRUCTURE.md) and [Method guide](docs/METHOD.md).

## Review release contents

Included:

- canonical model, losses, training, preprocessing, tiled inference, and evaluation code;
- verified MoNuSeg membership, external final-split validation, and manuscript result tables;
- updated figures from the current manuscript;
- limited, real training commands and a documented inference interface.

Not included during anonymous review:

- pretrained ChromaFlow checkpoints;
- complete third-party datasets;
- external baseline repositories, their checkpoints, or private experiment logs.

A compatible checkpoint is required to run predictions. Model weights will be released upon acceptance; see [Model zoo](docs/MODEL_ZOO.md).

## Installation

```bash
conda create -n chromaflow python=3.10 -y
conda activate chromaflow
pip install -r requirements.txt
pip install -e .
```

The canonical ResNet-34 path requires PyTorch, torchvision, NumPy, SciPy, scikit-image, Pillow, PyYAML, OpenCV, and Cellpose. `timm` is only required by optional encoder experiments.

## Training and evaluation

**Final CPM-17, CoNIC and NuInsSeg manifests must be supplied externally.** The old manifests have been removed. See [Datasets](docs/DATASETS.md) and [Alignment record](docs/MANUSCRIPT_ALIGNMENT.md).

Prepare static cluster labels, then train one seed:

```bash
bash scripts/prepare_labels.sh /path/to/project_or_data_root MO
bash scripts/train_canonical_mo.sh /path/to/project_or_data_root 0
```

Evaluate a user-provided checkpoint:

```bash
bash scripts/evaluate.sh \
  /path/to/project_or_data_root MO test \
  /path/to/best_val_aji.pth runs/eval_MO_test
```

The canonical run uses 150 epochs, random 224x224 crops from 250x250 prepared patches, batch size 16, Adam at `1e-4`, validation every 10 epochs, and best-validation-AJI checkpoint selection. See [Training protocol](docs/TRAINING.md), [Inference guide](docs/INFERENCE.md), and [Data format](docs/DATA_FORMAT.md).

## Results and visualizations

ChromaFlow obtains the highest AJI among weakly supervised methods in all four in-domain settings and seven of eight cross-dataset directions. The complete paper tables are available as [structured results](docs/RESULTS.md); visual comparisons, mechanism analyses, robustness, and failure cases are collected in [Visualizations](docs/VISUALIZATIONS.md).

![In-domain qualitative comparison](assets/paper/figure3_indomain.png)

## Documentation

- [Method and workflow](docs/METHOD.md)
- [Code structure](docs/CODE_STRUCTURE.md)
- [Datasets, sources, and splits](docs/DATASETS.md)
- [Training protocol](docs/TRAINING.md)
- [Inference and output format](docs/INFERENCE.md)
- [Paper result tables](docs/RESULTS.md)
- [Visualizations](docs/VISUALIZATIONS.md)
- [Model zoo and release status](docs/MODEL_ZOO.md)

## Reproducibility checks

```bash
python tools/validate_release.py
python tools/sync_results.py /path/to/final/manuscript.tex --check
```

The first command checks bundled manifest provenance, result schemas, assets and links, and reports required external manifests. To validate all final data splits, pass `--data-root DATA_ROOT --require-all-splits`. The second confirms that the committed CSV tables exactly match the final LaTeX manuscript.

## License and data notice

Repository code is provided for anonymous academic review. Dataset images retain their original licenses and terms; consult the source links in [Datasets](docs/DATASETS.md) before redistribution or downstream use.
