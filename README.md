# ChromaFlow

Anonymous review release for point-supervised nuclei instance segmentation.

ChromaFlow trains an image-only nuclei segmentation network from point annotations and stain-aware supervision. The public package contains model definitions, training losses, preprocessing utilities, full-image inference, decoding variants, and evaluation code. Datasets, checkpoints, logs, and external baseline repositories are intentionally excluded.

## Repository Layout

```text
chromaflow/          Core Python package
configs/             Reference training and inference settings
scripts/             Reproducible command-line entry points
docs/                Data format and implementation details
```

## Installation

```bash
conda create -n chromaflow python=3.10 -y
conda activate chromaflow
pip install -r requirements.txt
pip install -e .
```

The default ResNet-34 ChromaFlow path requires PyTorch, torchvision, NumPy, SciPy, scikit-image, Pillow, PyYAML, OpenCV, and Cellpose. `timm` is only needed for optional ViT/UNI/DINO encoder variants.

## Data

The training loader expects prepared patches under:

```text
data/official_clean/
  bonus_prepared/{MO,CPM}/images/{train,val,test}/*.png
  bonus_prepared/{MO,CPM}/ratio2.0/labels_clu/{train,val,test}/*_label_clu.png
  ours_prepared/{MO,CPM}/ratio2.0/pob_targets/{train,val,test}/*.npz
  core/monuseg/...
  core/cpm17/...
```

See `docs/DATA_FORMAT.md` for the exact fields.

## Training

Example MoNuSeg training:

```bash
bash scripts/train_mo.sh /path/to/project_or_data_root 0
```

Example CPM-17 training:

```bash
bash scripts/train_cpm.sh /path/to/project_or_data_root 0
```

The released commands train for 150 epochs, evaluate every 10 epochs, and select the final model using the best validation AJI checkpoint (`best_val_aji.pth`). Early stopping is disabled.

## Inference And Evaluation

Evaluate a checkpoint with the default full-image evaluator:

```bash
bash scripts/evaluate.sh /path/to/project_or_data_root MO test /path/to/checkpoint.pth
```

Run the seed-anchored S4 decoding variant with flip TTA:

```bash
bash scripts/decode_variants.sh /path/to/project_or_data_root MO test /path/to/checkpoint.pth
```

Outputs are written under `runs/` by default.

## Preprocessing Utility

To generate stain-aware Voronoi cluster labels from prepared point targets:

```bash
bash scripts/prepare_sav_labels.sh /path/to/project_or_data_root MO
```

This utility is optional for experiments that use the default `labels_clu` labels.
