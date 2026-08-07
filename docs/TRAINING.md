# Training protocol

This page records the canonical configuration used for the paper. It intentionally exposes the steps and principal hyperparameters needed to understand and reproduce the training protocol without publishing every exploratory script from the research workspace.

## Canonical configuration

| Item | Setting |
|---|---|
| Network | ResNet-34 U-Net with foreground, centerness, and scalar-potential heads |
| Input crop | 224 x 224 RGB |
| Prepared samples | 1176 patches per source-domain run |
| Epochs / batch size | 150 / 16 |
| Optimizer | Adam, betas `(0.9, 0.99)`, epsilon `1e-8` |
| Learning rate | `1e-4`, constant |
| Weight decay | `3e-4` |
| Validation | every 10 epochs |
| Selection rule | highest validation AJI; test data are never used for selection |
| Repetitions | seeds 0, 1, and 2 |
| Inference | 224 tile, 64 overlap, cosine blend, identity/hflip/vflip TTA |

The loss is

`L = 1.0 L_fg + 2.0 L_flow + 0.5 L_point + 0.75 L_div + 1.0 L_center + 0.5 L_H + 0.5 L_E`.

DSF uses `kappa=0.1 px^-1` and Gaussian scales `sigma_s=sigma_c=3`. CAR uses `tau+=tau-=0.5` and a positive soft target `alpha=0.7`.

## Label preparation

Only one interior point per nucleus is used as manual supervision. The preprocessing stage combines RGB and nearest-point-distance features with three-class K-means, assigns candidate regions to points with Voronoi cells, and protects a two-pixel disk around every point. It writes a static foreground/background/ignore cluster map consumed by the dataset loader.

```bash
bash scripts/prepare_sav_labels.sh /path/to/project_or_data_root MO
bash scripts/prepare_sav_labels.sh /path/to/project_or_data_root CPM
```

The expected folders and NPZ fields are documented in [Data format](DATA_FORMAT.md). Inspect a few generated maps before starting a long run; point disks must coincide with nuclei and instance IDs must remain intact.

## Augmentation

All runs use horizontal and vertical flips and 90-degree rotations. HED augmentation perturbs stain concentrations during training. MoNuSeg additionally uses scale jitter in `[0.8, 1.25]` to reduce the magnification mismatch with CPM-17. Augmentation is applied only to training samples.

## Train one seed

```bash
# MoNuSeg: includes the paper's scale jitter
bash scripts/train_canonical_mo.sh /path/to/project_or_data_root 0

# CPM-17
bash scripts/train_canonical_cpm.sh /path/to/project_or_data_root 0
```

The shell wrappers expose the exact main arguments while keeping the Python entry point available for controlled changes:

```bash
python -m chromaflow.train --help
```

## Run the three paper seeds

```bash
for seed in 0 1 2; do
  bash scripts/train_canonical_mo.sh /path/to/project_or_data_root "$seed"
done

for seed in 0 1 2; do
  bash scripts/train_canonical_cpm.sh /path/to/project_or_data_root "$seed"
done
```

Each run validates every ten epochs and stores the checkpoint with the highest source-validation AJI. Do not choose a checkpoint or decoding threshold using the target test set.

## Evaluate the selected checkpoint

```bash
bash scripts/evaluate.sh \
  /path/to/project_or_data_root MO test \
  /path/to/best_val_aji.pth runs/eval_MO_test

# Paper decoding variant with flip TTA
bash scripts/decode_variants.sh \
  /path/to/project_or_data_root MO test \
  /path/to/best_val_aji.pth runs/decode_MO_test
```

The anonymous release does not contain checkpoints or private server paths. The detailed experimental orchestration and exploratory hyperparameter scripts will be added after acceptance; the optimizer, losses, data preparation, model selection, and evaluation code used by the canonical path are already present.
