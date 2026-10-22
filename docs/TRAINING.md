# Training protocol

| Item | Canonical setting |
|---|---|
| Network | ImageNet-pretrained ResNet-34 U-Net, three 1x1 heads |
| Prepared patch / random crop | 250x250 / 224x224 |
| Epochs / batch size | 150 / 16 |
| Optimizer | Adam, betas (0.9, 0.999), epsilon 1e-8 |
| Learning rate / weight decay | fixed 1e-4 / 3e-4 |
| Validation / checkpoint | every 10 epochs / best source-validation AJI |
| Repetition seeds | 0, 1, 2 |
| Decoder | flow scale 70, 30 steps, area 15, NMS 0.4/8, S4_drop, expansion 1 |

The objective is

`L = L_fg + 2 L_flow + 2 L_point + 0.5 L_div + 0.75 L_center + L_H + 0.5 L_E`.

`L_fg` has unit weight. The CLI retains `--lambda-fg` for controlled ablations.
The code names `lambda_center`, `lambda_stain_soft`, and `lambda_eosin_neg`
correspond to manuscript lambda_ctr, lambda_H, and lambda_E.
DSF uses kappa=0.1 and max-Gaussian source/centerness scales of 3 pixels.
CAR thresholds are 0.5, with foreground target 0.7 and background target 0.

All datasets use flips, right-angle rotations and scale jitter [0.8,1.25].
HED augmentation is off for the main method; `--stain-aug` enables the separate
HED augmentation control. The range and probability of scale jitter are explicit
implementation settings retained from the existing code (probability 0.5).
No early stopping or learning-rate schedule is used in the canonical command.

## Prepare and train

First provide final, patient-disjoint manifests as in [Datasets](DATASETS.md),
then prepare targets as in [Data format](DATA_FORMAT.md).

```bash
bash scripts/prepare_labels.sh DATA_ROOT MO 1.0 0
bash scripts/train_canonical.sh DATA_ROOT MO 0 1.0 0
# Other datasets use the same settings:
bash scripts/train_canonical.sh DATA_ROOT CPM 0 1.0 0
bash scripts/train_canonical.sh DATA_ROOT CONIC 0 1.0 0
bash scripts/train_canonical.sh DATA_ROOT NUINSSEG 0 1.0 0
```

The YAML is a reference specification; the shell wrapper explicitly passes the
training arguments. Validation uses the same seed-refined decoder and flip TTA
as final evaluation. Test membership is read only for final evaluation.

## Fixed 5% point experiment

```bash
bash scripts/prepare_labels.sh SPARSE_DATA_ROOT CPM 0.05 0
for seed in 0 1 2; do
  bash scripts/train_canonical.sh SPARSE_DATA_ROOT CPM "$seed" 0.05 0
done
```

For each original training image, uniformly select `round(0.05*N)` points
without replacement (round-half-up, minimum one if N>0). Selection occurs once
before patch extraction and remains fixed across epochs and the shown training
seeds. Change the separate point-selection seed only when intentionally preparing
another point subset. Rebuild coarse labels, Voronoi flow and Gaussian targets
using retained points; there is no mask-based exclusion around discarded nuclei.
Selected coordinates are saved for auditing in `preparation.json`.
