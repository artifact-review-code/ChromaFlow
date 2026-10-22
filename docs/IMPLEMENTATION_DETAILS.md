# Implementation details

The canonical protocol is documented in [Training](TRAINING.md),
[Data format](DATA_FORMAT.md), [Datasets](DATASETS.md) and [Method](METHOD.md).
The code was aligned to the 2026-10-21 manuscript. See
[Alignment record](MANUSCRIPT_ALIGNMENT.md) for changes and data requirements.

RGB is scaled to [0,1] and ImageNet-normalized for the network. Stain channels
are extracted from RGB with the fixed Ruifrok-Johnston HED matrix (`rgb2hed`),
normalized with 1st/99th percentiles and epsilon=1e-6 over all prepared-image
pixels, then min-max normalized over the current augmented crop inside CAR.
Degenerate percentile ranges yield zero channels. Spatial derivatives use
centered interior differences and one-sided boundary differences.

The point-derived Gaussian source and centerness target are prepared before
cropping, retaining the contribution of points just outside a crop. The center
loss is `mean((1+2*target)*(sigmoid(logits)-target)**2)`. Point loss is zero for
an empty point set. Flow is dx,dy in training, swapped to dy,dx for Cellpose.
Targets use the fixed displacement denominator 250 even on 224x224 crops.
Uniform scale augmentation rescales displacement magnitude with spatial scale.

The public result CSVs are manuscript transcriptions, not outputs from running
this edited code. The `tools/sync_results.py ... --check` command checks that
transcription. New runs write their own evaluation results separately.
