# ChromaFlow method

ChromaFlow combines **Divergence-Sink Flow (DSF)** and
**Chromophore-Anchored Recovery (CAR)** with fixed point-derived coarse labels.
A ResNet-34 U-Net predicts foreground logits, centerness logits and scalar
potential phi. The flow is `v = -grad(phi)`.

## Fixed supervision

Training uses RGB, retained point positions and their derived targets.
The cluster map labels background=0, foreground=1 and ignore=2. Preparation
uses RGB/point-distance K-means within Voronoi cells and disk(2) refinement;
see [exact implementation conventions](DATA_FORMAT.md). The map is fixed.

## Divergence-Sink Flow

The direction target at pixel r is `(p_nearest-r)/250`, with both magnitude and
direction preserved. `L_flow` averages squared errors over two components and
weak-foreground pixels. It is not a Gaussian-weighted direction target.
The independent Gaussian source is `rho(r)=max_i exp(-||r-p_i||^2/(2*3^2))`.
The screened-Poisson residual is `div(v)+rho+kappa^2*phi`, kappa=0.1.
Since v=-grad(phi), zero residual gives `(Delta-kappa^2)*phi=rho`.
`L_div` averages its squared residual over weak foreground.

## Chromophore-Anchored Recovery

Fixed HED deconvolution extracts hematoxylin H and eosin E. First/99th-percentile
normalization uses all prepared-image pixels and epsilon=1e-6. After augmentation
and cropping, per-crop min-max normalization uses the same epsilon.
Scores are `q_plus=H` and `q_minus=min(E,1-H)`.
Within ignore pixels only, q_plus>0.5 supplies BCE target 0.7; q_minus>0.5
supplies BCE target 0. Empty anchor sets contribute zero.

## Objective and decoder

See [Training](TRAINING.md) for all seven loss weights. The foreground loss
uses coarse labels outside ignore; the point loss uses retained points only.
Centerness fits the max-Gaussian target with center-weighted MSE (gamma=2).

Inference uses foreground logits >=0, flow scale 70 and 30 Cellpose Euler steps
with interpolation. Basins smaller than 15 pixels are discarded. Centerness
NMS uses threshold 0.4 and minimum distance 8. Zero-seed basins are dropped,
one-seed basins retained, and multiple-seed basins partitioned by nearest seed.
Finally, `expand_labels(distance=1)` expands each instance with nearest-label
conflict resolution. [Inference](INFERENCE.md) records tiling and commands.

## Code map

| Concept | Implementation |
|---|---|
| Fixed original-image sampling and weak target preparation | `chromaflow/prepare.py` |
| Patient/WSI membership validation | `chromaflow/protocol.py` |
| Joint cropping and augmentation | `chromaflow/dataset.py` |
| Network and negative potential gradient | `chromaflow/model.py` |
| DSF, CAR and training objective | `chromaflow/losses.py` |
| Optimization and source-validation selection | `chromaflow/train.py` |
| Full-image inference and evaluation | `chromaflow/eval.py` |
| Seed refinement and decoder variants | `chromaflow/decode_variants.py` |

Other architectures/losses remain explicit research variants. Canonical commands
select the manuscript's ResNet-34 potential-head configuration.
