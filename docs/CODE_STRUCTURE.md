# Code structure

## Reviewer-oriented tree

```text
ChromaFlow/
├── chromaflow/
│   ├── model.py              # canonical ResNet-34 U-Net and potential head
│   ├── losses.py             # DSF, CAR, and base training objective
│   ├── dataset.py            # prepared patches, point targets, augmentation
│   ├── train.py              # optimization and validation selection
│   ├── eval.py               # tiled inference, decoding, and metrics
│   ├── decode_variants.py    # S4 decoding and flip TTA
│   ├── prepare.py            # original-image point sampling and K-means/Voronoi targets
│   ├── protocol.py           # patient/WSI split validation
│   ├── prep_sav_clu.py       # legacy stain-geodesic experiment
│   ├── tta.py                # test-time transforms
│   └── *_loss.py, uni_*.py   # optional research variants
├── configs/                  # canonical training and inference settings
├── scripts/                  # public entry points; train_canonical_*.sh is the paper path
├── docs/                     # reviewer documentation
├── splits/                   # verified MoNuSeg membership; other final manifests supplied externally
├── results/                  # machine-readable paper tables
├── assets/
│   └── paper/                # current paper figures
└── tools/                    # release validation and table synchronization
```

## Classification

- **Canonical:** files explicitly listed in the paper-to-code map. These are sufficient to understand ChromaFlow.
- **Optional:** alternative encoders, auxiliary losses, and decoding investigations retained for research context. They are not required by the reported canonical model.
- **Legacy compatibility:** older command arguments and decoding modes that reproduce historical experiments. Canonical scripts explicitly select the current settings.

The release does not reorganize optional files because doing so would break historical imports. Reviewers can ignore them unless studying ablations.
