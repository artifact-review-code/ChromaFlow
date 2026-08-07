# Inference guide

## Release status

The inference implementation and pre-generated reviewer examples are included. **No ChromaFlow checkpoint is distributed during anonymous review**, so a user-provided compatible checkpoint is required for a new image. Planned weights are listed in the [Model zoo](MODEL_ZOO.md).

## Accepted input and produced output

- Input: an RGB `.png`, `.jpg`, or dataset image readable by Pillow.
- Primary output: a 2-D integer instance-ID map (`0` is background), normally saved as `.npy`.
- Reviewer-friendly output: deterministic colored-instance PNG and a red boundary overlay.
- Optional diagnostics: foreground probability/logits, centerness, and two-channel flow.

The included examples follow this contract; each example folder contains `input.png`, `instances.npy`, `prediction.png`, and `overlay.png`.

## Paper evaluation command

For a prepared dataset, this is the preferred interface because it applies 224-pixel tiling, 64-pixel overlap, cosine blending, identity/horizontal/vertical-flip averaging, S4 decoding, and metric export consistently:

```bash
bash scripts/decode_variants.sh \
  /path/to/project_or_data_root MO test \
  <PATH_TO_CHECKPOINT> runs/decode_MO_test
```

The lower-level evaluator is useful when only the canonical Cellpose-style decoder is needed:

```bash
python -m chromaflow.eval \
  --root /path/to/project_or_data_root \
  --dataset MO --split test \
  --checkpoint <PATH_TO_CHECKPOINT> \
  --out-dir runs/MO_test \
  --tile-size 224 --tile-overlap 64 --tile-batch-size 16 \
  --tta hflip,vflip \
  --cellprob-mode logits --cellprob-threshold 0.0 \
  --flow-threshold 0.0 --min-size 15
```

## Minimal Python interface

The following shows the model construction, checkpoint loading, tiled prediction, and instance-map saving used by the command-line evaluator. Flip TTA is applied by the command above; this minimal snippet intentionally demonstrates a single-view prediction primitive.

```python
from pathlib import Path

import numpy as np
import torch

from chromaflow.eval import load_checkpoint, predict_instance_map
from chromaflow.model import NISFlowNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NISFlowNet(
    pretrained=False,
    backbone="resnet34",
    flow_param="potential",
).to(device)

checkpoint = load_checkpoint(Path("<PATH_TO_CHECKPOINT>"), device)
model.load_state_dict(checkpoint["state"], strict=True)
model.eval()

config = {
    "tile_size": 224,
    "tile_overlap": 64,
    "tile_batch_size": 16,
    "stitch_mode": "blend",
    "cellprob_mode": "logits",
    "flow_scale": 70.0,
    "niter": 30,
    "cellprob_threshold": 0.0,
    "flow_threshold": 0.0,
    "min_size": 15,
    "use_centerness_seeds": True,
    "seed_threshold": 0.4,
    "seed_min_distance": 8,
}

instances = predict_instance_map(
    model, Path("path/to/rgb_image.png"), config, device
)
np.save("instances.npy", instances.astype(np.int32))
```

Do not substitute a random checkpoint: it produces meaningless masks and is not representative of the paper. The files under `assets/demo/` are explicitly pre-generated predictions from the paper experiments, with sample IDs and crop coordinates in the [demo manifest](../assets/demo/manifest.json).
