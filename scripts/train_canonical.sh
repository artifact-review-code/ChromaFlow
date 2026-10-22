#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:?data root required}"
DATASET="${2:?MO, CPM, CONIC or NUINSSEG required}"
SEED="${3:-0}"
RATIO="${4:-1.0}"
POINT_SEED="${5:-0}"
python -m chromaflow.train \
  --root "$ROOT" --dataset "$DATASET" --method v3 \
  --epochs 150 --batch-size 16 --lr 1e-4 --weight-decay 3e-4 \
  --crop-size 224 --encoder resnet34 --flow-param potential \
  --lambda-fg 1.0 --lambda-flow 2.0 --lambda-point 2.0 \
  --lambda-div 0.5 --screen-kappa 0.1 --lambda-center 0.75 \
  --lambda-stain-soft 1.0 --lambda-eosin-neg 0.5 \
  --lr-schedule constant --ckpt-split val --eval-every 10 \
  --point-ratio "$RATIO" --point-subset-seed "$POINT_SEED" --seed "$SEED" \
  --scale-jitter --scale-jitter-min 0.8 --scale-jitter-max 1.25
