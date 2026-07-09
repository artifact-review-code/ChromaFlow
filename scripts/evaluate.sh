#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
DATASET="${2:-MO}"
SPLIT="${3:-test}"
CHECKPOINT="${4:?checkpoint path is required}"
OUT_DIR="${5:-runs/eval_${DATASET}_${SPLIT}}"

python -m chromaflow.eval \
  --root "$ROOT" \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --checkpoint "$CHECKPOINT" \
  --out-dir "$OUT_DIR" \
  --tile-size 224 \
  --tile-overlap 64 \
  --tile-batch-size 16 \
  --cellprob-mode logits \
  --cellprob-threshold 0.0 \
  --flow-threshold 0.0 \
  --min-size 15
