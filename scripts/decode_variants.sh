#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
DATASET="${2:-MO}"
SPLIT="${3:-test}"
CHECKPOINT="${4:?checkpoint path is required}"
OUT_DIR="${5:-runs/decode_${DATASET}_${SPLIT}}"

python -m chromaflow.decode_variants \
  --root "$ROOT" \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --checkpoint "$CHECKPOINT" \
  --out-dir "$OUT_DIR" \
  --variants S4_drop \
  --tta hflip,vflip \
  --mask-dilation 1 \
  --tile-size 224 \
  --tile-overlap 64 \
  --tile-batch-size 16
