#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:?data root required}"
DATASET="${2:?dataset required}"
RATIO="${3:-1.0}"
SEED="${4:-0}"
python -m chromaflow.prepare --root "$ROOT" --dataset "$DATASET" --point-ratio "$RATIO" --point-subset-seed "$SEED"
