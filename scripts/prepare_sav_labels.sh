#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
DATASET="${2:-MO}"

python -m chromaflow.prep_sav_clu \
  --root "$ROOT" \
  --dataset "$DATASET" \
  --subsets train val
