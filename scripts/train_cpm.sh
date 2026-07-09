#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
SEED="${2:-0}"

python -m chromaflow.train \
  --root "$ROOT" \
  --dataset CPM \
  --method v3 \
  --epochs 150 \
  --batch-size 16 \
  --lr 1e-4 \
  --weight-decay 3e-4 \
  --crop-size 224 \
  --encoder resnet34 \
  --flow-param potential \
  --lambda-fg 1.0 \
  --lambda-flow 2.0 \
  --lambda-point 0.5 \
  --lambda-div 0.75 \
  --screen-kappa 0.1 \
  --lambda-center 1.0 \
  --lambda-stain-soft 0.5 \
  --lambda-eosin-neg 0.5 \
  --lambda-spp 0.5 \
  --lambda-gap 0.5 \
  --lambda-msst 0.5 \
  --lambda-niche 0.3 \
  --lambda-idg 0.3 \
  --lambda-boundary 0.5 \
  --lambda-pace 0.05 \
  --lambda-dhs 1.0 \
  --lambda-hgfa 1.0 \
  --lr-schedule constant \
  --ckpt-split val \
  --eval-every 10 \
  --seed "$SEED" \
  --stain-aug
