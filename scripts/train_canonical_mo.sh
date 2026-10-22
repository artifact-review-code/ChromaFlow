#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/train_canonical.sh" "${1:?data root required}" MO "${2:-0}" "${3:-1.0}" "${4:-0}"
