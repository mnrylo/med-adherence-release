#!/usr/bin/env bash
# Shared environment of the run scripts: repository root, interpreter and PYTHONPATH.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$ROOT"
run() { echo "[$(date '+%F %T')] $*"; "$PY" -m "$@"; }
