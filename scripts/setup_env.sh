#!/usr/bin/env bash
# Create .venv with the pinned dependencies (Python 3.10 or newer).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
"${PYTHON:-python3}" -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/pip" install --upgrade pip
"$ROOT/.venv/bin/pip" install -r "$ROOT/requirements.txt"
