#!/usr/bin/env bash
# Every experiment that runs on a PC, in order: E1-E3, the decoder study, the embedded study (phases 0-1).
set -euo pipefail
D="$(dirname "$0")"
bash "$D/run_thesis.sh"
bash "$D/run_decoders.sh"
bash "$D/run_embedded.sh"
