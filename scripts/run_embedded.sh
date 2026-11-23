#!/usr/bin/env bash
# Embedded study, the PC part: phase 0 (causal filter, retraining, ONNX precisions, fixed-lag
# Viterbi, streaming LLR) and the phase-1 bundle, replay files and PC reference.
# Needs a finished scripts/run_thesis.sh; outputs go to outputs/embedded/.
source "$(dirname "$0")/env.sh"
KEY="$("$PY" -c 'from embedded import emb as E; print(E.CAUSAL_KEY)')"
OUT="$("$PY" -c 'from embedded import emb as E; print(E.CAUSAL_DIR)')"
run thesis.train_config --experiment E2 --key "$KEY" --folds 0 1 2 --seeds 0 1 2 --out "$OUT"
run thesis.train_config --experiment E2 --key "$KEY" --folds all --seeds 0 --out "$OUT"
run embedded.phase0_predict --variant zerophase
run embedded.phase0_predict --variant causal
run embedded.phase0_quant
run embedded.phase0_quant --batch 1 --skip-build --precisions int8_dynamic
run embedded.phase0_eval --workers "${WORKERS:-6}"
run embedded.phase0_report
run embedded.phase1_export
run embedded.phase1_reference
