#!/usr/bin/env bash
# Decoder study (stages D0-D3, D5 and analysis A4) on the saved predictions of E1/E2.
# Needs a finished scripts/run_thesis.sh; outputs go to outputs/decoders/.
source "$(dirname "$0")/env.sh"
W="${WORKERS:-8}"
run decoders.d0_reproduce --workers "$W"
run decoders.d0_report
run decoders.d1_run --workers "$W"
run decoders.d1b_run --workers "$W"
run decoders.d1_report
run decoders.d2_run --workers "$W"
run decoders.d2_report
run decoders.d3_run --workers "$W"
run decoders.d3_report
run decoders.a4_tuning_data --workers "$W"
run decoders.d5_report
