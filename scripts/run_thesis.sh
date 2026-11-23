#!/usr/bin/env bash
# Experiments E1-E3: staged sweep, short-window tracks, post-processing, model registry, reports.
# Every step resumes where it stopped; outputs go to outputs/thesis/.
source "$(dirname "$0")/env.sh"
run thesis.sweep "$@"
run thesis.extra_windows
run thesis.postproc --workers "${WORKERS:-6}"
run thesis.registry
run thesis.report
run thesis.results_chapter
