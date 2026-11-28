#!/usr/bin/env bash
# DWT + window attention: DEFAULT_DWT_CONFIG with spatial window attention.
# GPU, DWT variant and lambdas are set at the top of scripts/lib_experiments.sh. Finished configs are skipped and
# configs still training are not started twice. Results: final_results/<config>/, then the results table.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=run_dwt_window
. scripts/lib_experiments.sh
run_configs "${DEFAULT_DWT_CONFIG}_win"
