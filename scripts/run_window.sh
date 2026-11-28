#!/usr/bin/env bash
# Window attention alone: the original model with spatial window attention (--attn window).
# GPU, DWT variant and lambdas are set at the top of scripts/lib_experiments.sh. Finished configs are skipped and
# configs still training are not started twice. Results: final_results/<config>/, then the results table.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=run_window
. scripts/lib_experiments.sh
run_configs lowpass_win
