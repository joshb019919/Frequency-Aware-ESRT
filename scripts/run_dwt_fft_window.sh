#!/usr/bin/env bash
# DWT + FFT loss + window attention (V4): DEFAULT_DWT_CONFIG, best lambda of scripts/run_fft.sh, window attention.
# GPU, DWT variant and lambdas are set at the top of scripts/lib_experiments.sh. Finished configs are skipped and
# configs still training are not started twice. Results: final_results/<config>/, then the results table.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=run_dwt_fft_window
. scripts/lib_experiments.sh
LAMBDA=$(need_lambda); echo "lambda: $LAMBDA (best of the FFT sweep)"
run_configs "${DEFAULT_DWT_CONFIG}_fl$(fmt_lambda "$LAMBDA")_win"
