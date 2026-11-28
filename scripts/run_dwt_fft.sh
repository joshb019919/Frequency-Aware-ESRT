#!/usr/bin/env bash
# DWT + FFT loss (V3): the three DWT variants trained with the best lambda of scripts/run_fft.sh.
# GPU, DWT variant and lambdas are set at the top of scripts/lib_experiments.sh. Finished configs are skipped and
# configs still training are not started twice. Results: final_results/<config>/, then the results table.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=run_dwt_fft
. scripts/lib_experiments.sh
LAMBDA=$(need_lambda); echo "lambda: $LAMBDA (best of the FFT sweep)"
NAMES=(); for b in $DWT_CONFIGS; do NAMES+=("${b}_fl$(fmt_lambda "$LAMBDA")"); done
run_configs "${NAMES[@]}"
