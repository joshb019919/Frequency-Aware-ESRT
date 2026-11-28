#!/usr/bin/env bash
# FFT loss (V1 + lambda study): lowpass trained with each lambda in SWEEP_LAMBDAS.
# GPU, DWT variant and lambdas are set at the top of scripts/lib_experiments.sh. Finished configs are skipped and
# configs still training are not started twice. Results: final_results/<config>/, then the results table.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=run_fft
. scripts/lib_experiments.sh
NAMES=(); for l in ${SWEEP_LAMBDAS//,/ }; do NAMES+=("lowpass_fl$(fmt_lambda "$l")"); done
run_configs "${NAMES[@]}"
