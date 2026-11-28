#!/usr/bin/env bash
# Window-attention experiments (proposal stretch goal V4), separate from the FFT runs.
#
#   scripts/run_window_experiments.sh
#       lowpass_win: the original model with spatial window attention instead of ESRT's strip split
#       (--attn window, util/transformer.WindowAttention). Needs no lambda or DWT choice: run it any time.
#   LAMBDA=0.05 DWT_CONFIG=dwt_c3_gr1_skipnorm scripts/run_window_experiments.sh
#       also V4 = <DWT_CONFIG> + FFT loss (LAMBDA) + window attention (lowpass_win is skipped if already done).
#
# Same training settings as every other run (x2, patch 96, 1000 epochs, seed 1, --fast, bf16), one config per
# GPU in parallel (GPUs auto-detected; e.g. GPUS=1 to use only GPU 1), then tests on Set5 / Set14 / BSDS300 /
# Manga109, packaging into final_results/<config>/ and the summary table. ~7 h per config.
# Own run folders, experiment/run_window_gpu<N>/ (RUN=run_window): it never shares files with
# run_fft_experiments.sh, so both can run at the same time on different GPUs.
# Re-running resumes (finished configs and tests are skipped); DRY_RUN=1 previews every command.
# Env: LAMBDA, DWT_CONFIG (both or neither), RUN, and everything in scripts/lib_experiments.sh
#      (ROOT, BENCH, PY, EPOCHS, GPUS, TEST_FAST, DRY_RUN). Extra arguments are passed to experiments.py.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN=${RUN:-run_window}
# shellcheck source=lib_experiments.sh
. scripts/lib_experiments.sh

LAMBDA=${LAMBDA:-}; DWT_CONFIG=${DWT_CONFIG:-}
LOSS_ARGS=()
if [ -n "$LAMBDA$DWT_CONFIG" ]; then
  [ -n "$LAMBDA" ] && [ -n "$DWT_CONFIG" ] || { echo "set both LAMBDA and DWT_CONFIG for V4 (or neither for lowpass_win only)" >&2; exit 1; }
  case " $DWT_CONFIGS " in *" $DWT_CONFIG "*) ;; *) echo "DWT_CONFIG must be one of: $DWT_CONFIGS" >&2; exit 1 ;; esac
  NAMES=("${DWT_CONFIG}_fl$(fmt_lambda "$LAMBDA")_win" "lowpass_win")
  LOSS_ARGS=(--freq_lambdas "$LAMBDA")
else
  echo "LAMBDA / DWT_CONFIG not set: training only lowpass_win (window attention alone)"
  NAMES=("lowpass_win")
fi

train_parallel --window ${LOSS_ARGS[@]+"${LOSS_ARGS[@]}"} "$@"
test_and_package
summarize
