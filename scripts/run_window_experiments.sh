#!/usr/bin/env bash
# Window-attention experiments (proposal stretch goal V4 + the full window ablation), separate from the FFT runs.
#
# Trains every window-attention combination that can run yet (window attention: --attn window,
# util/transformer.WindowAttention), with no settings needed:
#
#   lowpass_win                  window
#   <dwt>_win                    DWT + window                  <dwt> = DEFAULT_DWT_CONFIG in lib_experiments.sh
#   lowpass_fl<l>_win            FFT + window                  only once the FFT sweep has finished:
#   <dwt>_fl<l>_win              DWT + FFT + window (V4)       <l> = its best lambda, picked automatically
#
#   scripts/run_window_experiments.sh                 # run it now, and again after the sweep for the FFT ones
#
# Finished configs are skipped and configs still training (in any run folder) are not started twice, so you
# can simply run it again at any time. Overrides: LAMBDA=, DWT_CONFIG=, CONFIGS="name ..." (only those).
#
# Same training settings as every other run (x2, patch 96, 1000 epochs, seed 1, --fast, bf16), one config per
# GPU (GPU 1 by default: DEFAULT_GPUS in lib_experiments.sh), then tests on Set5 / Set14 / BSDS300 /
# Manga109, packaging into final_results/<config>/ and the summary table. ~7 h per config.
# Own run folders, experiment/run_window_gpu<N>/ (RUN=run_window): it never shares files with
# run_fft_experiments.sh, so both can run at the same time on different GPUs.
# DRY_RUN=1 previews every command and the GPU split.
# Env: LAMBDA, DWT_CONFIG, CONFIGS (space-separated subset), RUN, and everything in scripts/lib_experiments.sh
#      (ROOT, BENCH, PY, EPOCHS, GPUS, TEST_FAST, DRY_RUN). Extra arguments are passed to experiments.py.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN=${RUN:-run_window}
# shellcheck source=lib_experiments.sh
. scripts/lib_experiments.sh

DWT_CONFIG=${DWT_CONFIG-$DEFAULT_DWT_CONFIG}   # DWT_CONFIG= (empty) leaves the DWT configs out
LAMBDA=${LAMBDA:-$(best_lambda)}                # empty until the FFT sweep is finished
[ -n "$LAMBDA" ] || echo "FFT sweep not finished yet: the FFT + window configs are left out (run this again later)"
if [ -n "$DWT_CONFIG" ]; then
  case " $DWT_CONFIGS " in *" $DWT_CONFIG "*) ;; *) echo "DWT_CONFIG must be one of: $DWT_CONFIGS" >&2; exit 1 ;; esac
fi
LOSS_ARGS=()
NAMES=("lowpass_win")                                                       # window
if [ -n "$DWT_CONFIG" ]; then NAMES+=("${DWT_CONFIG}_win"); fi              # DWT + window
if [ -n "$LAMBDA" ]; then
  l=$(fmt_lambda "$LAMBDA")
  NAMES+=("lowpass_fl${l}_win")                                              # FFT + window
  if [ -n "$DWT_CONFIG" ]; then NAMES+=("${DWT_CONFIG}_fl${l}_win"); fi     # DWT + FFT + window (V4)
  LOSS_ARGS=(--freq_lambdas "$LAMBDA")
fi
if [ -n "${CONFIGS:-}" ]; then
  for c in $CONFIGS; do
    case " ${NAMES[*]} " in *" $c "*) ;; *) echo "CONFIGS: $c is not one of: ${NAMES[*]} (set LAMBDA / DWT_CONFIG accordingly)" >&2; exit 1 ;; esac
  done
  read -ra NAMES <<< "$CONFIGS"
fi
echo "window configs: ${NAMES[*]}"

train_parallel --window ${LOSS_ARGS[@]+"${LOSS_ARGS[@]}"} "$@"
test_and_package
summarize
