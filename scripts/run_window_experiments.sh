#!/usr/bin/env bash
# Window-attention experiments (proposal stretch goal V4 + the full window ablation), separate from the FFT runs.
#
# The configs are built from what you have decided so far; each adds window attention (--attn window,
# util/transformer.WindowAttention) on top of the other changes:
#
#   (nothing set)          lowpass_win                     window                      needs no other result
#   DWT_CONFIG=<dwt>       + <dwt>_win                     DWT + window
#   LAMBDA=<l>             + lowpass_fl<l>_win             FFT + window
#   both                   + <dwt>_fl<l>_win               DWT + FFT + window (V4)
#
#   scripts/run_window_experiments.sh
#   LAMBDA=0.05 DWT_CONFIG=dwt_c3_gr1_skipnorm scripts/run_window_experiments.sh   # all four window rows
#   CONFIGS="dwt_c3_gr1_skipnorm_fl0.05_win" LAMBDA=0.05 DWT_CONFIG=dwt_c3_gr1_skipnorm scripts/run_window_experiments.sh
#                                                                                   # only the named config(s)
# Finished configs are always skipped, so running again with more set only trains what is new.
#
# Same training settings as every other run (x2, patch 96, 1000 epochs, seed 1, --fast, bf16), one config per
# GPU in parallel (GPUs auto-detected; e.g. GPUS=1 to use only GPU 1), then tests on Set5 / Set14 / BSDS300 /
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

LAMBDA=${LAMBDA:-}; DWT_CONFIG=${DWT_CONFIG:-}
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
