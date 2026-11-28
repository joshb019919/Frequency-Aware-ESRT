#!/usr/bin/env bash
# Baselines and frequency-loss (FFT) experiments for the report, run on the GPU server after
# scripts/download_data.sh. Window attention (V4) has its own script: scripts/run_window_experiments.sh.
#
#   baselines  (minutes, no training)
#              - B1/B2: bilinear + bicubic on Set5, Set14, BSDS300, Manga109   -> final_results/{bilinear,bicubic}/
#              - Set14 tests of the four committed models (lowpass + 3 DWT)      -> final_results/<config>/set14/
#              - data check: re-tests final_results/lowpass on this machine's Set5/BSDS300/Manga109 (fast and
#                no-fast) and compares per image with the committed results, so you know the new data and
#                test mode reproduce them (records which TEST_FAST setting to use)
#   sweep      V1 + lambda study: lowpass trained with each lambda in SWEEP_LAMBDAS (lib_experiments.sh)
#   dwt        V3: the three DWT configs trained with the sweep's best lambda (picked automatically; LAMBDA=
#              overrides it)
#              Each phase is 3 configs of ~7 h, trained one per GPU in parallel (2 GPUs: ~14 h, 3+: ~7 h).
#              Worker run folders: experiment/run_fft_gpu<N>/ (one experiments.py per GPU).
#
# Training uses the same settings as the DWT runs (x2, patch 96, 1000 epochs, seed 1, --fast, bf16): only the
# loss changes. final_results/lowpass is reused as the baseline/plot reference (copied into the run folder,
# never retrained). Each trained config is tested on Set5 (experiments.py) and on Set14/BSDS300/Manga109,
# then copied to final_results/<config>/. Re-running a phase resumes: finished configs/tests are skipped.
# Benchmark sets that are not prepared (e.g. Manga109 without its licence/mapping file) are skipped.
#
# Usage (repo root, GPU machine):
#   scripts/run_fft_experiments.sh baselines
#   scripts/run_fft_experiments.sh sweep
#   scripts/run_fft_experiments.sh dwt
# GPU, lambdas and DWT variant are set once at the top of scripts/lib_experiments.sh (default: GPU 1).
# Env: RUN (run id, default run_fft) and everything listed in scripts/lib_experiments.sh (ROOT, BENCH, PY,
#      EPOCHS, GPUS, TEST_FAST, DRY_RUN). Extra arguments after the phase are passed to experiments.py.
set -euo pipefail
cd "$(dirname "$0")/.."

PHASE=${1:?usage: $0 baselines|sweep|dwt [experiments.py args...]}; shift
RUN=${RUN:-run_fft}
# shellcheck source=lib_experiments.sh
. scripts/lib_experiments.sh

# ------------------------------------------------------------------------------------------- baselines
if [ "$PHASE" = baselines ]; then
  echo "== B1/B2: bilinear + bicubic"
  if [ "$DRY_RUN" = 1 ]; then echo "+ $PY scripts/interp_baselines.py --bench_root $BENCH"
  else "$PY" scripts/interp_baselines.py --bench_root "$BENCH"; fi

  echo "== data check: re-test final_results/lowpass on this machine and compare per image"
  best=""; for mode in --fast --no-fast; do
    ok=1; tested=0
    for set in set5 bsds300 manga109; do
      sub=$([ $set = set5 ] && echo "" || echo "$set")
      committed=final_results/lowpass/$sub/test_x${SCALE}_summary.json
      [ -f "$committed" ] || continue
      out=experiment/$RUN/verify_lowpass${mode//-/_}/$set
      test_one final_results/lowpass/epoch_$EPOCHS.pth lowpass $set "$out" "$mode"
      [ "$DRY_RUN" = 1 ] || [ -f "$out/test_x${SCALE}_summary.json" ] || continue
      [ "$DRY_RUN" = 1 ] && continue
      tested=1
      "$PY" scripts/compare_summaries.py "$committed" "$out/test_x${SCALE}_summary.json" --label "lowpass $set $mode" || ok=0
    done
    if [ "$DRY_RUN" != 1 ] && [ $tested = 1 ] && [ $ok = 1 ] && [ -z "$best" ]; then best=$mode; fi
  done
  if [ "$DRY_RUN" != 1 ]; then
    mkdir -p "$(dirname "$MODE_FILE")"
    if [ -n "$best" ]; then
      echo "$best" > "$MODE_FILE"
      echo "data check passed with $best: new results are directly comparable; later phases test with $best"
    else
      echo "WARNING: this machine's data/test mode does not reproduce the committed per-image numbers."
      echo "         Compare the MISMATCH lines above. For a fair table, re-test every model here (lowpass + DWT):"
      echo "         delete final_results/<config>/{test_x2*,bsds300,manga109} and re-run with a consistent mode."
    fi
  fi

  echo "== Set14 tests of the committed models (mode: $(cat "$MODE_FILE" 2>/dev/null || echo "$TEST_FAST"))"
  [ -f "$MODE_FILE" ] && TEST_FAST=$(cat "$MODE_FILE")
  for cfg in lowpass $DWT_CONFIGS; do
    echo "[$cfg] set14"
    ref=""; [ "$cfg" = lowpass ] || ref=final_results/lowpass/set14
    test_one "final_results/$cfg/epoch_$EPOCHS.pth" "$cfg" set14 "final_results/$cfg/set14" "$TEST_FAST" "$ref"
  done
  summarize
  exit 0
fi

# ------------------------------------------------------------------------------------------- sweep / dwt
NAMES=()
case "$PHASE" in
  sweep) LAMBDAS=${LAMBDAS:-$SWEEP_LAMBDAS}
         for l in ${LAMBDAS//,/ }; do NAMES+=("lowpass_fl$(fmt_lambda "$l")"); done ;;
  dwt)   LAMBDAS=${LAMBDA:-$(best_lambda)}
         [ -n "$LAMBDAS" ] || { echo "the sweep is not finished yet (no final_results/lowpass_fl* for every lambda); run it first, or set LAMBDA" >&2; exit 1; }
         echo "lambda: $LAMBDAS${LAMBDA:+ (set by LAMBDA)}${LAMBDA:- (best of the sweep)}"
         for b in $DWT_CONFIGS; do NAMES+=("${b}_fl$(fmt_lambda "$LAMBDAS")"); done ;;
  *) echo "phase must be baselines, sweep or dwt (window attention: scripts/run_window_experiments.sh)" >&2; exit 1 ;;
esac

train_parallel --freq_lambdas "$LAMBDAS" "$@"
test_and_package
summarize
