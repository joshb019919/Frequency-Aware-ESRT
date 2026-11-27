#!/usr/bin/env bash
# Frequency-loss (FFT-L1) experiments, packaged like the DWT results in final_results/.
#
#   Phase "sweep" (V1 + lambda study): lowpass trained with lambda in $LAMBDAS (default 0.01,0.05,0.1).
#   Phase "dwt"   (V3):                the three DWT configs trained with one lambda ($LAMBDA).
#
# Same settings as the DWT runs (x2, patch 96, 1000 epochs, seed 1, --fast, bf16): only the loss changes.
# The finished lowpass run in final_results/lowpass is copied into the run folder (not retrained), so it
# is the dashed reference line in every curve/test plot and experiments.py skips it.
#
# Each config is trained + tested on Set5 by experiments.py, then tested on BSDS300 and Manga109, then
# copied to final_results/<config>/ (epoch_1000.pth, curves, history.json, logs, test files, bsds300/,
# manga109/). Re-running the same command resumes: finished configs are skipped.
#
# Usage (from the repo root, on the GPU machine):
#   scripts/run_fft_experiments.sh sweep
#   LAMBDA=0.05 scripts/run_fft_experiments.sh dwt          # after picking the best lambda from the sweep
# Env: ROOT (DIV2K dir, default /data/data/div2k), RUN (run id, default run_fft), PY (python),
#      EPOCHS (default 1000), TEST_FAST (--fast or --no-fast for the BSDS300/Manga109 tests, default --fast),
#      DRY_RUN=1 (print every command, train/test nothing, copy nothing).
# Extra arguments after the phase are passed to experiments.py.
set -euo pipefail
cd "$(dirname "$0")/.."

PHASE=${1:?usage: $0 sweep|dwt [experiments.py args...]}; shift
ROOT=${ROOT:-/data/data/div2k}
RUN=${RUN:-run_fft}
PY=${PY:-python}
EPOCHS=${EPOCHS:-1000}
TEST_FAST=${TEST_FAST:---fast}
SCALE=2
DRY_RUN=${DRY_RUN:-0}
run() { if [ "$DRY_RUN" = 1 ]; then echo "+ $*"; else "$@"; fi; }

flags_for() {  # model flags of a base config (must match how it was trained)
  case "$1" in
    lowpass)                echo "" ;;
    dwt_c1_gr1_norm)        echo "--freq dwt --conv 1 --global_res 1 --ll_norm 1" ;;
    dwt_c3_gr1_skipnorm)    echo "--freq dwt --conv 3 --global_res 1 --dwt_skip 1 --ll_norm 1" ;;
    dwt_cband_gr1_skipnorm) echo "--freq dwt --conv band --global_res 1 --dwt_skip 1 --ll_norm 1" ;;
    *) echo "unknown config $1" >&2; exit 1 ;;
  esac
}

case "$PHASE" in
  sweep) LAMBDAS=${LAMBDAS:-0.01,0.05,0.1}; BASES="lowpass" ;;
  dwt)   LAMBDAS=${LAMBDA:?set LAMBDA to the best value from the sweep, e.g. LAMBDA=0.05}
         BASES="dwt_c3_gr1_skipnorm dwt_c1_gr1_norm dwt_cband_gr1_skipnorm" ;;
  *) echo "phase must be sweep or dwt" >&2; exit 1 ;;
esac

# Config names experiments.py will create: <base>_fl<lambda> ({:g} formatting, e.g. 0.10 -> 0.1).
NAMES=()
for b in $BASES; do
  for l in ${LAMBDAS//,/ }; do NAMES+=("${b}_fl$($PY -c "print('{:g}'.format(float('$l')))")"); done
done
ONLY=$(IFS=,; echo "${NAMES[*]}")

# 1. Seed the run folder with the finished lowpass baseline (reference for plots; never retrained).
REF=experiment/$RUN/exp_lowpass
if [ ! -f "$REF/epoch_$EPOCHS.pth" ]; then
  [ -f final_results/lowpass/epoch_$EPOCHS.pth ] || { echo "final_results/lowpass/epoch_$EPOCHS.pth missing" >&2; exit 1; }
  run mkdir -p "$REF"
  run cp -R final_results/lowpass/. "$REF/"
  echo "seeded $REF from final_results/lowpass"
fi

# 2. Train + Set5 test every config (resumes: configs with a final checkpoint are skipped).
DRY=(); [ "$DRY_RUN" = 1 ] && DRY=(--dry_run)
$PY experiments.py --root "$ROOT" --run_id "$RUN" --epochs "$EPOCHS" --scale $SCALE \
    --freq_lambdas "$LAMBDAS" --only "$ONLY" "${DRY[@]}" "$@"

# 3. BSDS300 + Manga109 tests, then package into final_results/<config>/.
for name in "${NAMES[@]}"; do
  exp=experiment/$RUN/exp_$name
  ckpt=$exp/epoch_$EPOCHS.pth
  [ -f "$ckpt" ] || [ "$DRY_RUN" = 1 ] || { echo "[$name] no final checkpoint, skipping tests"; continue; }
  base=${name%_fl*}
  for set in bsds300 manga109; do
    [ -f "$exp/$set/test_x${SCALE}_summary.json" ] && continue
    echo "[$name] testing on $set"
    # shellcheck disable=SC2046
    cmd=($PY test.py --test_sets $set --checkpoint "$ckpt" --upscale_factor $SCALE $(flags_for "$base")
         --output_folder "results/$RUN/exp_$name/$set" --plot_path "$exp/$set/test_x${SCALE}.png"
         --ref_test_dir "$REF/$set" $TEST_FAST)
    if [ "$DRY_RUN" = 1 ]; then
      echo "+ ${cmd[*]}"
    else
      mkdir -p "$exp/$set"
      "${cmd[@]}" 2>&1 | tee "$exp/$set/test.log"   # pipefail: a failed test stops the script
    fi
  done
  if [ "$DRY_RUN" = 1 ]; then echo "+ package $exp -> final_results/$name"; continue; fi
  out=final_results/$name
  mkdir -p "$out"
  for f in "epoch_$EPOCHS.pth" curves.png freq_curves.png history.json train.log test.log \
           test_x${SCALE}.png test_x${SCALE}_freq.png test_x${SCALE}_freq.csv test_x${SCALE}_summary.json; do
    if [ -f "$exp/$f" ]; then cp "$exp/$f" "$out/"; fi
  done
  for set in bsds300 manga109; do
    if [ -d "$exp/$set" ]; then mkdir -p "$out/$set"; cp "$exp/$set/"* "$out/$set/"; fi
  done
  echo "[$name] packaged into $out"
done

# 4. Comparison table over everything in final_results/ (DWT and FFT runs together).
if [ "$DRY_RUN" = 1 ]; then echo "+ $PY scripts/summarize_results.py"; else $PY scripts/summarize_results.py; fi
