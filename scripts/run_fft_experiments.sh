#!/usr/bin/env bash
# Everything the report needs, run on the GPU server after scripts/download_data.sh.
#
#   baselines  (minutes, no training)
#              - B1/B2: bilinear + bicubic on Set5, Set14, BSDS300, Manga109   -> final_results/{bilinear,bicubic}/
#              - Set14 tests of the four committed models (lowpass + 3 DWT)      -> final_results/<config>/set14/
#              - data check: re-tests final_results/lowpass on this machine's Set5/BSDS300/Manga109 (fast and
#                no-fast) and compares per image with the committed results, so you know the new data and
#                test mode reproduce them (prints which TEST_FAST setting to use)
#   sweep      V1 + lambda study: lowpass trained with lambda in $LAMBDAS (default 0.01,0.05,0.1)
#   dwt        V3: the three DWT configs trained with one lambda ($LAMBDA)
#              Each phase is 3 configs of ~7 h, trained one per GPU in parallel (2 GPUs: ~14 h, 3+: ~7 h).
#              Worker run folders: experiment/<RUN>_gpu<N>/ (one experiments.py per GPU, unchanged Python).
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
#   LAMBDA=0.05 scripts/run_fft_experiments.sh dwt
# Env: ROOT (DIV2K dir, default /data/data/div2k), BENCH (benchmark root, default /data/data),
#      RUN (run id, default run_fft), PY (default .venv/bin/python, else python), EPOCHS (default 1000),
#      GPUS (comma-separated GPU ids for parallel training; default: every GPU found, else 0),
#      TEST_FAST (--fast | --no-fast for Set14/BSDS300/Manga109 tests; default: what `baselines` found),
#      DRY_RUN=1 (print every command, run/copy nothing).
# Extra arguments after the phase are passed to experiments.py.
set -euo pipefail
cd "$(dirname "$0")/.."

PHASE=${1:?usage: $0 baselines|sweep|dwt [experiments.py args...]}; shift
ROOT=${ROOT:-/data/data/div2k}
BENCH=${BENCH:-/data/data}
RUN=${RUN:-run_fft}
if [ -z "${PY:-}" ]; then
  if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python; fi
fi
EPOCHS=${EPOCHS:-1000}
# GPUs used for parallel training (one config per GPU at a time): $GPUS, else $CUDA_VISIBLE_DEVICES,
# else every GPU nvidia-smi lists, else just 0. One GPU simply means one worker.
if [ -z "${GPUS:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then GPUS=$CUDA_VISIBLE_DEVICES
  elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=index --format=csv,noheader >/dev/null 2>&1; then
    GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ' | paste -sd, -)
  else GPUS=0; fi
fi
SCALE=2
DRY_RUN=${DRY_RUN:-0}
MODE_FILE=experiment/$RUN/test_mode   # written by `baselines`: the test mode that reproduces the committed numbers
if [ -z "${TEST_FAST:-}" ]; then
  if [ -f "$MODE_FILE" ]; then TEST_FAST=$(cat "$MODE_FILE"); else TEST_FAST=--fast; fi
fi
TEST_SETS="set14 bsds300 manga109"     # Set5 is tested by experiments.py (top level of each config folder)
DWT_CONFIGS="dwt_c3_gr1_skipnorm dwt_c1_gr1_norm dwt_cband_gr1_skipnorm"

flags_for() {  # model flags of a base config (must match how it was trained)
  case "$1" in
    lowpass)                echo "" ;;
    dwt_c1_gr1_norm)        echo "--freq dwt --conv 1 --global_res 1 --ll_norm 1" ;;
    dwt_c3_gr1_skipnorm)    echo "--freq dwt --conv 3 --global_res 1 --dwt_skip 1 --ll_norm 1" ;;
    dwt_cband_gr1_skipnorm) echo "--freq dwt --conv band --global_res 1 --dwt_skip 1 --ll_norm 1" ;;
    *) echo "unknown config $1" >&2; exit 1 ;;
  esac
}

have_set() {  # is benchmark $1 prepared on this machine?
  "$PY" -c "import sys; sys.path.insert(0, '.'); from data import benchmarks; benchmarks.resolve('$1', $SCALE, '$BENCH')" 2>/dev/null
}

# test_one CKPT BASE_CONFIG SET OUT_DIR MODE [REF_DIR]: test.py on one set, plots/summary/log in OUT_DIR.
test_one() {
  local ckpt=$1 base=$2 set=$3 out=$4 mode=$5 ref=${6:-}
  [ -f "$out/test_x${SCALE}_summary.json" ] && return 0
  if ! have_set "$set"; then echo "  [skip] $set is not prepared on this machine"; return 0; fi
  # shellcheck disable=SC2046
  local cmd=("$PY" test.py --test_sets "$set" --bench_root "$BENCH" --checkpoint "$ckpt" --upscale_factor $SCALE
             $(flags_for "$base") --output_folder "results/$RUN/$(basename "$(dirname "$ckpt")")/$set"
             --plot_path "$out/test_x${SCALE}.png" "$mode")
  if [ -n "$ref" ] && [ -d "$ref" ]; then cmd+=(--ref_test_dir "$ref"); fi
  if [ "$DRY_RUN" = 1 ]; then echo "+ ${cmd[*]}"; return 0; fi
  mkdir -p "$out"
  "${cmd[@]}" 2>&1 | tee "$out/test.log" | grep -E "^Mean PSNR" || true
  [ -f "$out/test_x${SCALE}_summary.json" ] || { echo "test.py failed, see $out/test.log" >&2; exit 1; }
}

seed_reference() {  # seed_reference RUN_ID: finished lowpass run -> that run folder (plot reference; never retrained)
  local ref=experiment/$1/exp_lowpass
  if [ ! -f "$ref/epoch_$EPOCHS.pth" ]; then
    [ -f "final_results/lowpass/epoch_$EPOCHS.pth" ] || { echo "final_results/lowpass/epoch_$EPOCHS.pth missing" >&2; exit 1; }
    if [ "$DRY_RUN" = 1 ]; then echo "+ cp -R final_results/lowpass/. $ref/"; else mkdir -p "$ref"; cp -R final_results/lowpass/. "$ref/"; fi
  fi
  for set in $TEST_SETS; do  # sets tested after seeding (e.g. set14 from `baselines`)
    if [ -d "final_results/lowpass/$set" ] && [ ! -d "$ref/$set" ] && [ "$DRY_RUN" != 1 ]; then
      cp -R "final_results/lowpass/$set" "$ref/$set"
    fi
  done
}

find_exp() {  # find_exp NAME: the worker run folder that holds NAME's final checkpoint (empty if none)
  local d
  for d in experiment/"$RUN"_gpu*/exp_"$1"; do
    if [ -f "$d/epoch_$EPOCHS.pth" ]; then echo "$d"; return 0; fi
  done
  return 0
}

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
    mkdir -p "experiment/$RUN"
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
  if [ "$DRY_RUN" = 1 ]; then echo "+ $PY scripts/summarize_results.py"; else "$PY" scripts/summarize_results.py; fi
  exit 0
fi

# ------------------------------------------------------------------------------------------- sweep / dwt
case "$PHASE" in
  sweep) LAMBDAS=${LAMBDAS:-0.01,0.05,0.1}; BASES="lowpass" ;;
  dwt)   LAMBDAS=${LAMBDA:?set LAMBDA to the best value from the sweep, e.g. LAMBDA=0.05}; BASES=$DWT_CONFIGS ;;
  *) echo "phase must be baselines, sweep or dwt" >&2; exit 1 ;;
esac

# Config names experiments.py will create: <base>_fl<lambda> ({:g} formatting, e.g. 0.10 -> 0.1).
NAMES=()
for b in $BASES; do
  for l in ${LAMBDAS//,/ }; do NAMES+=("${b}_fl$("$PY" -c "print('{:g}'.format(float('$l')))")"); done
done
ONLY=$(IFS=,; echo "${NAMES[*]}")

# Train + Set5 test. Configs that are not finished yet are spread round-robin over $GPUS; each GPU runs its
# own experiments.py with its own run id ($RUN_gpu<N>), so no two processes ever share a folder or a CSV
# (experiments.py is unchanged). Finished configs (in any worker folder) are skipped, so re-running resumes
# even with a different number of GPUs.
IFS=, read -ra GPU_LIST <<< "$GPUS"
TODO=()
for name in "${NAMES[@]}"; do
  if [ -n "$(find_exp "$name")" ]; then echo "[$name] already trained in $(find_exp "$name")"; else TODO+=("$name"); fi
done
echo "GPUs: ${GPU_LIST[*]}  |  configs to train: ${TODO[*]:-none}"
pids=(); logs=()
for i in "${!GPU_LIST[@]}"; do
  g=${GPU_LIST[$i]}; subset=()
  for k in "${!TODO[@]}"; do if [ $((k % ${#GPU_LIST[@]})) = "$i" ]; then subset+=("${TODO[$k]}"); fi; done
  [ ${#subset[@]} = 0 ] && continue
  only=$(IFS=,; echo "${subset[*]}")
  wrun=${RUN}_gpu$g
  seed_reference "$wrun"
  cmd=("$PY" experiments.py --root "$ROOT" --run_id "$wrun" --epochs "$EPOCHS" --scale $SCALE
       --freq_lambdas "$LAMBDAS" --only "$only" --force "$@")   # --force: only unfinished configs are passed,
                                                                  # so it just restarts interrupted ones
  log=experiment/$wrun/worker.log
  echo "GPU $g -> $only   (run folder experiment/$wrun, log $log)"
  if [ "$DRY_RUN" = 1 ]; then CUDA_VISIBLE_DEVICES=$g "${cmd[@]}" --dry_run | sed 's/^/    /'; continue; fi
  mkdir -p "experiment/$wrun"
  CUDA_VISIBLE_DEVICES=$g "${cmd[@]}" > "$log" 2>&1 &
  pids+=($!); logs+=("$log")
done
if [ ${#pids[@]} -gt 0 ]; then
  echo "training in parallel: tail -f experiment/${RUN}_gpu*/worker.log"
  failed=0
  for j in "${!pids[@]}"; do
    if ! wait "${pids[$j]}"; then echo "worker failed, see ${logs[$j]}" >&2; failed=1; fi
  done
  [ $failed = 0 ] || echo "some configs failed; finished ones are still tested and packaged below" >&2
fi
export CUDA_VISIBLE_DEVICES=${GPU_LIST[0]}   # the tests below are quick: run them on the first GPU

# Set14 / BSDS300 / Manga109 tests, then package into final_results/<config>/.
for name in "${NAMES[@]}"; do
  exp=$(find_exp "$name")
  if [ -z "$exp" ]; then
    [ "$DRY_RUN" = 1 ] || { echo "[$name] no final checkpoint, skipping tests"; continue; }
    exp=experiment/${RUN}_gpuN/exp_$name   # placeholder for the dry run
  fi
  ckpt=$exp/epoch_$EPOCHS.pth
  for set in $TEST_SETS; do
    echo "[$name] $set"
    test_one "$ckpt" "${name%_fl*}" "$set" "$exp/$set" "$TEST_FAST" "final_results/lowpass/$set"
  done
  if [ "$DRY_RUN" = 1 ]; then echo "+ package $exp -> final_results/$name"; continue; fi
  out=final_results/$name
  mkdir -p "$out"
  for f in "epoch_$EPOCHS.pth" curves.png freq_curves.png history.json train.log test.log \
           test_x${SCALE}.png test_x${SCALE}_freq.png test_x${SCALE}_freq.csv test_x${SCALE}_summary.json; do
    if [ -f "$exp/$f" ]; then cp "$exp/$f" "$out/"; fi
  done
  for set in $TEST_SETS; do
    if [ -d "$exp/$set" ]; then mkdir -p "$out/$set"; cp "$exp/$set/"* "$out/$set/"; fi
  done
  echo "[$name] packaged into $out"
done

if [ "$DRY_RUN" = 1 ]; then echo "+ $PY scripts/summarize_results.py"; else "$PY" scripts/summarize_results.py; fi
