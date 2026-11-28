# Shared helpers for scripts/run_fft_experiments.sh and scripts/run_window_experiments.sh (sourced, not run).
#
# Provides: defaults from the environment, GPU detection, train_parallel (one experiments.py per GPU),
# test_and_package (Set14/BSDS300/Manga109 tests + copy to final_results/<config>/), test_one, summarize.
# The caller must `cd` to the repo root and set RUN before sourcing; all paths are relative to the repo root.
#
# Env: ROOT (DIV2K dir, default div2k), BENCH (benchmark root, default "."; another location is linked into
#      the repo's set14/ bsds300/ manga109/ folders, because train.py validation looks there),
#      PY (default .venv/bin/python, else python), EPOCHS (default 1000),
#      GPUS (comma-separated GPU ids for parallel training; default: every GPU found, else 0),
#      TEST_FAST (--fast | --no-fast for the Set14/BSDS300/Manga109 tests; default: what `baselines` found,
#      else --fast), DRY_RUN=1 (print every command, run/copy nothing).

ROOT=${ROOT:-div2k}
BENCH=${BENCH:-.}
# train.py validates on Set5 + Set14 and experiments.py cannot pass it a benchmark root, so the sets must be
# reachable from the repo (data/benchmarks.py tries repo-relative paths first). Link them in if they live elsewhere.
if [ "$(cd "$BENCH" 2>/dev/null && pwd)" != "$PWD" ]; then
  for set in set14 bsds300 manga109; do
    [ -d "$BENCH/$set" ] || continue
    mkdir -p "$set"
    for sub in "$BENCH/$set"/*; do
      if [ -e "$sub" ] && [ ! -e "$set/$(basename "$sub")" ]; then ln -s "$sub" "$set/$(basename "$sub")"; fi
    done
  done
fi
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
MODE_FILE=experiment/run_fft/test_mode   # written by `run_fft_experiments.sh baselines` (data check)
if [ -z "${TEST_FAST:-}" ]; then
  if [ -f "$MODE_FILE" ]; then TEST_FAST=$(cat "$MODE_FILE"); else TEST_FAST=--fast; fi
fi
TEST_SETS="set14 bsds300 manga109"     # Set5 is tested by experiments.py (top level of each config folder)
DWT_CONFIGS="dwt_c3_gr1_skipnorm dwt_c1_gr1_norm dwt_cband_gr1_skipnorm"

fmt_lambda() { "$PY" -c "print('{:g}'.format(float('$1')))"; }   # experiments.py names: 0.10 -> 0.1

flags_for() {  # model flags of a config name (must match how it was trained): <base>[_fl<l>][_win]
  local name=$1 extra=""
  if [ "${name%_win}" != "$name" ]; then name=${name%_win}; extra="--attn window"; fi
  name=${name%_fl*}
  echo "$(base_flags "$name") $extra"
}

base_flags() {
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

# test_one CKPT CONFIG_NAME SET OUT_DIR MODE [REF_DIR]: test.py on one set, plots/summary/log in OUT_DIR.
test_one() {
  local ckpt=$1 name=$2 set=$3 out=$4 mode=$5 ref=${6:-}
  [ -f "$out/test_x${SCALE}_summary.json" ] && return 0
  if ! have_set "$set"; then echo "  [skip] $set is not prepared on this machine"; return 0; fi
  # shellcheck disable=SC2046
  local cmd=("$PY" test.py --test_sets "$set" --bench_root "$BENCH" --checkpoint "$ckpt" --upscale_factor $SCALE
             $(flags_for "$name") --output_folder "results/$RUN/$(basename "$(dirname "$ckpt")")/$set"
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

find_exp() {  # find_exp NAME: the worker run folder (of this $RUN) holding NAME's final checkpoint, or empty
  local d
  for d in experiment/"$RUN"_gpu*/exp_"$1"; do
    if [ -f "$d/epoch_$EPOCHS.pth" ]; then echo "$d"; return 0; fi
  done
  return 0
}

# train_parallel EXPERIMENTS_ARGS...: train every config in NAMES that is not finished yet, spread round-robin
# over $GPUS; each GPU runs its own experiments.py with its own run id ($RUN_gpu<N>), so no two processes
# ever share a folder or a CSV. EXPERIMENTS_ARGS (e.g. --freq_lambdas 0.05 --window) select the config
# copies; NAMES must be names experiments.py creates with them. Finished configs (in any worker folder) are
# skipped, so re-running resumes even with a different number of GPUs.
train_parallel() {
  local -a gpu_list todo subset cmd pids logs
  local name i k g only wrun log failed j
  IFS=, read -ra gpu_list <<< "$GPUS"
  todo=()
  for name in "${NAMES[@]}"; do
    if [ -n "$(find_exp "$name")" ]; then echo "[$name] already trained in $(find_exp "$name")"; else todo+=("$name"); fi
  done
  echo "GPUs: ${gpu_list[*]}  |  configs to train: ${todo[*]:-none}"
  pids=(); logs=()
  [ ${#todo[@]} -gt 0 ] || { export CUDA_VISIBLE_DEVICES=${gpu_list[0]}; return 0; }
  for i in "${!gpu_list[@]}"; do
    g=${gpu_list[$i]}; subset=()
    for k in "${!todo[@]}"; do if [ $((k % ${#gpu_list[@]})) = "$i" ]; then subset+=("${todo[$k]}"); fi; done
    [ ${#subset[@]} = 0 ] && continue
    only=$(IFS=,; echo "${subset[*]}")
    wrun=${RUN}_gpu$g
    seed_reference "$wrun"
    # --force: only unfinished configs are passed, so it just restarts interrupted ones
    cmd=("$PY" experiments.py --root "$ROOT" --run_id "$wrun" --epochs "$EPOCHS" --scale $SCALE "$@" --only "$only" --force)
    log=experiment/$wrun/worker.log
    echo "GPU $g -> $only   (run folder experiment/$wrun, log $log)"
    if [ "$DRY_RUN" = 1 ]; then CUDA_VISIBLE_DEVICES=$g "${cmd[@]}" --dry_run | sed 's/^/    /'; continue; fi
    mkdir -p "experiment/$wrun"
    CUDA_VISIBLE_DEVICES=$g "${cmd[@]}" > "$log" 2>&1 &
    pids+=($!); logs+=("$log")
  done
  if [ ${#pids[@]} -gt 0 ]; then
    echo "training in parallel: tail -F experiment/${RUN}_gpu*/worker.log"
    failed=0
    for j in "${!pids[@]}"; do
      if ! wait "${pids[$j]}"; then echo "worker failed, see ${logs[$j]}" >&2; failed=1; fi
    done
    [ $failed = 0 ] || echo "some configs failed; finished ones are still tested and packaged below" >&2
  fi
  export CUDA_VISIBLE_DEVICES=${gpu_list[0]}   # the tests that follow are quick: run them on the first GPU
}

# test_and_package: Set14 / BSDS300 / Manga109 tests of every config in NAMES, then copy to final_results/<config>/.
test_and_package() {
  local name exp ckpt set out f
  for name in "${NAMES[@]}"; do
    exp=$(find_exp "$name")
    if [ -z "$exp" ]; then
      [ "$DRY_RUN" = 1 ] || { echo "[$name] no final checkpoint, skipping tests"; continue; }
      exp=experiment/${RUN}_gpuN/exp_$name   # placeholder for the dry run
    fi
    ckpt=$exp/epoch_$EPOCHS.pth
    for set in $TEST_SETS; do
      echo "[$name] $set"
      test_one "$ckpt" "$name" "$set" "$exp/$set" "$TEST_FAST" "final_results/lowpass/$set"
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
}

summarize() {
  if [ "$DRY_RUN" = 1 ]; then echo "+ $PY scripts/summarize_results.py"; else "$PY" scripts/summarize_results.py; fi
}
