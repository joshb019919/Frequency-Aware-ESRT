#!/usr/bin/env bash
# Baselines (minutes, no training):
#   - B1/B2: bilinear + bicubic on Set5, Set14, BSDS300, Manga109   -> final_results/{bilinear,bicubic}/
#   - Set14 tests of the four committed models (lowpass + 3 DWT)      -> final_results/<config>/set14/
#   - data check: re-tests final_results/lowpass on this machine and compares per image with the committed
#     results (records the matching test mode for the other scripts)
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=run_fft
. scripts/lib_experiments.sh

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
