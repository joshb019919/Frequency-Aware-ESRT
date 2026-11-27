#!/usr/bin/env bash
# First-time dataset download + preparation for a new GPU server. Safe to re-run: every step whose
# output already exists is skipped. Run from anywhere; uses the repo's .venv python if present.
#
# By default everything goes inside the repo (no special permissions needed), into the placeholder folders
# div2k/, set14/, bsds300/, manga109/ and Test_Datasets/ (git-ignored apart from their .gitkeep). That is also
# where train.py/test.py find the benchmark sets without any extra flags. Needs ~20 GB free while running.
#
#   DIV2K train (x2)  ETH Zurich zips (~4.4 GB)  -> $DATA/div2k/DF2K_decoded/{DIV2K_train_HR,DIV2K_train_LR_bicubic/X2} (.npy)
#   Set5              github linzino7/Set5-dataset -> <repo>/Test_Datasets/{Set5,Set5_LR/x2}  (same source as the DWT runs)
#   Set14             github jbhuang0604/SelfExSR (the source named in the proposal) -> $DATA/set14/{Set14_HR,Set14_LR_x2}
#   BSDS300           Berkeley tgz (22 MB), test split -> $DATA/bsds300/{HR,LR/x2} via scripts/prepare_benchmarks.py
#   Manga109          licence-restricted, cannot be downloaded automatically (see the message at the end)
#
# Env: DATA (default "." = the repo root; paths are relative to it), PY (default <repo>/.venv/bin/python), KEEP_ZIPS=1 to keep the archives,
#      SKIP_DIV2K=1 to only fetch the small benchmark sets.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"   # all paths below are relative to the repo root
DATA=${DATA:-.}
if [ -z "${PY:-}" ]; then
  if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python; fi
fi
DL=$DATA/downloads
mkdir -p "$DATA" "$DL"

fetch() {  # fetch URL DEST: download with resume and retries, to a .part file first
  local url=$1 dest=$2
  [ -s "$dest" ] && return 0
  echo "  downloading $url"
  curl -fL --retry 5 --retry-delay 5 -C - -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
}

count() { [ -d "$1" ] && find "$1" -maxdepth 1 -type f -name "$2" | wc -l | tr -d ' ' || echo 0; }

# ---------------------------------------------------------------- DIV2K
if [ "${SKIP_DIV2K:-0}" != 1 ]; then
  NPY=$DATA/div2k/DF2K_decoded
  if [ "$(count "$NPY/DIV2K_train_HR" '*.npy')" -ge 800 ] && [ "$(count "$NPY/DIV2K_train_LR_bicubic/X2" '*.npy')" -ge 800 ]; then
    echo "[DIV2K] already prepared in $NPY"
  else
    echo "[DIV2K] downloading + converting (one-off, ~4.4 GB download, ~15 GB of .npy)"
    RAW=$DATA/div2k/DIV2K_raw
    mkdir -p "$RAW"
    for z in DIV2K_train_HR DIV2K_train_LR_bicubic_X2; do
      fetch "http://data.vision.ee.ethz.ch/cvl/DIV2K/$z.zip" "$DL/$z.zip"
    done
    [ "$(count "$RAW/DIV2K_train_HR" '*.png')" -ge 800 ] || unzip -q -o "$DL/DIV2K_train_HR.zip" -d "$RAW"
    [ "$(count "$RAW/DIV2K_train_LR_bicubic/X2" '*.png')" -ge 800 ] || unzip -q -o "$DL/DIV2K_train_LR_bicubic_X2.zip" -d "$RAW"
    mkdir -p "$(dirname "$NPY")"
    "$PY" "scripts/png2npy.py" --pathFrom "$RAW/" --pathTo "$NPY/" | grep -E "Converted (800|[0-9]*00) " | tail -2 || true
    echo "[DIV2K] HR $(count "$NPY/DIV2K_train_HR" '*.npy') / LR $(count "$NPY/DIV2K_train_LR_bicubic/X2" '*.npy') .npy files"
    [ "${KEEP_ZIPS:-0}" = 1 ] || { rm -rf "$RAW"; rm -f "$DL"/DIV2K_*.zip; }
  fi
fi

# ---------------------------------------------------------------- Set5 (repo-relative, as train.py expects)
S5=Test_Datasets
if [ "$(count "$S5/Set5" '*.png')" -eq 5 ] && [ "$(count "$S5/Set5_LR/x2" '*.png')" -eq 5 ]; then
  echo "[Set5] already in $S5"
else
  echo "[Set5] downloading"
  mkdir -p "$S5/Set5" "$S5/Set5_LR/x2"
  for n in baby bird butterfly head woman; do
    fetch "https://raw.githubusercontent.com/linzino7/Set5-dataset/main/Set5/HR/$n.png" "$S5/Set5/$n.png"
    fetch "https://raw.githubusercontent.com/linzino7/Set5-dataset/main/Set5/LR_bicubic/X2/${n}x2.png" "$S5/Set5_LR/x2/${n}x2.png"
  done
fi

# ---------------------------------------------------------------- Set14
S14=$DATA/set14
if [ "$(count "$S14/Set14_HR" '*.png')" -eq 14 ] && [ "$(count "$S14/Set14_LR_x2" '*.png')" -eq 14 ]; then
  echo "[Set14] already in $S14"
else
  echo "[Set14] downloading"
  mkdir -p "$S14/Set14_HR" "$S14/Set14_LR_x2"
  base=https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set14/image_SRF_2
  for i in $(seq -w 1 14); do
    fetch "$base/img_0${i}_SRF_2_HR.png" "$S14/Set14_HR/img_0$i.png"
    fetch "$base/img_0${i}_SRF_2_LR.png" "$S14/Set14_LR_x2/img_0$i.png"
  done
fi

# ---------------------------------------------------------------- BSDS300 (test split, 100 images)
B=$DATA/bsds300
if [ "$(count "$B/HR" '*.png')" -eq 100 ] && [ "$(count "$B/LR/x2" '*.png')" -eq 100 ]; then
  echo "[BSDS300] already in $B"
else
  echo "[BSDS300] downloading"
  fetch "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz" "$DL/BSDS300-images.tgz"
  mkdir -p "$B"
  [ -f "$B/iids_test.txt" ] || tar -xzf "$DL/BSDS300-images.tgz" -C "$B" --strip-components=1
  "$PY" "scripts/prepare_benchmarks.py" --data_root "$DATA" --sets bsds300
fi

# ---------------------------------------------------------------- Manga109 (manual)
M=$DATA/manga109
if [ "$(count "$M/HR" '*.png')" -eq 109 ]; then
  echo "[Manga109] already in $M"
elif [ -d "$M/images" ] && [ -f "scripts/manga109_official.json" ]; then
  echo "[Manga109] preparing the official 109-image benchmark"
  "$PY" "scripts/prepare_benchmarks.py" --data_root "$DATA" --sets manga109
else
  cat <<EOF
[Manga109] NOT prepared (needs manual steps):
  1. Manga109 is licence-restricted: request it at http://www.manga109.org/en/download.html and unpack it
     so that $M/images/<Book>/000.jpg exists.
  2. Get scripts/manga109_official.json from Josh (it defines the 109 benchmark crops used for the DWT
     results and is not in the repo), copy it to scripts/, then re-run this script.
  Everything else (training, Set5/Set14/BSDS300 results) works without it; Manga109 rows will show '-'.
EOF
fi

echo
"$PY" scripts/check_setup.py --root "$DATA/div2k" --bench_root "$DATA" || true
