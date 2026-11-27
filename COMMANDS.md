# Commands and flags

All commands run from the repo root with the project venv (`/data/venvs/ml/bin/python`, written `python` below).
Flags that take a single dash (`-nEpochs`) really do use a single dash.

| Script | What it does |
|---|---|
| `main.py` | Launcher: optional HF dataset download, then runs `train.py`, `test.py` or `experiments.py`. Flags after `--` are forwarded. |
| `train.py` | Trains one model; validates on Set14 + Set5; writes checkpoints, `curves.png`, `freq_curves.png`, `history.json`. |
| `test.py` | Evaluates a checkpoint (PSNR/SSIM on Y + frequency metrics), writes SR images and plots. |
| `experiments.py` | Trains + tests several variants (lowpass and DWT) into one timestamped run folder, writes summary CSVs. |
| `bench.py` | Speed benchmark (inference and training step timing). |
| `pytest -q tests` | Test suite (6 `test_experiments.py` tests fail since the gr0 configs were disabled). |

## Model variant flags (train.py, test.py, bench.py, main.py)

A checkpoint must be loaded with the same flags it was trained with.

| Flag | Values / default | Meaning |
|---|---|---|
| `--freq` | `lowpass` (default), `dwt` | Band split inside each `Updownblock`: original average-pool low/high split, or Haar DWT. |
| `--conv` | `1`, `3`, `band` (default 1; dwt only) | DWT high path: 1x1 conv, 3x3 conv, or shared `decoder_high` per band. |
| `--global_res` | `1` (default), `0` | `0` removes the global skip (dwt only). |
| `--dwt_skip` | `0` (default), `1` | Residual around the DWT high path (`--conv 1/3` only). |
| `--ll_norm` | `0` (default), `1` | LL amplitude normalisation (dwt only). |

## train.py

| Flag | Default | Meaning |
|---|---|---|
| `--root` | `/data0/luzs/dataset/` | Dataset directory (DIV2K `.npy`, see `logs/instructions.md`). |
| `--scale` | 2 | SR factor. Must match the validation/test LR folders. |
| `-nEpochs` | 1000 | Epochs to train. |
| `--batch_size` | 16 | Training batch size. |
| `--patch_size` | 192 | HR training patch. Use 96 for x2 to avoid OOM. |
| `--lr` / `--step_size` | 2e-4 / 200 | Initial learning rate; it is multiplied by gamma every `step_size` epochs. |
| `--gamma` | 0.5 | LR decay factor (declared `int`, so it cannot be set fractionally from the CLI). |
| `--seed` | 1 | Random seed. |
| `--threads` | 8 | Data-loader workers. |
| `--resume` / `--start-epoch` | none / 1 | Resume from a checkpoint and epoch number. |
| `--pretrained` | none | Initialise from a checkpoint (tail params of another scale are skipped). |
| `--save_dir` | `experiment/checkpoint_ESRT_x<scale>[variant]/` | Where checkpoints, plots and `history.json` go. |
| `--val_every` | 5 | Validate every N epochs (always at epoch 0 and after the last epoch). |
| `--val_sets` | `set14,set5` | Validation sets (any of set5, set14, b100, bsds300, manga109). |
| `--bench_root` | `/data/data` | Folder holding the benchmark datasets. |
| `--no_amp` | off | Plain fp32 training (disables bf16 autocast, channels_last, cudnn.benchmark). bf16 AMP is on by default. |
| `--fast` / `--no-fast` | on | `torch.compile` the training model (validation and checkpoints use the uncompiled module). |
| `--compile_mode` | `default` | `default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`. |
| `--plot` / `--no-plot` | on | Write `curves.png` and `freq_curves.png` at each validation. |
| `--ref_history` | none | `history.json` of a finished run (e.g. lowpass): its final level (mean of the last 10 points) is drawn as a dashed black line in the curves. |
| `--hf_metrics` / `--no-hf_metrics` | on | Frequency metrics at every validation (`===> Freq[set]` lines). |
| `--perceptual` | `both` | `both`, `lpips`, `dists`, `none`. |
| `--wavelets` | `haar,db2,bior2.2` | PyWavelets names for the per-subband error. |

Checkpoints are written every 10 epochs and at the last epoch (`epoch_<N>.pth`).

```bash
# lowpass baseline, x2, 1000 epochs
python train.py --root /data/data/div2k --scale 2 --patch_size 96 -nEpochs 1000 --save_dir experiments/lowpass

# DWT variant (3x3 conv, skip + LL norm), compared against lowpass in the curves
python train.py --root /data/data/div2k --scale 2 --patch_size 96 -nEpochs 1000 \
    --freq dwt --conv 3 --dwt_skip 1 --ll_norm 1 \
    --save_dir experiments/dwt_c3 --ref_history experiments/lowpass/history.json

# plain fp32, no compile (reference for AMP/--fast comparisons)
python train.py --root /data/data/div2k --scale 2 --patch_size 96 -nEpochs 1000 --no_amp --no-fast --save_dir experiments/lowpass_fp32
```

## test.py

Always pass `--checkpoint` (the default is a leftover) and the same variant flags used in training.

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint` | leftover path | Checkpoint to evaluate. |
| `--upscale_factor` | 2 | SR factor; must match the checkpoint. |
| `--test_sets` | `bsds300,manga109` | Benchmark sets to test (from `--bench_root`). |
| `--bench_root` | `/data/data` | Benchmark dataset folder. |
| `--test_hr_folder` + `--test_lr_folder` | none | Test one explicit folder pair instead of `--test_sets` (LR files named `<name>x<scale>.png`). Both required together. |
| `--output_folder` | `results/<set>/x<scale>` | SR images (with several sets: `<folder>/<set>`). |
| `--plot` / `--no-plot` | on | Write the plots below. |
| `--plot_path` | `<checkpoint dir>/test_x<scale>.png` | Summary plot path; the other files use the same stem. |
| `--ref_test_dir` | none | Folder holding a reference run's `test_x<scale>_summary.json` and `_freq.csv` (lowpass). Drawn as dashed black lines on the plots. |
| `--fast` / `--no-fast` | on | `torch.compile` + bf16 inference. About -0.015 dB versus fp32; use `--no-fast` for final numbers. |
| `--hf_metrics`, `--perceptual`, `--wavelets` | as in train.py | Frequency metrics. |

Outputs next to the plot path: `test_x2.png` (mean PSNR/SSIM bars + per-image PSNR), `test_x2_freq.png` (subband error, radial power ratio, HFEN/GMSD/LPIPS/DISTS on log axes), `test_x2_freq.csv` (per image), `test_x2_summary.json` (per-image PSNR/SSIM, used as the reference file by other runs). The last stdout line is `Mean PSNR: ..., SSIM: ...`.

```bash
# default sets (bsds300 + manga109)
python test.py --checkpoint experiments/lowpass/epoch_1000.pth --upscale_factor 2 --no-fast

# one set, own output folder and plot location, lowpass as the reference
python test.py --test_sets manga109 --checkpoint experiments/dwt_c3/epoch_1000.pth --upscale_factor 2 \
    --freq dwt --conv 3 --dwt_skip 1 --ll_norm 1 \
    --output_folder results/dwt_c3 --plot_path experiments/dwt_c3/manga109/test_x2.png \
    --ref_test_dir experiments/lowpass/manga109

# Set5 through explicit folders (what experiments.py does)
python test.py --test_hr_folder Test_Datasets/Set5/ --test_lr_folder Test_Datasets/Set5_LR/x2/ \
    --checkpoint experiments/lowpass/epoch_1000.pth --upscale_factor 2
```

## experiments.py

Trains and tests every config, with the same seed and settings. New run folder `experiment/run_<YYYYMMDD_HHMMSS>/exp_<name>/` per invocation (nothing is overwritten), SR images in `results/<run_id>/exp_<name>/`, summary CSVs `experiment/experiments_<ts>.csv` and `_epochs.csv`. Its test step uses Set5 only. Lowpass should run first: later configs get `--ref_history` / `--ref_test_dir` pointing at it automatically.

| Flag | Default | Meaning |
|---|---|---|
| `--root` | none | Dataset directory (required unless `--from_logs`). |
| `--epochs` | 10 | Epochs per config. |
| `--scale`, `--batch_size`, `--threads`, `--seed`, `--val_every` | 2, 16, 8, 1, 5 | Forwarded to train.py. |
| `--patch_size` | 96 | `96`, `144` or `both`. |
| `--only` | all | Comma-separated config names, e.g. `lowpass,dwt_c3_gr1_skipnorm`. |
| `--dwt_tweaks` | none | Subset of `skip,norm,skipnorm`: also run the DWT configs with those tweaks (named `<config>_<tweak>`). |
| `--fast` / `--no-fast`, `--no_amp` | on, off | Speed options for training and testing. |
| `--run_id` | new each run | Reuse a run folder; configs with a final checkpoint are skipped (resume). |
| `--force` | off | Retrain even if the final checkpoint exists; allow a non-empty config folder. |
| `--from_logs` | off | No training: rebuild a per-epoch CSV from existing `train.log` files. |
| `--dry_run` | off | Print the commands and exit. |
| `--test_hr_folder` / `--test_lr_folder` | `Test_Datasets/Set5/` / `.../Set5_LR/x<scale>/` | Test folders. |

```bash
# preview
python experiments.py --root /data/data/div2k --epochs 1000 --only lowpass,dwt_c3_gr1_skipnorm --dry_run

# the 1000-epoch comparison (fp32, no compile)
python experiments.py --root /data/data/div2k --scale 2 --patch_size 96 --epochs 1000 \
    --dwt_tweaks norm,skipnorm --only lowpass,dwt_c3_gr1_skipnorm,dwt_cband_gr1_skipnorm,dwt_c1_gr1_norm \
    --no-fast --no_amp

# resume an interrupted run
python experiments.py --root /data/data/div2k --epochs 1000 --run_id run_20261125_122259
```

## main.py

```bash
# train (download skipped), flags after -- go to train.py
python main.py --train --skip_download --data_dir /data/data/div2k -- -nEpochs 1 --patch_size 96

# test
python main.py --test --skip_download --data_dir /data/data/div2k --checkpoint experiments/lowpass/epoch_1000.pth --scale 2

# experiments
python main.py --experiments --skip_download --data_dir /data/data/div2k -- --epochs 10
```

| Flag | Meaning |
|---|---|
| `--train` / `--test` / `--experiments` | Exactly one mode. |
| `--hf_repo`, `--hf_revision` | Hugging Face dataset to download (token from `HF_TOKEN`). |
| `--data_dir`, `--skip_download` | Local data folder; skip the download. |
| `--scale`, `--checkpoint`, `--test_hr_folder`, `--test_lr_folder`, `--output_folder` | Forwarded to `test.py`. |
| `--freq/--conv/--global_res/--dwt_skip/--ll_norm` | Variant flags, forwarded only when given. |
| `--fast` / `--no-fast` | Forwarded to the script. |

Each run appends a line to `timings.log`.

## bench.py

```bash
python bench.py --scale 2                      # amp + channels_last (default)
python bench.py --scale 2 --no_amp             # plain fp32
python bench.py --scale 2 --freq dwt --json bench_dwt.json
```

Flags: `--lr_size H W` (inference input, default 64 64), `--patch_size`, `--batch_size`, `--warmup`, `--iters`, `--device`, `--seed`, `--no_amp`, `--fast`, `--json <file>`.

## Data and benchmark preparation

```bash
python scripts/png2npy.py ...                          # convert PNG training data to .npy
python scripts/prepare_benchmarks.py                   # adds HR/ and LR/x{2,3,4}/ to /data/data/bsds300 and manga109
python scripts/prepare_benchmarks.py --manga109_pages middle   # old mid-page Manga109 stand-in
```

Expected layout: `<root>/DF2K_decoded/DIV2K_train_HR/` and `DIV2K_train_LR_bicubic/X<scale>/` as `.npy`; `Test_Datasets/Set5/` (HR) and `Test_Datasets/Set5_LR/x<scale>/`.

## Packaging results for upload

`final_results/<config>/` holds `epoch_1000.pth`, `curves.png`, `freq_curves.png`, `history.json`, `train.log`, the Set5 test files, and `bsds300/` and `manga109/` subfolders with their own plots, CSV, summary JSON and `test.log`. Intermediate checkpoints live in `experiments/<run>/exp_<name>/chkpts/` and are not uploaded.

## Notes

- Everything under `experiments/` is git-ignored (`.gitignore` line `experiments/`), including `chkpts/`. `final_results/` is not.
- The code writes to `experiment/` (singular) by default; pass `--save_dir` or rename the folder to match the layout above.
- `--fast` and bf16 AMP cost about 0.02-0.1 dB versus fp32 on the lowpass model; test with `--no-fast` for reported numbers.
