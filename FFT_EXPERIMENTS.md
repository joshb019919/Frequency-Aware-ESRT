# Report experiments on a GPU server: runbook

Produces everything still missing for the report, next to the DWT results already in `final_results/`.

| Phase | What | Proposal | Time |
|---|---|---|---|
| — | `lowpass` + 3 DWT configs (Set5, BSDS300, Manga109) | V0, V2 | done, in `final_results/` |
| `baselines` | bilinear + bicubic on all sets; Set14 tests of the 4 committed models; data check | B1, B2 | minutes |
| `sweep` | `lowpass` + FFT loss, λ = 0.01 / 0.05 / 0.1 | V1 + λ study | ~7 h per config |
| `dwt` | the 3 DWT configs + FFT loss at the best λ | V3 | ~7 h per config |

Training uses the same settings as the DWT runs (×2, patch 96, 1000 epochs, seed 1, batch 16, `--fast`, bf16). Only the loss changes:

L = ‖SR − HR‖₁ + λ · ‖FFT(SR) − FFT(HR)‖₁

Configs train **one per GPU in parallel**. GPUs are detected automatically, so two GPUs make each phase about 14 h. All commands run from the repo root.

## 1. Clone and create the venv

```bash
git clone -b dev/fft https://github.com/joshb019919/Frequency-Aware-ESRT.git
cd Frequency-Aware-ESRT
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # match `nvidia-smi` CUDA version
.venv/bin/pip install -r requirements.txt
```

All the scripts below use `.venv/bin/python` automatically. Set `PY=/path/to/python` to override.

## 2. Download the data (first time only, re-runnable)

```bash
DATA=/data/data scripts/download_data.sh
```

| Set | Source | Goes to |
|---|---|---|
| DIV2K train ×2 (~4.4 GB, converted to `.npy`) | ETH Zurich | `$DATA/div2k/DF2K_decoded/` |
| Set5 | `linzino7/Set5-dataset` (same as the DWT runs) | `Test_Datasets/` in the repo (git-ignored) |
| Set14 | Jia-Bin Huang's SelfExSR (named in the proposal) | `$DATA/set14/` |
| BSDS300 test split | Berkeley | `$DATA/bsds300/` |
| Manga109 | **manual**: licence from manga109.org, unpacked to `$DATA/manga109/images/`, plus `scripts/manga109_official.json` from Josh | `$DATA/manga109/` |

The script ends by running `scripts/check_setup.py`, which must report "all required checks passed". Manga109 is the one optional set: without it, the Manga109 column shows `-` and everything else still runs.

If `$DATA` is not `/data/data`, pass `ROOT=$DATA/div2k BENCH=$DATA` to the run commands below.

## 3. Run

Use `tmux`, because the training phases run for hours:

```bash
tmux new -s report
scripts/run_fft_experiments.sh baselines      # minutes: B1/B2, Set14 tests, data check
scripts/run_fft_experiments.sh sweep          # V1: lowpass with λ = 0.01, 0.05, 0.1
```

Pick the λ with the best PSNR in the printed table (look at all test sets, not only Set5), then:

```bash
LAMBDA=0.05 scripts/run_fft_experiments.sh dwt
```

- **Preview first:** `DRY_RUN=1 scripts/run_fft_experiments.sh <phase>` prints every command and the GPU assignment without running anything.
- **Progress:** `tail -f experiment/run_fft_gpu*/worker.log` (per config: `experiment/run_fft_gpu<N>/exp_<config>/train.log`).
- **GPUs:** detected automatically. Use `GPUS=1` for one specific GPU, or `GPUS=0,1,2,3` for a list. Each GPU gets its own run folder `experiment/run_fft_gpu<N>/`, so parallel workers never share files.
- **Resume:** run the same command again. Finished configs are skipped, interrupted ones restart, and this works even with a different number of GPUs.

**About the data check** (in `baselines`): it re-tests the committed `lowpass` checkpoint on this machine's data, in both `--fast` and `--no-fast` mode, and compares every image with the committed numbers.
- On a match, it records the mode and the later phases test the same way, so the new rows are directly comparable with the DWT rows.
- On a mismatch, it prints a warning. Usually this means different benchmark files; then re-test the committed models here before comparing.

## 4. Save and push the results

```bash
.venv/bin/python scripts/summarize_results.py --csv final_results/summary.csv
cp experiment/experiments_run_fft_gpu*.csv final_results/ 2>/dev/null || true
git add final_results
git commit -m "Add baseline, Set14 and FFT-loss results"
git push origin dev/fft
```

Each `final_results/<config>/` folder then holds:
- `epoch_1000.pth`
- `curves.png` and `freq_curves.png`
- `history.json`, `train.log` and `test.log`
- the Set5 test files at the top level, plus `set14/`, `bsds300/` and `manga109/` subfolders

In each test folder:
- `test_x2.png`, the PSNR/SSIM plot
- `test_x2_freq.png` and `test_x2_freq.csv`, the frequency metrics
- `test_x2_summary.json`, the per-image PSNR/SSIM

`final_results/bilinear/` and `bicubic/` hold the summary JSONs. `final_results/summary.csv` is the report table.

`experiment/`, `results/` and `Test_Datasets/` are git-ignored, so intermediate checkpoints, SR images and datasets stay on the server.

## Reporting notes
- Bicubic reproduces the literature values here: Set5 33.64 dB (paper 33.66), Set14 30.31 (30.24), BSDS300 test 29.55 (BSD100: 29.56). This is the proposal's sanity check.
- Set5 and Set14 are also the validation sets during training. Checkpoints are never selected by validation score (always epoch 1000), so testing on them is fine, but mention it.
- The DWT differences are 0.01–0.04 dB from a single seed, below the `--fast`/bf16 effect (0.02–0.1 dB, see COMMANDS.md). Treat FFT gains the same way. If they are this small, call them "no measurable difference".
- The loss is computed on RGB because the model is RGB-only. PSNR/SSIM are on Y, like everything else.
