# Report experiments on a GPU server: runbook

Produces everything still missing for the report, next to the DWT results already in `final_results/`.

| Phase | What | Proposal | Time |
|---|---|---|---|
| — | `lowpass` + 3 DWT configs (Set5, BSDS300, Manga109) | V0, V2 | done, in `final_results/` |
| `baselines` | bilinear + bicubic on all sets; Set14 tests of the 4 committed models; data check | B1, B2 | minutes |
| `sweep` | `lowpass` + FFT loss, λ = 0.01 / 0.05 / 0.1 | V1 + λ study | ~7 h per config |
| `dwt` | the 3 DWT configs + FFT loss at the best λ | V3 | ~7 h per config |
| `run_window_experiments.sh` | window attention alone, + DWT, + FFT, + DWT + FFT (V4): the full window ablation | V4 (stretch) | ~7 h per config |

Training uses the same settings as the DWT runs (×2, patch 96, 1000 epochs, seed 1, batch 16, `--fast`, bf16). Only the loss changes:

L = ‖SR − HR‖₁ + λ · ‖FFT(SR) − FFT(HR)‖₁

Configs train **one per GPU in parallel**. GPUs are detected automatically, so two GPUs make each phase about 14 h. All commands run from the repo root.

## 1. Clone and create the venv

```bash
git clone -b dev/fft https://github.com/joshb019919/Frequency-Aware-ESRT.git
cd Frequency-Aware-ESRT
python3 -m venv --without-pip .venv          # works even without the python3-venv system package
source .venv/bin/activate
curl -sS https://bootstrap.pypa.io/get-pip.py | python
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # match `nvidia-smi` CUDA version
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"   # e.g. True 2
```

Run `source .venv/bin/activate` in every new terminal or tmux window. The scripts also use `.venv/bin/python` on their own; set `PY=/path/to/python` to override.

## 2. Download the data (first time only, re-runnable)

Run every command from the `Frequency-Aware-ESRT` folder. All data goes into folders **inside the repo**, so no special permissions are needed. Those folders are already in the clone (empty, with a `.gitkeep`) and git-ignored, so the data is never committed. You need about 20 GB free.

```bash
scripts/download_data.sh
```

| Set | Source | Repo folder |
|---|---|---|
| DIV2K train ×2 (~4.4 GB, converted to `.npy`) | ETH Zurich | `div2k/DF2K_decoded/` |
| Set5 | `linzino7/Set5-dataset` (same as the DWT runs) | `Test_Datasets/Set5`, `Test_Datasets/Set5_LR/x2` |
| Set14 | Jia-Bin Huang's SelfExSR (named in the proposal) | `set14/Set14_HR`, `set14/Set14_LR_x2` |
| BSDS300 test split | Berkeley | `bsds300/HR`, `bsds300/LR/x2` |
| Manga109 | **manual**: licence from manga109.org, unpack to `manga109/images/`, plus `scripts/manga109_official.json` from Josh; then re-run the script | `manga109/HR`, `manga109/LR/x2` |

These are the paths `train.py`, `test.py` and `experiments.py` already look for (relative to the repo), so no path flags are needed anywhere. The script ends with `scripts/check_setup.py`; everything except Manga109 should be `ok`. Without Manga109, the Manga109 column shows `-` and the rest still runs.

If the data already lives somewhere else, `ROOT=/path/div2k BENCH=/path scripts/run_fft_experiments.sh ...` links those benchmark folders into the repo, where training validation needs them.

## 3. Run

Use `tmux`, because the training phases run for hours:

```bash
tmux new -s report
source .venv/bin/activate
scripts/run_fft_experiments.sh baselines      # minutes: B1/B2, Set14 tests, data check
scripts/run_fft_experiments.sh sweep          # V1: lowpass with λ = 0.01, 0.05, 0.1
```

Pick the λ with the best PSNR in the printed table (look at all test sets, not only Set5), then:

```bash
LAMBDA=0.05 scripts/run_fft_experiments.sh dwt
```

Stretch goal V4 has its own script, `scripts/run_window_experiments.sh`, with its own run folders (`experiment/run_window_gpu<N>/`). It can run at the same time as the FFT phases on a free GPU and never shares files with them.

It builds its configs from whatever you have decided so far:

| Set | Adds | Combination |
|---|---|---|
| nothing | `lowpass_win` | window |
| `DWT_CONFIG=<dwt>` | `<dwt>_win` | DWT + window |
| `LAMBDA=<λ>` | `lowpass_fl<λ>_win` | FFT + window |
| both | `<dwt>_fl<λ>_win` | DWT + FFT + window (**V4**) |

```bash
scripts/run_window_experiments.sh                                                  # window only
LAMBDA=0.05 DWT_CONFIG=dwt_c3_gr1_skipnorm scripts/run_window_experiments.sh       # all four window rows
GPUS=1 DWT_CONFIG=dwt_cband_gr1_skipnorm CONFIGS=dwt_cband_gr1_skipnorm_win scripts/run_window_experiments.sh
                                                                                   # one named config on GPU 1
```

- **Finished configs are skipped,** so running again with more set only trains what's new.
- **A config still training doesn't count as finished,** because it has no final checkpoint yet. While one is running, use `CONFIGS=` to start others without retraining it.
- **Together with the FFT phases,** this covers all 8 combinations of DWT, FFT and window attention.

**What window attention changes** (`--attn window`, `util/transformer.WindowAttention`):
- **Square windows instead of strips.** ESRT's feature split groups the tokens into 4 horizontal strips, so a token never sees its vertical neighbours in another strip. Here tokens attend inside 24 × 24 spatial windows, Swin-style.
- **Shift inside the single block.** ESRT has only one transformer block, so instead of alternating blocks, half the heads use windows shifted by 12, which links neighbouring windows.
- **Masking and position bias.** Odd image sizes are handled by masking padded tokens, and there is a learned relative-position bias.
- **Same layers and the same training cost.** On a 48 × 48 patch that's 4 groups of 576 tokens, like the strips. The only extra parameters are the 17,672 of the position bias, about 2.6 %.
- **Cheaper at test time.** The cost is linear in image size, while ESRT's strips grow with the image.

- **Preview first:** `DRY_RUN=1 scripts/run_fft_experiments.sh <phase>` (or `DRY_RUN=1 scripts/run_window_experiments.sh`) prints every command and the GPU assignment without running anything.
- **Progress:** `tail -F experiment/run_fft_gpu*/worker.log` (window runs: `experiment/run_window_gpu*/worker.log`) (per config: `experiment/run_fft_gpu<N>/exp_<config>/train.log`).
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

`experiment/`, `results/`, `downloads/` and the dataset folders' contents are git-ignored, so intermediate checkpoints, SR images and datasets stay on the server.

## Reporting notes
- Proposal table rows: B1/B2 = `bilinear`/`bicubic`, V0 = `lowpass`, V1 = `lowpass_fl<λ>`, V2 = `dwt_*`, V3 = `dwt_*_fl<λ>`, V4 = `<DWT_CONFIG>_fl<λ>_win`. The `lowpass_win` ablation is extra.
- Bicubic reproduces the literature values here: Set5 33.64 dB (paper 33.66), Set14 30.31 (30.24), BSDS300 test 29.55 (BSD100: 29.56). This is the proposal's sanity check.
- Set5 and Set14 are also the validation sets during training. Checkpoints are never selected by validation score (always epoch 1000), so testing on them is fine, but mention it.
- The DWT differences are 0.01–0.04 dB from a single seed, below the `--fast`/bf16 effect (0.02–0.1 dB, see COMMANDS.md). Treat FFT gains the same way. If they are this small, call them "no measurable difference".
- The loss is computed on RGB because the model is RGB-only. PSNR/SSIM are on Y, like everything else.
