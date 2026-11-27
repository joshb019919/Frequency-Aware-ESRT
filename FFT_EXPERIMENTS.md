# Frequency-loss (FFT) experiments: GPU server runbook

These runs give the FFT-loss rows (V1 and V3 in the proposal) to go next to the DWT results already in `final_results/`.
The settings match the DWT runs (×2, patch 96, 1000 epochs, seed 1, batch 16, `--fast`, bf16). Only the loss changes:

L = ‖SR − HR‖₁ + λ · ‖FFT(SR) − FFT(HR)‖₁ (`losses.py`, `--freq_loss_lambda`)

| Phase | Configs | Proposal | Runs (≈7 h each) |
|---|---|---|---|
| — | `lowpass`, `dwt_c1_gr1_norm`, `dwt_c3_gr1_skipnorm`, `dwt_cband_gr1_skipnorm` | V0, V2 | already in `final_results/` |
| `sweep` | `lowpass_fl0.01`, `lowpass_fl0.05`, `lowpass_fl0.1` | V1 + λ study | 3 |
| `dwt` | the three DWT configs at the best λ, e.g. `dwt_c3_gr1_skipnorm_fl0.05` | V3 | 3 |

The finished `final_results/lowpass` is reused as the baseline and the plot reference. It is not retrained.

## 1. Clone and install

```bash
git clone -b dev/fft https://github.com/joshb019919/Frequency-Aware-ESRT.git
cd Frequency-Aware-ESRT
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # match the server's CUDA
pip install -r requirements.txt
```

## 2. Data

All paths are the defaults. If yours differ, pass `--root` / `--bench_root` to `check_setup.py` and set `ROOT=` for the run.

| What | Where |
|---|---|
| DIV2K train, `.npy` (`scripts/png2npy.py`) | `/data/data/div2k/DF2K_decoded/DIV2K_train_HR/`, `.../DIV2K_train_LR_bicubic/X2/` |
| Set5 (validation + experiments.py test) | `Test_Datasets/Set5/`, `Test_Datasets/Set5_LR/x2/` (in the repo folder) |
| Set14 (validation) | `/data/data/set14/Set14_HR/`, `/data/data/set14/Set14_LR_x2/` |
| BSDS300 test split, Manga109 | `/data/data/bsds300/`, `/data/data/manga109/`, then `python scripts/prepare_benchmarks.py` |

**Manga109:** the DWT results used the official 109-image list. That needs `scripts/manga109_official.json`, which is **not in the repo**. Copy it from the machine that produced the DWT results, or copy the prepared `/data/data/manga109/HR` and `LR/x2` folders. Without it, the Manga109 numbers are not comparable.

If you're on the same server as the DWT runs, this is all already in place.

## 3. Preflight

```bash
python scripts/check_setup.py              # must end with "all required checks passed"
DRY_RUN=1 scripts/run_fft_experiments.sh sweep   # prints every command, runs nothing
```

## 4. Run

Use `tmux` or `nohup`, since each phase takes about 21 h on one GPU.

```bash
tmux new -s fft
scripts/run_fft_experiments.sh sweep 2>&1 | tee fft_sweep.log
```

Pick the λ with the best PSNR (averaged over Set5, BSDS300 and Manga109) from the table printed at the end, then:

```bash
LAMBDA=0.05 scripts/run_fft_experiments.sh dwt 2>&1 | tee fft_dwt.log
```

Re-running the same command resumes: finished configs are skipped. Logs while training: `tail -f experiment/run_fft/exp_<config>/train.log`.

The script, for each config:
1. trains and tests on Set5 (`experiments.py`; summary CSVs go to `experiment/experiments_run_fft*.csv`);
2. tests on BSDS300 and Manga109, with lowpass as the dashed reference line;
3. copies everything to `final_results/<config>/` in the same layout as the DWT folders;
4. prints the combined DWT + FFT table (`python scripts/summarize_results.py --csv final_results/summary.csv` writes it as a CSV).

Optional env vars: `RUN` (run id, default `run_fft`), `ROOT`, `PY`, `EPOCHS`, `TEST_FAST=--no-fast` (fp32 testing; then re-test lowpass the same way for a fair comparison).

## 5. Bring the results back

```bash
cp experiment/experiments_run_fft*.csv final_results/     # optional: per-run summary CSVs
git add final_results && git commit -m "Add FFT-loss experiment results" && git push
```

`experiment/` and `results/` are git-ignored, so intermediate checkpoints and SR images stay on the server.

## Reporting notes
- The DWT differences are 0.01–0.04 dB from a single seed, below the `--fast`/bf16 effect (0.02–0.1 dB, see COMMANDS.md). Report FFT gains on the same footing. If they are this small, say "no measurable difference" unless they're confirmed with more seeds or `--no-fast` tests.
- The loss is computed on RGB because the model is RGB-only. PSNR/SSIM are on Y, as for the DWT runs.
