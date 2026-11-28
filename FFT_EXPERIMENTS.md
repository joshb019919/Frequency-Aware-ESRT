# Report experiments on a GPU server: runbook

One script per experiment. Each one trains exactly its configs, tests them on Set5 / Set14 / BSDS300 / Manga109, copies everything to `final_results/<config>/` (with a `run_info.json` saying how it was made) and prints the results table. Run them from the `Frequency-Aware-ESRT` folder, inside `tmux`.

| Script | Experiment | Proposal | Configs | Time |
|---|---|---|---|---|
| `scripts/run_baselines.sh` | bilinear, bicubic; Set14 tests of the committed models | B1, B2 | – | minutes (done) |
| `scripts/run_fft.sh` | FFT loss, λ = 0.01 / 0.05 / 0.1 | V1 + λ study | `lowpass_fl<λ>` | ~7 h each |
| `scripts/run_dwt_fft.sh` | DWT + FFT, best λ | V3 | `dwt_*_fl<λ>` (3 variants) | ~7 h each |
| `scripts/run_window.sh` | window attention | – | `lowpass_win` | ~7 h |
| `scripts/run_dwt_window.sh` | DWT + window | – | `dwt_cband_gr1_skipnorm_win` | ~7 h |
| `scripts/run_fft_window.sh` | FFT + window, best λ | – | `lowpass_fl<λ>_win` | ~7 h |
| `scripts/run_dwt_fft_window.sh` | DWT + FFT + window, best λ | V4 | `dwt_cband_gr1_skipnorm_fl<λ>_win` | ~7 h |

Already in `final_results/`: `lowpass` (V0) and the three DWT variants (V2).

- **λ:** the scripts with "best λ" read it from the finished FFT sweep, picking the λ with the best mean PSNR gain over lowpass. Until the sweep has finished, they stop with a message.
- **Settings:** the choices are set once, at the top of `scripts/lib_experiments.sh`: `DEFAULT_GPUS=1` (`0,1` = both GPUs), `DEFAULT_DWT_CONFIG=dwt_cband_gr1_skipnorm` and `SWEEP_LAMBDAS=0.01,0.05,0.1`.
- **Safe to re-run:** finished configs are skipped and configs still training are not started twice. A script that was interrupted continues where it stopped; a config cut off mid-training restarts from epoch 1.
- **One at a time on one GPU:** start the next script when the previous one prints its table. With `DEFAULT_GPUS=0,1`, a script with several configs trains them in parallel.
- **Preview:** `DRY_RUN=1 scripts/<script>.sh` prints what it would run, without running it.
- **Progress:** `tail -F experiment/*_gpu*/worker.log`. Each script's run folder is `experiment/run_<name>_gpu<N>/`.

## Order on the server

```bash
cd ~/Frequency-Aware-ESRT && source .venv/bin/activate && git pull
scripts/run_dwt_window.sh          # needs no λ: can run while the FFT sweep is still going
# ... after the FFT sweep has finished:
scripts/run_dwt_fft.sh
scripts/run_fft_window.sh
scripts/run_dwt_fft_window.sh
```

The FFT sweep and `lowpass_win` are already running from the older scripts, which is fine: the new scripts recognise their results. While that sweep is running, **don't start `scripts/run_fft.sh`**, because it would train λ = 0.1 a second time. Run it once after the sweep has finished: it only adds `run_info.json` to the λ results.

## Setup (once)

### Clone and create the venv

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

### Download the data (first time only, re-runnable)

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

If the data already lives somewhere else, `ROOT=/path/div2k BENCH=/path scripts/run_<name>.sh` links those benchmark folders into the repo, where training validation needs them.


## Save and push the results

```bash
.venv/bin/python scripts/summarize_results.py --csv final_results/summary.csv
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


## What the changes are

- **FFT loss** (`--freq_loss_lambda`, `losses.py`): L = ‖SR − HR‖₁ + λ · ‖FFT(SR) − FFT(HR)‖₁, with an orthonormal FFT over real and imaginary parts, computed on RGB.
- **DWT** (`--freq dwt`, `model/dwt.py`): Haar wavelet split in the high-frequency module, from Josh's experiments.
- **Window attention** (`--attn window`, `util/transformer.WindowAttention`): tokens attend inside 24 × 24 spatial windows instead of ESRT's 4 horizontal strips.
  - Half the heads use windows shifted by 12, since ESRT has only one transformer block.
  - It uses the same layers as before, plus a relative-position bias (+17,672 params, about 2.6 %).
  - Training cost is the same, and it is cheaper on large test images.

## Reporting notes
- Proposal rows: B1/B2 = `bilinear`/`bicubic`, V0 = `lowpass`, V1 = `lowpass_fl<λ>`, V2 = `dwt_*`, V3 = `dwt_*_fl<λ>`, V4 = `dwt_cband_gr1_skipnorm_fl<λ>_win`. The other window rows complete the DWT × FFT × window ablation.
- Bicubic reproduces the literature values here: Set5 33.64 dB (paper 33.66), Set14 30.31 (30.24), BSDS300 test 29.55 (BSD100: 29.56).
- Set5 and Set14 are also the validation sets during training. Checkpoints are never selected by validation score (always epoch 1000), so testing on them is fine, but mention it.
- The DWT differences are 0.01–0.04 dB from a single seed, below the `--fast`/bf16 effect (0.02–0.1 dB). Judge every other change the same way; differences this small are "no measurable difference".
