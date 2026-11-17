# AI Changes

## train.py: added a main guard
- **Problem:** on Python 3.14 Linux defaults to `forkserver`. DataLoader workers re-imported `train.py`, which re-ran the whole script and raised `RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase`.
- **Fix:**
  - Moved the setup and training loop into `main()` and added `if __name__ == '__main__': main()`.
  - Kept the imports and the argparse `parser` at module level.
  - `train`, `valid` and `save_checkpoint` now take `model`, `args` and the other objects they need as parameters instead of using globals.
  - No intended behaviour change.
- **Checked:** the file compiles and `import train` has no side effects.
- **Not run end to end:** that needs the DF2K `.npy` data and `Test_Datasets/Set5`.

## forward_chop: scale fix (train.py, test.py)
- **`test.py`:** replaced the hard-coded `scale = 4` with a `scale` parameter, and the call passes `opt.upscale_factor`. x2 and x3 models now stitch at the right size.
- **`train.py`:** the recursive call now forwards `scale`. It was missing, which would raise a `TypeError` on large images.
- **Checked:** both files compile. Not run on real data.

## logs/
- **`claude.txt`:** the original change log.
- **`changes.md`:** filled in the root vs `baseline/` differences.
- **`cite.txt`:** added BibTeX for Set5, BSD100/BSD300, Manga109 and Urban100.
- **`AIChanges.md`:** this file.

## Noticed, not changed
- `compute_ssim` uses the removed `multichannel` argument. On scikit-image >= 0.21 it needs `channel_axis`.
- `Test_Datasets/Set5` and `Set5_LR/x<scale>` must be created by hand.
