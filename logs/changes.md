## train.py
    Removed architecture from import line `from model import architecture, esrt`
    Put main training pipe in a `main()` function and called from `if __name__ == "__main__"` guard
        Otherwise, it'll try to call more subprocesses before the first one is done setting up, 
        forcing `ConnectionResetError: [Errno 104] Connection reset by peer`
    `train`, `valid` and `save_checkpoint` now take `model`, `optimizer`, `l1_criterion`, data loaders, `args` and `device` as parameters
        Previously they read these from module-level globals, which no longer exist after moving the setup into `main()`
    Added `--amp` flag (`store_true`, default False) for fp16 mixed-precision training
        Only the argument is added so far; nothing in the training loop uses it yet
    `forward_chop` now forwards `scale` in its recursive call (it was missing, which would raise a `TypeError` on large images)
    Added docstrings, a module docstring and explanatory comments throughout (no behaviour change)

## main.py
    Put `environ.get("HF_TOKEN")` on main.py:66 and appropriate import stmt.
        Passed as `token=` to `snapshot_download` so private/gated Hugging Face datasets can be downloaded
    Added docstrings and explanatory comments (no behaviour change)

## test.py
    `forward_chop` takes a `scale` parameter instead of hard-coding `scale = 4`; the call passes `opt.upscale_factor`
        The hard-coded x4 gave wrong output sizes for x2/x3 models; the recursive call forwards `scale` too
    Added docstrings and explanatory comments

## utils.py
    Replaced `from skimage.measure import compare_psnr as psnr` / `compare_ssim as ssim`
        with `from skimage.metrics import peak_signal_noise_ratio as psnr` / `structural_similarity as ssim`
        `compare_psnr`/`compare_ssim` were removed in scikit-image >= 0.18
    Added docstrings and explanatory comments

## model/dwt.py
    New file, placeholder only (docstring, no code)
        Intended for the discrete wavelet transform used to split features into frequency sub-bands in the frequency-aware variant

## util/__init__.py
    Added a module docstring describing the contents of util/

## Docstring / comment only (no code changes)
    bench.py, data/common.py, data/DIV2K.py, data/RealSR.py, data/image_folder.py, data/Set5_val.py,
    model/common.py, model/esrt.py, model/block.py, scripts/png2npy.py, FLOPs/profile.py, FLOPs/count_hooks.py,
    util/tools.py, util/transformer.py, util/nonlocal.py, util/position.py, util/non.py
