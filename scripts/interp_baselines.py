"""Interpolation baselines B1 (bilinear) and B2 (bicubic) from the proposal.

Upsamples each benchmark LR image to HR size and scores it exactly like test.py: HR mod-cropped to the
scale, `scale` border pixels shaved, PSNR/SSIM on the quantised Y channel (utils.compute_psnr/ssim).
No GPU or training needed.

Interpolation is PIL's (Keys cubic, a = -0.5, the kernel MATLAB's imresize uses), so bicubic on Set5 x2
should land near the 33.66 dB reported in the SR literature (the proposal's sanity check).

Results go to final_results/<method>/ in the same layout as the model runs, so
scripts/summarize_results.py lists them next to lowpass/DWT/FFT:
    final_results/bicubic/test_x2_summary.json        Set5
    final_results/bicubic/<set>/test_x2_summary.json  set14, bsds300, manga109

  python scripts/interp_baselines.py                       # bilinear + bicubic, all four sets
  python scripts/interp_baselines.py --sets set5 --methods bicubic
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import utils  # noqa: E402
from data import benchmarks  # noqa: E402

METHODS = {"bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC}
SET5_KEY = "Set5"  # test.py names the Set5 run after its folder; keep the same key


def y_channel(rgb):
    """Quantised Y of YCbCr for a uint8 RGB image (same as test.py: skimage rgb2ycbcr + utils.quantize)."""
    import skimage.color as sc
    return utils.quantize(sc.rgb2ycbcr(rgb)[:, :, 0])


def score(sr, hr, scale):
    """(psnr, ssim) on Y after shaving `scale` border pixels, as test.py does."""
    a, b = y_channel(utils.shave(sr, scale)), y_channel(utils.shave(hr, scale))
    return utils.compute_psnr(a, b), utils.compute_ssim(a, b)


def upsample(lr, size, method):
    """uint8 HxWx3 LR -> uint8 RGB of `size` (W, H) with PIL interpolation."""
    return np.asarray(Image.fromarray(lr).resize(size, METHODS[method]))


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def run_set(hr_dir, lr_dir, scale, methods):
    """{method: {"psnr": [...], "ssim": [...]}} over every image of one benchmark (sorted by name)."""
    ext, lr_ext = utils.detect_image_ext(hr_dir), utils.detect_image_ext(lr_dir)
    out = {m: {"psnr": [], "ssim": []} for m in methods}
    for path in sorted(utils.get_list(hr_dir, ext=ext)):
        stem = os.path.splitext(os.path.basename(path))[0]
        hr = utils.modcrop(load_rgb(path), scale)
        lr = load_rgb(benchmarks.find_lr(lr_dir, stem, scale, lr_ext))
        if lr.shape[0] * scale != hr.shape[0] or lr.shape[1] * scale != hr.shape[1]:
            raise ValueError("{}: LR {} x{} != mod-cropped HR {}".format(stem, lr.shape[:2], scale, hr.shape[:2]))
        for m in methods:
            p, s = score(upsample(lr, (hr.shape[1], hr.shape[0]), m), hr, scale)
            out[m]["psnr"].append(float(p))
            out[m]["ssim"].append(float(s))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sets", default="set5,set14,bsds300,manga109")
    p.add_argument("--methods", default="bilinear,bicubic")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--bench_root", default=".", help="benchmark root, relative to the repo root")
    p.add_argument("--out_root", default="final_results")
    a = p.parse_args()
    os.chdir(ROOT)  # benchmark paths such as Test_Datasets/Set5 are relative to the repo

    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    bad = set(methods) - set(METHODS)
    if bad:
        p.error("unknown method(s) {}; choose from {}".format(sorted(bad), sorted(METHODS)))
    for name in benchmarks.parse_sets(a.sets):
        try:
            hr_dir, lr_dir = benchmarks.resolve(name, a.scale, a.bench_root)
        except FileNotFoundError as e:
            print("[skip] {}".format(e))
            continue
        res = run_set(hr_dir, lr_dir, a.scale, methods)
        for m in methods:
            sub = "" if name == "set5" else name
            d = os.path.join(a.out_root, m, sub)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "test_x{}_summary.json".format(a.scale)), "w") as f:
                json.dump({SET5_KEY if name == "set5" else name: res[m]}, f)
            r = res[m]
            print("{:<9} {:<9} {:>3} images  PSNR {:.3f}  SSIM {:.4f}".format(
                m, name, len(r["psnr"]), np.mean(r["psnr"]), np.mean(r["ssim"])))
            if name == "set5" and m == "bicubic":
                print("          (literature: 33.66 / 0.9299; a small gap comes from the LR images and SSIM settings)")


if __name__ == "__main__":
    main()
