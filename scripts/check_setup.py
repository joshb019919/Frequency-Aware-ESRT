"""Preflight for the FFT experiments on a GPU server: run this before scripts/run_fft_experiments.sh.

Checks CUDA, the Python packages, the DIV2K training data, the benchmark sets (with the image counts
the DWT results were produced with) and the lowpass baseline in final_results/. Prints one line per
check and exits 1 if anything required is missing.

  python scripts/check_setup.py                       # data inside the repo (the default)
  python scripts/check_setup.py --root /elsewhere/div2k --bench_root /elsewhere
"""
import argparse
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PACKAGES = ["numpy", "scipy", "skimage", "cv2", "PIL", "matplotlib", "pywt", "thop"]
OPTIONAL = {"lpips": "LPIPS metric dropped", "DISTS_pytorch": "DISTS metric dropped"}
# Benchmark image counts used for the committed DWT results (Manga109 = official 109-image list).
EXPECTED = {"set5": 5, "set14": 14, "bsds300": 100, "manga109": 109}
IMG = (".png", ".bmp", ".jpg", ".npy")

failures = []


def report(ok, msg, required=True):
    print("[{}] {}".format("ok" if ok else ("FAIL" if required else "warn"), msg))
    if not ok and required:
        failures.append(msg)


def count(d):
    return len([f for f in os.listdir(d) if f.lower().endswith(IMG)]) if os.path.isdir(d) else 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="div2k", help="train.py --root (holds DF2K_decoded/); relative to the repo root")
    p.add_argument("--bench_root", default=".", help="benchmark root (set14/, bsds300/, manga109/)")
    p.add_argument("--scale", type=int, default=2)
    a = p.parse_args()
    os.chdir(ROOT)

    try:
        import torch
        cuda = torch.cuda.is_available()
        report(cuda, "torch {} CUDA {}{}".format(torch.__version__, torch.version.cuda,
                                                 " - " + torch.cuda.get_device_name(0) if cuda else " not available"))
        if cuda:
            report(torch.cuda.is_bf16_supported(), "bf16 supported (default AMP); otherwise pass --no_amp", False)
    except ImportError:
        report(False, "torch not installed")
    for m in PACKAGES:
        try:
            importlib.import_module(m)
            report(True, "import " + m)
        except ImportError:
            report(False, "import {} (pip install -r requirements.txt)".format(m))
    for m, why in OPTIONAL.items():
        try:
            importlib.import_module(m)
            report(True, "import " + m)
        except ImportError:
            report(False, "import {} missing: {} (the DWT results include it)".format(m, why), False)

    d = os.path.join(a.root, "DF2K_decoded")
    hr, lr = os.path.join(d, "DIV2K_train_HR"), os.path.join(d, "DIV2K_train_LR_bicubic", "X{}".format(a.scale))
    for path in (hr, lr):
        n = count(path)
        report(n >= 800, "{}: {} images (need 800 .npy; see scripts/png2npy.py)".format(path, n))

    from data import benchmarks
    for name, want in EXPECTED.items():
        try:
            h, l = benchmarks.resolve(name, a.scale, a.bench_root)
            nh, nl = count(h), count(l)
            report(nh == nl == want, "{}: HR {} / LR {} images in {} (expected {})".format(name, nh, nl, h, want))
        except FileNotFoundError as e:
            hint = ""
            if name == "manga109" and not os.path.exists("scripts/manga109_official.json"):
                hint = " - scripts/manga109_official.json is not in the repo; copy it from the machine that made the DWT results"
            report(False, str(e) + hint)

    ref = "final_results/lowpass"
    need = ["epoch_1000.pth", "history.json", "test_x{}_summary.json".format(a.scale),
            "bsds300/test_x{}_summary.json".format(a.scale), "manga109/test_x{}_summary.json".format(a.scale)]
    missing = [f for f in need if not os.path.exists(os.path.join(ref, f))]
    report(not missing, "lowpass baseline in {}{}".format(ref, "" if not missing else ": missing " + ", ".join(missing)))

    print("\n{} problem(s)".format(len(failures)) if failures else "\nall required checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
