"""Frequency-oriented SR metrics (beyond PSNR/SSIM) for validation and testing.

All numpy metrics work on the quantised Y (luma) channel in [0, 255], the same image PSNR/SSIM use:

  sb_<wavelet>_<LL|LH|HL|HH>  per-subband PSNR (dB, higher is better) of a 1-level DWT of SR vs HR.
                              Wavelets default to haar, db2, bior2.2; scoring with wavelets other than
                              the model's own Haar guards against a Haar-specific advantage. pywt naming:
                              LH = horizontal detail (cH), HL = vertical detail (cV), HH = diagonal (cD).
  sbrel_<wavelet>_<band>      same subbands as relative error ||SR-HR||^2 / ||HR||^2 within the band
                              (lower is better; 1 = the band is as wrong as having nothing). Unlike band
                              PSNR it is not dominated by the small absolute energy of the HH band.
  psd_r0..psd_r7              SR/HR ratio of radially averaged power spectrum in 8 bands from DC (r0) to
                              Nyquist (r7); 1 is ideal, < 1 means that band is smoothed away.
  psd_hf_ratio                SR/HR power ratio over the upper half of the frequency range (r4..r7).
  psd_lsd                     log-spectral distance over the 8 bands (dB, lower is better).
  hfen                        high-frequency error norm: ||LoG(sr) - LoG(hr)|| / ||LoG(hr)||, sigma 1.5
                              (lower is better).
  gmsd                        gradient magnitude similarity deviation (Xue et al. 2014, lower is better).
  lpips, dists                perceptual distances on RGB (lower is better); need the optional packages
                              `lpips` and `DISTS_pytorch` (pretrained weights download on first use).
                              A missing package or failed weight download disables that metric with a
                              warning instead of failing the run.
"""
import csv
import functools
import math
import os

import numpy as np
import pywt
from scipy import ndimage

import utils

DEFAULT_WAVELETS = "haar,db2,bior2.2"
BANDS = ("LL", "LH", "HL", "HH")
N_RADIAL = 8
PSNR_CAP = 100.0  # reported instead of inf when a band matches exactly


def add_args(parser):
    """Add --hf_metrics/--no-hf_metrics, --perceptual and --wavelets (shared by train.py and test.py)."""
    import argparse
    parser.add_argument("--hf_metrics", action=argparse.BooleanOptionalAction, default=True,
                        help="also compute frequency metrics (subband PSNR, spectrum ratio, HFEN, GMSD, "
                             "LPIPS/DISTS) at every validation / test")
    parser.add_argument("--perceptual", choices=["both", "lpips", "dists", "none"], default="both",
                        help="perceptual metrics to include with --hf_metrics")
    parser.add_argument("--wavelets", type=str, default=DEFAULT_WAVELETS,
                        help="comma-separated PyWavelets names for the per-subband error")


def parse_wavelets(text):
    """'haar,db2' -> ['haar', 'db2']; raises ValueError for names pywt does not know."""
    names = [t.strip() for t in text.split(",") if t.strip()]
    if not names:
        raise ValueError("--wavelets is empty")
    for n in names:
        if n not in pywt.wavelist(kind="discrete"):
            raise ValueError("unknown wavelet {!r} (see pywt.wavelist(kind='discrete'))".format(n))
    return names


def wname(w):
    """Key-safe wavelet name: 'bior2.2' -> 'bior22'."""
    return w.replace(".", "")


def _psnr(mse, peak=255.0):
    return PSNR_CAP if mse < 1e-10 else min(PSNR_CAP, 10.0 * math.log10(peak * peak / mse))


def subband_errors(sr, hr, wavelets):
    """Per-subband PSNR of a 1-level DWT (periodization mode, so haar/db2 are orthonormal)."""
    out = {}
    sr, hr = sr.astype(np.float64), hr.astype(np.float64)
    for w in wavelets:
        a = pywt.dwt2(sr, w, mode="periodization")
        b = pywt.dwt2(hr, w, mode="periodization")
        for band, ca, cb in zip(BANDS, (a[0],) + tuple(a[1]), (b[0],) + tuple(b[1])):
            sse = float(np.sum((ca - cb) ** 2))
            out["sb_{}_{}".format(wname(w), band)] = _psnr(sse / ca.size)
            out["sbrel_{}_{}".format(wname(w), band)] = sse / max(float(np.sum(cb ** 2)), 1e-12)
    return out


@functools.lru_cache(maxsize=16)
def _radial_bins(h, w):
    yy, xx = np.indices((h, w))
    r = np.hypot((yy - h // 2) / (h / 2.0), (xx - w // 2) / (w / 2.0))  # 1.0 = Nyquist on the axes
    bins = np.floor(r * N_RADIAL).astype(np.int64)
    bins[bins >= N_RADIAL] = -1  # corners beyond Nyquist are ignored
    return bins


def radial_power(img):
    """Mean power in N_RADIAL radial frequency bands (Hann-windowed 2-D FFT, DC removed)."""
    img = img.astype(np.float64)
    h, w = img.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    p = np.abs(np.fft.fftshift(np.fft.fft2((img - img.mean()) * win))) ** 2
    bins = _radial_bins(h, w)
    keep = bins >= 0
    cnt = np.bincount(bins[keep], minlength=N_RADIAL)
    return np.bincount(bins[keep], weights=p[keep], minlength=N_RADIAL) / np.maximum(cnt, 1)


def spectrum_metrics(sr, hr, eps=1e-12):
    """Radial power-spectrum ratio per band, upper-half ratio and log-spectral distance."""
    ps, ph = radial_power(sr) + eps, radial_power(hr) + eps
    out = {"psd_r{}".format(k): float(ps[k] / ph[k]) for k in range(N_RADIAL)}
    half = N_RADIAL // 2
    out["psd_hf_ratio"] = float(ps[half:].sum() / ph[half:].sum())
    out["psd_lsd"] = float(np.sqrt(np.mean((10.0 * np.log10(ps / ph)) ** 2)))
    return out


def hfen(sr, hr, sigma=1.5):
    """High-frequency error norm: relative L2 error of the Laplacian-of-Gaussian responses."""
    ls = ndimage.gaussian_laplace(sr.astype(np.float64), sigma)
    lh = ndimage.gaussian_laplace(hr.astype(np.float64), sigma)
    return float(np.linalg.norm(ls - lh) / max(np.linalg.norm(lh), 1e-12))


_PREWITT = np.array([[1, 0, -1]] * 3, dtype=np.float64) / 3.0


def gmsd(sr, hr, c=170.0):
    """Gradient magnitude similarity deviation (2x2 average-pooled, Prewitt gradients, c=170)."""
    def grad_mag(x):
        x = ndimage.uniform_filter(x.astype(np.float64), size=2)[::2, ::2]
        gx = ndimage.convolve(x, _PREWITT, mode="nearest")
        gy = ndimage.convolve(x, _PREWITT.T, mode="nearest")
        return np.hypot(gx, gy)
    m1, m2 = grad_mag(sr), grad_mag(hr)
    gms = (2 * m1 * m2 + c) / (m1 ** 2 + m2 ** 2 + c)
    return float(np.std(gms))


class Perceptual:
    """Lazy LPIPS (alex) / DISTS evaluator on uint8 RGB images; unavailable metrics are dropped with a warning."""

    def __init__(self, which="both", device="cpu"):
        self.device, self.models = device, {}
        if which in ("both", "lpips"):
            self._load("lpips", self._make_lpips)
        if which in ("both", "dists"):
            self._load("dists", self._make_dists)

    @staticmethod
    def _make_lpips():
        import lpips
        return lpips.LPIPS(net="alex", verbose=False)

    @staticmethod
    def _make_dists():
        from DISTS_pytorch import DISTS
        return DISTS()

    def _load(self, name, make):
        try:
            m = make().to(self.device).eval()
            for p in m.parameters():
                p.requires_grad_(False)
            self.models[name] = m
        except Exception as e:  # missing package or no network for the pretrained weights
            print("[metrics_freq] {} disabled: {}".format(name, e))

    def __call__(self, sr_rgb, hr_rgb):
        import torch
        if not self.models:
            return {}

        def to_t(a):
            return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None].float().div(255.0).to(self.device)
        sr, hr = to_t(sr_rgb), to_t(hr_rgb)
        out = {}
        with torch.no_grad():
            if "lpips" in self.models:
                out["lpips"] = float(self.models["lpips"](sr, hr, normalize=True))
            if "dists" in self.models:
                out["dists"] = float(self.models["dists"](sr, hr))
        return out


class FreqEvaluator:
    """Computes every frequency metric for one SR/HR pair of (shaved) uint8 RGB images."""

    def __init__(self, wavelets=None, perceptual="both", device="cpu"):
        self.wavelets = list(wavelets or parse_wavelets(DEFAULT_WAVELETS))
        self.perceptual = Perceptual(perceptual, device) if perceptual != "none" else None

    @classmethod
    def from_args(cls, args, device="cpu"):
        return cls(parse_wavelets(args.wavelets), args.perceptual, device)

    def __call__(self, sr_rgb, hr_rgb):
        import skimage.color as sc
        sr = utils.quantize(sc.rgb2ycbcr(sr_rgb)[:, :, 0])
        hr = utils.quantize(sc.rgb2ycbcr(hr_rgb)[:, :, 0])
        out = subband_errors(sr, hr, self.wavelets)
        out.update(spectrum_metrics(sr, hr))
        out["hfen"] = hfen(sr, hr)
        out["gmsd"] = gmsd(sr, hr)
        if self.perceptual is not None:
            out.update(self.perceptual(sr_rgb, hr_rgb))
        return out


def mean_metrics(rows):
    """Mean of every numeric key over a list of per-image metric dicts."""
    keys = [k for k in rows[0] if k != "image"] if rows else []
    return {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}


def summary_line(m, wavelets=None):
    """Compact one-line log summary of a mean-metrics dict (full dict goes to history.json / CSV)."""
    parts = ["sbHH({}) {:.2f}".format(w, m["sb_{}_HH".format(wname(w))])
             for w in (wavelets or ()) if "sb_{}_HH".format(wname(w)) in m]
    for key, fmt in (("psd_hf_ratio", "{:.3f}"), ("psd_lsd", "{:.2f}"), ("hfen", "{:.4f}"),
                     ("gmsd", "{:.4f}"), ("lpips", "{:.4f}"), ("dists", "{:.4f}")):
        if key in m:
            parts.append(("{} " + fmt).format(key, m[key]))
    return ", ".join(parts)


def write_csv(path, results):
    """results: {set: [per-image metric dicts with an 'image' key]} -> one CSV row per image."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rows = [dict(set=s, **r) for s, rs in results.items() for r in rs]
    if not rows:
        return
    keys = ["set", "image"] + sorted({k for r in rows for k in r} - {"set", "image"})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
