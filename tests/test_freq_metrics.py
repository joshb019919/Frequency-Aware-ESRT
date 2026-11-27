"""Tests for metrics_freq (subband / spectrum / HFEN / GMSD / perceptual) and its plot + train/test hooks."""
import argparse
import csv
import json
import os
import sys

import numpy as np
import pytest
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import metrics_freq as M  # noqa: E402
import plotting  # noqa: E402
from data import benchmarks  # noqa: E402


def _texture(seed=0, size=(96, 80)):
    """Sharp random-ish test image (uint8, HxW) with edges and fine texture."""
    rng = np.random.RandomState(seed)
    img = ndimage.gaussian_filter(rng.rand(*size), 0.8) * 255
    img[20:60, 30:50] = 255 - img[20:60, 30:50]
    return img.clip(0, 255).round().astype(np.uint8)


def _blur(img, sigma):
    return ndimage.gaussian_filter(img.astype(float), sigma).round().astype(np.uint8)


def _rgb(y):
    return np.stack([y, y, y], -1)


def test_parse_wavelets():
    assert M.parse_wavelets("haar, db2 ,bior2.2") == ["haar", "db2", "bior2.2"]
    assert M.wname("bior2.2") == "bior22"
    for bad in ("nope", ""):
        with pytest.raises(ValueError):
            M.parse_wavelets(bad)


def test_identical_images_are_perfect():
    y = _texture()
    sb = M.subband_errors(y, y, ["haar", "db2", "bior2.2"])
    assert len([k for k in sb if k.startswith("sb_")]) == 12 and len([k for k in sb if k.startswith("sbrel_")]) == 12
    assert all(v == M.PSNR_CAP for k, v in sb.items() if k.startswith("sb_"))
    assert all(v == 0 for k, v in sb.items() if k.startswith("sbrel_"))
    sp = M.spectrum_metrics(y, y)
    assert sp["psd_hf_ratio"] == pytest.approx(1.0) and sp["psd_lsd"] == pytest.approx(0.0, abs=1e-9)
    assert all(sp["psd_r%d" % k] == pytest.approx(1.0) for k in range(M.N_RADIAL))
    assert M.hfen(y, y) == 0 and M.gmsd(y, y) == pytest.approx(0.0, abs=1e-12)


def test_haar_subbands_are_orthonormal_parseval():
    """Orthonormal Haar: the per-band squared errors add up to the pixel-domain squared error."""
    a, b = _texture(1), _texture(2)
    pix = float(np.sum((a.astype(float) - b) ** 2))
    n = a.size / 4
    total = sum(10 ** (-M.subband_errors(a, b, ["haar"])["sb_haar_%s" % bd] / 10) * 255 ** 2 * n for bd in M.BANDS)
    assert total == pytest.approx(pix, rel=1e-6)


def test_blur_lowers_hf_content_and_is_monotonic():
    hr = _texture()
    mild, strong = _blur(hr, 0.6), _blur(hr, 1.5)
    r1 = {**M.subband_errors(mild, hr, ["haar"]), **M.spectrum_metrics(mild, hr)}
    r2 = {**M.subband_errors(strong, hr, ["haar"]), **M.spectrum_metrics(strong, hr)}
    for r in (r1, r2):
        assert r["psd_hf_ratio"] < 1 and r["sbrel_haar_HH"] > r["sbrel_haar_LL"]  # HF is lost more than LL
    assert r2["psd_hf_ratio"] < r1["psd_hf_ratio"] and r2["psd_lsd"] > r1["psd_lsd"]
    assert r2["sbrel_haar_HH"] > r1["sbrel_haar_HH"] and r2["sb_haar_HH"] < r1["sb_haar_HH"]
    assert 0 < M.hfen(mild, hr) < M.hfen(strong, hr)
    assert 0 < M.gmsd(mild, hr) < M.gmsd(strong, hr)


def test_oversharpened_output_gives_hf_ratio_above_one():
    hr = _blur(_texture(), 1.0)
    sharp = (hr.astype(float) + 2.0 * (hr - ndimage.gaussian_filter(hr.astype(float), 1.5))).clip(0, 255).astype(np.uint8)
    assert M.spectrum_metrics(sharp, hr)["psd_hf_ratio"] > 1


def test_odd_sizes_and_shapes_work():
    y = _texture(size=(51, 67))
    out = {**M.subband_errors(_blur(y, 1), y, ["db2", "bior2.2"]), **M.spectrum_metrics(_blur(y, 1), y)}
    assert all(np.isfinite(v) for v in out.values())
    assert np.isfinite(M.hfen(_blur(y, 1), y)) and np.isfinite(M.gmsd(_blur(y, 1), y))


def test_freq_evaluator_without_perceptual_and_mean_summary():
    hr = _texture()
    ev = M.FreqEvaluator(["haar", "db2"], perceptual="none")
    r1, r2 = ev(_rgb(_blur(hr, 0.6)), _rgb(hr)), ev(_rgb(_blur(hr, 1.2)), _rgb(hr))
    assert "lpips" not in r1 and "hfen" in r1 and "sb_db2_HH" in r1 and "sb_bior22_HH" not in r1
    mean = M.mean_metrics([r1, r2])
    assert mean["hfen"] == pytest.approx((r1["hfen"] + r2["hfen"]) / 2)
    line = M.summary_line(mean, ["haar", "db2"])
    assert "sbHH(haar)" in line and "hfen" in line and "lpips" not in line


def test_perceptual_missing_package_is_dropped_with_warning(monkeypatch, capsys):
    monkeypatch.setattr(M.Perceptual, "_make_lpips", staticmethod(lambda: (_ for _ in ()).throw(ImportError("no lpips"))))
    monkeypatch.setattr(M.Perceptual, "_make_dists", staticmethod(lambda: (_ for _ in ()).throw(ImportError("no dists"))))
    p = M.Perceptual("both", "cpu")
    assert p.models == {} and p(np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8, 3), np.uint8)) == {}
    out = capsys.readouterr().out
    assert "lpips disabled" in out and "dists disabled" in out


def test_perceptual_real_models_when_available():
    pytest.importorskip("lpips")
    pytest.importorskip("DISTS_pytorch")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = M.Perceptual("both", dev)
    if set(p.models) != {"lpips", "dists"}:
        pytest.skip("pretrained weights unavailable")
    x = (np.random.RandomState(0).rand(64, 64, 3) * 255).astype(np.uint8)
    same, diff = p(x, x), p(_rgb(_blur(x[..., 0], 2)), x)
    assert same["lpips"] == pytest.approx(0, abs=1e-5) and same["dists"] == pytest.approx(0, abs=1e-4)
    assert diff["lpips"] > 0 and diff["dists"] > 0
    assert not any(q.grad is not None for m in p.models.values() for q in m.parameters())  # no backprop state


def _rows(n=3, sets=("a",)):
    ev = M.FreqEvaluator(["haar", "db2"], perceptual="none")
    hr = _texture()
    return {s: [dict(image="im%d" % i, **ev(_rgb(_blur(hr, 0.5 + 0.3 * i)), _rgb(hr))) for i in range(n)] for s in sets}


def test_write_csv(tmp_path):
    res = _rows(sets=("bsds300", "manga109"))
    p = str(tmp_path / "x/f.csv")
    M.write_csv(p, res)
    rows = list(csv.DictReader(open(p)))
    assert len(rows) == 6 and {r["set"] for r in rows} == {"bsds300", "manga109"} and "hfen" in rows[0]


def test_plot_test_freq(tmp_path):
    p = str(tmp_path / "t_freq.png")
    assert plotting.plot_test_freq(p, _rows(sets=("a", "b"))) is True and os.path.getsize(p) > 1000


def test_trainval_plotter_freq_curves_and_history(tmp_path):
    pl = plotting.TrainValPlotter(str(tmp_path))
    rows = _rows(sets=("set14", "set5"))
    for e in (0, 5, 10):
        pl.add_val(e, {"set14": (30.0, 0.9), "set5": (31.0, 0.91)})
        pl.add_freq(e, {n: M.mean_metrics(r) for n, r in rows.items()})
    pl.save()
    assert os.path.getsize(tmp_path / "freq_curves.png") > 1000
    h = json.load(open(tmp_path / "history.json"))
    assert set(h["freq"]) == {"set14", "set5"} and set(h["freq"]["set5"]) == {"0", "5", "10"}


def test_no_freq_data_means_no_freq_plot(tmp_path):
    pl = plotting.TrainValPlotter(str(tmp_path))
    pl.add_val(0, {"s": (1.0, 0.1)})
    pl.save()
    assert os.path.exists(tmp_path / "curves.png") and not os.path.exists(tmp_path / "freq_curves.png")


def test_valid_fills_extras_and_prints_freq_line(tmp_path, capsys, monkeypatch):
    import train
    from PIL import Image
    monkeypatch.setattr(torch, "inference_mode", torch.no_grad)  # CPU tensor2np quirk, see test_plotting_benchmarks
    for i in range(2):
        rs = np.random.RandomState(i)
        Image.fromarray((rs.rand(96, 96, 3) * 255).astype(np.uint8)).save(str(tmp_path / "HR_{}.png".format(i)))
    (tmp_path / "HR").mkdir()
    (tmp_path / "LR").mkdir()
    for i in range(2):
        os.rename(str(tmp_path / "HR_{}.png".format(i)), str(tmp_path / "HR" / "im{}.png".format(i)))
        Image.fromarray((np.random.RandomState(i).rand(48, 48, 3) * 255).astype(np.uint8)).save(
            str(tmp_path / "LR" / "im{}x2.png".format(i)))

    class Up(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.interpolate(x, scale_factor=2)

    ld = DataLoader(benchmarks.PairedVal(str(tmp_path / "HR"), str(tmp_path / "LR"), 2), batch_size=1)
    ev = M.FreqEvaluator(["haar"], perceptual="none")
    extras = {}
    res = train.valid(2, Up(), {"s": ld}, argparse.Namespace(cuda=False, scale=2, isY=True), torch.device("cpu"), 3, ev, extras)
    out = capsys.readouterr().out
    assert "===> Freq[s] sbHH(haar)" in out and "(weights after 3 epochs)" in out
    assert set(extras) == {"s"} and "hfen" in extras["s"] and set(res) == {"s"}
    # without an evaluator nothing extra is printed or stored
    extras2 = {}
    train.valid(2, Up(), {"s": ld}, argparse.Namespace(cuda=False, scale=2, isY=True), torch.device("cpu"), 3, None, extras2)
    assert extras2 == {} and "Freq[" not in capsys.readouterr().out
