"""Tests for benchmark registry, dataset prep, plotting and the multi-set valid()/train() hooks."""
import argparse
import json
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from data import benchmarks  # noqa: E402
import plotting  # noqa: E402
import prepare_benchmarks as prep  # noqa: E402


def _img(path, size=(48, 40), seed=0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.random.RandomState(seed).randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_parse_sets():
    assert benchmarks.parse_sets("set14,set5") == ["set14", "set5"]
    assert benchmarks.parse_sets(" Set5 , set5,b100 ") == ["set5", "b100"]
    with pytest.raises(ValueError):
        benchmarks.parse_sets("set5,nope")
    assert benchmarks.DEFAULT_VAL_SETS == "set14,set5" and benchmarks.DEFAULT_TEST_SETS == "bsds300,manga109"


def test_resolve_and_missing(tmp_path):
    (tmp_path / "bsds300/HR").mkdir(parents=True)
    (tmp_path / "bsds300/LR/x2").mkdir(parents=True)
    hr, lr = benchmarks.resolve("bsds300", 2, str(tmp_path))
    assert hr.endswith("bsds300/HR") and lr.endswith("bsds300/LR/x2")
    with pytest.raises(FileNotFoundError, match="prepare_benchmarks"):
        benchmarks.resolve("manga109", 2, str(tmp_path))


def test_paired_val_matches_by_name_for_both_conventions(tmp_path):
    for hr_name, lr_name in (("a", "ax2"), ("b", "bx2")):  # Set5/B100 style
        _img(str(tmp_path / "s5/HR" / (hr_name + ".png")))
        _img(str(tmp_path / "s5/LR" / (lr_name + ".png")), (24, 20))
    for n in ("a", "b"):  # Set14 style: LR has the same name as HR
        _img(str(tmp_path / "s14/HR" / (n + ".png")))
        _img(str(tmp_path / "s14/LR" / (n + ".png")), (24, 20))
    for d in ("s5", "s14"):
        ds = benchmarks.PairedVal(str(tmp_path / d / "HR"), str(tmp_path / d / "LR"), 2)
        assert len(ds) == 2
        lr, hr = ds[0]
        assert lr.shape == (3, 20, 24) and hr.shape == (3, 40, 48) and 0 <= float(hr.max()) <= 1
    with pytest.raises(FileNotFoundError):
        os.remove(str(tmp_path / "s14/LR/b.png"))
        benchmarks.PairedVal(str(tmp_path / "s14/HR"), str(tmp_path / "s14/LR"), 2)


def test_find_lr_both_conventions(tmp_path):
    _img(str(tmp_path / "ax2.png"))
    _img(str(tmp_path / "b.png"))
    assert benchmarks.find_lr(str(tmp_path), "a", 2, ".png").endswith("ax2.png")
    assert benchmarks.find_lr(str(tmp_path), "b", 2, ".png").endswith("b.png")


def test_prepare_benchmarks_write_pair_sizes_and_idempotent(tmp_path):
    src = Image.fromarray(np.zeros((41, 53, 3), np.uint8))  # odd sizes force modcrop
    n = prep.write_pair(src, "im", str(tmp_path), (2, 3, 4), False)
    assert n == 4
    assert Image.open(tmp_path / "HR/im.png").size == (53, 41)  # HR is stored untouched
    for s in (2, 3, 4):
        assert Image.open(tmp_path / "LR/x{}/imx{}.png".format(s, s)).size == (53 // s, 41 // s)
    assert prep.write_pair(src, "im", str(tmp_path), (2, 3, 4), False) == 0  # never rewrites existing files


def test_manga109_middle_items_picks_middle_page(tmp_path):
    for book, pages in (("A", 5), ("B", 4)):
        for p in range(pages):
            _img(str(tmp_path / "images" / book / "{:03d}.jpg".format(p)), (8, 8))
    items = {n: path for n, path, box in prep.manga109_middle_items(str(tmp_path))}
    assert set(items) == {"A_002", "B_002"}


def test_manga109_official_mapping_and_crops(tmp_path):
    m = json.load(open(os.path.join(ROOT, "scripts", "manga109_official.json")))
    assert len(m) == 109 and all(e["page"] == "000.jpg" and e["side"] in "LR" for e in m.values())
    assert "ARMS" in m and "ThatsIzumiko_000" in m  # names follow the upstream outputs
    assert [k for k, e in m.items() if e["side"] == "L"] == ["LoveHina_vol01"]
    items = {n: (path, box) for n, path, box in prep.manga109_official_items(str(tmp_path))}
    assert len(items) == 109
    assert items["ARMS"][1] == (827, 0, 827 + 826, 1170) and items["LoveHina_vol01"][1] == (0, 0, 826, 1170)
    assert items["ARMS"][0].endswith("images/ARMS/000.jpg")
    for _, box in items.values():
        assert box[2] - box[0] == 826 and box[3] - box[1] == 1170  # official size, 1654-wide spreads


def test_train_val_plotter_writes_files(tmp_path):
    pl = plotting.TrainValPlotter(str(tmp_path / "out"), title="t")
    for e in range(1, 4):
        pl.add_loss(e, 0.1 / e)
    pl.add_val(0, {"set14": (10.0, 0.1), "set5": (11.0, 0.2)})
    pl.add_val(2, {"set14": (30.0, 0.9), "set5": (31.0, 0.91)})
    pl.save()
    assert os.path.getsize(tmp_path / "out/curves.png") > 1000
    h = json.load(open(tmp_path / "out/history.json"))
    assert set(h["val"]) == {"set14", "set5"} and len(h["loss"]) == 3


def test_plotter_never_raises(tmp_path):
    f = tmp_path / "file"
    f.write_text("x")
    pl = plotting.TrainValPlotter(str(f / "sub"))  # cannot mkdir under a file
    pl.add_loss(1, 0.5)
    pl.save()  # prints a warning, must not raise


def test_plot_test_summary(tmp_path):
    res = {"bsds300": {"psnr": [30.0, 31.0], "ssim": [0.9, 0.91]}, "manga109": {"psnr": [36.0, 38.0], "ssim": [0.97, 0.98]}}
    p = str(tmp_path / "x/test.png")
    assert plotting.plot_test_summary(p, res, "t") is True and os.path.getsize(p) > 1000


class _Up(nn.Module):
    """x2 nearest upsample plus one conv, so there is something to backprop through."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x):
        return self.conv(nn.functional.interpolate(x, scale_factor=2))


def _val_dir(tmp_path, name):
    for i in range(2):
        _img(str(tmp_path / name / "HR" / "im{}.png".format(i)), (96, 96), seed=i)
        _img(str(tmp_path / name / "LR" / "im{}x2.png".format(i)), (48, 48), seed=i)
    return str(tmp_path / name / "HR"), str(tmp_path / name / "LR")


@pytest.fixture
def cpu_valid(monkeypatch):
    """utils.tensor2np clamps in place, which fails on CPU for inference-mode tensors (CUDA copies
    first, so real runs are fine); use no_grad in these CPU-only tests."""
    monkeypatch.setattr(torch, "inference_mode", torch.no_grad)


def test_valid_multi_set_prints_legacy_line_and_returns_dict(tmp_path, capsys, cpu_valid):
    import train
    loaders = {n: DataLoader(benchmarks.PairedVal(*_val_dir(tmp_path, n), 2), batch_size=1) for n in ("set14", "set5")}
    args = argparse.Namespace(cuda=False, scale=2, isY=True)
    res = train.valid(2, _Up(), loaders, args, torch.device("cpu"), 7)
    out = capsys.readouterr().out
    assert set(res) == {"set14", "set5"} and all(np.isfinite(v).all() for v in res.values())
    assert "===> Valid[set14] psnr:" in out and "===> Valid[set5] psnr:" in out
    import re  # the line experiments.py parses must still be present, once, with the mean
    m = re.findall(r"^===> Valid\. psnr: ([\d.]+), ssim: ([\d.]+) \(weights after 7 epochs\)$", out, re.M)
    assert len(m) == 1
    assert abs(float(m[0][0]) - np.mean([v[0] for v in res.values()])) < 1e-3


def test_valid_single_loader_still_works(tmp_path, capsys, cpu_valid):
    import train
    ld = DataLoader(benchmarks.PairedVal(*_val_dir(tmp_path, "one"), 2), batch_size=1)
    res = train.valid(2, _Up(), ld, argparse.Namespace(cuda=False, scale=2, isY=True), torch.device("cpu"), 0)
    assert list(res) == ["val"] and "===> Valid. psnr:" in capsys.readouterr().out


def test_train_epoch_returns_finite_loss_and_all_grads(capsys):
    """One backward pass: train() returns the mean L1 and every parameter gets a finite grad."""
    import train
    model, ld = _Up(), DataLoader([(torch.rand(3, 8, 8), torch.rand(3, 16, 16)) for _ in range(2)], batch_size=2)
    args = argparse.Namespace(cuda=False, use_amp=False, step_size=200, lr=1e-3, gamma=0.5)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = train.train(1, model, opt, nn.L1Loss(), ld, args, torch.device("cpu"))
    assert isinstance(loss, float) and np.isfinite(loss) and loss > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.shape == p.shape for p in model.parameters())


def test_reference_lines_drawn(tmp_path):
    """load_reference averages the last points; curves/freq_curves still render with a reference."""
    import json
    import plotting
    hist = {"loss": {"1": 0.02, "2": 0.01}, "val": {"set5": {"0": [10, 0.1], "5": [30, 0.9]}},
            "freq": {"set5": {"5": {"hfen": 0.2, "sbrel_haar_HH": 0.9, "sb_haar_HH": 30.0}}}}
    (tmp_path / "h.json").write_text(json.dumps(hist))
    ref = plotting.load_reference(str(tmp_path / "h.json"), last_n=2)
    assert ref["val"]["set5"] == (20.0, 0.5) and abs(ref["loss"] - 0.015) < 1e-9
    pl = plotting.TrainValPlotter(str(tmp_path / "out"), reference=ref)
    pl.add_loss(1, 0.03)
    pl.add_val(5, {"set5": (31.0, 0.91)})
    pl.add_freq(5, {"set5": {"hfen": 0.1, "sbrel_haar_HH": 0.8, "sb_haar_HH": 31.0}})
    pl.save()
    assert (tmp_path / "out" / "curves.png").exists() and (tmp_path / "out" / "freq_curves.png").exists()
    assert plotting.load_reference(str(tmp_path / "missing.json")) is None
