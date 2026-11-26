"""Tests for the frequency-aware loss (losses.py) and its wiring into train.py / experiments.py."""
import argparse
import os
import sys

import pytest
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import losses  # noqa: E402
import experiments  # noqa: E402


def _pair(seed=0, shape=(2, 3, 24, 20)):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(shape, generator=g), torch.rand(shape, generator=g)


def test_lambda_zero_is_exactly_l1():
    sr, hr = _pair()
    total, l1, freq = losses.FrequencyAwareLoss(0.0)(sr, hr)
    assert torch.equal(total, nn.L1Loss()(sr, hr)) and torch.equal(total, l1) and float(freq) == 0.0


def test_total_is_l1_plus_lambda_freq():
    sr, hr = _pair(1)
    total, l1, freq = losses.FrequencyAwareLoss(0.05)(sr, hr)
    assert torch.allclose(total, l1 + 0.05 * freq) and float(freq) > 0


def test_fft_l1_matches_manual_definition():
    sr, hr = _pair(2)
    d = torch.fft.fft2(sr, norm='ortho') - torch.fft.fft2(hr, norm='ortho')
    ref = torch.cat([d.real.abs().flatten(), d.imag.abs().flatten()]).mean()
    assert torch.allclose(losses.fft_l1(sr, hr), ref)
    assert float(losses.fft_l1(hr, hr)) == 0.0


def test_orthonormal_fft_scale_is_patch_size_independent():
    """Ortho norm keeps the term roughly constant as the patch grows (lambda stays meaningful)."""
    small = losses.fft_l1(*_pair(3, (4, 3, 48, 48)))
    big = losses.fft_l1(*_pair(3, (4, 3, 144, 144)))
    assert 0.8 < float(big / small) < 1.25


def test_penalises_lost_high_frequency_more_than_l1_does():
    """Blurring removes high-frequency detail; the FFT term grows relative to its L1 on a pure shift in DC."""
    torch.manual_seed(0)
    hr = torch.rand(1, 1, 32, 32)
    blurred = nn.functional.avg_pool2d(hr, 3, stride=1, padding=1, count_include_pad=False)
    brighter = hr + (blurred - hr).abs().mean()  # same pixel L1, but all error at DC
    l1_blur, l1_dc = nn.L1Loss()(blurred, hr), nn.L1Loss()(brighter, hr)
    assert torch.allclose(l1_blur, l1_dc, atol=1e-6)
    assert float(losses.fft_l1(blurred, hr)) > 5 * float(losses.fft_l1(brighter, hr))


def test_gradients_finite_and_bf16_input_ok():
    sr, hr = _pair(4)
    sr = sr.to(torch.bfloat16).requires_grad_()
    total, _, _ = losses.FrequencyAwareLoss(0.1)(sr, hr)
    assert total.dtype == torch.float32
    total.backward()
    assert sr.grad is not None and torch.isfinite(sr.grad.float()).all()


def test_negative_lambda_rejected():
    with pytest.raises(ValueError):
        losses.FrequencyAwareLoss(-0.1)


def test_loss_tag():
    assert losses.loss_tag(0) == "" and losses.loss_tag(0.05) == "_fl0.05" and losses.loss_tag(0.1) == "_fl0.1"


def test_train_epoch_with_freq_loss_logs_both_terms(capsys):
    import train
    model = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(3, 3, 3, padding=1))
    ld = torch.utils.data.DataLoader([(torch.rand(3, 8, 8), torch.rand(3, 16, 16)) for _ in range(2)], batch_size=2)
    args = argparse.Namespace(cuda=False, use_amp=False, step_size=200, lr=1e-3, gamma=0.5)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = train.train(1, model, opt, losses.FrequencyAwareLoss(0.05), ld, args, torch.device("cpu"))
    out = capsys.readouterr().out
    assert loss > 0 and "avg Loss_l1:" in out and "avg Loss_freq:" in out
    rows = experiments.parse_train_log(out)
    assert rows[0]["train_loss_l1"] == pytest.approx(loss, abs=1e-5) and rows[0]["train_loss_freq"] > 0


def test_default_save_dir_includes_lambda():
    import train
    a = train.parser.parse_args(["--scale", "2", "--freq_loss_lambda", "0.05"])
    assert train.default_save_dir(a) == "experiment/checkpoint_ESRT_x2_fl0.05"
    a = train.parser.parse_args(["--scale", "4", "--freq", "dwt", "--conv", "1"])
    assert train.default_save_dir(a) == "experiment/checkpoint_ESRT_x4_dwt_c1_gr1"


def test_experiment_configs_with_lambdas_and_test_cmd_drops_loss_flag():
    base = experiments.build_configs()
    cfgs = experiments.build_configs(freq_lambdas=(0.01, 0.1))
    names = [n for n, _, _ in cfgs]
    assert len(cfgs) == 3 * len(base) and len(set(names)) == len(names)
    assert "lowpass_fl0.01" in names and "dwt_c1_gr1_fl0.1" in names
    name, flags, _ = next(c for c in cfgs if c[0] == "dwt_c1_gr1_fl0.1")
    a = argparse.Namespace(root="/d", scale=2, epochs=10, batch_size=16, threads=0, seed=1, val_every=5,
                           fast=False, no_amp=False, test_hr_folder="Test_Datasets/Set5/", test_lr_folder=None)
    tr, te = experiments.train_cmd(name, flags, a), experiments.test_cmd(name, flags, a)
    assert tr[tr.index("--freq_loss_lambda") + 1] == "0.1"
    assert "--freq_loss_lambda" not in te and "--freq" in te and "0.1" not in te


def test_experiments_rejects_bad_lambdas(capsys):
    for bad in ("abc", "0", "-0.1"):
        with pytest.raises(SystemExit):
            experiments.main(["--root", "/d", "--freq_lambdas", bad, "--dry_run"])
