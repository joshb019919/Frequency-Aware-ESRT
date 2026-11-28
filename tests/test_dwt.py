"""Tests for the Haar DWT module and the --freq / --conv / --global_res ESRT variants."""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import dwt  # noqa: E402
from model.esrt import ESRT  # noqa: E402

CONVS = [1, 3, "band"]


def test_dwt_idwt_round_trip_exact():
    x = torch.rand(2, 5, 12, 14, dtype=torch.float64)
    rec = dwt.haar_idwt2(*dwt.haar_dwt2(x))
    assert rec.shape == x.shape
    assert torch.allclose(rec, x, atol=1e-12, rtol=0)


def test_dwt_is_orthonormal_and_shapes():
    x = torch.rand(1, 3, 8, 6, dtype=torch.float64)
    bands = dwt.haar_dwt2(x)
    assert all(b.shape == (1, 3, 4, 3) for b in bands)
    assert torch.isclose(sum((b ** 2).sum() for b in bands), (x ** 2).sum())
    # LL is 2 x the 2x2 average pool
    assert torch.allclose(bands[0], 2 * F.avg_pool2d(x, 2))


def test_pad_even_then_crop_round_trip():
    x = torch.rand(1, 2, 7, 9)
    xp, (h, w) = dwt.pad_even(x)
    assert xp.shape[-2:] == (8, 10) and (h, w) == (7, 9)
    assert torch.allclose(dwt.haar_idwt2(*dwt.haar_dwt2(xp))[..., :h, :w], x, atol=1e-6)


@pytest.mark.parametrize("global_res", [1, 0])
@pytest.mark.parametrize("conv", CONVS)
def test_esrt_output_shape(conv, global_res):
    model = ESRT(upscale=2, freq="dwt", dwt_conv=conv, global_res=bool(global_res)).eval()
    for hw in [(16, 16), (15, 17)]:  # even and odd LR sizes
        x = torch.rand(1, 3, *hw)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 3, 2 * hw[0], 2 * hw[1]) and y.dtype == x.dtype


@pytest.mark.parametrize("global_res", [1, 0])
@pytest.mark.parametrize("conv", CONVS)
def test_single_backward_pass(conv, global_res):
    torch.manual_seed(1)
    model = ESRT(upscale=2, freq="dwt", dwt_conv=conv, global_res=bool(global_res)).train()
    x = torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(2))
    target = torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(3))
    loss = F.l1_loss(model(x), target)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is None:
            assert name.endswith("weight1.scale"), "no grad for " + name
            continue
        assert p.grad.shape == p.shape and torch.isfinite(p.grad).all(), name
    # new DWT-path params must all receive gradients
    for name, p in model.named_parameters():
        if "high_reduce" in name or "high_expand" in name:
            assert p.grad is not None, name


def test_global_res_off_removes_only_the_global_skip():
    on = ESRT(upscale=2, freq="dwt", global_res=True)
    off = ESRT(upscale=2, freq="dwt", global_res=False)
    assert any(k.startswith("up.") for k in on.state_dict())
    assert not any(k.startswith("up.") for k in off.state_dict())
    assert {k for k in on.state_dict() if not k.startswith("up.")} == set(off.state_dict())


def test_invalid_flag_combinations_raise():
    with pytest.raises(ValueError, match="only valid with --freq dwt"):
        ESRT(upscale=2, dwt_conv=3)
    with pytest.raises(ValueError, match="global_res 0 is only valid with --freq dwt"):
        ESRT(upscale=2, global_res=False)
    with pytest.raises(ValueError):
        ESRT(upscale=2, freq="dwt", dwt_conv=2)
    with pytest.raises(ValueError):
        ESRT(upscale=2, freq="haar")


def test_default_model_unchanged_by_new_options():
    keys = set(ESRT(upscale=2).state_dict())
    assert any("alise2" in k for k in keys)
    assert not any("high_reduce" in k or "high_expand" in k for k in keys)
    assert sum(p.numel() for p in ESRT(upscale=2).parameters()) == 677783
