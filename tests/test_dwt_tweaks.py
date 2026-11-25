"""Tests for the --dwt_skip and --ll_norm architectural tweaks."""
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.esrt import ESRT, Updownblock, model_kwargs, variant_tag  # noqa: E402
import argparse  # noqa: E402
from model import esrt  # noqa: E402


def _args(*argv):
    p = argparse.ArgumentParser()
    esrt.add_model_args(p)
    return p.parse_args(list(argv))


def test_tweaks_off_by_default_and_tags():
    a = _args()
    assert model_kwargs(a) == {} and variant_tag(a) == ""
    a = _args("--freq", "dwt", "--dwt_skip", "1", "--ll_norm", "1")
    assert model_kwargs(a)["dwt_skip"] and model_kwargs(a)["ll_norm"]
    assert variant_tag(a) == "_dwt_c1_gr1_sk_nm"
    assert variant_tag(_args("--freq", "dwt")) == "_dwt_c1_gr1"  # existing tags unchanged


def test_tweaks_error_with_lowpass():
    with pytest.raises(ValueError, match="only valid with --freq dwt"):
        ESRT(upscale=2, dwt_skip=True)
    with pytest.raises(ValueError, match="only valid with --freq dwt"):
        ESRT(upscale=2, ll_norm=True)


def test_no_new_parameters():
    base = ESRT(upscale=2, freq="dwt", dwt_conv=1)
    both = ESRT(upscale=2, freq="dwt", dwt_conv=1, dwt_skip=True, ll_norm=True)
    assert list(base.state_dict()) == list(both.state_dict())
    assert sum(p.numel() for p in base.parameters()) == sum(p.numel() for p in both.parameters())


@pytest.mark.parametrize("conv", [1, 3])
def test_skip_starts_as_pass_through_of_detail_bands(conv):
    """With zero-init expand the high-path weights cannot influence the output at init."""
    torch.manual_seed(0)
    blk = Updownblock(32, "dwt", conv, dwt_skip=True).eval()
    assert float(blk.high_expand.conv.weight.abs().sum()) == 0.0
    x = torch.rand(1, 32, 8, 8)
    with torch.no_grad():
        y0 = blk(x)
        for p in blk.decoder_high.parameters():
            p.add_(torch.randn_like(p))
        y1 = blk(x)
    assert torch.equal(y0, y1)
    plain = Updownblock(32, "dwt", conv, dwt_skip=False).eval()
    plain.load_state_dict(blk.state_dict())
    plain.high_expand.conv.weight.data.normal_()
    with torch.no_grad():
        assert not torch.equal(plain(x), y1)


def test_ll_norm_is_identity_rescale_around_decoder_low():
    torch.manual_seed(0)
    a = Updownblock(32, "dwt", 1, ll_norm=False).eval()
    b = Updownblock(32, "dwt", 1, ll_norm=True).eval()
    b.load_state_dict(a.state_dict())
    x = torch.rand(1, 32, 8, 8)
    with torch.no_grad():
        assert not torch.allclose(a(x), b(x))           # differs with the real (nonlinear) decoder_low
        a.decoder_low = nn.Identity(); b.decoder_low = nn.Identity()
        assert torch.allclose(a(x), b(x), atol=1e-5)    # x0.5 ... x2 cancels exactly


@pytest.mark.parametrize("flags", [dict(dwt_skip=True), dict(ll_norm=True), dict(dwt_skip=True, ll_norm=True)])
@pytest.mark.parametrize("conv", [1, 3, "band"])
def test_tweaked_backward_pass(conv, flags):
    torch.manual_seed(1)
    m = ESRT(upscale=2, freq="dwt", dwt_conv=conv, **flags).train()
    x = torch.rand(1, 3, 15, 17, generator=torch.Generator().manual_seed(2))
    t = torch.rand(1, 3, 30, 34, generator=torch.Generator().manual_seed(3))
    out = m(x)
    assert out.shape == t.shape
    loss = F.l1_loss(out, t)
    assert torch.isfinite(loss)
    loss.backward()
    for n, p in m.named_parameters():
        if p.grad is None:
            assert n.endswith("weight1.scale"), n
        else:
            assert torch.isfinite(p.grad).all() and p.grad.shape == p.shape, n
    if flags.get("dwt_skip") and conv != "band":  # zero-init expand must still get a gradient
        assert all(p.grad.abs().sum() > 0 for n, p in m.named_parameters() if "high_expand" in n)
