"""Tests for the speed options: channels_last, bf16 autocast, inference_mode, compile fallback."""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import utils  # noqa: E402
from model import dwt  # noqa: E402
from model.esrt import ESRT  # noqa: E402

VARIANTS = [{}, dict(freq="dwt", dwt_conv=1), dict(freq="dwt", dwt_conv=3), dict(freq="dwt", dwt_conv="band")]


def test_functional_idwt_round_trip_still_exact():
    x = torch.rand(2, 3, 10, 12, dtype=torch.float64)
    assert torch.allclose(dwt.haar_idwt2(*dwt.haar_dwt2(x)), x, atol=1e-12, rtol=0)


@pytest.mark.parametrize("kw", VARIANTS)
def test_channels_last_matches_contiguous(kw):
    torch.manual_seed(0)
    m = ESRT(upscale=2, **kw).eval()
    x = torch.rand(1, 3, 16, 18)
    with torch.inference_mode():
        a = m(x)
        m = m.to(memory_format=torch.channels_last)
        b = m(x.contiguous(memory_format=torch.channels_last))
    assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("kw", VARIANTS)
def test_bf16_autocast_forward_backward(kw):
    torch.manual_seed(0)
    m = ESRT(upscale=2, **kw).train()
    x = torch.rand(1, 3, 16, 16)
    t = torch.rand(1, 3, 32, 32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = m(x)
    loss = F.l1_loss(out.float(), t)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    # bf16 stays close to fp32 on the same weights
    with torch.inference_mode():
        ref = m.eval()(x)
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        lo = m(x)
    assert (lo.float() - ref).abs().max() < 0.1


def test_inference_mode_equals_no_grad():
    torch.manual_seed(0)
    m = ESRT(upscale=2).eval()
    x = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        a = m(x)
    with torch.inference_mode():
        b = m(x)
    assert torch.equal(a, b)


def test_compile_model_graph_capture_works():
    """Dynamo traces and runs the model (aot_eager avoids needing a C compiler)."""
    torch.manual_seed(0)
    m = ESRT(upscale=2, freq="dwt").eval()
    x = torch.rand(1, 3, 16, 16)
    cm = utils.compile_model(m, x, backend="aot_eager")
    assert cm is not m
    with torch.inference_mode():
        assert torch.allclose(cm(x), m(x), atol=1e-5)


def test_compile_model_falls_back_when_compile_fails(capsys):
    m = ESRT(upscale=2).eval()
    out = utils.compile_model(m, torch.rand(1, 3, 16, 16), backend="no_such_backend")
    assert out is m
    assert "continuing eager" in capsys.readouterr().out
