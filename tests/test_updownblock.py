"""Contract tests for Updownblock.split_bands (low/high frequency split)."""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.esrt import Updownblock  # noqa: E402


def test_split_bands_shapes_and_values():
    torch.manual_seed(0)
    blk = Updownblock(32).eval()
    feat = torch.rand(2, 32, 12, 14)
    low, high = blk.split_bands(feat, feat.shape[-2:])
    assert low.shape == (2, 32, 6, 7) and high.shape == feat.shape
    assert low.dtype == high.dtype == feat.dtype
    assert torch.allclose(low, F.avg_pool2d(feat, 2))
    up = F.interpolate(low, size=feat.shape[-2:], mode="bilinear", align_corners=True)
    assert torch.allclose(high, feat - up)


def test_forward_preserves_shape():
    blk = Updownblock(32).eval()
    x = torch.rand(1, 32, 12, 14)
    with torch.no_grad():
        assert blk(x).shape == x.shape


def test_single_backward_pass():
    """One backward pass through ESRT x2: finite loss, finite grads, right shapes.

    `one_module.weight1` is never used in forward, so those Scale params
    legitimately have no grad (pre-existing); any other missing grad is a bug.
    """
    from model.esrt import ESRT

    torch.manual_seed(1)
    model = ESRT(upscale=2).train()
    x = torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(2))
    target = torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(3))
    out = model(x)
    assert out.shape == target.shape
    loss = F.l1_loss(out, target)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is None:
            assert name.endswith("weight1.scale"), f"no grad for {name}"
            continue
        assert p.grad.shape == p.shape, name
        assert torch.isfinite(p.grad).all(), name
