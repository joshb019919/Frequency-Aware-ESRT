"""Tests for the spatial window attention (util/transformer.WindowAttention, --attn window)."""
import argparse
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from util.transformer import WindowAttention, MLABlock, EffAttention  # noqa: E402
from model import esrt  # noqa: E402


def reference(att, x, H, W):
    """Naive O(N^2) window attention: one explicit token-pair mask per head group, no partitioning.

    Token (i, j) of a group with shift s sits at rolled position ((i - s) mod Hp, (j - s) mod Wp) of the
    padded grid; it may attend to real tokens in the same M x M window and (if shifted) the same Swin region.
    """
    M = att.window_size
    Hp, Wp = -(-H // M) * M, -(-W // M) * M
    t = att.reduce(x)
    B, N, C = t.shape
    nh, hd = att.num_heads, C // att.num_heads
    qkv = att.qkv(t).view(B, N, 3, nh, hd).permute(2, 0, 3, 1, 4)      # (3, B, h, N, hd)
    q, k, v = qkv[0], qkv[1], qkv[2]
    ii, jj = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    ii, jj = ii.flatten(), jj.flatten()
    g = nh // 2 if att.shift else nh
    out = torch.empty(B, nh, N, hd)
    for lo, hi, shift in ((0, g, False), (g, nh, True)):
        if lo == hi:
            continue
        s = M // 2 if shift else 0
        ri, rj = (ii - s) % Hp, (jj - s) % Wp
        win = (ri // M) * (Wp // M) + rj // M
        allowed = win[:, None] == win[None, :]
        if s:
            reg = lambda r, n: torch.where(r < n - M, 0, torch.where(r < n - s, 1, 2))
            region = reg(ri, Hp) * 3 + reg(rj, Wp)
            allowed &= region[:, None] == region[None, :]
        logits = (q[:, lo:hi] @ k[:, lo:hi].transpose(-2, -1)) * att.scale
        if att.rel_pos_bias:
            di, dj = ri[:, None] % M - ri[None, :] % M + (M - 1), rj[:, None] % M - rj[None, :] % M + (M - 1)
            rpb = att.rel_bias_table[(di * (2 * M - 1) + dj).flatten()].view(N, N, -1)[..., lo:hi].permute(2, 0, 1)
            logits = logits + rpb
        logits = logits.masked_fill(~allowed, float("-inf"))
        out[:, lo:hi] = logits.softmax(-1) @ v[:, lo:hi]
    return att.proj(out.permute(0, 2, 1, 3).reshape(B, N, C))


@pytest.mark.parametrize("H,W,M,shift", [(8, 8, 8, False), (8, 8, 4, True), (10, 13, 4, True),
                                         (7, 5, 4, False), (12, 9, 6, True), (3, 3, 4, True)])
@pytest.mark.parametrize("rpb", [False, True])
def test_matches_naive_reference(H, W, M, shift, rpb):
    torch.manual_seed(0)
    att = WindowAttention(32, num_heads=8, window_size=M, shift=shift, rel_pos_bias=rpb).eval()
    if rpb:
        torch.nn.init.normal_(att.rel_bias_table, std=0.5)   # make the bias matter
    x = torch.randn(2, H * W, 32)
    with torch.no_grad():
        assert torch.allclose(att(x, (H, W)), reference(att, x, H, W), atol=1e-5)


def test_one_window_without_shift_is_full_attention():
    """If the window covers the whole grid, it is plain global attention over all tokens."""
    torch.manual_seed(1)
    att = WindowAttention(32, num_heads=8, window_size=8, shift=False, rel_pos_bias=False).eval()
    x = torch.randn(1, 64, 32)
    t = att.reduce(x)
    q, k, v = att.qkv(t).view(1, 64, 3, 8, 2).permute(2, 0, 3, 1, 4)
    full = att.proj(((q @ k.transpose(-2, -1)) * att.scale).softmax(-1).matmul(v).permute(0, 2, 1, 3).reshape(1, 64, 16))
    with torch.no_grad():
        assert torch.allclose(att(x, (8, 8)), full, atol=1e-5)


def test_locality_and_shift_connect_windows():
    """No shift: a change in one window never reaches another. Shift: it reaches the neighbour window."""
    torch.manual_seed(2)
    H = W = 8
    for shift, expect_cross in ((False, False), (True, True)):
        att = WindowAttention(32, num_heads=8, window_size=4, shift=shift, rel_pos_bias=False).eval()
        x = torch.randn(1, H * W, 32)
        y = x.clone()
        y[0, 3 * W + 3] += 5.0                      # token (3, 3), bottom-right of the top-left window
        with torch.no_grad():
            d = (att(x, (H, W)) - att(y, (H, W))).abs().sum(-1).view(H, W)
        assert d[:4, :4].max() > 0                  # its own window always changes
        assert (d[4:, 4:].max() > 0) == expect_cross


def test_padding_does_not_leak():
    """Real tokens' outputs depend only on real tokens: crop-invariance of the top-left windows."""
    torch.manual_seed(3)
    att = WindowAttention(32, num_heads=8, window_size=4, shift=False).eval()
    x = torch.randn(1, 6 * 6, 32)
    big = torch.randn(1, 8 * 8, 32)
    big.view(1, 8, 8, 32)[:, :6, :6] = x.view(1, 6, 6, 32)
    with torch.no_grad():
        a = att(x, (6, 6)).view(1, 6, 6, 32)[:, :4, :4]       # first window: fully real in both grids
        b = att(big, (8, 8)).view(1, 8, 8, 32)[:, :4, :4]
    assert torch.allclose(a, b, atol=1e-5)


def test_same_layers_as_effattention():
    e, w = EffAttention(288), WindowAttention(288, window_size=24)
    shapes = lambda m: {k: v.shape for k, v in m.state_dict().items() if k != "rel_bias_table"}
    assert shapes(e) == shapes(w)
    assert w.rel_bias_table.shape == (47 * 47, 8)


def test_training_patch_cost_matches_feature_split():
    """M = 24 on a 48x48 token grid: 4 windows of 576 tokens, like ESRT's 4 strips of 12 x 48."""
    assert esrt.WINDOW_SIZE == 24 and (48 // 24) ** 2 == 4 and 24 * 24 == 48 * 48 // 4


@pytest.mark.parametrize("hw", [(48, 48), (37, 53), (5, 7)])
def test_esrt_window_forward_backward_any_size(hw):
    torch.manual_seed(4)
    m = esrt.ESRT(upscale=2, attn="window", window_size=8)
    x = torch.rand(1, 3, *hw, requires_grad=False)
    y = m(x)
    assert y.shape == (1, 3, 2 * hw[0], 2 * hw[1])
    y.mean().backward()
    no_grad = {n for n, p in m.named_parameters() if p.grad is None}
    # The original ESRT never uses one_module.weight1 (12 scalars); the window model must add nothing to that.
    assert no_grad and all(n.endswith(".weight1.scale") for n in no_grad)
    att = {n: p for n, p in m.named_parameters() if ".attention.atten." in n}
    assert "body.0.attention.atten.rel_bias_table" in att
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in att.values())
    assert all(torch.isfinite(p.grad).all() for n, p in m.named_parameters() if n not in no_grad)


def test_dwt_plus_window_builds():
    m = esrt.ESRT(upscale=2, freq="dwt", dwt_conv="3", dwt_skip=True, ll_norm=True, attn="window")
    assert m(torch.rand(1, 3, 12, 12)).shape == (1, 3, 24, 24)


def _parse(argv):
    p = argparse.ArgumentParser()
    esrt.add_model_args(p)
    return p.parse_args(argv)


def test_defaults_unchanged_tags_and_kwargs():
    assert esrt.model_kwargs(_parse([])) == {} and esrt.variant_tag(_parse([])) == ""
    dwt = ["--freq", "dwt", "--conv", "3", "--dwt_skip", "1", "--ll_norm", "1"]
    assert esrt.variant_tag(_parse(dwt)) == "_dwt_c3_gr1_sk_nm"
    assert "attn" not in esrt.model_kwargs(_parse(dwt))
    assert esrt.variant_tag(_parse(["--attn", "window"])) == "_win24"
    assert esrt.variant_tag(_parse(dwt + ["--attn", "window", "--window_size", "16", "--window_shift", "0"])) \
        == "_dwt_c3_gr1_sk_nm_win16ns"
    kw = esrt.model_kwargs(_parse(["--attn", "window"]))
    assert kw["attn"] == "window" and kw["window_size"] == 24 and kw["window_shift"] is True
    with pytest.raises(ValueError):   # window options without --attn window
        esrt.ESRT(upscale=2, **esrt.model_kwargs(_parse(["--window_size", "8"])))


@pytest.mark.parametrize("cfg,flags", [("lowpass", {}),
                                       ("dwt_c3_gr1_skipnorm", dict(freq="dwt", dwt_conv="3", dwt_skip=True, ll_norm=True))])
def test_committed_checkpoints_still_load_exactly(cfg, flags):
    path = os.path.join(ROOT, "final_results", cfg, "epoch_1000.pth")
    if not os.path.exists(path):
        pytest.skip("no committed checkpoint")
    sd = torch.load(path, map_location="cpu")
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    m = esrt.ESRT(upscale=2, **flags)
    own = m.state_dict()
    assert set(own) == set(sd)
    assert all(own[k].shape == sd[k].shape for k in own)


def test_window_params_only_add_position_bias():
    base = sum(p.numel() for p in esrt.ESRT(upscale=2).parameters())
    win = sum(p.numel() for p in esrt.ESRT(upscale=2, attn="window").parameters())
    assert win - base == 47 * 47 * 8
