"""Orthonormal 2-D Haar DWT / inverse DWT for (B, C, H, W) tensors.

Role in ESRT: used by the `--freq dwt` variant of Updownblock to split features
into one low band (LL) and three directional detail bands. Same math as the
numpy prototype ../dwt_test.py (lo = (a+b)/sqrt2, hi = (a-b)/sqrt2).

Band naming follows dwt_test.haar_dwt2: the first letter is the horizontal
filter (along W), the second the vertical filter (along H). The transform is
exactly invertible (haar_idwt2(*haar_dwt2(x)) == x up to float rounding).
H and W must be even; use pad_even / the crop in the caller for odd sizes.
"""
import torch
import torch.nn.functional as F


def pad_even(x):
    """Reflect-pad the bottom/right edge so H and W are even. Returns (x, (H, W))."""
    h, w = x.shape[-2:]
    if h % 2 or w % 2:
        x = F.pad(x, (0, w % 2, 0, h % 2), mode='reflect')
    return x, (h, w)


def haar_dwt2(x):
    """(B, C, H, W) -> LL, LH, HL, HH, each (B, C, H/2, W/2)."""
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]
    ll = (a + b + c + d) / 2
    lh = (a + b - c - d) / 2
    hl = (a - b + c - d) / 2
    hh = (a - b - c + d) / 2
    return ll, lh, hl, hh


def haar_idwt2(ll, lh, hl, hh):
    """Inverse of haar_dwt2: four (B, C, h, w) bands -> (B, C, 2h, 2w)."""
    a = (ll + lh + hl + hh) / 2
    b = (ll + lh - hl - hh) / 2
    c = (ll - lh + hl - hh) / 2
    d = (ll - lh - hl + hh) / 2
    # (B, C, h, 2, w, 2) -> (B, C, 2h, 2w); functional (no in-place strided writes).
    top = torch.stack([a, b], dim=-1)
    bottom = torch.stack([c, d], dim=-1)
    out = torch.stack([top, bottom], dim=-3)
    return out.reshape(ll.shape[:-2] + (ll.shape[-2] * 2, ll.shape[-1] * 2))
