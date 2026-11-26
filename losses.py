"""Frequency-aware training loss (proposal Change 2).

    L = ||SR - HR||_1 + lambda * ||F(SR) - F(HR)||_1

F is the 2-D FFT over (H, W) of each channel. The L1 distance is taken over the
real and imaginary parts (as in the proposal), averaged like nn.L1Loss.

The FFT uses norm='ortho' (unitary: sum |F(x)|^2 == sum |x|^2), so the frequency
term is on roughly the same scale as the pixel term and does not depend on the
patch size. That keeps one lambda meaningful across --patch_size settings.
The FFT always runs in fp32 (it is not autocast-safe in bf16).

lambda = 0 gives exactly the original L1 loss, and the FFT is skipped.
"""
import torch
import torch.nn as nn


def fft_l1(sr, hr):
    """Mean |Re/Im(FFT(sr)) - Re/Im(FFT(hr))| over all coefficients (orthonormal 2-D FFT)."""
    f_sr = torch.fft.fft2(sr.float(), norm='ortho')
    f_hr = torch.fft.fft2(hr.float(), norm='ortho')
    return (torch.view_as_real(f_sr) - torch.view_as_real(f_hr)).abs().mean()


class FrequencyAwareLoss(nn.Module):
    """L1 + lambda * FFT-L1. Calling it returns (total, l1, freq); freq is 0 when lambda == 0."""

    def __init__(self, freq_lambda=0.0):
        super().__init__()
        if freq_lambda < 0:
            raise ValueError("freq_lambda must be >= 0, got {}".format(freq_lambda))
        self.freq_lambda = float(freq_lambda)
        self.l1 = nn.L1Loss()

    def forward(self, sr, hr):
        sr = sr.float()
        l1 = self.l1(sr, hr)
        if self.freq_lambda == 0:
            return l1, l1, torch.zeros((), device=l1.device)
        freq = fft_l1(sr, hr)
        return l1 + self.freq_lambda * freq, l1, freq


def loss_tag(freq_lambda):
    """Folder/name suffix for a lambda ('' for plain L1), e.g. 0.05 -> '_fl0.05'."""
    return "" if not freq_lambda else "_fl{:g}".format(freq_lambda)
