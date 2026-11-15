"""Generic building blocks shared by the ESRT model (EDSR/RCAN-style helpers).

Role in ESRT: esrt.py takes its same-padding conv factory (default_conv), the
learnable Scale multipliers used on every residual branch, and the sub-pixel
Upsampler that produces the final high-resolution output from here.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

def default_conv(in_channels, out_channels, kernel_size, bias=True, groups = 1):
    """Conv2d with 'same' padding (kernel_size // 2), so H and W are preserved for odd kernels."""
    wn = lambda x:torch.nn.utils.weight_norm(x)
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, groups = groups)

class Scale(nn.Module):
    """Learnable scalar multiplier: forward(x) = x * s, with s a trainable parameter."""
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale
        
class MeanShift(nn.Conv2d):
    """1x1 conv that subtracts (sign=-1) or adds (sign=+1) the dataset RGB mean,
    optionally dividing by the std. Currently unused by ESRT.

    NOTE: `self.requires_grad = False` only sets a plain attribute; it does not
    freeze the weights (that would need `p.requires_grad = False` per parameter).
    """
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    """Conv2d -> optional BatchNorm -> optional activation."""
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, bias=False,
                 bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(in_channels, out_channels, kernel_size,
                       padding=(kernel_size//2), stride=stride, bias=bias)]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    """EDSR residual block: x + res_scale * conv(act(conv(x)))."""
    def __init__(self, conv, n_feats, kernel_size,
                 bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class LuConv(nn.Module):
    """Two convs with a LeakyReLU between them and *no* skip connection
    (`res_scale` is stored but unused)."""
    def __init__(self, conv, n_feats, kernel_size,
                 bias=True, bn=False, act=nn.LeakyReLU(0.05), res_scale=1):
        super(LuConv, self).__init__()
        #self.scale1 = Scale(1)
        #self.scale2 = Scale(1)
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        return res
        
class Upsampler(nn.Sequential):
    """Sub-pixel upsampler: conv to r^2*C channels + PixelShuffle(r).

    Supports scale = 2^n (n stages of x2) and scale = 3 (one x3 stage).
    `act` may be False, 'relu' or 'prelu'; `bn` adds BatchNorm after each shuffle.
    Output channels = n_feats, spatial size multiplied by `scale`.
    """
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

