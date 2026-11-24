"""ESRT: Efficient Super-Resolution Transformer.

Role in ESRT: this file IS the model. Everything else (data, training,
evaluation, utilities) exists to feed, train or measure the ESRT class below.

Architecture overview (see ESRT class):
    head conv -> Un body (3 Updownblocks + efficient transformer attention)
    -> tail (upsampler) ; plus a global skip path that upsamples the head
    features directly and is added to the tail output.

Building blocks in this file:
    CALayer      channel attention (squeeze-and-excite)
    one_conv     lightweight residual conv pair with learnable scales
    BasicConv    conv (+BN/ReLU/resize) helper
    one_module   small residual block built from two one_conv + channel attention
    Updownblock  splits features into low/high frequency and processes each
    Un           stack of Updownblocks followed by the transformer block (MLABlock)
"""
from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from util.non  import NONLocalBlock2D
from util.tools import extract_image_patches,\
    reduce_mean, reduce_sum, same_padding, reverse_patches

from util.transformer import drop_path, DropPath, PatchEmbed, Mlp,MLABlock
from util.position import PositionEmbeddingLearned, PositionEmbeddingSine
import pdb
import math
from model import dwt

FREQ_CHOICES = ('lowpass', 'dwt')
CONV_CHOICES = (1, 3, 'band')


def add_model_args(parser):
    """Add the frequency-split flags (--freq, --conv, --global_res) to an argparse parser."""
    parser.add_argument('--freq', choices=FREQ_CHOICES, default='lowpass',
                        help='Updownblock band split: lowpass (original) or Haar dwt')
    parser.add_argument('--conv', choices=['1', '3', 'band'], default=None,
                        help='dwt high path: 1 = 1x1 conv 3C->C, 3 = 3x3 conv 3C->C, '
                             'band = shared decoder_high per band (default 1; needs --freq dwt)')
    parser.add_argument('--global_res', type=int, choices=[0, 1], default=1,
                        help='1 = keep the global skip in ESRT (default); 0 = remove it (needs --freq dwt)')


def model_kwargs(args):
    """ESRT keyword arguments from parsed args (empty for the default lowpass model)."""
    kw = {}
    if args.freq != 'lowpass' or args.conv is not None or args.global_res != 1:
        kw = dict(freq=args.freq, dwt_conv=args.conv, global_res=bool(args.global_res))
    return kw


def variant_tag(args):
    """Folder suffix identifying the flag combination ('' for the default model)."""
    if args.freq == 'lowpass':
        return ''
    return '_dwt_c{}_gr{}'.format(args.conv or 1, args.global_res)


def make_model(upscale=4, freq='lowpass', dwt_conv=None, global_res=True):
    """Factory returning an ESRT model for the given upscale factor."""
    # inpu = torch.randn(1, 3, 320, 180).cpu()
    # flops, params = profile(RTC(upscale).cpu(), inputs=(inpu,))
    # print(params)
    # print(flops)
    return ESRT(upscale=upscale, freq=freq, dwt_conv=dwt_conv, global_res=global_res)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    """Squeeze-and-excitation channel attention.

    Global-average-pools each channel to a scalar, passes the vector through a
    bottleneck (channel -> channel//reduction -> channel) with a sigmoid, and
    rescales the input channels by the resulting weights.
    """
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)    # (B, C, 1, 1) channel descriptor
        y = self.conv_du(y)     # per-channel weights in (0, 1)
        return x * y


class one_conv(nn.Module):
    """Residual bottleneck conv: out = w1 * x + w2 * conv1(act(conv(x))).

    `conv` reduces channels to `growth_rate`, `conv1` restores them to the input
    width. w1/w2 are learnable scalar weights (common.Scale, initialised to 1).
    Set relu=False to drop the PReLU between the two convs.
    """
    def __init__(self,inchanels,growth_rate,kernel_size = 3, relu = True):
        super(one_conv,self).__init__()
        wn = lambda x:torch.nn.utils.weight_norm(x)
        self.conv = nn.Conv2d(inchanels,growth_rate,kernel_size=kernel_size,padding = kernel_size>>1,stride= 1)
        self.flag = relu
        self.conv1 = nn.Conv2d(growth_rate,inchanels,kernel_size=kernel_size,padding = kernel_size>>1,stride= 1)
        if relu:
            self.relu = nn.PReLU(growth_rate)
        self.weight1 = common.Scale(1)
        self.weight2 = common.Scale(1)
    def forward(self,x):
        if self.flag == False:
            output = self.weight1(x) + self.weight2(self.conv1(self.conv(x)))
        else:
            output = self.weight1(x) + self.weight2(self.conv1(self.relu(self.conv(x))))
        return output#torch.cat((x,output),1)
        
class BasicConv(nn.Module):
    """Conv2d (or ConvTranspose2d if fan=True) with optional BatchNorm, ReLU and
    bilinear resize to a fixed `up_size` x `up_size` output (disabled when 0).
    Applied in that order: conv -> bn -> relu -> resize.
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1, dilation=1, groups=1, relu=True,
                 bn=False, bias=False, up_size=0,fan=False):
        super(BasicConv, self).__init__()
        wn = lambda x:torch.nn.utils.weight_norm(x)
        self.out_channels = out_planes
        self.in_channels = in_planes
        if fan:
            self.conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.up_size = up_size
        self.up_sample = nn.Upsample(size=(up_size, up_size), mode='bilinear') if up_size != 0 else None
        
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.up_size > 0:
            x = self.up_sample(x)
        return x

class one_module(nn.Module):
    """Residual feature block.

    Two chained one_conv layers; their outputs are concatenated, fused by a 1x1
    conv (`alise`), reweighted by channel attention, refined by a 3x3 conv, and
    added back to the input with learnable scales.
    """
    def __init__(self, n_feats):
        super(one_module, self).__init__()
        self.layer1 = one_conv(n_feats, n_feats//2,3)
        self.layer2 = one_conv(n_feats, n_feats//2,3)
        # self.layer3 = one_conv(n_feats, n_feats//2,3)
        self.layer4 = BasicConv(n_feats, n_feats, 3,1,1)
        self.alise = BasicConv(2*n_feats, n_feats, 1,1,0)
        self.atten = CALayer(n_feats)
        self.weight1 = common.Scale(1)
        self.weight2 = common.Scale(1)
        self.weight3 = common.Scale(1)
        self.weight4 = common.Scale(1)
        self.weight5 = common.Scale(1)
    def forward(self, x):

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        # x3 = self.layer3(x2)
        # pdb.set_trace()
        x4 = self.layer4(self.atten(self.alise(torch.cat([self.weight2(x2),self.weight3(x1)],1))))
        return self.weight4(x)+self.weight5(x4)

class Updownblock(nn.Module):
    """Frequency-split block (the "frequency-aware" part of this variant).

    Features are encoded, then average-pooled 2x to get a low-frequency
    (coarse) branch. The high-frequency residual is the encoded feature minus the
    upsampled low-frequency one. Each band is processed by its own decoder, the
    bands are recombined (concat -> 1x1 conv -> channel attention -> refine),
    and a skip connection from the block input is added.
    """
    def __init__(self, n_feats, freq='lowpass', dwt_conv=1):
        super(Updownblock, self).__init__()
        self.freq = freq
        self.dwt_conv = dwt_conv
        self.encoder = one_module(n_feats)
        self.decoder_low = one_module(n_feats) #nn.Sequential(one_module(n_feats),
        #                     one_module(n_feats),
        #                     one_module(n_feats))
        self.decoder_high = one_module(n_feats)
        self.alise = one_module(n_feats)
        if freq == 'lowpass':
            self.alise2 = BasicConv(2*n_feats, n_feats, 1,1,0) #one_module(n_feats)
        elif dwt_conv in (1, 3):
            # Detail bands are signed, so no ReLU on these projections.
            k = 1 if dwt_conv == 1 else 3
            self.high_reduce = BasicConv(3*n_feats, n_feats, k, 1, k//2, relu=False)
            self.high_expand = BasicConv(n_feats, 3*n_feats, 1, 1, 0, relu=False)
        self.down = nn.AvgPool2d(kernel_size=2)
        self.att = CALayer(n_feats)

    def split_bands(self, feat, size):
        """Split `feat` into (low, high) frequency bands.

        low: 2x average-pooled features (half resolution).
        high: `feat` minus `low` bilinearly upsampled to `size` (full resolution).
        """
        low = self.down(feat)
        high = feat - F.interpolate(low, size=size, mode='bilinear', align_corners=True)
        return low, high

    def _forward_dwt(self, x):
        """Haar variant: DWT -> low path on LL, high path on LH/HL/HH -> exact IDWT."""
        x1 = self.encoder(x)
        xp, (h, w) = dwt.pad_even(x1)
        ll, lh, hl, hh = dwt.haar_dwt2(xp.float())  # Haar math stays fp32 even under autocast
        for i in range(5):
            ll = self.decoder_low(ll)
        if self.dwt_conv == 'band':
            b = ll.shape[0]
            out = self.decoder_high(torch.cat([lh, hl, hh], dim=0))
            lh, hl, hh = out[:b], out[b:2*b], out[2*b:]
        else:
            high = self.decoder_high(self.high_reduce(torch.cat([lh, hl, hh], dim=1)))
            lh, hl, hh = torch.chunk(self.high_expand(high), 3, dim=1)
        y = dwt.haar_idwt2(ll.float(), lh.float(), hl.float(), hh.float())[..., :h, :w]
        return self.alise(self.att(y)) + x

    def forward(self, x):
        if self.freq == 'dwt':
            return self._forward_dwt(x)
        x1 = self.encoder(x)
        # x2: low-frequency branch at half resolution; high: full-res detail residual.
        x2, high = self.split_bands(x1, x.size()[-2:])
        # The same decoder_low block (shared weights) is applied 5 times.
        for i in range(5):
            x2 = self.decoder_low(x2)
        x3 = x2
        # x3 = self.decoder_low(x2)
        high1 = self.decoder_high(high)
        # Upsample the processed low-frequency branch back to full resolution.
        x4 = F.interpolate(x3, size = x.size()[-2:], mode='bilinear', align_corners=True)
        return self.alise(self.att(self.alise2(torch.cat([x4,high1],dim=1))))+ x

class Un(nn.Module):
    """ESRT body block: 3 chained Updownblocks + transformer attention.

    The three block outputs are concatenated and reduced to `n_feats` channels,
    turned into 3x3 patch tokens, passed through the MLABlock (efficient
    multi-head attention + MLP), folded back to a feature map, and added to the
    input through a learnable-scaled residual. `wn` is unused.

    The attention dim=288 equals n_feats(32) * 3 * 3, i.e. one flattened 3x3
    patch per token, so it is tied to n_feats=32.
    """
    def __init__(self,n_feats, wn, freq='lowpass', dwt_conv=1):
        super(Un, self).__init__()
        self.encoder1 = Updownblock(n_feats, freq, dwt_conv)
        self.encoder2 = Updownblock(n_feats, freq, dwt_conv)
        self.encoder3 = Updownblock(n_feats, freq, dwt_conv)
        self.reduce = common.default_conv(3*n_feats, n_feats, 3)
        self.weight2 = common.Scale(1)
        self.weight1 = common.Scale(1)
        self.attention = MLABlock(n_feat=n_feats, dim=288) 
        self.alise = common.default_conv(n_feats, n_feats, 3)

    def forward(self,x):
        # out = self.encoder3(self.encoder2(self.encoder1(x)))
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        out = x3
        b,c,h,w = x3.shape
        # (B, 3C, H, W) -> (B, C, H, W) -> tokens (B, H*W, C*9) after attention
        out = self.attention(self.reduce(torch.cat([x1,x2,x3],dim=1)))
        out = out.permute(0,2,1)                           # (B, C*9, H*W) for Fold
        out = reverse_patches(out, (h,w), (3,3), 1, 1)     # fold tokens back to (B, C, H, W)
        out = self.alise(out)

        return self.weight1(x) + self.weight2(out)
        
class ESRT(nn.Module):
    """Top-level super-resolution network.

    Args:
        upscale: SR factor (2^n or 3, as supported by common.Upsampler).
        conv: conv factory (default: 'same'-padded nn.Conv2d).

    Input: LR RGB image (B, 3, H, W) in [0, 1]. Output: (B, 3, scale*H, scale*W).
    Uses n_feats=32 feature channels and a single `Un` body block.
    """
    def __init__(self, upscale=4, conv=common.default_conv, freq='lowpass', dwt_conv=None, global_res=True):
        super(ESRT, self).__init__()
        if freq not in FREQ_CHOICES:
            raise ValueError("freq must be one of {}, got {!r}".format(FREQ_CHOICES, freq))
        if freq == 'lowpass' and dwt_conv is not None:
            raise ValueError("--conv is only valid with --freq dwt")
        if freq == 'lowpass' and not global_res:
            raise ValueError("--global_res 0 is only valid with --freq dwt")
        if dwt_conv is None:
            dwt_conv = 1
        dwt_conv = dwt_conv if dwt_conv == 'band' else int(dwt_conv)
        if dwt_conv not in CONV_CHOICES:
            raise ValueError("--conv must be one of {}, got {!r}".format(CONV_CHOICES, dwt_conv))
        self.global_res = global_res
        wn = lambda x:torch.nn.utils.weight_norm(x)
        n_feats = 32
        n_blocks = 1
        kernel_size = 3
        scale = upscale#args.scale[0] #gaile
        act = nn.ReLU(True)
        #self.up_sample = F.interpolate(scale_factor=2, mode='nearest')
        self.n_blocks = n_blocks
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        # self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        
        # NOTE: the mean shift is disabled; inputs are used as-is in [0, 1].
        # define head module (shallow feature extraction)
        modules_head = [conv(3, n_feats, kernel_size)]
        
        # define body module
        modules_body = nn.ModuleList()
        for i in range(n_blocks):
            modules_body.append(
                Un(n_feats=n_feats, wn = wn, freq=freq, dwt_conv=dwt_conv))

        # define tail module (upsample deep features, project back to RGB)
        modules_tail = [

            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 3, kernel_size)]


        # Global skip path: upsample the shallow (head) features straight to RGB.
        if global_res:
            self.up = nn.Sequential(common.Upsampler(conv,scale,n_feats,act=False),
                              BasicConv(n_feats, 3,3,1,1))
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        self.reduce = conv(n_blocks*n_feats, n_feats, kernel_size)


    def forward(self, x1,x2 = None, test=False):
        """Super-resolve `x1` (B, 3, H, W). `x2` and `test` are unused (kept for API compatibility)."""
        # x1 = self.sub_mean(x1)
        x1 = self.head(x1)
        res2 = x1  # shallow features, reused by the global skip path
        #res2 = x2
        body_out = []
        for i in range(self.n_blocks):
            x1 = self.body[i](x1)
            body_out.append(x1)
        res1 = torch.cat(body_out,1)
        res1 = self.reduce(res1)  # fuse outputs of all body blocks


        x1 = self.tail(res1)
        if self.global_res:
            x1 = self.up(res2) + x1   # add the global skip (long-range residual)
        # x1 = self.add_mean(x1)
        #x2 = self.tail(res2)
        return x1

    def load_state_dict(self, state_dict, strict=False):
        """Lenient weight loader (overrides nn.Module.load_state_dict; returns None).

        Copies each tensor whose name exists in this model. A shape mismatch in a
        'tail' (upsampler) parameter is skipped with a message so a model
        pre-trained at another scale can be reused; any other mismatch raises.
        With strict=True, unexpected/missing keys (other than 'tail') raise KeyError.
        """
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
        #MSRB_out = []from model import common

