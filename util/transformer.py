"""Transformer components for ESRT: stochastic depth, patch embedding, MLP, the
memory-efficient attention (EffAttention) and the full block (MLABlock).

Role in ESRT: the "transformer" half of the model. MLABlock is instantiated
inside the Un body block in model/esrt.py to add long-range (global) context to
the CNN features cheaply.
"""
from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from util.tools import extract_image_patches,\
    reduce_mean, reduce_sum, same_padding, reverse_patches
from util.position import PositionEmbeddingLearned, PositionEmbeddingSine
import pdb
import math

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        # Only active in training mode; identity at eval time.
        return drop_path(x, self.drop_prob, self.training)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    # Keep each sample's residual branch with prob keep_prob (rescaled by
    # 1/keep_prob so the expectation is unchanged); drop the whole branch otherwise.
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding: a strided conv turns non-overlapping
    patch_size x patch_size patches of a (B, C, H, W) image into a token
    sequence (B, num_patches, embed_dim). Input must be exactly img_size.
    Not used by the current ESRT (MLABlock unfolds patches itself).
    """
    def __init__(self, img_size=48, patch_size=2, in_chans=64, embed_dim=768):
        super().__init__()
        img_size = tuple((img_size,img_size))
        patch_size = tuple((patch_size,patch_size))
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape #16*64*48*48
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # pdb.set_trace()
        x = self.proj(x).flatten(2).transpose(1, 2)#64*48*48->768*6*6->768*36->36*768
        return x
class Mlp(nn.Module):
    """Transformer feed-forward network: Linear -> activation -> dropout -> Linear -> dropout.
    The hidden width defaults to in_features // 4 (a bottleneck, unlike the usual 4x expansion)."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features//4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x




class EffAttention(nn.Module):
    """Efficient multi-head self-attention (the "EMHA" of ESRT).

    Tokens are first projected from `dim` to `dim // 2` channels to cut cost.
    The token sequence is then split into 4 contiguous chunks and attention is
    computed *within each chunk only*, so the cost is ~1/4 of full attention.
    Chunk outputs are concatenated and projected back to `dim`.

    Input/output: (B, N, dim).
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.reduce = nn.Linear(dim, dim//2, bias=qkv_bias)
        self.qkv = nn.Linear(dim//2, dim//2 * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim//2, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        # Debug prints of the attention configuration (shown once at model construction).
        print('scale', self.scale)
        print(dim)
        print(dim//num_heads)
        # self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x = self.reduce(x)          # (B, N, dim) -> (B, N, dim//2)
        B, N, C = x.shape
        # pdb.set_trace()
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q = x.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # k = x.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # v = x.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv: 3*16*8*37*96
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        # pdb.set_trace()
        
        # Split the N tokens into chunks of N//4 and attend within each chunk.
        # (If N is not divisible by 4, split yields a short extra chunk.)
        q_all = torch.split(q, math.ceil(N//4), dim=-2)
        k_all = torch.split(k, math.ceil(N//4), dim=-2)
        v_all = torch.split(v, math.ceil(N//4), dim=-2)        

        output = []
        for q,k,v in zip(q_all, k_all, v_all):
            # Scaled dot-product attention inside this chunk.
            attn = (q @ k.transpose(-2, -1)) * self.scale   #16*8*37*37
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            
            trans_x = (attn @ v).transpose(1, 2)#.reshape(B, N, C)

            output.append(trans_x)
        # pdb.set_trace()
        # attn = torch.cat(att, dim=-2)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C) #16*37*768
        # Each chunk output is (B, chunk_N, heads, head_dim); concatenating on
        # dim=1 restores the full token order, then heads are merged into C.
        x = torch.cat(output,dim=1)
        x = x.reshape(B,N,C)
        # pdb.set_trace()
        x = self.proj(x)
        # x = self.proj_drop(x)
        return x




ATTN_CHOICES = ('split', 'window')
_MASKED = -1e4  # additive mask value: finite, so a fully-masked (padding-only) row cannot produce NaN


class WindowAttention(nn.Module):
    """Spatial window attention, a drop-in replacement for EffAttention (proposal stretch goal V4).

    ESRT's Feature Split cuts the flattened token sequence into 4 contiguous segments, i.e. horizontal
    strips of the image, so a token never sees its vertical neighbours in another strip. Here tokens
    attend inside M x M spatial windows instead (Swin-style), with:

    * the same reduce -> qkv -> proj layers as EffAttention (same width, heads and scale), so the only
      change is *which* tokens attend to each other (plus a small relative position bias);
    * shifted windows inside the single block: ESRT has one transformer block, so instead of
      alternating regular/shifted blocks as Swin does, half of the heads use regular windows and half
      use windows shifted by M // 2 (`shift=True`). This connects neighbouring windows in one layer at
      the same cost;
    * exact masking: feature maps whose H/W are not multiples of M are padded, and the padded tokens are
      masked out as keys; shifted windows are masked so that tokens wrapped around by the roll do not
      attend to each other (standard Swin mask). Results for real tokens therefore do not depend on the
      padding, and any image size works (full-size test images, forward_chop tiles);
    * a learned relative position bias per head (as in Swin), (2M-1)^2 x heads parameters.

    With M = 24 a training patch (48 x 48 LR tokens) gives 4 windows of 576 tokens, the same number and
    size of attention groups as ESRT's 4 strips of 12 x 48 = 576, so training cost is equal and the
    ablation isolates the grouping shape. Unlike the strips (N/4 tokens each, growing with the image),
    the cost stays linear in N at test time.

    Input/output: (B, N, dim) with N == H * W; `hw` = (H, W) of the token grid.
    """
    def __init__(self, dim, num_heads=8, window_size=24, shift=True, qkv_bias=False, qk_scale=None,
                 rel_pos_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = int(window_size)
        if self.window_size < 2:
            raise ValueError("window_size must be >= 2, got {}".format(window_size))
        self.shift = bool(shift) and num_heads >= 2
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5   # same as EffAttention (head_dim of the full dim)
        self.reduce = nn.Linear(dim, dim // 2, bias=qkv_bias)
        self.qkv = nn.Linear(dim // 2, dim // 2 * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim // 2, dim)
        M = self.window_size
        self.rel_pos_bias = bool(rel_pos_bias)
        if self.rel_pos_bias:
            self.rel_bias_table = nn.Parameter(torch.zeros((2 * M - 1) ** 2, num_heads))
            nn.init.trunc_normal_(self.rel_bias_table, std=0.02)
            coords = torch.stack(torch.meshgrid(torch.arange(M), torch.arange(M), indexing='ij')).flatten(1)
            rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0) + (M - 1)   # (M², M², 2)
            self.register_buffer('rel_index', rel[..., 0] * (2 * M - 1) + rel[..., 1], persistent=False)

    def _region_ids(self, Hp, Wp, s, device):
        """Swin region labels on the rolled (Hp, Wp) grid: tokens may only attend within one region."""
        ids = torch.zeros(Hp, Wp, dtype=torch.long, device=device)
        M, cnt = self.window_size, 0
        for hs in (slice(0, -M), slice(-M, -s), slice(-s, None)):
            for ws in (slice(0, -M), slice(-M, -s), slice(-s, None)):
                ids[hs, ws] = cnt
                cnt += 1
        return ids

    def _windows(self, t, M):
        """(B, h, Hp, Wp, d) -> (B, h, nW, M*M, d)."""
        B, h, Hp, Wp, d = t.shape
        t = t.view(B, h, Hp // M, M, Wp // M, M, d).permute(0, 1, 2, 4, 3, 5, 6)
        return t.reshape(B, h, (Hp // M) * (Wp // M), M * M, d)

    def _attend(self, q, k, v, H, W, shift, heads):
        """Window attention for the head group `heads` (a slice). q/k/v: (B, h, H, W, d) -> (B, h, H, W, d)."""
        M = self.window_size
        B, h, _, _, d = q.shape
        Hp, Wp = -(-H // M) * M, -(-W // M) * M
        pad = (0, 0, 0, Wp - W, 0, Hp - H)
        q, k, v = (F.pad(t, pad) for t in (q, k, v))
        valid = F.pad(torch.ones(H, W, device=q.device), (0, Wp - W, 0, Hp - H))    # 1 = real token
        s = M // 2 if shift else 0
        if s:
            q, k, v = (torch.roll(t, shifts=(-s, -s), dims=(2, 3)) for t in (q, k, v))
            valid = torch.roll(valid, shifts=(-s, -s), dims=(0, 1))
        qw, kw, vw = (self._windows(t, M) for t in (q, k, v))                 # (B, h, nW, L, d)
        nW, L = qw.shape[2], M * M
        # Additive mask (nW, L, L): padded keys always; with a shift, keys from another region too.
        vk = valid.view(Hp // M, M, Wp // M, M).permute(0, 2, 1, 3).reshape(nW, 1, L)
        allowed = vk.expand(nW, L, L) > 0
        if s:
            r = self._region_ids(Hp, Wp, s, q.device).view(Hp // M, M, Wp // M, M).permute(0, 2, 1, 3).reshape(nW, L)
            allowed = allowed & (r[:, :, None] == r[:, None, :])
        bias = torch.zeros(nW, L, L, device=q.device).masked_fill(~allowed, _MASKED)
        bias = bias.unsqueeze(0)                                                 # (1, nW, L, L)
        if self.rel_pos_bias:
            rpb = self.rel_bias_table[self.rel_index.reshape(-1)].view(L, L, -1)[..., heads].permute(2, 0, 1)
            bias = bias + rpb[:, None]                                           # (h, nW, L, L)
        bias = bias.expand(h, nW, L, L).reshape(1, h * nW, L, L).to(qw.dtype)
        out = F.scaled_dot_product_attention(qw.reshape(B, h * nW, L, d), kw.reshape(B, h * nW, L, d),
                                             vw.reshape(B, h * nW, L, d), attn_mask=bias, scale=self.scale)
        out = out.view(B, h, Hp // M, Wp // M, M, M, d).permute(0, 1, 2, 4, 3, 5, 6).reshape(B, h, Hp, Wp, d)
        if s:
            out = torch.roll(out, shifts=(s, s), dims=(2, 3))
        return out[:, :, :H, :W]

    def forward(self, x, hw):
        H, W = hw
        x = self.reduce(x)                                   # (B, N, dim) -> (B, N, C=dim//2)
        B, N, C = x.shape
        if N != H * W:
            raise ValueError("WindowAttention got {} tokens for a {}x{} grid".format(N, H, W))
        hd = C // self.num_heads
        # Same channel layout as EffAttention: (3, heads, head_dim).
        qkv = self.qkv(x).view(B, H, W, 3, self.num_heads, hd).permute(3, 0, 4, 1, 2, 5)   # (3, B, h, H, W, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]
        g = self.num_heads // 2 if self.shift else self.num_heads
        outs = []
        for lo, hi, shift in ((0, g, False), (g, self.num_heads, True)):
            if lo == hi:
                continue
            outs.append(self._attend(q[:, lo:hi], k[:, lo:hi], v[:, lo:hi], H, W, shift, slice(lo, hi)))
        out = torch.cat(outs, dim=1)                         # (B, h, H, W, hd)
        x = out.permute(0, 2, 3, 1, 4).reshape(B, N, C)
        return self.proj(x)


## Base block
class MLABlock(nn.Module):
    """Pre-norm transformer block over 3x3 image patches.

    Takes a feature map (B, C, H, W), unfolds it into H*W overlapping 3x3
    patches (tokens of size dim = C*9), then applies
        x = x + EffAttention(LN(x));  x = x + Mlp(LN(x)).
    Returns tokens (B, H*W, dim); the caller folds them back into a feature map.
    Only `dim` and `drop`/`act_layer` are honoured; the other constructor
    arguments are accepted for compatibility but unused (attention is
    hard-coded to 8 heads, no bias, no dropout).

    `attn` picks the token grouping: 'split' (default, ESRT's EffAttention, unchanged) or 'window'
    (WindowAttention with `window_size` x `window_size` windows, half the heads shifted if
    `window_shift`). The default builds exactly the original module, so existing checkpoints load.
    """
    def __init__(
        self, n_feat = 64,dim=768, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm, attn='split', window_size=24,
                 window_shift=True):
        super(MLABlock, self).__init__()
        self.dim = dim
        if attn not in ATTN_CHOICES:
            raise ValueError("attn must be one of {}, got {!r}".format(ATTN_CHOICES, attn))
        self.attn_type = attn
        if attn == 'window':
            self.atten = WindowAttention(self.dim, num_heads=8, window_size=window_size, shift=window_shift)
        else:
            self.atten = EffAttention(self.dim, num_heads=8, qkv_bias=False, qk_scale=None, \
                             attn_drop=0., proj_drop=0.)
        self.norm1 = nn.LayerNorm(self.dim)
        # self.posi = PositionEmbeddingLearned(n_feat)
        self.mlp = Mlp(in_features=dim, hidden_features=dim//4, act_layer=act_layer, drop=drop)
        self.norm2 = nn.LayerNorm(self.dim)
    def forward(self, x):
        # pdb.set_trace()
        B = x.shape[0]
        # posi = self.posi(x)
        # x = posi + x
        # x = self.patch_embed(x) # 16*36*768
        # print(x.shape)
        hw = x.shape[-2:]  # token grid = feature-map size (one 3x3-patch token per position)
        # Unfold into 3x3 patches: (B, C, H, W) -> (B, C*9, H*W) -> tokens (B, H*W, C*9)
        x = extract_image_patches(x, ksizes=[3, 3],
                                      strides=[1,1],
                                      rates=[1, 1],
                                      padding='same')#   16*2304*576
        x = x.permute(0,2,1)

        if self.attn_type == 'window':
            x = x + self.atten(self.norm1(x), hw)
        else:
            x = x + self.atten(self.norm1(x))
        x = x + self.mlp(self.norm2(x))#self.drop_path(self.mlp(self.norm2(x)))
        return x