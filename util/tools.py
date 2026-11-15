"""Tensor utilities for patch-based attention: TensorFlow-style 'same' padding,
patch extraction (unfold) / re-assembly (fold), and multi-axis reductions.

Role in ESRT: the transformer works on 3x3 feature patches; this file unfolds
the feature map into patch tokens (extract_image_patches) and folds the
attention output back into a feature map (reverse_patches).
"""
import os
import torch
import numpy as np
from PIL import Image

import torch.nn.functional as F

def normalize(x):
    """Map a [0, 1] tensor to [-1, 1] *in place*."""
    return x.mul_(2).add_(-1)

def same_padding(images, ksizes, strides, rates):
    """Zero-pad a (B, C, H, W) tensor so a sliding window yields ceil(H/stride) outputs
    (TensorFlow 'SAME' semantics). Extra padding goes to the bottom/right.

    Args:
        ksizes: [kh, kw] window size. strides: [sh, sw]. rates: [dh, dw] dilation.
    """
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :param padding: 'same' (pad so every position gets a patch) or 'valid' (no padding)
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()
    
    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks
def reverse_patches(images, out_size, ksizes, strides, padding):
    """
    Inverse of extract_image_patches: fold patch columns back into an image,
    summing overlapping values (torch.nn.Fold).
    :param images: [batch, channels*k*k, L] patch tensor, L = number of patches
    :param out_size: (H, W) of the image to reconstruct
    :param ksizes: (kh, kw) patch size
    :param strides: stride of the patch grid
    :param padding: padding that was applied when the patches were extracted
    :return: [batch, channels, H, W] tensor
    """
    unfold = torch.nn.Fold(output_size = out_size, 
                            kernel_size=ksizes, 
                            dilation=1, 
                            padding=padding, 
                            stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks
def reduce_mean(x, axis=None, keepdim=False):
    """Mean over several axes (all axes if `axis` is None/empty), like tf.reduce_mean."""
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x


def reduce_std(x, axis=None, keepdim=False):
    """Std-dev applied successively over several axes (all if None/empty).

    NOTE: this is a chain of per-axis stds, not the joint std over the axes.
    """
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.std(x, dim=i, keepdim=keepdim)
    return x


def reduce_sum(x, axis=None, keepdim=False):
    """Sum over several axes (all axes if `axis` is None/empty), like tf.reduce_sum."""
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x