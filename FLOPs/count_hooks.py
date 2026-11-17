"""Forward hooks that estimate the op count of individual layer types.

Each hook has the PyTorch forward-hook signature hook(module, inputs, output)
and stores its result in `module.total_ops` (a 1-element tensor), which
profile.py later sums. Here one multiply-accumulate counts as one op.

Role in ESRT: the per-layer cost formulas behind FLOPs/profile.py.
"""
import argparse

import torch
import torch.nn as nn

multiply_adds = 1  # 1 = count a multiply-add as a single op (MAC); 2 = count both


def count_convNd(m, x, y):
    """Ops for Conv1d/2d/3d: output_elements * cin * (kernel_size + bias) / groups."""
    x = x[0]
    cin = m.in_channels
    batch_size = x.size(0)

    kernel_ops = m.weight.size()[2:].numel()
    bias_ops = 1 if m.bias is not None else 0
    ops_per_element = kernel_ops + bias_ops
    output_elements = y.nelement()

    # cout x oW x oH
    total_ops = batch_size * cin * output_elements * ops_per_element // m.groups
    # total_ops = batch_size * output_elements * (cin * kernel_ops // m.groups + bias_ops)
    m.total_ops = torch.Tensor([int(total_ops)])


def count_conv2d(m, x, y):
    """Conv2d-specific variant of count_convNd (not registered by default)."""
    x = x[0]

    cin = m.in_channels
    cout = m.out_channels
    kh, kw = m.kernel_size
    batch_size = x.size()[0]

    out_h = y.size(2)
    out_w = y.size(3)

    # ops per output element
    # kernel_mul = kh * kw * cin
    # kernel_add = kh * kw * cin - 1
    kernel_ops = multiply_adds * kh * kw
    bias_ops = 1 if m.bias is not None else 0
    ops_per_element = kernel_ops + bias_ops

    # total ops
    # num_out_elements = y.numel()
    output_elements = batch_size * out_w * out_h * cout
    total_ops = output_elements * ops_per_element * cin // m.groups

    m.total_ops = torch.Tensor([int(total_ops)])


def count_convtranspose2d(m, x, y):
    """Ops for ConvTranspose2d: every output element touches the whole weight tensor."""
    x = x[0]

    cin = m.in_channels
    cout = m.out_channels
    kh, kw = m.kernel_size
    batch_size = x.size()[0]

    out_h = y.size(2)
    out_w = y.size(3)

    # ops per output element
    # kernel_mul = kh * kw * cin
    # kernel_add = kh * kw * cin - 1
    kernel_ops = multiply_adds * kh * kw * cin // m.groups
    bias_ops = 1 if m.bias is not None else 0
    ops_per_element = kernel_ops + bias_ops

    # total ops
    # num_out_elements = y.numel()
    # output_elements = batch_size * out_w * out_h * cout
    ops_per_element = m.weight.nelement()
    output_elements = y.nelement()
    total_ops = output_elements * ops_per_element

    m.total_ops = torch.Tensor([int(total_ops)])


def count_bn(m, x, y):
    """BatchNorm: 4 ops per element."""
    x = x[0]

    nelements = x.numel()
    # subtract, divide, gamma, beta
    total_ops = 4 * nelements

    m.total_ops = torch.Tensor([int(total_ops)])


def count_relu(m, x, y):
    """ReLU-like activations: 1 op per element."""
    x = x[0]

    nelements = x.numel()
    total_ops = nelements

    m.total_ops = torch.Tensor([int(total_ops)])


def count_sigmoid(m, x, y):
    """Sigmoid: exp + add + divide per element (3 ops)."""
    x = x[0]
    nelements = x.numel()

    total_exp = nelements
    total_add = nelements
    total_div = nelements

    total_ops = total_exp + total_add + total_div
    m.total_ops = torch.Tensor([int(total_ops)])

def count_pixelshuffle(m, x, y):
    """PixelShuffle: one memory move per element, counted as 1 op."""
    x = x[0]
    nelements = x.numel()
    total_ops = nelements
    m.total_ops = torch.Tensor([int(total_ops)])


def count_softmax(m, x, y):
    """Softmax over (batch, features): exp + add + divide per feature (not registered by default)."""
    x = x[0]

    batch_size, nfeatures = x.size()

    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)

    m.total_ops = torch.Tensor([int(total_ops)])


def count_maxpool(m, x, y):
    """MaxPool: kernel_size comparisons per output element."""
    kernel_ops = torch.prod(torch.Tensor([m.kernel_size]))
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops = torch.Tensor([int(total_ops)])


def count_adap_maxpool(m, x, y):
    """AdaptiveMaxPool: window size inferred as input_size // output_size."""
    kernel = torch.Tensor([*(x[0].shape[2:])]) // torch.Tensor(list((m.output_size,))).squeeze()
    kernel_ops = torch.prod(kernel)
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops = torch.Tensor([int(total_ops)])


def count_avgpool(m, x, y):
    """AvgPool: (kernel_size adds + 1 divide) per output element."""
    total_add = torch.prod(torch.Tensor([m.kernel_size]))
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops = torch.Tensor([int(total_ops)])


def count_adap_avgpool(m, x, y):
    """AdaptiveAvgPool: window size inferred as input_size // output_size, plus 1 divide."""
    kernel = torch.Tensor([*(x[0].shape[2:])]) // torch.Tensor(list((m.output_size,))).squeeze()
    total_add = torch.prod(kernel)
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops = torch.Tensor([int(total_ops)])


def count_linear(m, x, y):
    """Linear: (in_features multiplies + in_features-1 adds) per output element."""
    # per output element
    total_mul = m.in_features
    total_add = m.in_features - 1
    num_elements = y.numel()
    total_ops = (total_mul + total_add) * num_elements

    m.total_ops = torch.Tensor([int(total_ops)])
