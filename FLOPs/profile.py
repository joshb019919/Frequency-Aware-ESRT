"""Hook-based FLOP / parameter counter (derived from the pytorch-OpCounter
`thop` project). Registers a forward hook on every leaf module, runs one
forward pass on zeros, and sums the per-module op counts.

Role in ESRT: computes the model's FLOPs/parameters, the efficiency figures
reported alongside PSNR/SSIM for a lightweight SR network. Not needed for
training or inference.
"""
import logging

import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd

from .count_hooks import *

# Maps a leaf module type to the hook that computes its op count.
# A value of None means "counted as zero ops" (e.g. Dropout).
register_hooks = {
    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    nn.ConvTranspose2d: count_convtranspose2d,

    nn.BatchNorm1d: count_bn,
    nn.BatchNorm2d: count_bn,
    nn.BatchNorm3d: count_bn,

    nn.ReLU: count_relu,
    nn.ReLU6: count_relu,
    nn.LeakyReLU: count_relu,
    nn.PReLU: count_relu,

    nn.MaxPool1d: count_maxpool,
    nn.MaxPool2d: count_maxpool,
    nn.MaxPool3d: count_maxpool,
    nn.AdaptiveMaxPool1d: count_adap_maxpool,
    nn.AdaptiveMaxPool2d: count_adap_maxpool,
    nn.AdaptiveMaxPool3d: count_adap_maxpool,

    nn.AvgPool1d: count_avgpool,
    nn.AvgPool2d: count_avgpool,
    nn.AvgPool3d: count_avgpool,

    nn.AdaptiveAvgPool1d: count_adap_avgpool,
    nn.AdaptiveAvgPool2d: count_adap_avgpool,
    nn.AdaptiveAvgPool3d: count_adap_avgpool,
    nn.Linear: count_linear,
    nn.Dropout: None,
    nn.PixelShuffle: count_pixelshuffle,
    nn.Sigmoid: count_sigmoid,
}


def profile(model, input_size, custom_ops={}, device="cpu"):
    """Count operations and parameters of `model` for one forward pass.

    Args:
        model: nn.Module to analyse (temporarily switched to eval mode).
        input_size: shape of the dummy input, e.g. (1, 3, 64, 64).
        custom_ops: optional {module_type: hook_fn} overriding/extending register_hooks.
        device: device used for the profiling pass.

    Returns:
        (total_ops, total_params) as floats. Ops follow the hook definitions in
        count_hooks.py (multiply-accumulates counted as one op for convs).
        Leaf modules with no registered hook are reported as "Not implemented"
        and contribute 0 ops.
    """
    handler_collection = []

    def add_hooks(m):
        # Only instrument leaf modules (containers are counted via their children).
        if len(list(m.children())) > 0:
            return

        # Per-module accumulators filled in by the hooks / parameter count below.
        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()])

        m_type = type(m)
        fn = None

        if m_type in custom_ops:
            fn = custom_ops[m_type]
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
        else:
            print("Not implemented for ", m)

        if fn is not None:
            #print("Register FLOP counter for module %s" % str(m))
            handler = m.register_forward_hook(fn)
            handler_collection.append(handler)

    original_device = model.parameters().__next__().device
    training = model.training

    model.eval().to(device)
    model.apply(add_hooks)

    # One forward pass triggers every hook and populates m.total_ops.
    x = torch.zeros(input_size).to(device)
    with torch.no_grad():
        model(x)

    total_ops = 0
    total_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0:  # skip for non-leaf module
            continue
        total_ops += m.total_ops
        total_params += m.total_params

    total_ops = total_ops.item()
    total_params = total_params.item()

    # Restore the model's original mode/device and remove the hooks.
    model.train(training).to(original_device)
    for handler in handler_collection:
        handler.remove()

    return total_ops, total_params
