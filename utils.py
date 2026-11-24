"""Shared helpers: metrics, image conversion, LR scheduling, timing, checkpoint loading.

Role in ESRT: small utility layer used by train.py and test.py (PSNR/SSIM,
tensor<->image conversion, border shaving, step LR decay, checkpoint loading).

NOTE: compare_psnr / compare_ssim were removed in scikit-image >= 0.18; with a
newer version use skimage.metrics.peak_signal_noise_ratio / structural_similarity.
"""
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os
import torch
from collections import OrderedDict

import time
class Timer():
    """Simple wall-clock stopwatch."""

    def __init__(self):
        self.v = time.time()

    def s(self):
        """Start/restart the timer."""
        self.v = time.time()

    def t(self):
        """Return seconds elapsed since the last start."""
        return time.time() - self.v


def time_text(t):
    """Format a duration in seconds as a short string ('12.3s', '4.5m' or '1.2h')."""
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)
def compute_psnr(im1, im2):
    """PSNR (dB) between two uint8 images of the same shape."""
    p = psnr(im1, im2)
    return p


def compute_ssim(im1, im2):
    """SSIM between two uint8 images, using the standard Gaussian-window settings
    (sigma=1.5, K1=0.01, K2=0.03) from the original SSIM paper.

    Treated as multichannel when the input is HxWx3, otherwise as grayscale.
    """
    isRGB = len(im1.shape) == 3 and im1.shape[-1] == 3
    s = ssim(im1, im2, K1=0.01, K2=0.03, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
             multichannel=isRGB)
    return s


def shave(im, border):
    """Remove `border` pixels from every side of an HxW(xC) image.

    `border` must be > 0 (a value of 0 would slice to an empty array).
    """
    border = [border, border]
    im = im[border[0]:-border[0], border[1]:-border[1], ...]
    return im


def modcrop(im, modulo):
    """Crop bottom/right so height and width are multiples of `modulo` (the SR scale)."""
    sz = im.shape
    h = np.int32(sz[0] / modulo) * modulo
    w = np.int32(sz[1] / modulo) * modulo
    ims = im[0:h, 0:w, ...]
    return ims


def get_list(path, ext):
    """List full paths of files in directory `path` whose names end with `ext`
    (unsorted, non-recursive)."""
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)]


def detect_image_ext(path, candidates=('.png', '.bmp')):
    """Return the first extension in `candidates` that has files in directory `path`.

    Raises FileNotFoundError (instead of silently yielding an empty list) when none match.
    """
    for ext in candidates:
        if get_list(path, ext):
            return ext
    raise FileNotFoundError("no {} images found in {}".format(" or ".join(candidates), path))


def convert_shape(img):
    """Convert a CHW float image in [0,1] to an HWC uint8 image in [0,255]."""
    img = np.transpose((img * 255.0).round(), (1, 2, 0))
    img = np.uint8(np.clip(img, 0, 255))
    return img


def quantize(img):
    """Clip to [0,255], round, and cast to uint8 (mimics saving as an 8-bit image)."""
    return img.clip(0, 255).round().astype(np.uint8)


def tensor2np(tensor, out_type=np.uint8, min_max=(0, 1)):
    """Convert a CHW torch tensor to an HWC numpy image.

    Values are clamped to `min_max`, rescaled to [0,1], and (for uint8 output)
    scaled to [0,255] and rounded.
    """
    tensor = tensor.float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0, 1]
    img_np = tensor.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()

    return img_np.astype(out_type)

def convert2np(tensor):
    """Convert a [0,1] image tensor (squeezable to CHW) to an HWC uint8 numpy array."""
    return tensor.cpu().mul(255).clamp(0, 255).byte().squeeze().permute(1, 2, 0).numpy()


def adjust_learning_rate(optimizer, epoch, step_size, lr_init, gamma):
    """Step-decay schedule: lr = lr_init * gamma ** (epoch // step_size), applied to all param groups."""
    factor = epoch // step_size
    lr = lr_init * (gamma ** factor)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_state_dict(path):
    """Load a checkpoint file and return its state dict with any DataParallel
    'module.' key prefix removed, so it fits a non-parallel model."""

    state_dict = torch.load(path)
    new_state_dcit = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            name = k[7:]  # drop the leading 'module.'
        else:
            name = k
        new_state_dcit[name] = v
    return new_state_dcit


def add_fast_arg(parser):
    """Add --fast: torch.compile + bf16 autocast (experimental speed option)."""
    parser.add_argument('--fast', action='store_true', default=False,
                        help='experimental: torch.compile the model and use bf16 autocast for '
                             'inference (training already uses bf16 AMP unless --no_amp)')


def compile_model(model, example, train=False, autocast=False, dynamic=None, backend='inductor'):
    """torch.compile `model`, forcing the real compile with `example` now.

    Falls back to the original eager model (with a warning) if compilation fails, e.g.
    inductor needs the Python development headers and a C compiler. With train=True the
    warm-up runs forward+backward so the training graph is compiled too; gradients are
    cleared afterwards (no optimizer step).
    """
    import torch
    try:
        cm = torch.compile(model, backend=backend, dynamic=dynamic)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
            if train:
                cm(example).float().mean().backward()
                model.zero_grad(set_to_none=True)
            else:
                with torch.inference_mode():
                    cm(example)
        return cm
    except Exception as e:  # compile errors surface lazily, at the first call
        print("[--fast] torch.compile unavailable, continuing eager: {}: {}".format(
            type(e).__name__, str(e).splitlines()[0][:200]))
        torch._dynamo.reset()
        model.zero_grad(set_to_none=True)
        return model
