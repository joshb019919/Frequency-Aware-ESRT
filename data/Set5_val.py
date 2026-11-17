"""Validation dataset: full-size LR/HR image pairs (e.g. Set5) loaded from two folders.

Role in ESRT: supplies the images train.py uses to report PSNR/SSIM after each
epoch, so training progress can be tracked on a standard benchmark.
"""
import torch.utils.data as data
from os.path import join
from os import listdir
from torchvision.transforms import Compose, ToTensor
from PIL import Image
import numpy as np


def img_modcrop(image, modulo):
    """Crop a PIL image (bottom/right) so width and height are multiples of `modulo`."""
    sz = image.size
    w = np.int32(sz[0] / modulo) * modulo
    h = np.int32(sz[1] / modulo) * modulo
    out = image.crop((0, 0, w, h))
    return out


def np2tensor():
    """Return a transform converting a PIL image to a CxHxW float tensor in [0, 1]."""
    return Compose([
        ToTensor(),
    ])


def is_image_file(filename):
    """True for .bmp, .png or .jpg files."""
    return any(filename.endswith(extension) for extension in [".bmp", ".png", ".jpg"])


def load_image(filepath):
    """Open an image with PIL and convert it to RGB."""
    return Image.open(filepath).convert('RGB')


class DatasetFromFolderVal(data.Dataset):
    """Pairs of (LR input, HR target) tensors for validation.

    Files in `hr_dir` and `lr_dir` are matched by sorted order, so both folders
    must contain the same images in the same alphabetical order. The HR image
    is mod-cropped to a multiple of `upscale` so it matches the LR size * scale.
    """
    def __init__(self, hr_dir, lr_dir, upscale):
        super(DatasetFromFolderVal, self).__init__()
        self.hr_filenames = sorted([join(hr_dir, x) for x in listdir(hr_dir) if is_image_file(x)])
        self.lr_filenames = sorted([join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)])
        self.upscale = upscale

    def __getitem__(self, index):
        """Return (lr_tensor, hr_tensor) for pair `index`, both in [0, 1]."""
        input = load_image(self.lr_filenames[index])
        target = load_image(self.hr_filenames[index])
        input = np2tensor()(input)
        target = np2tensor()(img_modcrop(target, self.upscale))

        return input, target

    def __len__(self):
        return len(self.lr_filenames)
