"""Generic image-folder dataset and file-scanning helpers.

Role in ESRT: provides make_dataset(), the recursive file lister imported by
data/RealSR.py; the ImageFolder dataset itself is not used by training.
"""
###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path

def default_flist_reader(flist):
    """Read a text file listing one image path per line; return the stripped paths."""
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.npy'
]


def is_image_file(filename):
    """True if `filename` ends with one of IMG_EXTENSIONS (images or .npy arrays)."""
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    """Recursively collect paths of all image files under `dir` (including `dir` itself).

    Raises AssertionError if `dir` is not a directory.
    """
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def default_loader(path):
    """Open an image with PIL and convert it to RGB."""
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):
    """Flat dataset over every image found under `root`.

    Args:
        root: directory to scan.
        transform: optional callable applied to each loaded image.
        return_paths: if True, __getitem__ returns (img, path).
        loader: function mapping a path to an image.
    """

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)