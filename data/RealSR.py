"""RealSR dataset (real-world LR/HR pairs, pre-decoded to .npy).

Role in ESRT: optional alternative training dataset for real-world degradations.
Not used by train.py by default (it uses DIV2K). `scale` is fixed to 1 because
the RealSR crops are stored already aligned at the same resolution.
"""
import os
from data import common
import numpy as np
import torch.utils.data as data
from data.image_folder import make_dataset


class srdata(data.Dataset):
    """Patch dataset over <root>/RealSR_decoded/{TrainGT,TrainLR}.

    `args` is the argparse namespace (root, test_every, n_train, batch_size,
    n_val, n_colors, rgb_range, patch_size). train=False selects the eval split.
    """
    def __init__(self, args, train=True):
        self.args = args
        self.train = train
        self.root = self.args.root
        self.split = 'train' if train else 'test'
        self.scale = 1
        self.repeat = args.test_every // (args.n_train // args.batch_size)  # 300 / (60/6) = 30
        self._set_filesystem(self.root)
        self.images_hr, self.images_lr = self._scan_npy()

    def _set_filesystem(self, dir_data):
        """Set HR (TrainGT) / LR (TrainLR) directories and the .npy file extension."""
        self.apath = dir_data + '/RealSR_decoded'
        self.dir_hr = os.path.join(self.apath, 'TrainGT')
        self.dir_lr = os.path.join(self.apath, 'TrainLR')
        self.ext = '.npy'

    def __getitem__(self, idx):
        """Load a pair, crop/augment, fix channels, and return CHW float tensors."""
        lr, hr = self._load_file(idx)
        lr, hr = self._get_patch(lr, hr)
        lr, hr = common.set_channel(lr, hr, n_channels=self.args.n_colors)
        lr_tensor, hr_tensor = common.np2Tensor(lr, hr, rgb_range=self.args.rgb_range)
        return lr_tensor, hr_tensor

    def __len__(self):
        """Train: n_train * repeat samples per epoch; eval: n_val."""
        if self.train:
            return self.args.n_train * self.repeat
        else:
            return self.args.n_val

    def _get_index(self, idx):
        """Map a (repeated) sample index to a file index (wraps modulo n_train in training)."""
        if self.train:
            return idx % self.args.n_train
        else:
            return idx

    def _get_patch(self, img_in, img_tar):
        """Training: random aligned crop + augmentation. Eval: crop target to match input size."""
        patch_size = self.args.patch_size
        scale = self.scale
        if self.train:
            img_in, img_tar = common.get_patch(
                img_in, img_tar, patch_size=patch_size, scale=scale)
            img_in, img_tar = common.augment(img_in, img_tar)

        else:
            ih, iw = img_in.shape[:2]
            img_tar = img_tar[0:ih * scale, 0:iw * scale, :]

        return img_in, img_tar

    def _scan(self):
        """Build file lists from numbered names ('000001.npy', '000001x1.npy', ...).

        Unused legacy path (see _scan_npy). It references `args.offset_val`,
        which train.py does not define, so it would fail for the eval split.
        """
        list_hr = []
        list_lr = []
        if self.train:
            idx_begin = 0
            idx_end = self.args.n_train
        else:
            idx_begin = self.args.n_train
            idx_end = self.args.offset_val + self.args.n_val

        for i in range(idx_begin + 1, idx_end + 1):
            filename = '{:0>6}'.format(i)
            list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
            list_lr.append(os.path.join(self.dir_lr, '{}x{}{}'.format(filename, self.scale, self.ext)))

        return list_hr, list_lr

    def _scan_npy(self):
        """Return sorted HR and LR file lists found by directory scan (kept paired by sort order)."""
        list_hr = sorted(make_dataset(self.dir_hr))
        list_lr = sorted(make_dataset(self.dir_lr))
        return list_hr, list_lr

    def _load_file(self, idx):
        """Load the (lr, hr) .npy arrays for sample `idx`."""
        idx = self._get_index(idx)
        lr = np.load(self.images_lr[idx])
        hr = np.load(self.images_hr[idx])
        return lr, hr
