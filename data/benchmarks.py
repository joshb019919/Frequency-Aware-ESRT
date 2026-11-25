"""Registry of SR benchmark sets (HR folder + LR folder per scale) shared by train.py and test.py.

Defaults: training validation uses set14 + set5, final testing uses bsds300 + manga109.
bsds300/manga109 need `python scripts/prepare_benchmarks.py` once to create their HR/LR folders.
"""
import os
from os import listdir
from os.path import join, splitext

from data.Set5_val import DatasetFromFolderVal, is_image_file

DEFAULT_VAL_SETS = "set14,set5"
DEFAULT_TEST_SETS = "bsds300,manga109"
DEFAULT_BENCH_ROOT = "/data/data"

# name -> (HR folder, LR folder template); relative paths are tried in the repo first, then under bench_root.
_LAYOUT = {
    "set5": ("Test_Datasets/Set5", "Test_Datasets/Set5_LR/x{s}"),
    "set14": ("set14/Set14_HR", "set14/Set14_LR_x{s}"),
    "b100": ("b100/HR", "b100/LR/x{s}"),
    "bsds300": ("bsds300/HR", "bsds300/LR/x{s}"),
    "manga109": ("manga109/HR", "manga109/LR/x{s}"),
}
NAMES = tuple(_LAYOUT)


def parse_sets(text):
    """'set14,set5' -> ['set14', 'set5'] (validated, order kept, duplicates dropped)."""
    out = []
    for n in (t.strip().lower() for t in text.split(",") if t.strip()):
        if n not in _LAYOUT:
            raise ValueError("unknown benchmark set {!r}; choose from {}".format(n, ", ".join(NAMES)))
        if n not in out:
            out.append(n)
    return out


def resolve(name, scale, bench_root=DEFAULT_BENCH_ROOT):
    """Return (hr_dir, lr_dir) for benchmark `name` at `scale`; raises FileNotFoundError with a hint."""
    hr, lr = _LAYOUT[name]
    lr = lr.format(s=scale)

    def pick(rel):
        for cand in (rel, join(bench_root, rel)):
            if os.path.isdir(cand):
                return cand
        return None

    hr_dir, lr_dir = pick(hr), pick(lr)
    if hr_dir is None or lr_dir is None:
        hint = " (run scripts/prepare_benchmarks.py)" if name in ("bsds300", "manga109") else ""
        raise FileNotFoundError("benchmark {!r} x{}: missing {} under {}{}".format(
            name, scale, hr if hr_dir is None else lr, bench_root, hint))
    return hr_dir, lr_dir


def find_lr(lr_dir, stem, scale, lr_ext):
    """LR file for HR image `stem`: '<stem>x<scale><ext>' (Set5/B100 style) or '<stem><ext>' (Set14)."""
    for cand in (join(lr_dir, "{}x{}{}".format(stem, scale, lr_ext)), join(lr_dir, stem + lr_ext)):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError("no LR image for {!r} in {}".format(stem, lr_dir))


class PairedVal(DatasetFromFolderVal):
    """Validation pairs matched by file name (not by sorted position), so naming schemes can differ."""

    def __init__(self, hr_dir, lr_dir, upscale):
        self.upscale = upscale
        lr_names = {splitext(f)[0]: f for f in listdir(lr_dir) if is_image_file(f)}
        self.hr_filenames, self.lr_filenames = [], []
        for f in sorted(x for x in listdir(hr_dir) if is_image_file(x)):
            stem = splitext(f)[0]
            lr = "{}x{}".format(stem, upscale)
            if lr not in lr_names:
                lr = stem
            if lr not in lr_names:
                raise FileNotFoundError("no LR image for {!r} in {}".format(stem, lr_dir))
            self.hr_filenames.append(join(hr_dir, f))
            self.lr_filenames.append(join(lr_dir, lr_names[lr]))
