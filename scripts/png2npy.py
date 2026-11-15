"""Convert a directory tree of .png images into per-image .npy arrays.

Role in ESRT: one-off dataset preparation step that produces the DF2K_decoded
folder layout expected by data/DIV2K.py (and train.py / main.py).

Decoding PNGs on every training step is slow; pre-decoding them to .npy lets
the DIV2K dataset load them with np.load. The directory structure under
--pathFrom is mirrored under --pathTo.

  python scripts/png2npy.py --pathFrom DIV2K/ --pathTo DF2K_decoded/ [--select X4]

Run it once per dataset before training. --pathFrom must end with '/' (the
output sub-path is computed by slicing off len(pathFrom) characters).
"""
import os
import argparse
import skimage.io as sio
import numpy as np
import pdb

parser = argparse.ArgumentParser(description='Pre-processing .png images')
parser.add_argument('--pathFrom', default='',
                    help='directory of images to convert')
parser.add_argument('--pathTo', default='',
                    help='directory of images to save')
parser.add_argument('--split', default=True,
                    help='save individual images')
parser.add_argument('--select', default='',
                    help='select certain path')

args = parser.parse_args()

# Walk the source tree; for each leaf directory (no sub-directories) convert its PNGs.
for (path, dirs, files) in os.walk(args.pathFrom):
    print(path)
    # Mirror the source sub-path under the destination root.
    targetDir = os.path.join(args.pathTo, path[len(args.pathFrom):])
    # pdb.set_trace()
    # --select: only process directories whose path contains this substring.
    if len(args.select) > 0 and path.find(args.select) == -1:
        continue

    if not os.path.exists(targetDir):
        os.mkdir(targetDir)

    if len(dirs) == 0:
        pack = {}  # unused leftover from the packed-file variant of this script
        n = 0
        for fileName in files:
            (idx, ext) = os.path.splitext(fileName)
            # pdb.set_trace()
            if ext == '.png':
                image = sio.imread(os.path.join(path, fileName))
                # NOTE: --split defaults to True and any CLI value is a non-empty
                # string (also truthy), so saving is effectively always on.
                if args.split:
                    np.save(os.path.join(targetDir, idx + '.npy'), image)
                n += 1
                if n % 100 == 0:
                    print('Converted ' + str(n) + ' images.')
