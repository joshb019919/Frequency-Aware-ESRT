"""Evaluate a trained ESRT checkpoint on a benchmark set.

Role in ESRT: the evaluation/inference entry point; produces the PSNR/SSIM and
timing numbers reported for the model and writes the super-resolved images.

For every image in --test_hr_folder this loads the matching LR image from
--test_lr_folder (named <name>x<scale>.<ext>), super-resolves it, saves the
result as PNG to --output_folder, and reports mean PSNR / SSIM (Y channel by
default) and mean GPU inference time. Requires CUDA (uses CUDA events).

Usually launched through main.py:  python main.py --test ...
"""
import argparse
import torch
import os
import numpy as np
import utils
import skimage.color as sc
import cv2
from model import  esrt

# Testing settings

parser = argparse.ArgumentParser(description='ESRT')
parser.add_argument("--test_hr_folder", type=str, default='Test_Datasets/Set5/',
                    help='the folder of the target images')
parser.add_argument("--test_lr_folder", type=str, default='Test_Datasets/Set5_LR/x2/',
                    help='the folder of the input images')
parser.add_argument("--output_folder", type=str, default='results/Set5/x2')
parser.add_argument("--checkpoint", type=str, default='checkpoints/IMDN_x2.pth',
                    help='checkpoint folder to use')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use cuda')
parser.add_argument("--upscale_factor", type=int, default=2,
                    help='upscaling factor')
parser.add_argument("--is_y", action='store_true', default=True,
                    help='evaluate on y channel, if False evaluate on RGB channels')
opt = parser.parse_args()

print(opt)
def forward_chop(model, x, scale, shave=10, min_size=60000):
    """Memory-saving inference: split the LR image into 4 overlapping quadrants.

    Each quadrant (half size + `shave` px overlap) is super-resolved separately
    and stitched back together without the overlap. Quadrants of area
    >= `min_size` pixels are split recursively.

    Args:
        model: SR network.
        x: LR input tensor (B, C, H, W).
        scale: upscaling factor of the model (x2/x3/x4).
        shave: overlap (LR pixels) between quadrants.
        min_size: pixel-area threshold below which a quadrant is run directly.

    Returns:
        SR tensor (B, C, scale*H, scale*W).
    """
    n_GPUs = 1  # leftover from the multi-GPU EDSR/RCAN version
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    # Four corner crops: top-left, top-right, bottom-left, bottom-right.
    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        sr_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            sr_batch = model(lr_batch)
            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            forward_chop(model, patch, scale, shave=shave, min_size=min_size) \
            for patch in lr_list
        ]

    # Convert all sizes to the HR domain.
    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    # Paste the non-overlapping part of each SR quadrant into the output canvas.
    output = x.new(b, c, h, w)
    output[:, :, 0:h_half, 0:w_half] \
        = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output
cuda = opt.cuda
device = torch.device('cuda' if cuda else 'cpu')

# Set5/Set14 ship as .bmp; other benchmarks as .png. The folder name is taken
# from the second-to-last path component, so --test_hr_folder needs a trailing '/'.
filepath = opt.test_hr_folder
if filepath.split('/')[-2] == 'Set5' or filepath.split('/')[-2] == 'Set14':
    ext = '.bmp'
else:
    ext = '.png'

filelist = utils.get_list(filepath, ext=ext)
# Per-image metrics, averaged at the end.
psnr_list = np.zeros(len(filelist))
ssim_list = np.zeros(len(filelist))
time_list = np.zeros(len(filelist))

# Build the model and load weights (utils strips any DataParallel 'module.'
# prefix; ESRT.load_state_dict tolerates missing/extra keys with strict=False).
model =  esrt.ESRT(upscale = opt.upscale_factor)#
model_dict = utils.load_state_dict(opt.checkpoint)
model.load_state_dict(model_dict, strict=False)#True)

i = 0
# CUDA events for accurate GPU timing of the forward pass.
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for imname in filelist:
    im_gt = cv2.imread(imname, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
    # Crop HR so its size is divisible by the scale (matches the LR image size).
    im_gt = utils.modcrop(im_gt, opt.upscale_factor)
    # LR file name convention: <name>x<scale><ext>
    im_l = cv2.imread(opt.test_lr_folder + imname.split('/')[-1].split('.')[0] + 'x' + str(opt.upscale_factor) + ext, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
    # Grayscale fallback: replicate to 3 channels. (cv2.IMREAD_COLOR always
    # returns 3 channels, so this branch is effectively never taken.)
    if len(im_gt.shape) < 3:
        im_gt = im_gt[..., np.newaxis]
        im_gt = np.concatenate([im_gt] * 3, 2)
        im_l = im_l[..., np.newaxis]
        im_l = np.concatenate([im_l] * 3, 2)
    # HWC uint8 [0,255] -> NCHW float tensor in [0,1].
    im_input = im_l / 255.0
    im_input = np.transpose(im_input, (2, 0, 1))
    im_input = im_input[np.newaxis, ...]
    im_input = torch.from_numpy(im_input).float()

    if cuda:
        model = model.to(device)
        im_input = im_input.to(device)

    with torch.no_grad():
        start.record()
        out = forward_chop(model, im_input, opt.upscale_factor) #model(im_input)
        end.record()
        torch.cuda.synchronize()
        time_list[i] = start.elapsed_time(end)  # milliseconds

    # Back to uint8 HWC, then shave `scale` border pixels (standard SR evaluation).
    out_img = utils.tensor2np(out.detach()[0])
    crop_size = opt.upscale_factor
    cropped_sr_img = utils.shave(out_img, crop_size)
    cropped_gt_img = utils.shave(im_gt, crop_size)
    # Metrics on the Y (luma) channel of YCbCr, or on RGB if --is_y is off.
    if opt.is_y is True:
        im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
        im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
    else:
        im_label = cropped_gt_img
        im_pre = cropped_sr_img
    psnr_list[i] = utils.compute_psnr(im_pre, im_label)
    ssim_list[i] = utils.compute_ssim(im_pre, im_label)


    # Save the SR image (RGB -> BGR for cv2) as <name>x<scale>.png.
    # NOTE: despite the name, `output_folder` here is the output *file* path.
    output_folder = os.path.join(opt.output_folder,
                                 imname.split('/')[-1].split('.')[0] + 'x' + str(opt.upscale_factor) + '.png')

    if not os.path.exists(opt.output_folder):
        os.makedirs(opt.output_folder)

    cv2.imwrite(output_folder, out_img[:, :, [2, 1, 0]])
    i += 1


print("Mean PSNR: {}, SSIM: {}, TIME: {} ms".format(np.mean(psnr_list), np.mean(ssim_list), np.mean(time_list)))
