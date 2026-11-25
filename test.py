"""Evaluate a trained ESRT checkpoint on a benchmark set.

Role in ESRT: the evaluation/inference entry point; produces the PSNR/SSIM and
timing numbers reported for the model and writes the super-resolved images.

For every image in each test set (default bsds300 + manga109, or one --test_hr_folder) this loads the matching LR image from
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
from data import benchmarks
import plotting

# Testing settings

parser = argparse.ArgumentParser(description='ESRT')
parser.add_argument("--test_hr_folder", type=str, default=None,
                    help='folder of target images; with --test_lr_folder, tests that single folder instead of --test_sets')
parser.add_argument("--test_lr_folder", type=str, default=None,
                    help='folder of the input images (named <name>x<scale>.<ext>)')
parser.add_argument("--test_sets", type=str, default=benchmarks.DEFAULT_TEST_SETS,
                    help='comma-separated benchmark sets tested when no folders are given, from: ' + ','.join(benchmarks.NAMES))
parser.add_argument("--bench_root", type=str, default=benchmarks.DEFAULT_BENCH_ROOT,
                    help='directory holding the benchmark datasets')
parser.add_argument("--output_folder", type=str, default=None,
                    help='SR output folder (default results/<set>/x<scale>; with several sets, <folder>/<set>)')
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True,
                    help='save a test summary plot (per-set mean PSNR/SSIM bars + per-image PSNR)')
parser.add_argument("--plot_path", type=str, default=None,
                    help='summary plot path (default <checkpoint dir>/test_x<scale>.png)')
utils.add_fast_arg(parser)
esrt.add_model_args(parser)  # must match the flags the checkpoint was trained with
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

def build_model():
    try:
        model = esrt.ESRT(upscale = opt.upscale_factor, **esrt.model_kwargs(opt))
    except ValueError as e:
        parser.error(str(e))
    model_dict = utils.load_state_dict(opt.checkpoint)
    # Fail loudly if the checkpoint was trained with different --freq/--conv/--global_res flags.
    _missing = [k for k in model.state_dict() if k not in model_dict and 'tail' not in k]
    if _missing:
        raise SystemExit("Checkpoint is missing {} model keys (e.g. {}); pass the same "
                         "--freq/--conv/--global_res flags used for training.".format(len(_missing), _missing[0]))
    model.load_state_dict(model_dict, strict=False)#True)
    model.eval()
    utils.enable_tf32(opt.fast and cuda)
    if cuda:
        model = model.to(device).to(memory_format=torch.channels_last)
    # --fast at test time = bf16 autocast + TF32 only. torch.compile is deliberately not used here:
    # dynamic=True fails in inductor for the DWT models (Triton int64 error, falls back to eager), and
    # static shapes recompile per distinct tile size, which costs far more than the ~1 s Set5 run.
    return model


def run_set(model, hr_folder, lr_folder, out_folder):
    """Super-resolve every image of one set; returns (per-image psnr, ssim, ms) arrays."""
    # Benchmarks ship as .png or .bmp (e.g. original Set5/Set14 are .bmp); detect which
    # one the HR and LR folders actually contain instead of guessing from the folder name.
    ext = utils.detect_image_ext(hr_folder)
    lr_ext = utils.detect_image_ext(lr_folder)
    filelist = sorted(utils.get_list(hr_folder, ext=ext))
    # Per-image metrics, averaged by the caller.
    psnr_list = np.zeros(len(filelist))
    ssim_list = np.zeros(len(filelist))
    time_list = np.zeros(len(filelist))
    # CUDA events for accurate GPU timing of the forward pass.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    os.makedirs(out_folder, exist_ok=True)

    for i, imname in enumerate(filelist):
        stem = os.path.splitext(os.path.basename(imname))[0]
        im_gt = cv2.imread(imname, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
        # Crop HR so its size is divisible by the scale (matches the LR image size).
        im_gt = utils.modcrop(im_gt, opt.upscale_factor)
        # LR file name convention: <name>x<scale><ext> (or plain <name><ext>, as in Set14)
        im_l = cv2.imread(benchmarks.find_lr(lr_folder, stem, opt.upscale_factor, lr_ext), cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
        # HWC uint8 [0,255] -> NCHW float tensor in [0,1].
        im_input = im_l / 255.0
        im_input = np.transpose(im_input, (2, 0, 1))
        im_input = im_input[np.newaxis, ...]
        im_input = torch.from_numpy(im_input).float()

        if cuda:
            im_input = im_input.to(device)

        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=bool(opt.fast and cuda)):
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
        cv2.imwrite(os.path.join(out_folder, stem + 'x' + str(opt.upscale_factor) + '.png'), out_img[:, :, [2, 1, 0]])
    return psnr_list, ssim_list, time_list


model = build_model()

if bool(opt.test_hr_folder) != bool(opt.test_lr_folder):
    parser.error("--test_hr_folder and --test_lr_folder must be given together")
if opt.test_hr_folder:
    # Single explicit folder pair (the original behaviour; experiments.py parses the final line).
    jobs = [(os.path.basename(os.path.normpath(opt.test_hr_folder)), opt.test_hr_folder, opt.test_lr_folder,
             opt.output_folder or 'results/Set5/x2')]
else:
    try:
        names = benchmarks.parse_sets(opt.test_sets)
        jobs = []
        for name in names:
            hr, lr = benchmarks.resolve(name, opt.upscale_factor, opt.bench_root)
            out = os.path.join(opt.output_folder, name) if opt.output_folder else \
                os.path.join('results', name, 'x{}'.format(opt.upscale_factor))
            jobs.append((name, hr, lr, out))
    except (ValueError, FileNotFoundError) as e:
        parser.error(str(e))

summary = {}
for name, hr, lr, out in jobs:
    psnr_list, ssim_list, time_list = run_set(model, hr, lr, out)
    summary[name] = {"psnr": psnr_list.tolist(), "ssim": ssim_list.tolist()}
    if len(jobs) > 1:
        print("===> Test set {} ({} images)".format(name, len(psnr_list)))
    print("Mean PSNR: {}, SSIM: {}, TIME: {} ms".format(np.mean(psnr_list), np.mean(ssim_list), np.mean(time_list)))

if opt.plot:
    plot_path = opt.plot_path or os.path.join(os.path.dirname(os.path.abspath(opt.checkpoint)),
                                              'test_x{}.png'.format(opt.upscale_factor))
    if plotting.plot_test_summary(plot_path, summary, title=os.path.basename(os.path.dirname(os.path.abspath(opt.checkpoint)))):
        print("Test summary plot saved to {}".format(plot_path))
