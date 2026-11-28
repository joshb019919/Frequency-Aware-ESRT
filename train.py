"""Train ESRT for single-image super-resolution.

Role in ESRT: the training entry point. It wires together the data pipeline
(data/), the network (model/esrt.py) and the helpers in utils.py.

Script-style entry point (everything runs inside main(), under an
`if __name__ == '__main__':` guard so DataLoader worker processes started with
the 'spawn'/'forkserver' method can re-import this file safely): parses CLI
args, builds the DIV2K training loader and the validation loaders (--val_sets, default set14+set5), trains with L1 loss + Adam and a
step-decay LR schedule, validates (Y-channel PSNR/SSIM) every epoch and saves a
checkpoint every 10 epochs to experiment/checkpoint_ESRT_x<scale>/.

Usually launched through main.py:  python main.py --train --hf_repo USER/DIV2K
"""
import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import esrt
from data import DIV2K, benchmarks
import metrics_freq
import plotting
import utils
import skimage.color as sc
import random
from collections import OrderedDict
import datetime
from importlib import import_module
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Training settings
parser = argparse.ArgumentParser(description="ESRT")
parser.add_argument("--batch_size", type=int, default=16, help="training batch size")
parser.add_argument("--testBatchSize", type=int, default=1, help="testing batch size")
parser.add_argument("-nEpochs", type=int, default=1000, help="number of epochs to train")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning Rate. Default=2e-4")
parser.add_argument("--step_size", type=int, default=200, help="learning rate decay per N epochs")
# NOTE: declared type=int but the default is a float; a value passed on the CLI
# would be parsed as int, so fractional decay factors can't be given on the command line.
parser.add_argument("--gamma", type=int, default=0.5, help="learning rate decay factor for step decay")
parser.add_argument("--cuda", action="store_true", default=True, help="use cuda")
parser.add_argument("--resume", default="", type=str, help="path to checkpoint")
parser.add_argument("--start-epoch", default=1, type=int, help="manual epoch number")
parser.add_argument("--threads", type=int, default=8, help="number of threads for data loading")
parser.add_argument("--root", type=str, default="/data0/luzs/dataset/", help='dataset directory')
parser.add_argument("--n_train", type=int, default=800, help="number of training set")
parser.add_argument("--n_val", type=int, default=1, help="number of validation set")
parser.add_argument("--test_every", type=int, default=1000)  # only used by RealSR.py to size an epoch
parser.add_argument("--scale", type=int, default=2, help="super-resolution scale")
parser.add_argument("--patch_size", type=int, default=192, help="output patch size")
parser.add_argument("--rgb_range", type=int, default=1, help="maxium value of RGB")
parser.add_argument("--n_colors", type=int, default=3, help="number of color channels to use")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained models")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--isY", action="store_true", default=True)  # evaluate PSNR/SSIM on the Y channel
parser.add_argument("--ext", type=str, default='.npy')  # training image format: '.npy' (pre-decoded) or '.png'
parser.add_argument("--phase", type=str, default='train')
parser.add_argument("--model", type=str, default='ESRT')  # currently unused; ESRT is hard-coded below
esrt.add_model_args(parser)  # --freq / --conv / --global_res
parser.add_argument("--amp", action="store_true", default=True, help="deprecated no-op: bf16 AMP is on by default (CUDA)")
parser.add_argument("--no_amp", action="store_true", default=False, help="disable bf16 AMP and channels_last (plain fp32 training)")
parser.add_argument("--val_every", type=int, default=5, help="validate every N epochs (and once after the last epoch)")
parser.add_argument("--ref_history", type=str, default=None,
                    help="history.json of a finished run (e.g. lowpass); its final level is drawn as a dashed black line in the curves")
parser.add_argument("--save_dir", type=str, default=None, help="checkpoint folder (default experiment/checkpoint_ESRT_x<scale>[variant]/)")
parser.add_argument("--val_sets", type=str, default=benchmarks.DEFAULT_VAL_SETS,
                    help="comma-separated validation sets, from: " + ",".join(benchmarks.NAMES))
parser.add_argument("--bench_root", type=str, default=benchmarks.DEFAULT_BENCH_ROOT,
                    help="directory holding the benchmark datasets (set14, bsds300, ...)")
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True,
                    help="write <save_dir>/curves.png (train loss | val PSNR | val SSIM), updated at each validation")
metrics_freq.add_args(parser)  # --hf_metrics / --perceptual / --wavelets
utils.add_fast_arg(parser)

def train(epoch, model, optimizer, l1_criterion, training_data_loader, args, device):
    """Run one training epoch over `training_data_loader`.

    Applies the step-decay LR schedule for `epoch`, then optimises the L1 loss
    between the model output and the HR patch, logging every 100 iterations.
    """
    model.train()
    utils.adjust_learning_rate(optimizer, epoch, args.step_size, args.lr, args.gamma)
    print('epoch =', epoch, 'lr = ', optimizer.param_groups[0]['lr'])
    loss_sum, n_iter = 0.0, 0  # running sum stays on the GPU: no per-iteration sync
    for iteration, (lr_tensor, hr_tensor) in enumerate(training_data_loader, 1):

        if args.cuda:
            lr_tensor = lr_tensor.to(device, non_blocking=True)  # ranges from [0, 1]
            hr_tensor = hr_tensor.to(device, non_blocking=True)  # ranges from [0, 1]
            if args.use_amp:
                lr_tensor = lr_tensor.contiguous(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.use_amp):
            sr_tensor = model(lr_tensor)
        loss_l1 = l1_criterion(sr_tensor.float(), hr_tensor)
        loss_sr = loss_l1

        loss_sr.backward()
        optimizer.step()
        loss_sum += loss_l1.detach()
        n_iter = iteration
        if iteration % 100 == 0:
            print("===> Epoch[{}]({}/{}): Loss_l1: {:.5f}".format(epoch, iteration, len(training_data_loader),
                                                                  loss_l1.item()))
    if n_iter:
        avg = float(loss_sum) / n_iter
        print("===> Epoch[{}] avg Loss_l1: {:.5f}".format(epoch, avg))
        return avg
def forward_chop(model, x, scale, shave=10, min_size=60000):
    """Memory-saving inference: split the LR image into 4 overlapping quadrants.

    Each quadrant (half size + `shave` px of overlap) is super-resolved and the
    results are stitched back, discarding the overlap borders. Quadrants whose
    area is >= `min_size` pixels are split recursively.

    Args:
        model: SR network mapping (B, C, h, w) -> (B, C, scale*h, scale*w).
        x: LR input tensor (B, C, H, W).
        scale: upscaling factor.
        shave: overlap (LR pixels) between quadrants to hide border artefacts.
        min_size: pixel-area threshold below which a quadrant is run directly.

    Returns:
        SR tensor (B, C, scale*H, scale*W).
    """
    n_GPUs = 1  # leftover from the multi-GPU EDSR/RCAN version; chunks run one at a time
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

def evaluate(scale, model, loader, args, device, freq_eval=None, freq_rows=None):
    """Mean (PSNR, SSIM) of `model` over one validation loader.

    Images are super-resolved with forward_chop, shaved by `scale` border
    pixels, and (if --isY) compared on the quantised Y channel of YCbCr. If `freq_eval` (a
    metrics_freq.FreqEvaluator) is given, its per-image metric dicts are appended to `freq_rows`.
    """
    model.eval()
    avg_psnr, avg_ssim = 0, 0
    for batch in loader:
        lr_tensor, hr_tensor = batch[0], batch[1]
        if args.cuda:
            lr_tensor = lr_tensor.to(device, non_blocking=True)
            hr_tensor = hr_tensor.to(device, non_blocking=True)

        with torch.inference_mode():
            pre = forward_chop(model, lr_tensor, scale)#model(lr_tensor)

        sr_img = utils.tensor2np(pre.detach()[0])
        gt_img = utils.tensor2np(hr_tensor.detach()[0])
        crop_size = args.scale
        cropped_sr_img = utils.shave(sr_img, crop_size)
        cropped_gt_img = utils.shave(gt_img, crop_size)
        if args.isY is True:
            im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
            im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
        else:
            im_label = cropped_gt_img
            im_pre = cropped_sr_img
        avg_psnr += utils.compute_psnr(im_pre, im_label)
        avg_ssim += utils.compute_ssim(im_pre, im_label)
        if freq_eval is not None and freq_rows is not None:
            freq_rows.append(freq_eval(cropped_sr_img, cropped_gt_img))
    return avg_psnr / len(loader), avg_ssim / len(loader)


def valid(scale, model, loaders, args, device, epochs_trained=0, freq_eval=None, extras=None):
    """Validate on one DataLoader or a {name: DataLoader} dict; print and return the scores.

    With several sets, prints one '===> Valid[name] ...' line per set, then the
    '===> Valid. psnr/ssim' line (the one experiments.py parses) holding the mean over the sets.
    Returns {name: (psnr, ssim)}. With `freq_eval`, also prints a '===> Freq[name]' line per set and,
    if `extras` is a dict, fills it with {name: mean frequency metrics}.
    """
    if not isinstance(loaders, dict):
        loaders = {"val": loaders}
    results = {}
    for name, loader in loaders.items():
        rows = [] if freq_eval is not None else None
        results[name] = evaluate(scale, model, loader, args, device, freq_eval, rows)
        if rows:
            mean = metrics_freq.mean_metrics(rows)
            print("===> Freq[{}] {} (weights after {} epochs)".format(
                name, metrics_freq.summary_line(mean, freq_eval.wavelets), epochs_trained))
            if extras is not None:
                extras[name] = mean
        if len(loaders) > 1:
            print("===> Valid[{}] psnr: {:.4f}, ssim: {:.4f} (weights after {} epochs)".format(
                name, results[name][0], results[name][1], epochs_trained))
    psnr = sum(r[0] for r in results.values()) / len(results)
    ssim = sum(r[1] for r in results.values()) / len(results)
    print("===> Valid. psnr: {:.4f}, ssim: {:.4f} (weights after {} epochs)".format(psnr, ssim, epochs_trained))
    return results


def save_checkpoint(epoch, model, args):
    """Save the model weights to experiment/checkpoint_ESRT_x<scale>/epoch_<epoch>.pth."""
    model_folder = args.save_dir or "experiment/checkpoint_ESRT_x{}{}/".format(args.scale, esrt.variant_tag(args))
    model_folder = model_folder.rstrip("/") + "/"
    model_out_path = model_folder + "epoch_{}.pth".format(epoch)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(model.state_dict(), model_out_path)
    print("===> Checkpoint saved to {}".format(model_out_path))

def print_network(net):
    """Print the total number of parameters in `net`."""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)


def main():
    args = parser.parse_args()
    print(args)
    args.use_amp = bool(args.cuda and torch.cuda.is_available() and not args.no_amp)
    if args.val_every < 1:
        parser.error("--val_every must be >= 1")
    utils.enable_tf32(args.use_amp)
    torch.backends.cudnn.benchmark = args.use_amp  # autotune conv algorithms (fixed-size training patches)
    # random seed
    seed = args.seed
    if seed is None:
        seed = random.randint(1, 10000)
    print("Ramdom Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)

    cuda = args.cuda
    device = torch.device('cuda' if cuda else 'cpu')

    print("===> Loading datasets")

    # Training: random HR/LR patch pairs from DIV2K (see data/DIV2K.py).
    # Validation: full-size images from every --val_sets benchmark (default set14 + set5).
    trainset = DIV2K.div2k(args)
    try:
        val_names = benchmarks.parse_sets(args.val_sets)
        val_loaders = {}
        for name in val_names:
            hr_dir, lr_dir = benchmarks.resolve(name, args.scale, args.bench_root)
            val_loaders[name] = DataLoader(dataset=benchmarks.PairedVal(hr_dir, lr_dir, args.scale),
                                           num_workers=args.threads, batch_size=args.testBatchSize,
                                           shuffle=False, pin_memory=True, persistent_workers=args.threads > 0)
    except (ValueError, FileNotFoundError) as e:
        parser.error(str(e))
    training_data_loader = DataLoader(dataset=trainset, num_workers=args.threads, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True,
                                     persistent_workers=args.threads > 0)

    freq_eval = None
    if args.hf_metrics:
        try:
            freq_eval = metrics_freq.FreqEvaluator.from_args(args, device if cuda else "cpu")
        except ValueError as e:
            parser.error(str(e))

    print("===> Building models")
    args.is_train = True


    try:
        model = esrt.ESRT(upscale = args.scale, **esrt.model_kwargs(args))
    except ValueError as e:
        parser.error(str(e))
    #architecture.IMDN(upscale=args.scale)

    l1_criterion = nn.L1Loss()

    print("===> Setting GPU")
    if cuda:
        model = model.to(device)
        if args.use_amp:
            model = model.to(memory_format=torch.channels_last)
        l1_criterion = l1_criterion.to(device)

    # Optionally initialise from a checkpoint (supports DataParallel 'module.' prefixes).
    if args.pretrained:

        if os.path.isfile(args.pretrained):
            print("===> loading models '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained)
            new_state_dcit = OrderedDict()
            for k, v in checkpoint.items():
                if 'module' in k:
                    name = k[7:]  # strip the leading 'module.' added by nn.DataParallel
                else:
                    name = k
                new_state_dcit[name] = v
            model_dict = model.state_dict()
            # Keep only keys that exist in the current model, and list any model
            # parameters the checkpoint does not provide.
            pretrained_dict = {k: v for k, v in new_state_dcit.items() if k in model_dict}

            for k, v in model_dict.items():
                if k not in pretrained_dict:
                    print(k)
            model.load_state_dict(pretrained_dict, strict=True)

        else:
            print("===> no models found at '{}'".format(args.pretrained))

    print("===> Setting Optimizer")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, fused=bool(cuda and torch.cuda.is_available()))

    # --fast: compile for training (checkpoints/validation keep using the plain `model`).
    train_model = model
    if args.fast and cuda:
        lp = args.patch_size // args.scale
        example = torch.rand(args.batch_size, 3, lp, lp, device=device)
        if args.use_amp:
            example = example.contiguous(memory_format=torch.channels_last)
        train_model = utils.compile_model(model, example, train=True, autocast=args.use_amp,
                                         mode=args.compile_mode)



    print("===> Training")
    print_network(model)
    code_start = datetime.datetime.now()
    timer = utils.Timer()
    save_dir = (args.save_dir or "experiment/checkpoint_ESRT_x{}{}/".format(args.scale, esrt.variant_tag(args))).rstrip("/")
    plotter = plotting.TrainValPlotter(save_dir, title=os.path.basename(save_dir),
                                       reference=plotting.load_reference(args.ref_history) if args.ref_history else None) if args.plot else None
    # Main loop. Note validation runs *before* training each epoch, so the reported
    # score lags the weights saved at the end of that epoch by one epoch.
    for epoch in range(args.start_epoch, args.nEpochs + 1):
        t_epoch_start = timer.t()
        epoch_start = datetime.datetime.now()
        if (epoch - args.start_epoch) % args.val_every == 0:
            extras = {}
            res = valid(args.scale, model, val_loaders, args, device, epoch - 1, freq_eval, extras)
            if plotter:
                plotter.add_val(epoch - 1, res)
                plotter.add_freq(epoch - 1, extras)
                plotter.save()
        loss = train(epoch, train_model, optimizer, l1_criterion, training_data_loader, args, device)
        if plotter:
            plotter.add_loss(epoch, loss)
        if epoch%10==0 or epoch == args.nEpochs:  # always keep the final epoch
            save_checkpoint(epoch, model, args)
        epoch_end = datetime.datetime.now()
        print('Epoch cost times: %s' % str(epoch_end-epoch_start))
        t = timer.t()
        # Fraction of training done -> used to extrapolate total run time (ETA).
        prog = (epoch-args.start_epoch+1)/(args.nEpochs + 1 - args.start_epoch + 1)
        # Print: this epoch's time, total elapsed, projected total.
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        print('{} {}/{}'.format(t_epoch, t_elapsed, t_all))
    extras = {}
    res = valid(args.scale, model, val_loaders, args, device, args.nEpochs, freq_eval, extras)  # score the final weights
    if plotter:
        plotter.add_val(args.nEpochs, res)
        plotter.add_freq(args.nEpochs, extras)
        plotter.save()
        print("===> Training curves saved to {}/curves.png".format(save_dir))
    code_end = datetime.datetime.now()
    print('Code cost times: %s' % str(code_end-code_start))


if __name__ == '__main__':
    main()
