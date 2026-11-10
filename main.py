"""Download a dataset from Hugging Face, then launch ESRT training or testing.

Examples:
  python main.py --train --hf_repo USER/DIV2K --scale 2 -- --patch_size 96
  python main.py --test  --hf_repo USER/benchmark --scale 4 \
      --test_hr_folder B100/HR --test_lr_folder B100/LR_bicubic/X4 \
      --checkpoint experiment/checkpoint/x4/epoch_990.pth

Anything after `--` is forwarded unchanged to train.py / test.py.
Note: train.py expects <data_dir>/DF2K_decoded/{DIV2K_train_HR,DIV2K_train_LR_bicubic/X<scale>}
as .npy files, so the HF repo should contain that layout (or run scripts/png2npy.py on it).
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser(description="ESRT launcher")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true", help="run train.py")
    mode.add_argument("--test", action="store_true", help="run test.py")
    p.add_argument("--hf_repo", default=None, help="HF dataset repo id, e.g. user/name")
    p.add_argument("--hf_revision", default=None, help="branch/tag/commit")
    p.add_argument("--data_dir", default=os.path.join(ROOT, "dataset"),
                   help="local download directory")
    p.add_argument("--skip_download", action="store_true",
                   help="use existing --data_dir without contacting Hugging Face")
    p.add_argument("--scale", type=int, default=2)
    # test-only options (paths are resolved relative to --data_dir if they exist there)
    p.add_argument("--test_hr_folder")
    p.add_argument("--test_lr_folder")
    p.add_argument("--checkpoint")
    p.add_argument("--output_folder")
    args, extra = p.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]

    if not args.skip_download:
        if not args.hf_repo:
            p.error("--hf_repo is required unless --skip_download is set")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=args.hf_repo, repo_type="dataset",
                          revision=args.hf_revision, local_dir=args.data_dir)

    def in_data(path):
        cand = os.path.join(args.data_dir, path)
        return cand if os.path.exists(cand) else path

    if args.train:
        cmd = [sys.executable, os.path.join(ROOT, "train.py"),
               "--root", args.data_dir, "--scale", str(args.scale)]
    else:
        if not (args.test_hr_folder and args.test_lr_folder and args.checkpoint):
            p.error("--test needs --test_hr_folder, --test_lr_folder and --checkpoint")
        cmd = [sys.executable, os.path.join(ROOT, "test.py"),
               "--test_hr_folder", in_data(args.test_hr_folder),
               "--test_lr_folder", in_data(args.test_lr_folder),
               "--checkpoint", args.checkpoint,
               "--upscale_factor", str(args.scale)]
        if args.output_folder:
            cmd += ["--output_folder", args.output_folder]
    cmd += extra
    print("Running:", " ".join(cmd))
    sys.exit(subprocess.call(cmd, cwd=ROOT))


if __name__ == "__main__":
    main()
