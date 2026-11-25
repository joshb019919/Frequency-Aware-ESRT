"""Download a dataset from Hugging Face, then launch ESRT training or testing.

Role in ESRT: top-level command-line launcher. ESRT (Efficient Super-Resolution
Transformer) is a lightweight image super-resolution network (CNN feature
extractor + efficient transformer attention); this script fetches the data and
runs train.py or test.py as a subprocess, logging run time to timings.log.

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
import time

# Repo root; train.py/test.py are launched from here so their relative paths resolve.
ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    """Parse launcher args, optionally download the dataset, then run train.py or test.py.

    Builds a subprocess command line for the chosen mode, forwards any extra
    arguments, times the run, appends a line to timings.log and exits with the
    child's return code.
    """
    p = argparse.ArgumentParser(description="ESRT launcher")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true", help="run train.py")
    mode.add_argument("--test", action="store_true", help="run test.py")
    mode.add_argument("--experiments", action="store_true",
                      help="run experiments.py (train+test every variant); extra args after -- go to it")
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
    # model variant flags, forwarded to train.py/test.py only when given
    p.add_argument("--freq", choices=["lowpass", "dwt"])
    p.add_argument("--conv", choices=["1", "3", "band"])
    p.add_argument("--global_res", choices=["0", "1"])
    p.add_argument("--dwt_skip", choices=["0", "1"])
    p.add_argument("--ll_norm", choices=["0", "1"])
    p.add_argument("--fast", action=argparse.BooleanOptionalAction, default=True,
                   help="torch.compile + bf16 + TF32 (default on; --no-fast disables)")

    # Unknown args are kept in `extra` and passed through to the child script.
    args, extra = p.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]

    # Fetch the dataset snapshot from the Hugging Face Hub (imported lazily so
    # --skip_download works without huggingface_hub installed).
    if not args.skip_download:
        if not args.hf_repo:
            p.error("--hf_repo is required unless --skip_download is set")
        from huggingface_hub import snapshot_download
        from os import environ
        snapshot_download(repo_id=args.hf_repo, repo_type="dataset",
                          revision=args.hf_revision, local_dir=args.data_dir,
                          token=environ.get("HF_TOKEN"))
    def in_data(path):
        """Resolve `path` inside --data_dir if it exists there, else return it unchanged."""
        cand = os.path.join(args.data_dir, path)
        return cand if os.path.exists(cand) else path

    if args.train:
        cmd = [sys.executable, os.path.join(ROOT, "train.py"),
               "--root", args.data_dir, "--scale", str(args.scale)]
    elif args.experiments:
        cmd = [sys.executable, os.path.join(ROOT, "experiments.py"),
               "--root", args.data_dir, "--scale", str(args.scale)]
    else:
        if not args.checkpoint or bool(args.test_hr_folder) != bool(args.test_lr_folder):
            p.error("--test needs --checkpoint (and --test_hr_folder with --test_lr_folder, or neither "
                    "to use test.py's default sets, bsds300 + manga109)")
        cmd = [sys.executable, os.path.join(ROOT, "test.py"),
               "--checkpoint", args.checkpoint,
               "--upscale_factor", str(args.scale)]
        if args.test_hr_folder:
            cmd += ["--test_hr_folder", in_data(args.test_hr_folder),
                    "--test_lr_folder", in_data(args.test_lr_folder)]
        
        if args.output_folder:
            cmd += ["--output_folder", args.output_folder]

    for flag in (() if args.experiments else ("freq", "conv", "global_res", "dwt_skip", "ll_norm")):
        if getattr(args, flag) is not None:
            cmd += ["--" + flag, getattr(args, flag)]
    cmd.append("--fast" if args.fast else "--no-fast")
    cmd += extra  # user-supplied passthrough flags go last so they can override
    print("Running:", " ".join(cmd))

    start = time.perf_counter()
    rc = subprocess.call(cmd, cwd=ROOT)
    elapsed = time.perf_counter() - start

    mode = "train" if args.train else "experiments" if args.experiments else "test"
    print("[main.py] {} finished in {:.1f}s (exit code {})".format(mode, elapsed, rc))
    
    # Append one tab-separated record per run: timestamp, mode, scale, duration, rc.
    with open(os.path.join(ROOT, "timings.log"), "a") as f:
        f.write("{}\t{}\tscale={}\t{:.1f}s\trc={}\n".format(
            time.strftime("%Y-%m-%d %H:%M:%S"), mode, args.scale, elapsed, rc))
    sys.exit(rc)


if __name__ == "__main__":
    main()
