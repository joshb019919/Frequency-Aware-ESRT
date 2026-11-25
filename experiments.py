"""Run every ESRT variant (train, then test on Set5) and collect the results.

Role in ESRT: experiment driver for comparing the original low-pass block against
the Haar-DWT variants. Each config is trained for --epochs with the same seed and
settings, evaluated with test.py, and appended to a results CSV. A failed config is
recorded and does not stop the others.

Configs (7 per patch size): lowpass; dwt with --conv {1,3,band} x --global_res {1,0}.
--dwt_tweaks skip,norm,skipnorm adds the dwt configs again with the architectural tweaks
(--dwt_skip 1 / --ll_norm 1 / both), named <config>_<tweak>; `skip` is not run for conv band.
--patch_size {96,144,both} (default 96) sets the HR training patch; `both` runs every
config at 96 and at 144. Names at 144 get a `_p144` suffix (96 keeps the plain name).

Examples:
  python experiments.py --root /data/data/div2k --epochs 10
  python experiments.py --root /data/data/div2k --only lowpass,dwt_c1_gr1 --dry_run
  python main.py --experiments --skip_download --data_dir /data/data/div2k -- --epochs 10

Each config writes checkpoints to experiment/exp_<name>/ (existing checkpoint
folders are never touched), logs to experiment/exp_<name>/{train,test}.log and SR
images to results/exp_<name>/. A config whose final checkpoint already exists is
not retrained unless --force is given.
"""
import argparse
import csv
import itertools
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))


PATCH_SIZES = (96, 144)  # HR training patch sizes; 96 is the default


TWEAKS = {"skip": ["--dwt_skip", "1"], "norm": ["--ll_norm", "1"],
          "skipnorm": ["--dwt_skip", "1", "--ll_norm", "1"]}


def build_configs(patch_sizes=(96,), tweaks=()):
    """Return [(name, [flags...], patch)] for every variant x patch size.

    Variants: lowpass + dwt x conv x global_res. A patch size other than 96 adds a
    `_p<patch>` suffix to the name so checkpoints/logs never collide.
    """
    base = [("lowpass", [])]
    # gr0 variants disabled: 10-epoch results showed global_res=0 is clearly worse
    # (re-add "0" to the list to run them again).
    for conv, gr in itertools.product(["1", "3", "band"], ["1"]):  # , "0"]):
        base.append(("dwt_c{}_gr{}".format(conv, gr),
                     ["--freq", "dwt", "--conv", conv, "--global_res", gr]))
    cfgs = []
    for ps in patch_sizes:
        suffix = "" if ps == 96 else "_p{}".format(ps)
        cfgs += [(n + suffix, f, ps) for n, f in base]
        for tw in tweaks:  # architectural tweaks apply to the dwt configs only
            for n, f in base[1:]:
                if tw == "skip" and "band" in f:
                    continue  # band's decoder_high already has an internal residual: skip is a no-op
                # c1_skipnorm disabled: PSNR regressed (34.40 vs 34.90 basic) despite high SSIM.
                if tw == "skipnorm" and n == "dwt_c1_gr1":
                    continue
                cfgs.append(("{}_{}{}".format(n, tw, suffix), f + TWEAKS[tw], ps))
    return cfgs


def exp_dir(name):
    return os.path.join("experiment", "exp_" + name)


def checkpoint_path(name, epochs):
    return os.path.join(exp_dir(name), "epoch_{}.pth".format(epochs))


def train_cmd(name, flags, a, patch=96):
    cmd = [sys.executable, os.path.join(ROOT, "train.py"), "--root", a.root,
           "--scale", str(a.scale), "-nEpochs", str(a.epochs), "--batch_size", str(a.batch_size),
           "--patch_size", str(patch), "--threads", str(a.threads),
           "--seed", str(a.seed), "--val_every", str(a.val_every),
           "--save_dir", exp_dir(name)] + flags
    cmd.append("--fast" if a.fast else "--no-fast")
    if a.no_amp:
        cmd.append("--no_amp")
    return cmd


def test_cmd(name, flags, a):
    hr = a.test_hr_folder
    lr = a.test_lr_folder or "Test_Datasets/Set5_LR/x{}/".format(a.scale)
    cmd = [sys.executable, os.path.join(ROOT, "test.py"), "--test_hr_folder", hr,
           "--test_lr_folder", lr, "--checkpoint", checkpoint_path(name, a.epochs),
           "--upscale_factor", str(a.scale),
           "--output_folder", os.path.join("results", "exp_" + name)] + flags
    cmd.append("--fast" if a.fast else "--no-fast")
    return cmd


def parse_test_output(text):
    """(psnr, ssim, ms) from test.py's 'Mean PSNR: x, SSIM: y, TIME: z ms' line, else Nones."""
    m = re.search(r"Mean PSNR: ([\d.]+), SSIM: ([\d.]+), TIME: ([\d.]+) ms", text)
    return tuple(float(g) for g in m.groups()) if m else (None, None, None)


def parse_params(text):
    m = re.search(r"Total number of parameters: (\d+)", text)
    return int(m.group(1)) if m else None


def _groups(cmd):
    """Split a command into option groups: leading words, then ('--flag', value...) tuples."""
    groups, cur = [], []
    for tok in cmd:
        if tok.startswith("-") and not tok[1:2].isdigit() and cur:
            groups.append(tuple(cur))
            cur = []
        cur.append(tok)
    if cur:
        groups.append(tuple(cur))
    return groups


def _shared_and_diffs(cmds):
    """(shared groups in order, [per-command groups not shared by all commands])."""
    gl = [_groups(c) for c in cmds]
    shared = [g for g in gl[0] if all(g in other for other in gl[1:])]
    return shared, [[g for g in groups if g not in shared] for groups in gl]


def format_dry_run(cfgs, a):
    """Dry-run text: settings shared by all experiments once, then only the differences per line."""
    rel = lambda g: " ".join(os.path.basename(t) if t.endswith(".py") else t for t in g)
    out = []
    t_shared, t_diffs = _shared_and_diffs([train_cmd(n, f, a, ps) for n, f, ps in cfgs])
    e_shared, e_diffs = _shared_and_diffs([test_cmd(n, f, a) for n, f, ps in cfgs])
    out.append("Shared train: " + " ".join(rel(g) for g in t_shared))
    out.append("Shared test:  " + " ".join(rel(g) for g in e_shared))
    out.append("Per experiment (train flags | test-only flags):")
    for (n, _, _), td, ed in zip(cfgs, t_diffs, e_diffs):
        ed = [g for g in ed if g not in td]  # model flags already shown on the train side
        out.append("  {:<16} {} | {}".format(n, " ".join(rel(g) for g in td),
                                             " ".join(rel(g) for g in ed)))
    return "\n".join(out)


EPOCH_COLS = ["epoch", "lr", "train_loss_l1", "epoch_s", "val_psnr", "val_ssim"]


def parse_train_log(text):
    """Per-epoch rows from a train.py log.

    Row `epoch` N holds that epoch's lr, mean L1 training loss and duration, plus the
    validation score of the weights *after* N epochs when one was run (epoch 0 = untrained
    weights, validated before the first epoch). Fields missing from the log stay None.
    """
    rows = {}
    row = lambda e: rows.setdefault(e, dict.fromkeys(EPOCH_COLS))
    cur = None
    for line in text.splitlines():
        m = re.match(r"epoch = (\d+) lr =\s+([\d.eE+-]+)", line)
        if m:
            cur = int(m.group(1))
            row(cur).update(epoch=cur, lr=float(m.group(2)))
            continue
        m = re.match(r"===> Epoch\[(\d+)\] avg Loss_l1: ([\d.]+)", line)
        if m:
            row(int(m.group(1))).update(epoch=int(m.group(1)), train_loss_l1=float(m.group(2)))
            continue
        m = re.match(r"Epoch cost times: (\d+):(\d+):([\d.]+)", line)
        if m and cur is not None:
            h, mi, sec = int(m.group(1)), int(m.group(2)), float(m.group(3))
            row(cur).update(epoch=cur, epoch_s=round(h * 3600 + mi * 60 + sec, 1))
            continue
        m = re.match(r"===> Valid\. psnr: ([\d.]+), ssim: ([\d.]+)(?: \(weights after (\d+) epochs\))?", line)
        if m:
            # logs from before the "(weights after N epochs)" suffix: validation preceded epoch cur+1
            k = int(m.group(3)) if m.group(3) is not None else (0 if cur is None else cur)
            row(k).update(epoch=k, val_psnr=float(m.group(1)), val_ssim=float(m.group(2)))
    return [rows[k] for k in sorted(rows)]


def write_epoch_csv(path, name, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name"] + EPOCH_COLS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(dict(r, name=name))


ECHO_RE = re.compile(r"^(epoch = |===> Epoch\[\d+\] avg|===> Valid|Epoch cost|===> Checkpoint|Total number|Mean PSNR)")


def run_logged(cmd, log_path, echo_prefix=""):
    """Run cmd from the repo root, streaming its output live to log_path.

    The log is flushed line by line (so `tail -f` works) and the child runs unbuffered.
    Per-epoch progress lines (epoch/lr, avg loss, validation, epoch time, checkpoint) are
    also echoed to the terminal with `echo_prefix`. Returns (rc, full text, seconds).
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    t = time.perf_counter()
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    lines = []
    with open(log_path, "w") as f, subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                                    stderr=subprocess.STDOUT, text=True, bufsize=1) as p:
        for line in p.stdout:
            f.write(line)
            f.flush()
            lines.append(line)
            if echo_prefix and ECHO_RE.match(line):
                print(echo_prefix + line.rstrip(), flush=True)
    return p.returncode, "".join(lines), time.perf_counter() - t


def main(argv=None):
    p = argparse.ArgumentParser(description="Run all ESRT variant experiments")
    p.add_argument("--root", default=None, help="dataset directory (train.py --root); required unless --from_logs")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--patch_size", choices=["96", "144", "both"], default="96",
                   help="HR training patch size; 'both' runs every config at 96 and 144 (default 96)")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--val_every", type=int, default=5)
    p.add_argument("--test_hr_folder", default="Test_Datasets/Set5/")
    p.add_argument("--test_lr_folder", default=None)
    p.add_argument("--dwt_tweaks", default="",
                   help="comma-separated subset of {}: also run the dwt configs with these tweaks".format(",".join(TWEAKS)))
    p.add_argument("--only", default=None, help="comma-separated config names")
    p.add_argument("--fast", action=argparse.BooleanOptionalAction, default=True,
                   help="torch.compile + bf16 + TF32 in train/test (default on; --no-fast disables)")
    p.add_argument("--no_amp", action="store_true", help="plain fp32 training")
    p.add_argument("--force", action="store_true", help="retrain even if the final checkpoint exists")
    p.add_argument("--from_logs", action="store_true",
                   help="no training: rebuild a per-epoch CSV from existing experiment/exp_*/train.log files")
    p.add_argument("--dry_run", "--dry-run", action="store_true", help="print the commands and exit")
    a = p.parse_args(argv)

    if not a.root and not a.from_logs:
        p.error("--root is required unless --from_logs is given")
    if a.from_logs:
        import glob
        out = os.path.join(ROOT, "experiment", "epochs_{}.csv".format(time.strftime("%Y%m%d_%H%M%S")))
        logs = sorted(glob.glob(os.path.join(ROOT, "experiment", "exp_*", "train.log")))
        for lp in logs:
            write_epoch_csv(out, os.path.basename(os.path.dirname(lp))[len("exp_"):], parse_train_log(open(lp).read()))
        print("{} log(s) -> {}".format(len(logs), out if logs else "nothing written"))
        return 0

    tweaks = tuple(t for t in a.dwt_tweaks.split(",") if t)
    if set(tweaks) - set(TWEAKS):
        p.error("--dwt_tweaks must be a comma-separated subset of {}".format(sorted(TWEAKS)))
    cfgs = build_configs(PATCH_SIZES if a.patch_size == "both" else (int(a.patch_size),), tweaks)
    if a.scale not in (2, 3, 4) or any(ps % a.scale for _, _, ps in cfgs):
        p.error("patch size must be divisible by --scale")
    if a.only:
        want = [n.strip() for n in a.only.split(",")]
        unknown = set(want) - {n for n, _, _ in cfgs}
        if unknown:
            p.error("unknown config(s): {}; choose from {}".format(sorted(unknown), [n for n, _, _ in cfgs]))
        cfgs = [c for c in cfgs if c[0] in want]

    if a.dry_run:
        print(format_dry_run(cfgs, a))
        return 0

    out_csv = os.path.join(ROOT, "experiment", "experiments_{}.csv".format(time.strftime("%Y%m%d_%H%M%S")))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_epochs_csv = out_csv.replace(".csv", "_epochs.csv")
    cols = ["name", "patch_size", "epochs", "rc_train", "rc_test", "params", "train_s", "set5_psnr", "set5_ssim",
            "test_ms_per_img", "checkpoint"]
    rows = []
    for name, flags, patch in cfgs:
        ckpt = checkpoint_path(name, a.epochs)
        params, train_s, rc_train = None, None, 0
        if os.path.exists(os.path.join(ROOT, ckpt)) and not a.force:
            print("[{}] final checkpoint exists, skipping training (use --force to retrain)".format(name))
        else:
            print("[{}] training {} epochs ...".format(name, a.epochs), flush=True)
            rc_train, text, train_s = run_logged(train_cmd(name, flags, a, patch),
                                                 os.path.join(ROOT, exp_dir(name), "train.log"),
                                                 "[{}] ".format(name))
            params = parse_params(text)
        log_path = os.path.join(ROOT, exp_dir(name), "train.log")
        if os.path.exists(log_path):
            ep_rows = parse_train_log(open(log_path).read())
            write_epoch_csv(out_epochs_csv, name, ep_rows)
        psnr = ssim = ms = None
        rc_test = None
        if rc_train == 0 and os.path.exists(os.path.join(ROOT, ckpt)):
            print("[{}] testing ...".format(name), flush=True)
            rc_test, text, _ = run_logged(test_cmd(name, flags, a),
                                          os.path.join(ROOT, exp_dir(name), "test.log"))
            psnr, ssim, ms = parse_test_output(text)
        rows.append(dict(name=name, patch_size=patch, epochs=a.epochs, rc_train=rc_train, rc_test=rc_test, params=params,
                         train_s=None if train_s is None else round(train_s, 1), set5_psnr=psnr,
                         set5_ssim=ssim, test_ms_per_img=ms, checkpoint=ckpt))
        with open(out_csv, "w", newline="") as f:  # rewritten after each config so partial runs are kept
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    print("\n{:<16}{:>9}{:>10}{:>9}{:>9}{:>8}".format("config", "params", "train_s", "PSNR", "SSIM", "rc"))
    fmt = lambda v, f: "-" if v is None else f.format(v)
    for r in rows:
        print("{:<16}{:>9}{:>10}{:>9}{:>9}{:>8}".format(
            r["name"], fmt(r["params"], "{}"), fmt(r["train_s"], "{}"), fmt(r["set5_psnr"], "{:.3f}"),
            fmt(r["set5_ssim"], "{:.4f}"), "{}/{}".format(r["rc_train"], r["rc_test"])))
    print("results:", out_csv)
    print("per-epoch:", out_epochs_csv)
    return 0 if all(r["rc_train"] == 0 and r["rc_test"] in (0, None) for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
