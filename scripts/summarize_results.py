"""Print (and optionally save) the report table: mean PSNR/SSIM per config and test set.

Reads final_results/<config>/test_x<scale>_summary.json (Set5) and the set14/, bsds300/ and manga109/
subfolders (a set that was not tested shows '-'). Delta columns are
PSNR differences against the lowpass baseline.

  python scripts/summarize_results.py                  # print a markdown table
  python scripts/summarize_results.py --csv final_results/summary.csv
"""
import argparse
import csv
import json
import os

SETS = (("Set5", ""), ("Set14", "set14"), ("BSDS300", "bsds300"), ("Manga109", "manga109"))
BASELINES = ("bilinear", "bicubic")  # scripts/interp_baselines.py


def mean_scores(path):
    """(psnr, ssim, n) means from one test_x<s>_summary.json, or None if missing."""
    if not os.path.exists(path):
        return None
    (_, v), = json.load(open(path)).items()
    n = len(v["psnr"])
    return sum(v["psnr"]) / n, sum(v["ssim"]) / n, n


def collect(root, scale):
    rows = {}
    for cfg in sorted(os.listdir(root)):
        d = os.path.join(root, cfg)
        if not os.path.isdir(d):
            continue
        scores = {name: mean_scores(os.path.join(d, sub, "test_x{}_summary.json".format(scale)))
                  for name, sub in SETS}
        if any(scores.values()):
            rows[cfg] = scores
    return rows


def order(cfg):
    """bilinear, bicubic, lowpass, lowpass_fl*, then dwt configs each followed by their _fl copies;
    a _win (window attention) config comes right after the same config without it."""
    if cfg in BASELINES:
        return (0, BASELINES.index(cfg), "", -1.0, 0)
    win = cfg.endswith("_win")
    base, _, lam = (cfg[:-4] if win else cfg).partition("_fl")
    try:
        lam = float(lam) if lam else -1.0
    except ValueError:  # an unexpected name: sort it at the end of its group
        lam = float("inf")
    return (1 if base == "lowpass" else 2, 0, base, lam, int(win))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="final_results")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--csv", default=None, help="also write the table to this CSV file")
    a = p.parse_args()

    rows = collect(a.root, a.scale)
    if not rows:
        raise SystemExit("no test summaries found under {}".format(a.root))
    base = rows.get("lowpass", {})
    fmt = lambda s: "-" if s is None else "{:.3f} / {:.4f}".format(s[0], s[1])
    delta = lambda s, b: "-" if s is None or b is None else "{:+.3f}".format(s[0] - b[0])

    head = ["config"] + [n for n, _ in SETS] + ["Δ " + n for n, _ in SETS]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    out = []
    for cfg in sorted(rows, key=order):
        r = rows[cfg]
        cells = [cfg] + [fmt(r[n]) for n, _ in SETS] + [delta(r[n], base.get(n)) for n, _ in SETS]
        print("| " + " | ".join(cells) + " |")
        out.append(dict(config=cfg, **{"{}_{}".format(n, k): (None if r[n] is None else r[n][i])
                                       for n, _ in SETS for i, k in enumerate(("psnr", "ssim", "n"))}))
    ns = {n: next((r[n][2] for r in rows.values() if r[n]), "?") for n, _ in SETS}
    print("\nPSNR (dB) / SSIM on Y, x{}; images per set: {}. Δ = PSNR minus lowpass.".format(
        a.scale, ", ".join("{} {}".format(k, v) for k, v in ns.items())))
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
        print("written:", a.csv)


if __name__ == "__main__":
    main()
