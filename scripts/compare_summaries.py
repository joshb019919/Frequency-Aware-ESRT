"""Compare two test_x<s>_summary.json files image by image (same checkpoint, same set).

Used to check that a new machine's benchmark data reproduce the committed results: re-test a committed
checkpoint, then compare. Prints the largest per-image PSNR difference; exit code 0 if every image is
within --tol dB (and the image counts match), 1 otherwise.

  python scripts/compare_summaries.py final_results/lowpass/bsds300/test_x2_summary.json new/test_x2_summary.json
"""
import argparse
import json
import sys


def load(path):
    (_, v), = json.load(open(path)).items()
    return v["psnr"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("reference")
    p.add_argument("candidate")
    p.add_argument("--tol", type=float, default=0.01, help="max allowed per-image PSNR difference in dB (bf16 runs differ by up to ~0.007 dB across GPUs)")
    p.add_argument("--label", default="")
    a = p.parse_args()
    ref, new = load(a.reference), load(a.candidate)
    tag = (a.label + ": ") if a.label else ""
    if len(ref) != len(new):
        print("{}MISMATCH image count {} vs {}".format(tag, len(ref), len(new)))
        sys.exit(1)
    diffs = [abs(x - y) for x, y in zip(ref, new)]
    worst = max(diffs)
    n_bad = sum(d > a.tol for d in diffs)
    mean_d = sum(new) / len(new) - sum(ref) / len(ref)
    ok = n_bad == 0
    print("{}{} {} images, max |dPSNR| {:.4f} dB, mean dPSNR {:+.4f} dB{}".format(
        tag, "MATCH" if ok else "MISMATCH", len(ref), worst, mean_d,
        "" if ok else ", {} image(s) over {} dB".format(n_bad, a.tol)))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
