"""Training / validation / test plots for train.py and test.py (matplotlib, Agg backend).

Plotting never aborts a run: if matplotlib is missing or saving fails, a warning is printed
and training/testing continues.
"""
import json
import math
import os
import re


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _epoch_axis(ax):
    """Log-style x-axis for epoch curves so the fast early drop is spread out (symlog keeps epoch 0)."""
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    ax.set_xscale("symlog", linthresh=1, linscale=0.4)
    lo, hi = ax.get_xlim()
    ticks = [0] + [10 ** k for k in range(0, 7) if 10 ** k <= max(hi, 1)]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(left=-0.3 if lo < 0.5 else lo)


def _zoom_y(ax, from_epoch=5):
    """Fit a linear y-axis to the trained part of the curves (x >= from_epoch) so the untrained
    epoch-0 point does not squash everything else; notes it when that point is off-scale."""
    ys = [y for ln in ax.lines for x, y in zip(ln.get_xdata(), ln.get_ydata()) if x >= from_epoch]
    allys = [y for ln in ax.lines if len(ln.get_xdata()) > 1 for y in ln.get_ydata()]
    if not ys:
        return
    lo, hi = min(ys), max(ys)
    pad = 0.08 * (hi - lo or abs(hi) or 1.0)
    ax.set_ylim(lo - pad, hi + pad)
    if any(y < lo - pad or y > hi + pad for y in allys):
        ax.text(0.99, 0.02, "epoch 0 off-scale", transform=ax.transAxes, ha="right", fontsize=7, color="gray")


def load_reference(path, last_n=10):
    """Reference level from a finished run's history.json (e.g. the lowpass experiment).

    Returns {"loss": float, "val": {set: (psnr, ssim)}, "freq": {set: {metric: value}}}, each the mean over
    the run's last `last_n` recorded points (the very last point alone jitters by ~0.03-0.08 dB).
    Returns None if the file is missing or unreadable.
    """
    try:
        with open(path) as f:
            h = json.load(f)

        def tail(d):
            return [d[k] for k in sorted(d, key=int)[-last_n:]]
        mean = lambda xs: sum(xs) / len(xs)
        ref = {"loss": mean(tail(h["loss"])) if h.get("loss") else None, "val": {}, "freq": {}}
        for name, d in h.get("val", {}).items():
            t = tail(d)
            ref["val"][name] = (mean([v[0] for v in t]), mean([v[1] for v in t]))
        for name, d in h.get("freq", {}).items():
            t = tail(d)
            ref["freq"][name] = {k: mean([m[k] for m in t if k in m]) for k in t[-1]}
        return ref
    except Exception as e:
        print("[plotting] reference not loaded from {}: {}".format(path, e))
        return None


def load_test_reference(ref_dir, scale):
    """Reference test results (e.g. the lowpass experiment) from a folder that test.py wrote into:
    test_x<scale>_summary.json (per-image PSNR/SSIM) and test_x<scale>_freq.csv (per-image frequency metrics).

    Returns {"summary": {set: {"psnr": [...], "ssim": [...]}} or None, "freq": {set: {metric: mean}} or None}.
    """
    import csv
    ref = {"summary": None, "freq": None}
    try:
        with open(os.path.join(ref_dir, "test_x%d_summary.json" % scale)) as f:
            ref["summary"] = json.load(f)
    except Exception as e:
        print("[plotting] reference summary not loaded from {}: {}".format(ref_dir, e))
    try:
        sums, cnt = {}, {}
        with open(os.path.join(ref_dir, "test_x%d_freq.csv" % scale)) as f:
            for row in csv.DictReader(f):
                d = sums.setdefault(row["set"], {})
                cnt[row["set"]] = cnt.get(row["set"], 0) + 1
                for k, v in row.items():
                    if k not in ("set", "image") and v not in ("", None):
                        d[k] = d.get(k, 0.0) + float(v)
        ref["freq"] = {n: {k: v / cnt[n] for k, v in d.items()} for n, d in sums.items()}
    except Exception as e:
        print("[plotting] reference frequency metrics not loaded from {}: {}".format(ref_dir, e))
    return ref


def _ref_segments(ax, xs, ys, half, first=True):
    """Short dashed black segments (one per bar/box) marking the reference level."""
    for j, (x, y) in enumerate(zip(xs, ys)):
        ax.hlines(y, x - half, x + half, colors="black", linestyles="--", lw=1.4, zorder=5,
                  label="lowpass final" if first and j == 0 else None)


def _ref_line(ax, y, first):
    """Dashed black horizontal reference line; only the first one per panel gets a legend label."""
    ax.axhline(y, color="black", ls="--", lw=1.0, zorder=5, label="lowpass final" if first else None)


class TrainValPlotter:
    """Collects per-epoch train loss and per-set validation PSNR/SSIM, drawn side by side.

    Layout: [train L1 loss | validation PSNR | validation SSIM], one line per validation set.
    Writes <out_dir>/curves.png (redrawn on every `save()`) and <out_dir>/history.json.
    """

    def __init__(self, out_dir, title="", reference=None):
        """reference: optional load_reference() result, drawn as dashed black horizontal lines (lowpass final)."""
        self.out_dir, self.title, self.reference = out_dir, title, reference
        self.loss = {}   # epoch -> mean L1
        self.val = {}    # set name -> {epochs_trained: (psnr, ssim)}
        self.freq = {}   # set name -> {epochs_trained: {metric: mean value}} (see metrics_freq.py)

    def add_loss(self, epoch, loss):
        if loss is not None:
            self.loss[epoch] = float(loss)

    def add_val(self, epochs_trained, results):
        """results: {set name: (psnr, ssim)} measured with weights after `epochs_trained` epochs."""
        for name, (p, s) in results.items():
            self.val.setdefault(name, {})[epochs_trained] = (float(p), float(s))

    def add_freq(self, epochs_trained, freq):
        """freq: {set name: {metric: mean value}} measured with weights after `epochs_trained` epochs."""
        for name, m in (freq or {}).items():
            self.freq.setdefault(name, {})[epochs_trained] = {k: float(v) for k, v in m.items()}

    def save(self):
        self._save_curves()
        self._save_freq_curves()

    def _save_freq_curves(self):
        """<out_dir>/freq_curves.png: per-set frequency metrics over validations (2 rows x 4 panels)."""
        if not self.freq:
            return
        try:
            plt = _plt()
            sample = next(iter(next(iter(self.freq.values())).values()))
            wl = [m.group(1) for k in sample for m in [re.fullmatch(r"sb_(\w+)_HH", k)] if m]  # --wavelets order
            styles = ["-", "--", ":", "-."]
            panels = [("HH-band relative error (lower better)", [("sbrel_%s_HH" % w, styles[i % 4], w) for i, w in enumerate(wl)], None),
                      ("HH-band PSNR (dB)", [("sb_%s_HH" % w, styles[i % 4], w) for i, w in enumerate(wl)], None),
                      ("HF power ratio SR/HR (1 = ideal)", [("psd_hf_ratio", "-", "")], 1.0),
                      ("Log-spectral distance (dB)", [("psd_lsd", "-", "")], None),
                      ("HFEN", [("hfen", "-", "")], None), ("GMSD", [("gmsd", "-", "")], None),
                      ("LPIPS", [("lpips", "-", "")], None), ("DISTS", [("dists", "-", "")], None)]
            fig, axes = plt.subplots(2, 4, figsize=(18, 7))
            for ax, (title, keys, ref) in zip(axes.flat, panels):
                drawn = False
                for ci, (name, d) in enumerate(sorted(self.freq.items())):
                    x = sorted(d)
                    for key, ls, suffix in keys:
                        if key in d[x[0]]:
                            ax.plot(x, [d[k][key] for k in x], ls, color="C%d" % ci, marker="o", ms=3,
                                    label=(name + " " + suffix).strip())
                            drawn = True
                if ref is not None:
                    ax.axhline(ref, color="gray", lw=0.8)
                if drawn and self.reference:
                    first = True
                    for name, m in sorted(self.reference["freq"].items()):
                        if name not in self.freq:
                            continue
                        for key, ls, suffix in keys:
                            if key in m:
                                _ref_line(ax, m[key], first)
                                first = False
                ax.set(title=title, xlabel="epochs trained (log)")
                _epoch_axis(ax)
                if drawn and "relative error" in title:
                    ax.set_yscale("log")  # untrained weights start orders of magnitude higher
                elif drawn:
                    _zoom_y(ax)
                if drawn:
                    ax.legend(fontsize=7)
                else:
                    ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
                ax.grid(alpha=0.3)
            if self.title:
                fig.suptitle(self.title)
            fig.tight_layout()
            fig.savefig(os.path.join(self.out_dir, "freq_curves.png"), dpi=100)
            plt.close(fig)
        except Exception as e:
            print("[plotting] skipped frequency curves: {}".format(e))

    def _save_curves(self):
        try:
            plt = _plt()
            os.makedirs(self.out_dir, exist_ok=True)
            fig, ax = plt.subplots(1, 3, figsize=(15, 4))
            if self.loss:
                e = sorted(self.loss)
                ax[0].plot(e, [self.loss[k] for k in e])
            if self.reference and self.reference.get("loss") is not None:
                _ref_line(ax[0], self.reference["loss"], True)
                ax[0].legend()
            ax[0].set(title="Train L1 loss", xlabel="epoch (log)", ylabel="L1")
            ax[0].set_yscale("log")
            for i, (idx, label) in enumerate(((0, "PSNR (dB, Y)"), (1, "SSIM (Y)")), start=1):
                for name, d in sorted(self.val.items()):
                    x = sorted(d)
                    ax[i].plot(x, [d[k][idx] for k in x], marker="o", ms=3, label=name)
                if self.reference:
                    first = True
                    for name in sorted(self.val):
                        if name in self.reference["val"]:
                            _ref_line(ax[i], self.reference["val"][name][idx], first)
                            first = False
                ax[i].set(title="Validation " + label.split()[0], xlabel="epochs trained (log)", ylabel=label)
                if self.val:
                    _zoom_y(ax[i])
                    ax[i].legend()
            for a in ax:
                a.grid(alpha=0.3)
                _epoch_axis(a)
            if self.title:
                fig.suptitle(self.title)
            fig.tight_layout()
            fig.savefig(os.path.join(self.out_dir, "curves.png"), dpi=110)
            plt.close(fig)
            with open(os.path.join(self.out_dir, "history.json"), "w") as f:
                json.dump({"loss": self.loss, "val": self.val, "freq": self.freq}, f)
        except Exception as e:  # plotting must never kill a training run
            print("[plotting] skipped training curves: {}".format(e))


def plot_test_summary(path, results, title="", reference=None):
    """Bar charts of mean PSNR / SSIM per test set plus per-image PSNR distribution.

    results: {set name: {"psnr": [per image], "ssim": [per image]}}. Saves a PNG to `path`.
    reference: optional load_test_reference() result; its per-set means are drawn as dashed black segments.
    Returns True on success.
    """
    try:
        plt = _plt()
        names = list(results)
        mean = lambda xs: sum(xs) / len(xs)
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for a, key, label in ((ax[0], "psnr", "Mean PSNR (dB, Y)"), (ax[1], "ssim", "Mean SSIM (Y)")):
            vals = [mean(results[n][key]) for n in names]
            bars = a.bar(names, vals)
            a.bar_label(bars, fmt="%.3f" if key == "ssim" else "%.2f")
            rv = {i: mean(reference["summary"][n][key]) for i, n in enumerate(names)
                  if reference and reference.get("summary") and n in reference["summary"]}
            if rv:
                _ref_segments(a, list(rv), list(rv.values()), 0.4)
                a.legend(fontsize=8)
            allv = vals + list(rv.values())
            a.set(title=label, ylim=(min(allv) * 0.95, max(allv) * 1.02))
        ax[2].boxplot([results[n]["psnr"] for n in names], showmeans=True)
        rb = {i + 1: mean(reference["summary"][n]["psnr"]) for i, n in enumerate(names)
              if reference and reference.get("summary") and n in reference["summary"]}
        if rb:
            _ref_segments(ax[2], list(rb), list(rb.values()), 0.4)
            ax[2].legend(fontsize=8)
        ax[2].set_xticks(range(1, len(names) + 1), labels=names)
        ax[2].set(title="Per-image PSNR", ylabel="dB")
        for a in ax:
            a.grid(alpha=0.3, axis="y")
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return True
    except Exception as e:
        print("[plotting] skipped test summary: {}".format(e))
        return False


def plot_test_freq(path, results, reference=None):
    """Frequency-metric summary: relative error per DWT subband (one panel per wavelet), radial power
    ratio curve, and HFEN / GMSD / LPIPS / DISTS bars, one colour per test set.

    results: {set name: [per-image metric dicts]} (see metrics_freq.FreqEvaluator). Returns True on success.
    reference: optional load_test_reference() result; the lowpass means are drawn as dashed black segments/curve.
    """
    try:
        import numpy as np
        plt = _plt()
        names = [n for n, rows in results.items() if rows]
        means = {n: {k: float(np.mean([r[k] for r in results[n] if k in r])) for k in results[n][0] if k != "image"}
                 for n in names}
        sample = means[names[0]]
        wl = [m.group(1) for k in sample for m in [re.fullmatch(r"sb_(\w+)_HH", k)] if m]  # --wavelets order
        scalars = [("hfen", "HFEN (lower better)"), ("gmsd", "GMSD (lower better)"),
                   ("lpips", "LPIPS (lower better)"), ("dists", "DISTS (lower better)")]
        n_panels = len(wl) + 1 + len(scalars)
        cols = 4
        fig, axes = plt.subplots(math.ceil(n_panels / cols), cols, figsize=(4.6 * cols, 3.8 * math.ceil(n_panels / cols)))
        axes = list(np.atleast_1d(axes).flat)
        width = 0.8 / len(names)
        rf = (reference or {}).get("freq") or {}
        for ax, w in zip(axes, wl):
            for i, n in enumerate(names):
                vals = [max(means[n]["sbrel_%s_%s" % (w, b)], 1e-6) for b in ("LL", "LH", "HL", "HH")]
                ax.bar(np.arange(4) + i * width, vals, width, label=n)
            for i, n in enumerate(names):
                if n in rf:
                    _ref_segments(ax, np.arange(4) + i * width,
                                  [max(rf[n].get("sbrel_%s_%s" % (w, b), 1e-6), 1e-6) for b in ("LL", "LH", "HL", "HH")],
                                  width / 2, first=(i == 0))
            ax.set_xticks(np.arange(4) + width * (len(names) - 1) / 2, labels=["LL", "LH", "HL", "HH"])
            ax.set(title="Subband relative error (%s)" % w, yscale="log")
        ax = axes[len(wl)]
        edges = [(k + 0.5) / 8 for k in range(8)]
        for n in names:
            ax.plot(edges, [means[n]["psd_r%d" % k] for k in range(8)], marker="o", label=n)
        first = True
        for n in names:
            if n in rf and "psd_r0" in rf[n]:
                ax.plot(edges, [rf[n]["psd_r%d" % k] for k in range(8)], color="black", ls="--", lw=1.2, zorder=5,
                        label="lowpass final" if first else None)
                first = False
        ax.axhline(1.0, color="gray", lw=0.8)
        ax.set(title="Power ratio SR/HR by radial band", xlabel="frequency (fraction of Nyquist)", yscale="log")
        for ax, (key, title) in zip(axes[len(wl) + 1:], scalars):
            if key in sample:
                bars = ax.bar(names, [max(means[n][key], 1e-6) for n in names],
                              color=["C%d" % i for i in range(len(names))])
                ax.bar_label(bars, fmt="%.4f")
                idx = [i for i, n in enumerate(names) if n in rf and key in rf[n]]
                if idx:
                    _ref_segments(ax, idx, [max(rf[names[i]][key], 1e-6) for i in idx], 0.4)
                ax.set_yscale("log")  # values differ by ~10x between sets
                lv = [means[n][key] for n in names] + [rf[names[i]][key] for i in idx]
                ax.set_ylim(max(min(lv), 1e-6) * 0.5, max(lv) * 1.6)  # keep bar and reference visibly close
            else:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
            ax.set(title=title)
        for ax in axes[n_panels:]:
            ax.axis("off")
        for ax in axes[:n_panels]:
            ax.grid(alpha=0.3, axis="y")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, dpi=100)
        plt.close(fig)
        return True
    except Exception as e:
        print("[plotting] skipped frequency summary: {}".format(e))
        return False
