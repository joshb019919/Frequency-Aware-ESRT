"""Training / validation / test plots for train.py and test.py (matplotlib, Agg backend).

Plotting never aborts a run: if matplotlib is missing or saving fails, a warning is printed
and training/testing continues.
"""
import json
import os


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


class TrainValPlotter:
    """Collects per-epoch train loss and per-set validation PSNR/SSIM, drawn side by side.

    Layout: [train L1 loss | validation PSNR | validation SSIM], one line per validation set.
    Writes <out_dir>/curves.png (redrawn on every `save()`) and <out_dir>/history.json.
    """

    def __init__(self, out_dir, title=""):
        self.out_dir, self.title = out_dir, title
        self.loss = {}   # epoch -> mean L1
        self.val = {}    # set name -> {epochs_trained: (psnr, ssim)}

    def add_loss(self, epoch, loss):
        if loss is not None:
            self.loss[epoch] = float(loss)

    def add_val(self, epochs_trained, results):
        """results: {set name: (psnr, ssim)} measured with weights after `epochs_trained` epochs."""
        for name, (p, s) in results.items():
            self.val.setdefault(name, {})[epochs_trained] = (float(p), float(s))

    def save(self):
        try:
            plt = _plt()
            os.makedirs(self.out_dir, exist_ok=True)
            fig, ax = plt.subplots(1, 3, figsize=(15, 4))
            if self.loss:
                e = sorted(self.loss)
                ax[0].plot(e, [self.loss[k] for k in e])
            ax[0].set(title="Train L1 loss", xlabel="epoch", ylabel="L1")
            ax[0].set_yscale("log")
            for i, (idx, label) in enumerate(((0, "PSNR (dB, Y)"), (1, "SSIM (Y)")), start=1):
                for name, d in sorted(self.val.items()):
                    x = sorted(d)
                    ax[i].plot(x, [d[k][idx] for k in x], marker="o", ms=3, label=name)
                ax[i].set(title="Validation " + label.split()[0], xlabel="epochs trained", ylabel=label)
                if self.val:
                    ax[i].legend()
            for a in ax:
                a.grid(alpha=0.3)
            if self.title:
                fig.suptitle(self.title)
            fig.tight_layout()
            fig.savefig(os.path.join(self.out_dir, "curves.png"), dpi=110)
            plt.close(fig)
            with open(os.path.join(self.out_dir, "history.json"), "w") as f:
                json.dump({"loss": self.loss, "val": self.val}, f)
        except Exception as e:  # plotting must never kill a training run
            print("[plotting] skipped training curves: {}".format(e))


def plot_test_summary(path, results, title=""):
    """Bar charts of mean PSNR / SSIM per test set plus per-image PSNR distribution.

    results: {set name: {"psnr": [per image], "ssim": [per image]}}. Saves a PNG to `path`.
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
            a.set(title=label, ylim=(min(vals) * 0.95, max(vals) * 1.02))
        ax[2].boxplot([results[n]["psnr"] for n in names], showmeans=True)
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
