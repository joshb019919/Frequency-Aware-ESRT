"""Tests for experiments.py (config list, command building, output parsing, CLI wiring)."""
import os
import subprocess
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
import experiments  # noqa: E402


def test_config_list_covers_every_variant():
    names = [n for n, _, _ in experiments.build_configs()]
    assert names[0] == "lowpass" and len(names) == len(set(names)) == 7
    assert {"dwt_c1_gr1", "dwt_c3_gr0", "dwt_cband_gr1", "dwt_cband_gr0"} <= set(names)
    flags = {n: f for n, f, _ in experiments.build_configs()}["dwt_c3_gr0"]
    assert flags == ["--freq", "dwt", "--conv", "3", "--global_res", "0"]


def test_patch_size_dimension():
    both = experiments.build_configs(experiments.PATCH_SIZES)
    assert len(both) == 14 and len({n for n, _, _ in both}) == 14
    assert {ps for _, _, ps in both} == {96, 144}
    assert all(n.endswith("_p144") == (ps == 144) for n, _, ps in both)
    assert [n for n, _, ps in experiments.build_configs() if ps != 96] == []  # default: 96 only


def test_commands_use_separate_checkpoint_dirs_and_same_settings():
    class A:  # minimal args
        root = "/d"; scale = 2; epochs = 10; batch_size = 16; patch_size = 0; threads = 8; seed = 1
        val_every = 5; fast = False; no_amp = False; test_hr_folder = "Test_Datasets/Set5/"; test_lr_folder = None
    a = A()
    c = experiments.train_cmd("lowpass", [], a, 96)
    assert c[c.index("--save_dir") + 1] == os.path.join("experiment", "exp_lowpass")
    assert c[c.index("-nEpochs") + 1] == "10" and c[c.index("--patch_size") + 1] == "96"
    assert experiments.train_cmd("lowpass_p144", [], a, 144)[c.index("--patch_size") + 1] == "144"
    t = experiments.test_cmd("dwt_c1_gr1", ["--freq", "dwt"], a)
    assert t[t.index("--checkpoint") + 1].endswith(os.path.join("exp_dwt_c1_gr1", "epoch_10.pth"))
    assert t[t.index("--test_lr_folder") + 1] == "Test_Datasets/Set5_LR/x2/" and "--freq" in t


def test_parse_outputs():
    assert experiments.parse_test_output("x\nMean PSNR: 37.29, SSIM: 0.9568, TIME: 167.0 ms\n") == (37.29, 0.9568, 167.0)
    assert experiments.parse_test_output("nothing") == (None, None, None)
    assert experiments.parse_params("Total number of parameters: 677783") == 677783


def test_dry_run_and_unknown_config():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "experiments.py"), "--root", "/d", "--dry_run"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    # shared settings appear once; only differences are listed per experiment line
    assert r.stdout.count("train.py") == 1 and r.stdout.count("test.py") == 1
    assert r.stdout.count("-nEpochs") == 1 and r.stdout.count("--seed") == 1
    lines = [l for l in r.stdout.splitlines() if l.startswith("  ")]
    assert len(lines) == 7 and lines[0].split()[0] == "lowpass"
    assert "--freq dwt --conv 3 --global_res 0" in [l for l in lines if l.split()[0] == "dwt_c3_gr0"][0]
    assert all("--root" not in l for l in lines)
    for ps, n in [("96", 7), ("144", 7), ("both", 14)]:
        r = subprocess.run([sys.executable, os.path.join(ROOT, "experiments.py"), "--root", "/d", "--dry_run",
                            "--patch_size", ps], capture_output=True, text=True)
        lines = [l for l in r.stdout.splitlines() if l.startswith("  ")]
        assert len(lines) == n
        if ps == "both":
            assert sum("--patch_size 144" in l for l in lines) == 7 and sum("--patch_size 96" in l for l in lines) == 7
        else:  # a single patch size is shared by all, so it is printed once, not per line
            assert ("--patch_size " + ps) in r.stdout.split("Per experiment")[0]
            assert all("--patch_size" not in l for l in lines)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "experiments.py"), "--root", "/d", "--only", "nope"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "unknown config" in r.stderr


def test_main_experiments_flag_forwards_to_experiments_py():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "main.py"), "--experiments", "--skip_download",
                        "--data_dir", "/d", "--", "--epochs", "10", "--dry_run"], capture_output=True, text=True, cwd=ROOT)
    assert "experiments.py --root /d --scale 2 --epochs 10 --dry_run" in r.stdout and r.returncode == 0
    r = subprocess.run([sys.executable, os.path.join(ROOT, "main.py"), "--experiments", "--train"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0  # mutually exclusive with --train


def test_dry_run_accepts_hyphen_spelling():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "experiments.py"), "--root", "/d", "--dry-run"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and len([l for l in r.stdout.splitlines() if l.startswith("  ")]) == 7


SAMPLE_LOG = """Total number of parameters: 677783
===> Valid. psnr: 12.8000, ssim: 0.0900 (weights after 0 epochs)
epoch = 1 lr =  0.0002
===> Epoch[1] avg Loss_l1: 0.08123
Epoch cost times: 0:01:05.500000
epoch = 2 lr =  0.0002
===> Epoch[2] avg Loss_l1: 0.05000
Epoch cost times: 0:00:06.250000
===> Valid. psnr: 30.5000, ssim: 0.8800 (weights after 2 epochs)
"""


def test_parse_train_log_per_epoch_rows():
    rows = experiments.parse_train_log(SAMPLE_LOG)
    assert [r["epoch"] for r in rows] == [0, 1, 2]
    assert rows[0]["val_psnr"] == 12.8 and rows[0]["train_loss_l1"] is None   # untrained weights
    assert rows[1] == dict(epoch=1, lr=0.0002, train_loss_l1=0.08123, epoch_s=65.5, val_psnr=None, val_ssim=None)
    assert rows[2]["val_psnr"] == 30.5 and rows[2]["train_loss_l1"] == 0.05 and rows[2]["epoch_s"] == 6.2


def test_parse_train_log_old_format_without_suffix():
    old = "===> Valid. psnr: 10.0, ssim: 0.1\nepoch = 1 lr =  0.0002\nEpoch cost times: 0:00:05.0\n===> Valid. psnr: 20.0, ssim: 0.5\n"
    rows = experiments.parse_train_log(old)
    assert rows[0]["val_psnr"] == 10.0 and rows[1]["val_psnr"] == 20.0  # score before epoch 2 = after epoch 1


def test_from_logs_writes_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "ROOT", str(tmp_path))
    d = tmp_path / "experiment" / "exp_lowpass"
    d.mkdir(parents=True)
    (d / "train.log").write_text(SAMPLE_LOG)
    assert experiments.main(["--from_logs"]) == 0
    out = list((tmp_path / "experiment").glob("epochs_*.csv"))[0].read_text().splitlines()
    assert out[0] == "name,epoch,lr,train_loss_l1,epoch_s,val_psnr,val_ssim" and len(out) == 4
    assert out[2].startswith("lowpass,1,0.0002,0.08123,65.5")


def test_run_logged_streams_and_echoes(tmp_path, capsys):
    code = "print('epoch = 1 lr = 0.1'); print('noise'); print('===> Epoch[1] avg Loss_l1: 0.5')"
    log = tmp_path / "x" / "train.log"
    rc, text, secs = experiments.run_logged([sys.executable, "-c", code], str(log), "[t] ")
    out = capsys.readouterr().out
    assert rc == 0 and "noise" in log.read_text() and "noise" not in out
    assert "[t] epoch = 1 lr = 0.1" in out and "[t] ===> Epoch[1] avg Loss_l1: 0.5" in out
