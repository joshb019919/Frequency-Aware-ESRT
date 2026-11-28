"""Timestamped run folders in experiments.py: nothing from an earlier run may ever be overwritten."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import experiments  # noqa: E402


class A:
    root = "/d"; scale = 2; epochs = 10; batch_size = 16; threads = 8; seed = 1; val_every = 5
    fast = False; no_amp = False; test_hr_folder = "Test_Datasets/Set5/"; test_lr_folder = None

    def __init__(self, run_id):
        self.run_id = run_id


def test_paths_include_run_id_and_legacy_unchanged():
    assert experiments.exp_dir("x") == os.path.join("experiment", "exp_x")  # legacy flat layout
    assert experiments.exp_dir("x", "run_1") == os.path.join("experiment", "run_1", "exp_x")
    assert experiments.results_dir("x", "run_1") == os.path.join("results", "run_1", "exp_x")
    assert experiments.checkpoint_path("x", 5, "run_1") == os.path.join("experiment", "run_1", "exp_x", "epoch_5.pth")


def test_commands_for_different_runs_never_share_paths():
    a1, a2 = A("run_20261125_100000"), A("run_20261125_110000")
    t1, t2 = experiments.train_cmd("lowpass", [], a1, 96), experiments.train_cmd("lowpass", [], a2, 96)
    assert t1[t1.index("--save_dir") + 1] != t2[t2.index("--save_dir") + 1]
    assert "run_20261125_100000" in t1[t1.index("--save_dir") + 1]
    e1 = experiments.test_cmd("lowpass", [], a1)
    assert "run_20261125_100000" in e1[e1.index("--checkpoint") + 1] and "run_20261125_100000" in e1[e1.index("--output_folder") + 1]


def test_new_run_id_format():
    import re
    assert re.fullmatch(r"run_\d{8}_\d{6}", experiments.new_run_id())


def test_unique_path(tmp_path):
    p = str(tmp_path / "e.csv")
    assert experiments.unique_path(p) == p
    open(p, "w").close()
    assert experiments.unique_path(p) == str(tmp_path / "e_2.csv")
    open(str(tmp_path / "e_2.csv"), "w").close()
    assert experiments.unique_path(p) == str(tmp_path / "e_3.csv")


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    """Run main() in a temp ROOT with train/test replaced by a recorder that creates the final checkpoint."""
    monkeypatch.setattr(experiments, "ROOT", str(tmp_path))
    calls = []

    def fake_run_logged(cmd, log_path, echo_prefix=""):
        calls.append((cmd, log_path))
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        if "train.py" in cmd[1]:
            out = cmd[cmd.index("--save_dir") + 1]
            os.makedirs(os.path.join(str(tmp_path), out), exist_ok=True)
            open(os.path.join(str(tmp_path), out, "epoch_1.pth"), "w").close()
            text = "Total number of parameters: 5\n"
        else:
            text = "Mean PSNR: 30.0, SSIM: 0.9, TIME: 5.0 ms\n"
        open(log_path, "w").write(text)
        return 0, text, 1.0
    monkeypatch.setattr(experiments, "run_logged", fake_run_logged)
    return tmp_path, calls


def _run(*extra):
    return experiments.main(["--root", "/d", "--epochs", "1", "--only", "lowpass", "--no-fast"] + list(extra))


def test_two_invocations_use_separate_folders_and_keep_legacy_untouched(fake_env, monkeypatch):
    tmp, calls = fake_env
    legacy = tmp / "experiment" / "exp_lowpass"
    legacy.mkdir(parents=True)
    (legacy / "epoch_200.pth").write_text("precious")
    (legacy / "train.log").write_text("old log")
    ids = iter(["run_20261125_100000", "run_20261125_100001"])
    monkeypatch.setattr(experiments, "new_run_id", lambda: next(ids))
    assert _run() == 0 and _run() == 0
    for rid in ("run_20261125_100000", "run_20261125_100001"):
        assert (tmp / "experiment" / rid / "exp_lowpass" / "train.log").exists()
        assert (tmp / "experiment" / rid / "exp_lowpass" / "test.log").exists()
    assert (legacy / "epoch_200.pth").read_text() == "precious" and (legacy / "train.log").read_text() == "old log"
    csvs = [p.name for p in (tmp / "experiment").glob("experiments_*.csv")]
    assert len([c for c in csvs if "_epochs" not in c]) == 2 and len([c for c in csvs if "_epochs" in c]) == 2
    train_calls = [c for c, _ in calls if "train.py" in c[1]]
    assert len({c[c.index("--save_dir") + 1] for c in train_calls}) == 2


def test_reusing_run_id_skips_finished_config_and_uses_new_csv_name(fake_env):
    tmp, calls = fake_env
    assert _run("--run_id", "run_a") == 0
    n_train = sum("train.py" in c[1] for c, _ in calls)
    assert _run("--run_id", "run_a") == 0  # final checkpoint exists -> not retrained
    assert sum("train.py" in c[1] for c, _ in calls) == n_train
    assert sorted(p.name for p in (tmp / "experiment").glob("experiments_a*.csv") if "_epochs" not in p.name) == \
        ["experiments_a.csv", "experiments_a_2.csv"]  # second CSV did not overwrite the first


def test_unfinished_run_folder_is_refused_without_force(fake_env):
    tmp, calls = fake_env
    d = tmp / "experiment" / "run_b" / "exp_lowpass"
    d.mkdir(parents=True)
    (d / "epoch_0.pth").write_text("partial")  # files but no epoch_1.pth (the final checkpoint)
    with pytest.raises(SystemExit):
        _run("--run_id", "run_b")
    assert calls == [] and (d / "epoch_0.pth").read_text() == "partial"
    assert _run("--run_id", "run_b", "--force") == 0  # explicit override is allowed


def test_dry_run_prints_run_id(capsys):
    assert experiments.main(["--root", "/d", "--only", "lowpass", "--dry_run"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Run id: run_") and "--save_dir experiment/run_" in out


def test_from_logs_finds_run_folders(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "ROOT", str(tmp_path))
    log = "epoch = 1 lr =  0.0002\n===> Epoch[1] avg Loss_l1: 0.1\n"
    for d in (tmp_path / "experiment" / "exp_old", tmp_path / "experiment" / "run_x" / "exp_new"):
        d.mkdir(parents=True)
        (d / "train.log").write_text(log)
    assert experiments.main(["--from_logs"]) == 0
    text = list((tmp_path / "experiment").glob("epochs_*.csv"))[0].read_text()
    assert "\nold," in text and "\nnew@run_x," in text
