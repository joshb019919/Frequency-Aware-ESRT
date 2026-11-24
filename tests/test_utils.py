"""Tests for utils.detect_image_ext (test.py image-extension detection)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import utils  # noqa: E402


def _touch(d, *names):
    for n in names:
        (d / n).write_bytes(b"")


def test_detects_png(tmp_path):
    _touch(tmp_path, "a.png", "b.png")
    assert utils.detect_image_ext(str(tmp_path)) == ".png"


def test_detects_bmp(tmp_path):
    _touch(tmp_path, "a.bmp")
    assert utils.detect_image_ext(str(tmp_path)) == ".bmp"


def test_prefers_png_when_both(tmp_path):
    _touch(tmp_path, "a.png", "a.bmp")
    assert utils.detect_image_ext(str(tmp_path)) == ".png"


def test_clear_error_when_none(tmp_path):
    _touch(tmp_path, "notes.txt")
    with pytest.raises(FileNotFoundError, match="no .png or .bmp images"):
        utils.detect_image_ext(str(tmp_path))
