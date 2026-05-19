"""Tests for mind_nerve._runtime_dir.runtime_socket_dir()."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest


def test_xdg_runtime_dir_used_when_set_and_writable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When XDG_RUNTIME_DIR is set and writable, it is returned as-is."""
    xdg_dir = tmp_path / "xdg"
    xdg_dir.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(xdg_dir))

    import mind_nerve._runtime_dir as rd

    result = rd.runtime_socket_dir()
    assert result == xdg_dir


def test_cache_dir_created_when_xdg_unset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When XDG_RUNTIME_DIR is unset, ~/.cache/mind-nerve/run is created at mode 0700."""
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    # Patch Path.home() so we don't touch the real home dir.
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

    import mind_nerve._runtime_dir as rd

    result = rd.runtime_socket_dir()

    expected = fake_home / ".cache" / "mind-nerve" / "run"
    assert result == expected, f"expected {expected}, got {result}"
    assert result.exists(), "directory was not created"
    mode = stat.S_IMODE(result.stat().st_mode)
    assert mode == 0o700, f"expected mode 0700, got {oct(mode)}"


def test_returned_path_is_writable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The returned path must be writable by the current process."""
    xdg_dir = tmp_path / "xdg"
    xdg_dir.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(xdg_dir))

    import mind_nerve._runtime_dir as rd

    result = rd.runtime_socket_dir()
    assert os.access(result, os.W_OK), f"returned path {result} is not writable"
