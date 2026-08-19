"""Regression tests for the MIND_NERVE_RUNTIME_DIR pin logic in the installer.

Audit finding (HIGH): ``_mcp_entry`` used to pin ``MIND_NERVE_RUNTIME_DIR`` to
``~/.local/share/mind-nerve-runtime`` (the "dash" path) unconditionally, with
no existence/populated check. A fresh install never creates that dir — the
auto-seed target is ``~/.local/share/mind-nerve/runtime`` (the "slash" path,
see ``inference._default_user_runtime_dir``) — and
``inference._resolve_runtime_dir`` raises ``FileNotFoundError`` for an
explicitly-pinned dir that does not exist. Pinning the non-existent dash path
therefore crashed every generated MCP config (``mind-nerve-mcp``, including
the Windows in-process fallback in ``mcp_server.py``) on first use.

The fix mirrors ``existingRuntimeDir()`` in
``integrations/installer/src/install.ts``: only pin when a runtime dir is
actually populated (``manifest.json`` or ``route_table.jsonl`` present),
otherwise omit the pin and let ``inference`` auto-resolve/seed at runtime.

Also covers the secondary leak: ``_mcp_entry`` is reused for the unrelated
``mind-mem-mcp`` server (``_register_mind_mem_in``) and must never inject the
mind-nerve-specific env var into that server's env.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def fake_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect the installer's HOME to a temp directory (never touch real ~)."""
    from mind_nerve import installer

    monkeypatch.setattr(installer, "HOME", tmp_path)
    monkeypatch.delenv("MIND_NERVE_RUNTIME_DIR", raising=False)
    return tmp_path


def _populate(dash_dir: Path, marker: str = "manifest.json") -> None:
    dash_dir.mkdir(parents=True, exist_ok=True)
    (dash_dir / marker).write_text("{}" if marker.endswith(".json") else "")


class TestMcpEntryRuntimeDirPin:
    """_mcp_entry() must only pin a runtime dir that actually exists."""

    def test_unpopulated_dash_dir_is_not_pinned(self, fake_home: Path) -> None:
        from mind_nerve.installer import _mcp_entry

        entry = _mcp_entry()

        assert "MIND_NERVE_RUNTIME_DIR" not in entry["env"], (
            "a fresh install has no populated runtime dir yet -- pinning a "
            "nonexistent dir makes inference._resolve_runtime_dir raise "
            "FileNotFoundError on first use"
        )

    def test_missing_dash_dir_entirely_is_not_pinned(self, fake_home: Path) -> None:
        from mind_nerve.installer import _mcp_entry

        # Dash dir does not exist at all (not even created empty).
        dash = fake_home / ".local" / "share" / "mind-nerve-runtime"
        assert not dash.exists()

        entry = _mcp_entry()

        assert "MIND_NERVE_RUNTIME_DIR" not in entry["env"]

    def test_populated_dash_dir_via_manifest_is_pinned(self, fake_home: Path) -> None:
        from mind_nerve.installer import _mcp_entry

        dash = fake_home / ".local" / "share" / "mind-nerve-runtime"
        _populate(dash, "manifest.json")

        entry = _mcp_entry()

        assert entry["env"]["MIND_NERVE_RUNTIME_DIR"] == str(dash)

    def test_populated_dash_dir_via_route_table_jsonl_is_pinned(self, fake_home: Path) -> None:
        from mind_nerve.installer import _mcp_entry

        dash = fake_home / ".local" / "share" / "mind-nerve-runtime"
        _populate(dash, "route_table.jsonl")

        entry = _mcp_entry()

        assert entry["env"]["MIND_NERVE_RUNTIME_DIR"] == str(dash)

    def test_caller_supplied_env_is_not_overridden(self, fake_home: Path) -> None:
        from mind_nerve.installer import _mcp_entry

        dash = fake_home / ".local" / "share" / "mind-nerve-runtime"
        _populate(dash)

        entry = _mcp_entry(env={"MIND_NERVE_RUNTIME_DIR": "/explicit/override"})

        assert entry["env"]["MIND_NERVE_RUNTIME_DIR"] == "/explicit/override"

    def test_env_var_pin_respected_when_populated(
        self, fake_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An operator-set $MIND_NERVE_RUNTIME_DIR that IS populated wins over
        the dash default (matches existingRuntimeDir() in install.ts)."""
        from mind_nerve.installer import _mcp_entry

        custom = fake_home / "custom-runtime"
        _populate(custom)
        monkeypatch.setenv("MIND_NERVE_RUNTIME_DIR", str(custom))

        entry = _mcp_entry()

        assert entry["env"]["MIND_NERVE_RUNTIME_DIR"] == str(custom)

    def test_env_var_pin_unpopulated_is_not_pinned(
        self, fake_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An operator-set $MIND_NERVE_RUNTIME_DIR pointing at an empty dir
        must not be echoed back as a pin either."""
        from mind_nerve.installer import _mcp_entry

        custom = fake_home / "custom-runtime"
        custom.mkdir()
        monkeypatch.setenv("MIND_NERVE_RUNTIME_DIR", str(custom))

        entry = _mcp_entry()

        assert "MIND_NERVE_RUNTIME_DIR" not in entry["env"]


class TestMcpEntryScopedToMindNerve:
    """The mind-nerve-specific env var must never leak into mind-mem's entry."""

    def test_mind_mem_entry_never_pins_runtime_dir(self, fake_home: Path) -> None:
        from mind_nerve.installer import _mcp_entry

        dash = fake_home / ".local" / "share" / "mind-nerve-runtime"
        _populate(dash)

        entry = _mcp_entry("mind-mem-mcp", pin_runtime_dir=False)

        assert "MIND_NERVE_RUNTIME_DIR" not in entry["env"]
        assert entry["command"] == "mind-mem-mcp"

    def test_register_mind_mem_json_does_not_leak_runtime_dir(self, fake_home: Path) -> None:
        from mind_nerve.installer import _register_mind_mem_in

        dash = fake_home / ".local" / "share" / "mind-nerve-runtime"
        _populate(dash)

        cfg = fake_home / ".cursor" / "mcp.json"
        cfg.parent.mkdir(parents=True)

        _register_mind_mem_in(cfg, "json")

        saved = json.loads(cfg.read_text())
        assert "MIND_NERVE_RUNTIME_DIR" not in saved["mcpServers"]["mind-mem"]["env"]


class TestInstallCodexRuntimeDirPin:
    """install_codex()'s generated TOML block follows the same populated-only rule."""

    def test_fresh_install_omits_runtime_dir_env(self, fake_home: Path) -> None:
        from mind_nerve.installer import install_codex

        install_codex({})

        text = (fake_home / ".codex" / "config.toml").read_text()
        assert "MIND_NERVE_RUNTIME_DIR" not in text
        assert "env = {}" in text

    def test_populated_dash_dir_pins_runtime_dir_env(self, fake_home: Path) -> None:
        from mind_nerve.installer import install_codex

        dash = fake_home / ".local" / "share" / "mind-nerve-runtime"
        _populate(dash)

        install_codex({})

        text = (fake_home / ".codex" / "config.toml").read_text()
        assert f'MIND_NERVE_RUNTIME_DIR = "{dash}"' in text
