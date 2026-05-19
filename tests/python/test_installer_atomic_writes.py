"""Per-target atomic-write + rollback round-trip tests.

Stream D / audit P0 follow-up to the safe_write hardening in commit
``aac27fd``. Verifies that every public-named install target:

1. Writes its user config via :func:`mind_nerve.installer.safe_write`
   — i.e. it creates a ``<path>.bak`` containing the previous bytes,
   lands the new content via a temp file in the same dir, and atomically
   renames over the target (`os.replace`).
2. Survives :func:`mind_nerve.installer.rollback_last`, which must
   restore the original bytes from each ``.bak`` after a write.

The tests reach into ``mind_nerve.installer.HOME`` via monkeypatch so
they never touch the real user home directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def fake_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect the installer's HOME to a temp directory.

    Multiple module-level attributes in :mod:`mind_nerve.installer`
    reference ``HOME`` either at import time (the dicts) or directly
    inside install functions (``HOME / ...``). We rebind the module
    attribute itself; install functions read it fresh each call.
    """
    from mind_nerve import installer

    monkeypatch.setattr(installer, "HOME", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Atomic write coverage — per public-named target
# ---------------------------------------------------------------------------


class TestAtomicWriteCoverage:
    """Every public-named target writes through safe_write and produces .bak."""

    def test_claude_desktop_round_trips_through_safe_write(self, fake_home: Path) -> None:
        from mind_nerve.installer import install_claude_desktop

        cfg = fake_home / ".config" / "Claude" / "claude_desktop_config.json"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(json.dumps({"mcpServers": {"old": {"command": "x"}}}) + "\n")

        result = install_claude_desktop({})

        assert result["installed"] is True
        bak = cfg.with_suffix(cfg.suffix + ".bak")
        assert bak.exists(), "claude-desktop installer must produce a .bak"
        assert "old" in json.loads(bak.read_text())["mcpServers"]
        assert "mind-nerve" in json.loads(cfg.read_text())["mcpServers"]

    def test_cursor_round_trips_through_safe_write(self, fake_home: Path) -> None:
        from mind_nerve.installer import install_cursor

        cfg = fake_home / ".cursor" / "mcp.json"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(json.dumps({"mcpServers": {"old": {"command": "x"}}}) + "\n")

        install_cursor({})

        bak = cfg.with_suffix(cfg.suffix + ".bak")
        assert bak.exists(), "cursor installer must produce a .bak"
        assert "old" in json.loads(bak.read_text())["mcpServers"]
        assert "mind-nerve" in json.loads(cfg.read_text())["mcpServers"]

    def test_codex_round_trips_through_safe_write(self, fake_home: Path) -> None:
        from mind_nerve.installer import install_codex

        cfg = fake_home / ".codex" / "config.toml"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text('[some-other]\nkey = "value"\n')

        install_codex({})

        bak = cfg.with_suffix(cfg.suffix + ".bak")
        assert bak.exists(), "codex installer must produce a .bak"
        assert "[some-other]" in bak.read_text()
        assert "[mcp_servers.mind-nerve]" in cfg.read_text()

    def test_gemini_round_trips_through_safe_write(self, fake_home: Path) -> None:
        from mind_nerve.installer import install_gemini

        # Pre-create an extension manifest so we can verify the .bak path.
        manifest = fake_home / ".gemini" / "extensions" / "mind-nerve" / "extension.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"name": "mind-nerve", "version": "old"}) + "\n")

        install_gemini({})

        bak = manifest.with_suffix(manifest.suffix + ".bak")
        assert bak.exists(), "gemini installer must produce a .bak"
        assert json.loads(bak.read_text())["version"] == "old"
        assert "mcpServers" in json.loads(manifest.read_text())

    def test_vibe_round_trips_through_safe_write(self, fake_home: Path) -> None:
        from mind_nerve.installer import install_vibe

        cfg = fake_home / ".vibe" / "mcp.json"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(json.dumps({"mcpServers": {"old": {"command": "x"}}}) + "\n")

        install_vibe({})

        bak = cfg.with_suffix(cfg.suffix + ".bak")
        assert bak.exists(), "vibe installer must produce a .bak"
        assert "old" in json.loads(bak.read_text())["mcpServers"]
        assert "mind-nerve" in json.loads(cfg.read_text())["mcpServers"]

    def test_claude_code_hook_round_trips_through_safe_write(self, fake_home: Path) -> None:
        from mind_nerve.installer import install_claude_code_hook

        cfg = fake_home / ".claude" / "settings.json"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(json.dumps({"hooks": {"Other": []}}) + "\n")

        install_claude_code_hook({})

        bak = cfg.with_suffix(cfg.suffix + ".bak")
        assert bak.exists(), "claude-code-hook installer must produce a .bak"
        assert json.loads(bak.read_text()) == {"hooks": {"Other": []}}
        saved = json.loads(cfg.read_text())
        assert "UserPromptSubmit" in saved["hooks"]

    def test_no_temp_files_left_after_write(self, fake_home: Path) -> None:
        """safe_write must clean up its NamedTemporaryFile via os.replace."""
        from mind_nerve.installer import install_cursor

        cfg = fake_home / ".cursor" / "mcp.json"
        cfg.parent.mkdir(parents=True, exist_ok=True)

        install_cursor({})

        leftovers = [p for p in cfg.parent.iterdir() if p.name not in {"mcp.json", "mcp.json.bak"}]
        assert leftovers == [], f"unexpected temp files remain: {leftovers}"


# ---------------------------------------------------------------------------
# rollback_last — restore from .bak
# ---------------------------------------------------------------------------


class TestRollbackLast:
    """Per-target round-trip: original → install → mutate → rollback → original."""

    def test_rollback_unknown_target_returns_error(self, fake_home: Path) -> None:
        from mind_nerve.installer import rollback_last

        result = rollback_last("not-a-target")
        assert result["errors"], "unknown target must report an error"
        assert result["restored"] == []

    def test_rollback_with_no_bak_reports_missing(self, fake_home: Path) -> None:
        from mind_nerve.installer import rollback_last

        result = rollback_last("cursor")
        # Nothing was ever installed, so every probed path is "missing".
        assert result["restored"] == []
        assert result["missing"], "rollback with no .bak should report missing paths"
        assert result["errors"] == []

    def test_cursor_install_then_rollback_restores_original(self, fake_home: Path) -> None:
        from mind_nerve.installer import install_cursor, rollback_last

        cfg = fake_home / ".cursor" / "mcp.json"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        original = json.dumps({"mcpServers": {"only-mine": {"command": "x"}}}, indent=2) + "\n"
        cfg.write_text(original)

        install_cursor({})
        bak = cfg.with_suffix(cfg.suffix + ".bak")
        assert bak.exists()

        result = rollback_last("cursor")
        assert str(cfg) in result["restored"]
        assert result["errors"] == []
        assert cfg.read_text() == original, "rollback did not restore original bytes"

    def test_codex_install_then_rollback_restores_original(self, fake_home: Path) -> None:
        from mind_nerve.installer import install_codex, rollback_last

        cfg = fake_home / ".codex" / "config.toml"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        original = '[mcp_servers.other]\ncommand = "other"\n'
        cfg.write_text(original)

        install_codex({})
        assert "[mcp_servers.mind-nerve]" in cfg.read_text()

        result = rollback_last("codex")
        assert str(cfg) in result["restored"]
        assert cfg.read_text() == original

    def test_gemini_install_then_rollback_restores_original(self, fake_home: Path) -> None:
        from mind_nerve.installer import install_gemini, rollback_last

        manifest = fake_home / ".gemini" / "extensions" / "mind-nerve" / "extension.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        original = json.dumps({"name": "mind-nerve", "version": "old-but-mine"}) + "\n"
        manifest.write_text(original)

        install_gemini({})
        assert "mcpServers" in json.loads(manifest.read_text())

        result = rollback_last("gemini")
        assert str(manifest) in result["restored"]
        assert manifest.read_text() == original

    def test_vibe_install_then_rollback_restores_original(self, fake_home: Path) -> None:
        from mind_nerve.installer import install_vibe, rollback_last

        cfg = fake_home / ".vibe" / "mcp.json"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        original = json.dumps({"mcpServers": {"keepme": {"command": "y"}}}, indent=2) + "\n"
        cfg.write_text(original)

        install_vibe({})
        result = rollback_last("vibe")

        assert str(cfg) in result["restored"]
        assert cfg.read_text() == original

    def test_claude_code_hook_install_then_rollback_restores_original(
        self, fake_home: Path
    ) -> None:
        from mind_nerve.installer import install_claude_code_hook, rollback_last

        cfg = fake_home / ".claude" / "settings.json"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        original = json.dumps({"hooks": {"UserPromptSubmit": []}}, indent=2) + "\n"
        cfg.write_text(original)

        install_claude_code_hook({})

        # The "claude" alias rollback covers both ~/.claude.json AND
        # ~/.claude/settings.json — settings.json is what the hook
        # installer mutates, so the alias path must restore it.
        result = rollback_last("claude")
        assert str(cfg) in result["restored"]
        assert cfg.read_text() == original

    def test_rollback_is_idempotent(self, fake_home: Path) -> None:
        """A second rollback after the first is a no-op: .bak still exists,
        target now equals .bak, so re-restoring the same bytes is harmless."""
        from mind_nerve.installer import install_cursor, rollback_last

        cfg = fake_home / ".cursor" / "mcp.json"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        original = json.dumps({"mcpServers": {}}, indent=2) + "\n"
        cfg.write_text(original)

        install_cursor({})
        rollback_last("cursor")
        rollback_last("cursor")

        assert cfg.read_text() == original


# ---------------------------------------------------------------------------
# Audit gate: every public-named target's installer mentions safe_write.
# This is a source-level check that prevents a regression where someone
# adds a new write path that bypasses the atomic helper.
# ---------------------------------------------------------------------------


class TestInstallerSourceContract:
    def test_every_public_installer_uses_safe_write(self) -> None:
        import inspect

        from mind_nerve import installer

        public_named_installers = [
            installer.install_claude_desktop,
            installer.install_cursor,
            installer.install_codex,
            installer.install_gemini,
            installer.install_vibe,
            installer.install_claude_code_hook,
        ]
        for fn in public_named_installers:
            src = inspect.getsource(fn)
            assert "safe_write(" in src, (
                f"{fn.__name__} must write through safe_write() — "
                f"direct write_text/write_bytes bypasses the .bak contract."
            )

    def test_safe_write_preserves_atomic_rename_contract(self) -> None:
        """Source-level invariant: safe_write uses os.replace, not write_text."""
        import inspect

        from mind_nerve.installer import safe_write

        src = inspect.getsource(safe_write)
        assert "os.replace(" in src
        assert "NamedTemporaryFile" in src
        # Must back up before writing.
        assert '".bak"' in src
