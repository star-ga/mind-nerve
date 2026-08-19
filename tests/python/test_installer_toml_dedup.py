"""Regression tests for TOML MCP-block self-healing idempotency.

Audit finding (MEDIUM): ``install_codex`` / ``_register_mind_mem_in`` matched
an existing ``[mcp_servers.<name>]`` block with a regex and replaced it via
``pattern.sub(block, existing_text)``. ``re.sub``'s default ``count=0``
replaces EVERY match, so a config already left with TWO duplicate headers (by
the older non-idempotent bug) got each of the two normalized independently --
still two headers, which codex rejects as a duplicate TOML table. The fix
(``_upsert_toml_block``) strips ALL matches first, then appends a single
fresh block, so one re-run collapses any number of pre-existing duplicates
down to exactly one.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def fake_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    from mind_nerve import installer

    monkeypatch.setattr(installer, "HOME", tmp_path)
    monkeypatch.delenv("MIND_NERVE_RUNTIME_DIR", raising=False)
    return tmp_path


class TestUpsertTomlBlockUnit:
    """Unit coverage of the shared dedup helper."""

    def test_no_prior_match_appends(self) -> None:
        import re

        from mind_nerve.installer import _upsert_toml_block

        pattern = re.compile(r"\n?\[mcp_servers\.mind-nerve\][\s\S]*?(?=\n\[|\Z)", re.DOTALL)
        existing = '[other]\nkey = "value"\n'
        block = '\n[mcp_servers.mind-nerve]\ncommand = "x"\nargs = []\nenv = {}\n'

        updated = _upsert_toml_block(existing, pattern, block)

        assert updated.count("[mcp_servers.mind-nerve]") == 1
        assert '[other]\nkey = "value"' in updated

    def test_single_prior_match_replaced_not_duplicated(self) -> None:
        import re

        from mind_nerve.installer import _upsert_toml_block

        pattern = re.compile(r"\n?\[mcp_servers\.mind-nerve\][\s\S]*?(?=\n\[|\Z)", re.DOTALL)
        existing = '[other]\nkey = "value"\n\n[mcp_servers.mind-nerve]\ncommand = "old"\nargs = []\nenv = {}\n'
        block = '\n[mcp_servers.mind-nerve]\ncommand = "new"\nargs = []\nenv = {}\n'

        updated = _upsert_toml_block(existing, pattern, block)

        assert updated.count("[mcp_servers.mind-nerve]") == 1
        assert 'command = "new"' in updated
        assert 'command = "old"' not in updated

    def test_two_prior_duplicate_matches_collapse_to_one(self) -> None:
        """The exact corruption shape from the old bug: two headers already
        on disk. A single call must self-heal to exactly one."""
        import re

        from mind_nerve.installer import _upsert_toml_block

        pattern = re.compile(r"\n?\[mcp_servers\.mind-nerve\][\s\S]*?(?=\n\[|\Z)", re.DOTALL)
        existing = (
            '[other]\nkey = "value"\n'
            '\n[mcp_servers.mind-nerve]\ncommand = "dup1"\nargs = []\nenv = {}\n'
            '\n[mcp_servers.mind-nerve]\ncommand = "dup2"\nargs = []\nenv = {}\n'
        )
        block = '\n[mcp_servers.mind-nerve]\ncommand = "healed"\nargs = []\nenv = {}\n'

        updated = _upsert_toml_block(existing, pattern, block)

        assert updated.count("[mcp_servers.mind-nerve]") == 1
        assert 'command = "healed"' in updated


class TestInstallCodexSelfHeals:
    """install_codex() end-to-end: a config already corrupted with two
    duplicate mind-nerve blocks must end with exactly one after a re-run."""

    def test_two_preexisting_blocks_become_one(self, fake_home: Path) -> None:
        from mind_nerve.installer import install_codex

        cfg = fake_home / ".codex" / "config.toml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(
            '[mcp_servers.other]\ncommand = "other"\n'
            '\n[mcp_servers.mind-nerve]\ncommand = "mind-nerve-mcp"\nargs = []\nenv = {}\n'
            '\n[mcp_servers.mind-nerve]\ncommand = "mind-nerve-mcp"\nargs = []\nenv = {}\n'
        )

        install_codex({})

        text = cfg.read_text()
        assert text.count("[mcp_servers.mind-nerve]") == 1
        assert "[mcp_servers.other]" in text

    def test_normal_single_reinstall_stays_single(self, fake_home: Path) -> None:
        from mind_nerve.installer import install_codex

        install_codex({})
        install_codex({})

        text = (fake_home / ".codex" / "config.toml").read_text()
        assert text.count("[mcp_servers.mind-nerve]") == 1


class TestRegisterMindMemSelfHeals:
    def test_two_preexisting_mind_mem_blocks_become_one(self, fake_home: Path) -> None:
        from mind_nerve.installer import _register_mind_mem_in

        cfg = fake_home / ".codex" / "config.toml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(
            '[mcp_servers.other]\ncommand = "other"\n'
            '\n[mcp_servers.mind-mem]\ncommand = "mind-mem-mcp"\nargs = []\nenv = {}\n'
            '\n[mcp_servers.mind-mem]\ncommand = "mind-mem-mcp"\nargs = []\nenv = {}\n'
        )

        _register_mind_mem_in(cfg, "toml")

        text = cfg.read_text()
        assert text.count("[mcp_servers.mind-mem]") == 1
        assert "[mcp_servers.other]" in text
