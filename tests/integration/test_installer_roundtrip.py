"""Integration test: installer `safe_write` write-then-read round-trip.

Audit P0: every installer write goes through `safe_write` which (1)
backs up the original to `.bak` before writing and (2) atomic-replaces
the target via tempfile + os.replace. This test exercises:

  1. Round-trip — write a config, read it back, content matches.
  2. Failure-mode resilience — when `os.replace` raises mid-write the
     original file must remain intact and the `.bak` of the prior
     content must still be present.
  3. Backup semantics — the `.bak` always reflects the previous
     committed bytes, never the new ones.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.mark.integration
def test_safe_write_roundtrip_json(tmp_path: Path) -> None:
    """Write a JSON config via safe_write, read it back byte-identically."""
    from mind_nerve.installer import safe_write

    target = tmp_path / "settings.json"
    payload = {
        "mcpServers": {
            "mind-nerve": {"command": "mind-nerve-mcp", "args": []},
        },
        "version": 1,
    }
    serialized = json.dumps(payload, indent=2) + "\n"
    safe_write(target, serialized)

    assert target.exists(), "target file was not created"
    assert target.read_text() == serialized, "round-trip content mismatch"
    parsed = json.loads(target.read_text())
    assert parsed == payload


@pytest.mark.integration
def test_safe_write_roundtrip_toml(tmp_path: Path) -> None:
    """TOML round-trip — installer also writes TOML for some CLIs."""
    from mind_nerve.installer import safe_write

    target = tmp_path / "config.toml"
    content = '\n[mcp_servers.mind-nerve]\ncommand = "mind-nerve-mcp"\nargs = []\nenv = {}\n'
    safe_write(target, content)
    assert target.read_text() == content


@pytest.mark.integration
def test_safe_write_failure_preserves_original(tmp_path: Path) -> None:
    """If `os.replace` raises, the original file must be preserved on disk
    and the `.bak` containing the original bytes must still exist.
    """
    from mind_nerve.installer import safe_write

    target = tmp_path / "settings.json"
    original = '{"original": true, "version": 1}\n'
    target.write_text(original)

    # Patch os.replace to simulate an atomic-rename failure. The backup
    # is created BEFORE the temp file rename in safe_write — so the
    # backup must still appear, but the live file must remain unchanged.
    with patch("mind_nerve.installer.os.replace") as mock_replace:
        mock_replace.side_effect = OSError("simulated rename failure")
        with pytest.raises(OSError, match="simulated rename failure"):
            safe_write(target, '{"updated": "value"}\n')

    # Live target must still hold the original content.
    assert target.exists(), "target file vanished after failed write"
    assert target.read_text() == original, "target was corrupted by failed write"

    # The .bak must exist and hold the original bytes (backup happens
    # before the rename so the failure does not erase it).
    bak = target.with_suffix(target.suffix + ".bak")
    assert bak.exists(), ".bak was not created before the failing rename"
    assert bak.read_text() == original


@pytest.mark.integration
def test_safe_write_bak_holds_previous_content(tmp_path: Path) -> None:
    """After a successful write the .bak must hold the bytes that were
    there before the write — not the new content."""
    from mind_nerve.installer import safe_write

    target = tmp_path / "settings.json"
    v1 = '{"v": 1}\n'
    v2 = '{"v": 2}\n'

    safe_write(target, v1)
    bak_v1 = target.with_suffix(target.suffix + ".bak")
    assert not bak_v1.exists(), "first write should not create .bak"

    safe_write(target, v2)
    assert target.read_text() == v2
    assert bak_v1.exists()
    assert bak_v1.read_text() == v1, ".bak should hold the pre-write bytes"


@pytest.mark.integration
def test_safe_write_creates_parent_dirs(tmp_path: Path) -> None:
    """safe_write must create the parent directory tree if absent — this
    is the common path when installing into a fresh user CLI config dir.
    """
    from mind_nerve.installer import safe_write

    target = tmp_path / "deep" / "nested" / "path" / "config.json"
    safe_write(target, '{"x": 1}\n')
    assert target.exists()
    assert json.loads(target.read_text()) == {"x": 1}


@pytest.mark.integration
def test_safe_write_no_tempfile_left_behind(tmp_path: Path) -> None:
    """After a successful write only the target (and optionally the .bak)
    may remain in the parent dir — no stray tempfile.NamedTemporaryFile
    leftovers."""
    from mind_nerve.installer import safe_write

    target = tmp_path / "state.json"
    safe_write(target, '{"a": 1}\n')
    safe_write(target, '{"a": 2}\n')

    allowed = {target.name, target.name + ".bak"}
    found = {p.name for p in tmp_path.iterdir()}
    extra = found - allowed
    assert not extra, f"unexpected temp files left behind: {extra}"
