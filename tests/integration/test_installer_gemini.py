"""Integration tests for the Gemini CLI extension installer.

The Gemini CLI loads extensions from ~/.gemini/extensions/<name>/extension.json.
These tests verify that install_gemini() writes the manifest correctly,
is idempotent, and is promoted out of STUB_CLIS into INSTALLERS.

All filesystem operations are redirected to a tmp_path fixture — no network
access, no real ~/.gemini directory touched.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mind_nerve.installer import INSTALLERS, STUB_CLIS, install_gemini


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_home(tmp_path: Path) -> Path:
    """Return a fake home directory under tmp_path, mimicking ~."""
    h = tmp_path / "home"
    h.mkdir()
    return h


# ---------------------------------------------------------------------------
# Promotion checks
# ---------------------------------------------------------------------------


def test_gemini_not_in_stub_clis() -> None:
    """gemini must be removed from STUB_CLIS once the real installer exists."""
    assert "gemini" not in STUB_CLIS, (
        "gemini is still listed as a stub — move it to INSTALLERS"
    )


def test_gemini_in_installers() -> None:
    """install_gemini must be registered in the INSTALLERS dispatch table."""
    assert "gemini" in INSTALLERS, "gemini must be present in INSTALLERS"
    assert INSTALLERS["gemini"] is install_gemini


# ---------------------------------------------------------------------------
# Install behaviour
# ---------------------------------------------------------------------------


def test_install_gemini_creates_extension_manifest(tmp_path: Path) -> None:
    """install_gemini() must write ~/.gemini/extensions/mind-nerve/extension.json."""
    fake_home = _fake_home(tmp_path)
    with patch("mind_nerve.installer.HOME", fake_home):
        result = install_gemini({})

    manifest_path = fake_home / ".gemini" / "extensions" / "mind-nerve" / "extension.json"
    assert manifest_path.exists(), "extension.json not created"
    assert result.get("installed") is True
    assert "path" in result


def test_install_gemini_manifest_has_required_keys(tmp_path: Path) -> None:
    """The extension.json must contain name, version, and mcpServers keys."""
    fake_home = _fake_home(tmp_path)
    with patch("mind_nerve.installer.HOME", fake_home):
        install_gemini({})

    manifest_path = fake_home / ".gemini" / "extensions" / "mind-nerve" / "extension.json"
    data = json.loads(manifest_path.read_text())

    assert "name" in data, "extension.json missing 'name'"
    assert "version" in data, "extension.json missing 'version'"
    assert "mcpServers" in data, "extension.json missing 'mcpServers'"
    assert "mind-nerve" in data["mcpServers"], "mcpServers missing 'mind-nerve' entry"


def test_install_gemini_mcp_entry_has_command(tmp_path: Path) -> None:
    """The mind-nerve MCP entry must specify the server command."""
    fake_home = _fake_home(tmp_path)
    with patch("mind_nerve.installer.HOME", fake_home):
        install_gemini({})

    manifest_path = fake_home / ".gemini" / "extensions" / "mind-nerve" / "extension.json"
    data = json.loads(manifest_path.read_text())
    entry = data["mcpServers"]["mind-nerve"]

    assert "command" in entry, "MCP entry missing 'command'"
    assert entry["command"] == "mind-nerve-mcp"


def test_install_gemini_creates_parent_dirs(tmp_path: Path) -> None:
    """install_gemini() must create ~/.gemini/extensions/mind-nerve/ if absent."""
    fake_home = _fake_home(tmp_path)
    ext_dir = fake_home / ".gemini" / "extensions" / "mind-nerve"
    assert not ext_dir.exists(), "precondition: dir should not exist yet"

    with patch("mind_nerve.installer.HOME", fake_home):
        install_gemini({})

    assert ext_dir.is_dir(), "extension directory was not created"


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_install_gemini_is_idempotent(tmp_path: Path) -> None:
    """Running install_gemini() twice must not duplicate or corrupt the manifest."""
    fake_home = _fake_home(tmp_path)
    with patch("mind_nerve.installer.HOME", fake_home):
        install_gemini({})
        result2 = install_gemini({})  # second run

    assert result2.get("installed") is True

    manifest_path = fake_home / ".gemini" / "extensions" / "mind-nerve" / "extension.json"
    data = json.loads(manifest_path.read_text())
    # mcpServers must contain exactly one mind-nerve entry, not two
    assert isinstance(data["mcpServers"]["mind-nerve"], dict)


def test_install_gemini_idempotent_preserves_other_extensions(tmp_path: Path) -> None:
    """Existing extension entries in the parent dir must not be modified."""
    fake_home = _fake_home(tmp_path)
    # Pre-create a sibling extension that should not be touched.
    other_ext = fake_home / ".gemini" / "extensions" / "other-ext"
    other_ext.mkdir(parents=True)
    sentinel = other_ext / "extension.json"
    sentinel.write_text('{"name":"other-ext","version":"1.0","mcpServers":{}}')

    with patch("mind_nerve.installer.HOME", fake_home):
        install_gemini({})

    assert sentinel.read_text() == '{"name":"other-ext","version":"1.0","mcpServers":{}}', (
        "install_gemini() must not touch sibling extension manifests"
    )
