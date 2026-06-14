"""Integration tests for the vibe (Mistral) CLI MCP installer.

vibe reads its MCP server list from ~/.vibe/mcp.json under the key
"mcpServers", matching the standard JSON MCP shape used by Claude Desktop
and Cursor. Tests verify correct write, idempotency, and promotion out of
STUB_CLIS.

All filesystem operations are redirected to tmp_path — no real ~/.vibe
directory is touched, no network access.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from mind_nerve.installer import INSTALLERS, STUB_CLIS, install_vibe

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_home(tmp_path: Path) -> Path:
    h = tmp_path / "home"
    h.mkdir()
    return h


# ---------------------------------------------------------------------------
# Promotion checks
# ---------------------------------------------------------------------------


def test_vibe_not_in_stub_clis() -> None:
    """vibe must be removed from STUB_CLIS once the real installer exists."""
    assert "vibe" not in STUB_CLIS, "vibe is still listed as a stub"


def test_vibe_in_installers() -> None:
    """install_vibe must be registered in the INSTALLERS dispatch table."""
    assert "vibe" in INSTALLERS
    assert INSTALLERS["vibe"] is install_vibe


# ---------------------------------------------------------------------------
# Install behaviour
# ---------------------------------------------------------------------------


def test_install_vibe_creates_mcp_json(tmp_path: Path) -> None:
    """install_vibe() must write ~/.vibe/mcp.json."""
    fake_home = _fake_home(tmp_path)
    with patch("mind_nerve.installer.HOME", fake_home):
        result = install_vibe({})

    cfg_path = fake_home / ".vibe" / "mcp.json"
    assert cfg_path.exists(), "~/.vibe/mcp.json was not created"
    assert result.get("installed") is True
    assert "path" in result


def test_install_vibe_mcp_json_has_mcp_servers_key(tmp_path: Path) -> None:
    """The written mcp.json must have a top-level 'mcpServers' key."""
    fake_home = _fake_home(tmp_path)
    with patch("mind_nerve.installer.HOME", fake_home):
        install_vibe({})

    data = json.loads((fake_home / ".vibe" / "mcp.json").read_text())
    assert "mcpServers" in data, "mcp.json missing 'mcpServers'"
    assert "mind-nerve" in data["mcpServers"], "mcpServers missing 'mind-nerve'"


def test_install_vibe_mcp_entry_command(tmp_path: Path) -> None:
    """The mind-nerve MCP entry must set command = 'mind-nerve-mcp'."""
    fake_home = _fake_home(tmp_path)
    with patch("mind_nerve.installer.HOME", fake_home):
        install_vibe({})

    data = json.loads((fake_home / ".vibe" / "mcp.json").read_text())
    entry = data["mcpServers"]["mind-nerve"]
    assert entry.get("command") == "mind-nerve-mcp"


def test_install_vibe_creates_parent_dir(tmp_path: Path) -> None:
    """install_vibe() must mkdir ~/.vibe/ if it doesn't exist."""
    fake_home = _fake_home(tmp_path)
    assert not (fake_home / ".vibe").exists()

    with patch("mind_nerve.installer.HOME", fake_home):
        install_vibe({})

    assert (fake_home / ".vibe").is_dir()


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_install_vibe_is_idempotent(tmp_path: Path) -> None:
    """Running install_vibe() twice must not duplicate the mind-nerve entry."""
    fake_home = _fake_home(tmp_path)
    with patch("mind_nerve.installer.HOME", fake_home):
        install_vibe({})
        result2 = install_vibe({})

    assert result2.get("installed") is True
    data = json.loads((fake_home / ".vibe" / "mcp.json").read_text())
    # Exactly one mind-nerve entry — not a list, not doubled
    assert isinstance(data["mcpServers"]["mind-nerve"], dict)


def test_install_vibe_idempotent_preserves_other_servers(tmp_path: Path) -> None:
    """Existing mcp.json entries for other servers must be preserved."""
    fake_home = _fake_home(tmp_path)
    cfg_path = fake_home / ".vibe" / "mcp.json"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(
        json.dumps(
            {"mcpServers": {"other-server": {"command": "other-mcp", "args": [], "env": {}}}},
            indent=2,
        )
        + "\n"
    )

    with patch("mind_nerve.installer.HOME", fake_home):
        install_vibe({})

    data = json.loads(cfg_path.read_text())
    assert "other-server" in data["mcpServers"], (
        "install_vibe() must not remove existing server entries"
    )
    assert "mind-nerve" in data["mcpServers"]
