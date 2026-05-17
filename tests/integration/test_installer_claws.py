"""Integration tests for the claw-family CLI installers.

openclaw, nanoclaw, and nemoclaw are STARGA-internal agent runtimes that
share the same JSON mcpServers config shape. Each writes to its own config
path: ~/.openclaw/mcp.json, ~/.nanoclaw/mcp.json, ~/.nemoclaw/mcp.json.

Tests verify correct write, idempotency, and promotion out of STUB_CLIS.

All filesystem operations are redirected to tmp_path — no real config
directories are touched, no network access.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mind_nerve.installer import (
    INSTALLERS,
    STUB_CLIS,
    install_nanoclaw,
    install_nemoclaw,
    install_openclaw,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_home(tmp_path: Path) -> Path:
    h = tmp_path / "home"
    h.mkdir()
    return h


CLAW_NAMES = ["openclaw", "nanoclaw", "nemoclaw"]
CLAW_INSTALLERS = {
    "openclaw": install_openclaw,
    "nanoclaw": install_nanoclaw,
    "nemoclaw": install_nemoclaw,
}
CLAW_PATHS = {
    "openclaw": ".openclaw/mcp.json",
    "nanoclaw": ".nanoclaw/mcp.json",
    "nemoclaw": ".nemoclaw/mcp.json",
}


# ---------------------------------------------------------------------------
# Promotion checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", CLAW_NAMES)
def test_claw_not_in_stub_clis(name: str) -> None:
    """Each claw must be removed from STUB_CLIS once its installer exists."""
    assert name not in STUB_CLIS, f"{name} is still listed as a stub"


@pytest.mark.parametrize("name", CLAW_NAMES)
def test_claw_in_installers(name: str) -> None:
    """Each claw installer must appear in the INSTALLERS dispatch table."""
    assert name in INSTALLERS, f"{name} missing from INSTALLERS"
    assert INSTALLERS[name] is CLAW_INSTALLERS[name]


# ---------------------------------------------------------------------------
# Install behaviour — parametrized over all three claws
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", CLAW_NAMES)
def test_claw_creates_mcp_json(tmp_path: Path, name: str) -> None:
    """Each claw installer must create ~/.{claw}/mcp.json."""
    fake_home = _fake_home(tmp_path)
    installer = CLAW_INSTALLERS[name]
    with patch("mind_nerve.installer.HOME", fake_home):
        result = installer({})

    cfg_path = fake_home / CLAW_PATHS[name]
    assert cfg_path.exists(), f"~/{CLAW_PATHS[name]} was not created"
    assert result.get("installed") is True
    assert "path" in result


@pytest.mark.parametrize("name", CLAW_NAMES)
def test_claw_mcp_json_has_mcpservers(tmp_path: Path, name: str) -> None:
    """The written mcp.json must have 'mcpServers.mind-nerve'."""
    fake_home = _fake_home(tmp_path)
    installer = CLAW_INSTALLERS[name]
    with patch("mind_nerve.installer.HOME", fake_home):
        installer({})

    cfg_path = fake_home / CLAW_PATHS[name]
    data = json.loads(cfg_path.read_text())
    assert "mcpServers" in data
    assert "mind-nerve" in data["mcpServers"]


@pytest.mark.parametrize("name", CLAW_NAMES)
def test_claw_mcp_entry_command(tmp_path: Path, name: str) -> None:
    """The mind-nerve entry must have command = 'mind-nerve-mcp'."""
    fake_home = _fake_home(tmp_path)
    installer = CLAW_INSTALLERS[name]
    with patch("mind_nerve.installer.HOME", fake_home):
        installer({})

    cfg_path = fake_home / CLAW_PATHS[name]
    data = json.loads(cfg_path.read_text())
    assert data["mcpServers"]["mind-nerve"]["command"] == "mind-nerve-mcp"


@pytest.mark.parametrize("name", CLAW_NAMES)
def test_claw_creates_parent_dirs(tmp_path: Path, name: str) -> None:
    """Each installer must mkdir the config dir if absent."""
    fake_home = _fake_home(tmp_path)
    cfg_dir = fake_home / Path(CLAW_PATHS[name]).parent
    assert not cfg_dir.exists()

    installer = CLAW_INSTALLERS[name]
    with patch("mind_nerve.installer.HOME", fake_home):
        installer({})

    assert cfg_dir.is_dir()


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", CLAW_NAMES)
def test_claw_is_idempotent(tmp_path: Path, name: str) -> None:
    """Running each claw installer twice must not duplicate the entry."""
    fake_home = _fake_home(tmp_path)
    installer = CLAW_INSTALLERS[name]
    with patch("mind_nerve.installer.HOME", fake_home):
        installer({})
        result2 = installer({})

    assert result2.get("installed") is True
    cfg_path = fake_home / CLAW_PATHS[name]
    data = json.loads(cfg_path.read_text())
    assert isinstance(data["mcpServers"]["mind-nerve"], dict)


@pytest.mark.parametrize("name", CLAW_NAMES)
def test_claw_idempotent_preserves_other_servers(tmp_path: Path, name: str) -> None:
    """Existing mcpServers entries for other tools must be preserved."""
    fake_home = _fake_home(tmp_path)
    cfg_path = fake_home / CLAW_PATHS[name]
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(json.dumps({
        "mcpServers": {
            "some-other-tool": {"command": "other-mcp", "args": [], "env": {}}
        }
    }, indent=2) + "\n")

    installer = CLAW_INSTALLERS[name]
    with patch("mind_nerve.installer.HOME", fake_home):
        installer({})

    data = json.loads(cfg_path.read_text())
    assert "some-other-tool" in data["mcpServers"], (
        f"{name} installer must not remove existing server entries"
    )
    assert "mind-nerve" in data["mcpServers"]


# ---------------------------------------------------------------------------
# Isolation: each claw writes to its own path
# ---------------------------------------------------------------------------


def test_claw_paths_are_distinct(tmp_path: Path) -> None:
    """The three claw installers must write to three distinct config files."""
    fake_home = _fake_home(tmp_path)
    for name in CLAW_NAMES:
        installer = CLAW_INSTALLERS[name]
        with patch("mind_nerve.installer.HOME", fake_home):
            installer({})

    paths = [fake_home / CLAW_PATHS[n] for n in CLAW_NAMES]
    assert len(set(paths)) == 3, "claw installers wrote to overlapping paths"
    for p in paths:
        assert p.exists(), f"{p} was not written"
