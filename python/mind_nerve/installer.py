"""mind-nerve cross-CLI installer.

The universal install path is **MCP registration**. Most agent CLIs
(Claude Code, Cursor, Codex, Claude Desktop) accept an MCP server in
their config; mind-nerve ships an MCP server (`mind-nerve-mcp`)
exposing a single `mind_nerve_route` tool. The installer's job is
patching each CLI's MCP config to add the mind-nerve entry.

For CLIs that don't speak MCP (aider, plain hook-only Claude Code
setups), we fall back to per-CLI shim formats. These are noted in
each install function.

Usage:
    mind-nerve-install list
    mind-nerve-install detect
    mind-nerve-install install --cli claude-code
    mind-nerve-install install --cli all
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HOME = Path(os.path.expanduser("~"))


# CLIs grouped by install mechanism.
MCP_CAPABLE = {
    "claude-code": {"detect": HOME / ".claude" / "settings.json", "method": "claude_cli"},
    "claude-desktop": {
        "detect": HOME / ".config" / "Claude" / "claude_desktop_config.json",
        "method": "json_mcp_servers",
    },
    "cursor": {"detect": HOME / ".cursor" / "mcp.json", "method": "json_mcp_servers"},
    "codex": {"detect": HOME / ".codex" / "config.toml", "method": "toml_mcp_servers"},
}
HOOK_BASED = {
    "claude-code-hook": {
        "detect": HOME / ".claude" / "settings.json",
        "method": "claude_user_prompt_hook",
    },
}
STUB_CLIS = [
    "gemini",  # extension format not verified yet
    "windsurf",  # dir not standardly present
    "aider",  # no MCP support; needs bespoke integration
    "vibe",
    "openclaw",
    "nanoclaw",
    "nemoclaw",
    "continue",
    "cline",
    "roo",
    "zed",
    "copilot",
    "cody",
]


def _mcp_entry() -> dict:
    """The mind-nerve MCP server registration entry, used by every CLI."""
    return {
        "command": "mind-nerve-mcp",
        "args": [],
        "env": {},
    }


# ---------------------------------------------------------------------------
# Per-CLI installers
# ---------------------------------------------------------------------------


def install_claude_code(cfg: dict) -> dict:
    """Use the `claude mcp add` CLI command (preferred, validated by claude itself)."""
    if not shutil.which("claude"):
        return _install_claude_code_manual()
    try:
        r = subprocess.run(
            ["claude", "mcp", "add", "mind-nerve", "mind-nerve-mcp"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0:
            return {"installed": True, "method": "claude_mcp_add", "stdout": r.stdout.strip()}
        # If it's already added, treat as ok.
        if "already exists" in (r.stdout + r.stderr).lower():
            return {"installed": True, "method": "claude_mcp_add", "note": "already exists"}
        return _install_claude_code_manual()
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"installed": False, "error": str(exc)}


def _install_claude_code_manual() -> dict:
    """Fall back: patch ~/.claude.json directly (Claude Code's project config).

    Claude Code stores MCP servers per-project in ~/.claude.json. Patching
    the top-level `mcpServers` makes them visible across all projects.
    """
    cfg_path = HOME / ".claude.json"
    existing: dict = {}
    if cfg_path.exists():
        try:
            existing = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            existing = {}
    servers = existing.setdefault("mcpServers", {})
    servers["mind-nerve"] = _mcp_entry()
    cfg_path.write_text(json.dumps(existing, indent=2) + "\n")
    return {"installed": True, "method": "manual_claude_json", "path": str(cfg_path)}


def install_claude_desktop(cfg: dict) -> dict:
    """Patch ~/.config/Claude/claude_desktop_config.json (Linux) or macOS path."""
    candidates = [
        HOME / ".config" / "Claude" / "claude_desktop_config.json",
        HOME / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",
    ]
    cfg_path = next((c for c in candidates if c.parent.exists()), candidates[0])
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if cfg_path.exists():
        try:
            existing = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            existing = {}
    servers = existing.setdefault("mcpServers", {})
    servers["mind-nerve"] = _mcp_entry()
    cfg_path.write_text(json.dumps(existing, indent=2) + "\n")
    return {"installed": True, "method": "json_mcp_servers", "path": str(cfg_path)}


def install_cursor(cfg: dict) -> dict:
    """Patch ~/.cursor/mcp.json."""
    cfg_path = HOME / ".cursor" / "mcp.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if cfg_path.exists():
        try:
            existing = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            existing = {}
    servers = existing.setdefault("mcpServers", {})
    servers["mind-nerve"] = _mcp_entry()
    cfg_path.write_text(json.dumps(existing, indent=2) + "\n")
    return {"installed": True, "method": "json_mcp_servers", "path": str(cfg_path)}


def install_codex(cfg: dict) -> dict:
    """Patch ~/.codex/config.toml — add `[mcp_servers.mind-nerve]` block.

    Codex's config is TOML; we read-edit-write minimally to preserve
    unrelated blocks. Idempotent: re-running replaces the section.
    """
    cfg_path = HOME / ".codex" / "config.toml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    block = '\n[mcp_servers.mind-nerve]\ncommand = "mind-nerve-mcp"\nargs = []\nenv = {}\n'
    existing = ""
    if cfg_path.exists():
        existing = cfg_path.read_text()

    # If a [mcp_servers.mind-nerve] block already exists, replace it.
    import re

    pattern = re.compile(
        r"\n?\[mcp_servers\.mind-nerve\][^\[]*?(?=\n\[|\Z)",
        re.DOTALL,
    )
    if pattern.search(existing):
        updated = pattern.sub(block, existing)
    else:
        updated = existing.rstrip() + block

    cfg_path.write_text(updated)
    return {"installed": True, "method": "toml_mcp_servers", "path": str(cfg_path)}


def install_claude_code_hook(cfg: dict) -> dict:
    """Alternative claude-code path: UserPromptSubmit hook instead of MCP.

    Patches ~/.claude/settings.json `hooks.UserPromptSubmit` with a
    `mind-nerve route` command. Use when you want the preselector
    *inline* in the prompt path rather than as an explicit MCP tool.
    """
    cfg_path = HOME / ".claude" / "settings.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if cfg_path.exists():
        try:
            existing = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            existing = {}
    hooks = existing.setdefault("hooks", {})
    ups = hooks.setdefault("UserPromptSubmit", [])
    # Idempotency: drop any existing mind-nerve entry first
    ups = [
        e
        for e in ups
        if not any("mind-nerve" in (h.get("command", "") or "") for h in (e.get("hooks") or []))
    ]
    ups.append(
        {
            "matcher": "",
            "hooks": [{"type": "command", "command": "mind-nerve route --top-k 5 --ids-only"}],
        }
    )
    hooks["UserPromptSubmit"] = ups
    cfg_path.write_text(json.dumps(existing, indent=2) + "\n")
    return {"installed": True, "method": "claude_user_prompt_hook", "path": str(cfg_path)}


INSTALLERS = {
    "claude-code": install_claude_code,
    "claude-code-hook": install_claude_code_hook,
    "claude-desktop": install_claude_desktop,
    "cursor": install_cursor,
    "codex": install_codex,
}


# ---------------------------------------------------------------------------
# Detect + list + main
# ---------------------------------------------------------------------------


def detect() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for cli, info in {**MCP_CAPABLE, **HOOK_BASED}.items():
        present = info["detect"].exists()
        out[cli] = {
            "config_probe": str(info["detect"]),
            "present": present,
            "method": info["method"],
            "status": "supported",
        }
    for cli in STUB_CLIS:
        out[cli] = {"present": False, "status": "stub_v0.1.1"}
    return out


def cmd_list(args) -> int:
    print("MCP-capable CLIs (preferred path — single binary):")
    for cli in MCP_CAPABLE:
        print(f"  - {cli}")
    print("\nHook-based fallback:")
    for cli in HOOK_BASED:
        print(f"  - {cli}")
    print("\nStub CLIs (v0.1.1+, integration TBD):")
    for cli in STUB_CLIS:
        print(f"  - {cli}")
    return 0


def cmd_detect(args) -> int:
    print(json.dumps(detect(), indent=2))
    return 0


def cmd_install(args) -> int:
    known = set(INSTALLERS) | {"all"}
    if args.cli not in known:
        print(
            f"unsupported CLI: {args.cli}\nrun `mind-nerve-install list` to see options",
            file=sys.stderr,
        )
        return 2

    targets = list(INSTALLERS) if args.cli == "all" else [args.cli]
    # Don't install both claude-code and claude-code-hook in 'all' mode —
    # they're alternative integrations of the same CLI; prefer MCP.
    if args.cli == "all":
        targets = [t for t in targets if t != "claude-code-hook"]

    results: dict[str, dict] = {}
    for cli in targets:
        try:
            results[cli] = INSTALLERS[cli](MCP_CAPABLE.get(cli) or HOOK_BASED.get(cli) or {})
        except Exception as exc:  # noqa: BLE001
            results[cli] = {"installed": False, "error": str(exc)}
    print(json.dumps(results, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mind-nerve-install")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List supported CLIs")
    p_list.set_defaults(func=cmd_list)

    p_det = sub.add_parser("detect", help="Detect which supported CLIs are present on this host")
    p_det.set_defaults(func=cmd_detect)

    p_ins = sub.add_parser("install", help="Install the mind-nerve hook for a CLI")
    p_ins.add_argument(
        "--cli", required=True, help="One of: " + ", ".join(list(INSTALLERS) + ["all"])
    )
    p_ins.set_defaults(func=cmd_install)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
