"""mind-nerve cross-CLI installer.

The universal install path is **MCP registration**. Most agent CLIs
(Claude Code, Cursor, Codex, Claude Desktop) accept an MCP server in
their config; mind-nerve ships an MCP server (`mind-nerve-mcp`)
exposing a single `mind_nerve_route` tool. The installer's job is
patching each CLI's MCP config to add the mind-nerve entry.

For CLIs that don't speak MCP (aider, plain hook-only Claude Code
setups), we fall back to per-CLI shim formats. These are noted in
each install function.

Optional add-ons (only Claude Code right now):

* ``--with-preselect`` — wire the SessionStart + UserPromptSubmit hooks
  that project the top-K skills into ``~/.claude/skills``. For most users
  this renames the existing ``~/.claude/skills`` to
  ``~/.claude/skills.full`` once, then projects from ``.full`` back into
  the original path on every prompt. A cross-CLI shared catalog at
  ``~/.agents/skills`` is detected automatically if present.
* ``--with-mind-mem`` — also register the ``mind-mem-mcp`` server,
  if installed on PATH. mind-nerve does intent routing; mind-mem
  provides durable memory. Together they bracket the prompt path.

Usage:
    mind-nerve-install list
    mind-nerve-install detect
    mind-nerve-install install --cli claude-code
    mind-nerve-install install --cli claude-code --with-preselect
    mind-nerve-install install --cli claude-code --with-preselect --with-mind-mem
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
    "gemini": {
        "detect": HOME / ".gemini" / "extensions",
        "method": "gemini_extension_manifest",
    },
    "vibe": {"detect": HOME / ".vibe" / "mcp.json", "method": "json_mcp_servers"},
    "openclaw": {"detect": HOME / ".openclaw" / "mcp.json", "method": "json_mcp_servers"},
    "nanoclaw": {"detect": HOME / ".nanoclaw" / "mcp.json", "method": "json_mcp_servers"},
    "nemoclaw": {"detect": HOME / ".nemoclaw" / "mcp.json", "method": "json_mcp_servers"},
}
HOOK_BASED = {
    "claude-code-hook": {
        "detect": HOME / ".claude" / "settings.json",
        "method": "claude_user_prompt_hook",
    },
}
STUB_CLIS = [
    "windsurf",  # dir not standardly present
    "aider",  # no MCP support; needs bespoke integration
    "continue",
    "cline",
    "roo",
    "zed",
    "copilot",
    "cody",
]


def _mcp_entry(command: str = "mind-nerve-mcp", env: dict | None = None) -> dict:
    """Generic MCP server registration entry, used by every CLI."""
    return {
        "command": command,
        "args": [],
        "env": env or {},
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


def install_gemini(cfg: dict) -> dict:
    """Write a Gemini CLI extension manifest at ~/.gemini/extensions/mind-nerve/.

    The Gemini CLI discovers extensions by scanning ~/.gemini/extensions/. Each
    extension lives in its own subdirectory and must contain an extension.json
    manifest. The manifest declares the extension name, version, and optional
    MCP server registrations. Idempotent: re-running replaces the manifest in
    place without touching sibling extension directories.

    Reference: https://github.com/google-gemini/gemini-cli/blob/main/docs/extension.md
    """
    ext_dir = HOME / ".gemini" / "extensions" / "mind-nerve"
    ext_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = ext_dir / "extension.json"

    manifest = {
        "name": "mind-nerve",
        "version": "1",
        "description": "mind-nerve intent router — preselects top-K skills/routes before prompt dispatch",
        "mcpServers": {
            "mind-nerve": _mcp_entry(),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {"installed": True, "method": "gemini_extension_manifest", "path": str(manifest_path)}


def install_vibe(cfg: dict) -> dict:
    """Write the vibe (Mistral CLI) MCP config at ~/.vibe/mcp.json.

    vibe follows the same JSON ``mcpServers`` shape as Claude Desktop and
    Cursor. Idempotent: re-running updates the mind-nerve entry without
    removing other server registrations already present in the file.
    """
    cfg_path = HOME / ".vibe" / "mcp.json"
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


def _install_claw(claw_name: str) -> dict:
    """Shared implementation for the claw-family installers (openclaw / nanoclaw / nemoclaw).

    All three runtimes share the same JSON ``mcpServers`` config shape, each
    rooted at ~/.<claw>/mcp.json. Idempotent: re-running updates the
    mind-nerve entry without removing other server registrations.
    """
    cfg_path = HOME / f".{claw_name}" / "mcp.json"
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


def install_openclaw(cfg: dict) -> dict:
    """Wire mind-nerve MCP into ~/.openclaw/mcp.json."""
    return _install_claw("openclaw")


def install_nanoclaw(cfg: dict) -> dict:
    """Wire mind-nerve MCP into ~/.nanoclaw/mcp.json."""
    return _install_claw("nanoclaw")


def install_nemoclaw(cfg: dict) -> dict:
    """Wire mind-nerve MCP into ~/.nemoclaw/mcp.json."""
    return _install_claw("nemoclaw")


INSTALLERS = {
    "claude-code": install_claude_code,
    "claude-code-hook": install_claude_code_hook,
    "claude-desktop": install_claude_desktop,
    "cursor": install_cursor,
    "codex": install_codex,
    "gemini": install_gemini,
    "vibe": install_vibe,
    "openclaw": install_openclaw,
    "nanoclaw": install_nanoclaw,
    "nemoclaw": install_nemoclaw,
}


# ---------------------------------------------------------------------------
# --with-preselect: SessionStart + UserPromptSubmit hooks for Claude Code
# ---------------------------------------------------------------------------


def _looks_like_projection_dir(p: Path) -> bool:
    """Return True if ``p`` is a real directory whose children are mostly symlinks.

    That's the fingerprint of an in-place projection from a previous run:
    we must not rename it into ``.full``. Heuristic: at least one child and
    more than half are symlinks.
    """
    if not p.is_dir() or p.is_symlink():
        return False
    children = list(p.iterdir())
    if not children:
        return False
    link_count = sum(1 for c in children if c.is_symlink())
    return link_count > len(children) // 2


def _detect_skill_layout() -> dict:
    """Decide the source/projection layout for the preselect hook.

    Probed in this order — most users hit case 4 (regular) or 5 (empty):

    1. ``~/.claude/skills.full/`` exists → preselect was already installed
       on a previous run. Use ``.full`` as source, project into
       ``~/.claude/skills``.
    2. ``~/.claude/skills`` is a symlink → some other tool already moved
       the catalog elsewhere. Leave it in place; source = symlink target.
    3. ``~/.agents/skills/`` exists with at least one ``SKILL.md`` →
       cross-CLI shared catalog (Codex/Gemini/Vibe/Claude pointed at the
       same directory). Project from there.
    4. ``~/.claude/skills`` is a real directory of real skills (each
       subdir has its own ``SKILL.md``) → typical Claude Code user.
       Rename to ``~/.claude/skills.full`` so we can project in place.
    5. Empty / absent → create ``.full`` so the hook has somewhere to
       grow into; meanwhile it fails-open until the first skill arrives.
    """
    sk = HOME / ".claude" / "skills"
    full = HOME / ".claude" / "skills.full"
    agents = HOME / ".agents" / "skills"

    if full.is_dir():
        return {
            "layout": "preselect_already_installed",
            "source_dir": str(full),
            "projected_dir": str(sk),
            "action": "none",
        }
    if sk.is_symlink():
        return {
            "layout": "symlinked_catalog",
            "source_dir": str(sk.resolve()),
            "projected_dir": str(sk),
            "action": "none",
        }
    if agents.is_dir() and any(agents.glob("*/SKILL.md")):
        return {
            "layout": "shared_catalog_dir",
            "source_dir": str(agents),
            "projected_dir": str(sk),
            "action": "none",
        }
    if _looks_like_projection_dir(sk):
        # Real dir of symlinks but no detected source — leave it alone and
        # let the user set MIND_NERVE_SOURCE_DIR manually.
        return {
            "layout": "projection_without_known_source",
            "source_dir": str(full),
            "projected_dir": str(sk),
            "action": "none",
            "note": "set MIND_NERVE_SOURCE_DIR to your real skill catalog",
        }
    if sk.is_dir() and any(sk.glob("*/SKILL.md")):
        return {
            "layout": "regular_populated",
            "source_dir": str(full),
            "projected_dir": str(sk),
            "action": "rename_to_full",
        }
    return {
        "layout": "empty_or_absent",
        "source_dir": str(full),
        "projected_dir": str(sk),
        "action": "create_full",
    }


def _install_preselect_hook(layout: dict) -> dict:
    """Patch ~/.claude/settings.json with the SessionStart + UserPromptSubmit hooks.

    SessionStart spawns the daemon idempotently (no-op if already running).
    UserPromptSubmit projects the top-K skills into ``MIND_NERVE_PROJECTED_DIR``.
    """
    cfg_path = HOME / ".claude" / "settings.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if cfg_path.exists():
        try:
            existing = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            existing = {}

    source_dir = layout["source_dir"]
    projected_dir = layout["projected_dir"]
    env_export = f'MIND_NERVE_SOURCE_DIR="{source_dir}" MIND_NERVE_PROJECTED_DIR="{projected_dir}"'

    hooks = existing.setdefault("hooks", {})

    ss = hooks.setdefault("SessionStart", [])
    ss = [
        e
        for e in ss
        if not any(
            "mind-nerve-routed-ensure" in (h.get("command", "") or "")
            for h in (e.get("hooks") or [])
        )
    ]
    ss.append(
        {
            "matcher": "",
            "hooks": [{"type": "command", "command": "mind-nerve-routed-ensure"}],
        }
    )
    hooks["SessionStart"] = ss

    ups = hooks.setdefault("UserPromptSubmit", [])
    ups = [
        e
        for e in ups
        if not any(
            "mind-nerve-preselect" in (h.get("command", "") or "")
            or "mind-nerve route" in (h.get("command", "") or "")
            for h in (e.get("hooks") or [])
        )
    ]
    ups.append(
        {
            "matcher": "",
            "hooks": [
                {
                    "type": "command",
                    "command": f"{env_export} mind-nerve-preselect",
                }
            ],
        }
    )
    hooks["UserPromptSubmit"] = ups

    cfg_path.write_text(json.dumps(existing, indent=2) + "\n")
    return {"installed": True, "method": "session_and_prompt_hooks", "path": str(cfg_path)}


def install_claude_code_preselect() -> dict:
    """Wire SessionStart + UserPromptSubmit hooks for skill preselection.

    Migrates a regular user's existing ``~/.claude/skills`` directory to
    ``~/.claude/skills.full`` so the hook can rewrite the original path
    as a top-K projection on every turn. Idempotent.
    """
    layout = _detect_skill_layout()
    actions: list[str] = []

    if layout["action"] == "rename_to_full":
        src = HOME / ".claude" / "skills"
        dst = HOME / ".claude" / "skills.full"
        try:
            os.rename(src, dst)
            actions.append(f"renamed {src} -> {dst}")
        except OSError as exc:
            return {"installed": False, "error": f"rename failed: {exc}", "layout": layout}
    elif layout["action"] == "create_full":
        full = HOME / ".claude" / "skills.full"
        try:
            full.mkdir(parents=True, exist_ok=True)
            actions.append(f"created empty {full}")
            # Remove an empty real `skills` dir so the projection can take its place.
            sk = HOME / ".claude" / "skills"
            if sk.is_dir() and not sk.is_symlink() and not any(sk.iterdir()):
                sk.rmdir()
                actions.append(f"removed empty {sk}")
        except OSError as exc:
            return {
                "installed": False,
                "error": f"create skills.full failed: {exc}",
                "layout": layout,
            }

    hook_result = _install_preselect_hook(layout)
    hook_result["layout"] = layout["layout"]
    hook_result["source_dir"] = layout["source_dir"]
    hook_result["projected_dir"] = layout["projected_dir"]
    hook_result["actions"] = actions
    if not shutil.which("mind-nerve-preselect") or not shutil.which("mind-nerve-routed-ensure"):
        hook_result["warning"] = (
            "mind-nerve-preselect / mind-nerve-routed-ensure not found on PATH; "
            "make sure the mind-nerve venv's bin/ is on PATH for the shell that "
            "launches Claude Code."
        )
    return hook_result


# ---------------------------------------------------------------------------
# --with-mind-mem: register mind-mem-mcp alongside mind-nerve
# ---------------------------------------------------------------------------


def _register_mind_mem_in(cfg_path: Path, fmt: str) -> dict:
    """Idempotent mind-mem MCP entry write into a JSON or TOML CLI config."""
    if fmt == "json":
        existing: dict = {}
        if cfg_path.exists():
            try:
                existing = json.loads(cfg_path.read_text())
            except json.JSONDecodeError:
                existing = {}
        servers = existing.setdefault("mcpServers", {})
        servers["mind-mem"] = _mcp_entry("mind-mem-mcp")
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(existing, indent=2) + "\n")
        return {"installed": True, "path": str(cfg_path), "fmt": fmt}

    if fmt == "toml":
        import re

        block = '\n[mcp_servers.mind-mem]\ncommand = "mind-mem-mcp"\nargs = []\nenv = {}\n'
        existing_text = cfg_path.read_text() if cfg_path.exists() else ""
        pattern = re.compile(
            r"\n?\[mcp_servers\.mind-mem\][^\[]*?(?=\n\[|\Z)",
            re.DOTALL,
        )
        updated = (
            pattern.sub(block, existing_text)
            if pattern.search(existing_text)
            else (existing_text.rstrip() + block)
        )
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(updated)
        return {"installed": True, "path": str(cfg_path), "fmt": fmt}

    return {"installed": False, "error": f"unknown fmt: {fmt}"}


def install_mind_mem_companion(targets: list[str]) -> dict:
    """Register the ``mind-mem-mcp`` MCP server next to ``mind-nerve``.

    Only writes to CLI configs the user is already installing mind-nerve
    into. If ``mind-mem-mcp`` is not on PATH we still write the config
    entry (idempotent) but flag a warning.
    """
    if not shutil.which("mind-mem-mcp"):
        warning = (
            "mind-mem-mcp not found on PATH; install mind-mem first (`pip install mind-mem[mcp]`)"
        )
    else:
        warning = None

    results: dict[str, dict] = {}
    for cli in targets:
        try:
            if cli == "claude-code":
                if shutil.which("claude"):
                    r = subprocess.run(
                        ["claude", "mcp", "add", "mind-mem", "mind-mem-mcp"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if r.returncode == 0 or "already exists" in (r.stdout + r.stderr).lower():
                        results[cli] = {"installed": True, "method": "claude_mcp_add"}
                    else:
                        results[cli] = _register_mind_mem_in(HOME / ".claude.json", "json")
                else:
                    results[cli] = _register_mind_mem_in(HOME / ".claude.json", "json")
            elif cli == "claude-desktop":
                candidates = [
                    HOME / ".config" / "Claude" / "claude_desktop_config.json",
                    HOME
                    / "Library"
                    / "Application Support"
                    / "Claude"
                    / "claude_desktop_config.json",
                ]
                cfg_path = next((c for c in candidates if c.parent.exists()), candidates[0])
                results[cli] = _register_mind_mem_in(cfg_path, "json")
            elif cli == "cursor":
                results[cli] = _register_mind_mem_in(HOME / ".cursor" / "mcp.json", "json")
            elif cli == "codex":
                results[cli] = _register_mind_mem_in(HOME / ".codex" / "config.toml", "toml")
            elif cli == "vibe":
                results[cli] = _register_mind_mem_in(HOME / ".vibe" / "mcp.json", "json")
            elif cli in {"openclaw", "nanoclaw", "nemoclaw"}:
                results[cli] = _register_mind_mem_in(HOME / f".{cli}" / "mcp.json", "json")
            elif cli == "gemini":
                # Gemini uses extension.json, not a flat mcp.json; inject into
                # the existing extension manifest produced by install_gemini().
                ext_manifest = HOME / ".gemini" / "extensions" / "mind-nerve" / "extension.json"
                results[cli] = (
                    _register_mind_mem_in(ext_manifest, "json")
                    if ext_manifest.exists()
                    else {
                        "installed": False,
                        "error": "install mind-nerve for gemini first (mind-nerve install --cli gemini)",
                    }
                )
            else:
                results[cli] = {"installed": False, "error": f"not supported for {cli}"}
        except Exception as exc:  # noqa: BLE001
            results[cli] = {"installed": False, "error": str(exc)}

    out: dict = {"results": results}
    if warning:
        out["warning"] = warning
    return out


# ---------------------------------------------------------------------------
# Detect + list + main
# ---------------------------------------------------------------------------


def detect() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for cli, info in {**MCP_CAPABLE, **HOOK_BASED}.items():
        detect_path: Path = info["detect"]  # type: ignore[assignment]
        present = detect_path.exists()
        out[cli] = {
            "config_probe": str(info["detect"]),
            "present": present,
            "method": info["method"],
            "status": "supported",
        }
    for cli in STUB_CLIS:
        out[cli] = {"present": False, "status": "stub_v0.1.1"}
    out["__addons__"] = {
        "mind-mem-mcp": {"present": bool(shutil.which("mind-mem-mcp"))},
        "skill_layout": _detect_skill_layout(),
    }
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
    print("\nOptional add-ons (claude-code only):")
    print("  --with-preselect  SessionStart + UserPromptSubmit hooks for top-K skill projection")
    print("  --with-mind-mem   also register mind-mem-mcp (requires `pip install mind-mem[mcp]`)")
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

    if args.with_preselect:
        if "claude-code" in targets:
            results["__preselect__"] = install_claude_code_preselect()
        else:
            results["__preselect__"] = {
                "installed": False,
                "reason": "--with-preselect only applies to claude-code right now",
            }

    if args.with_mind_mem:
        results["__mind_mem__"] = install_mind_mem_companion(
            [t for t in targets if t in {"claude-code", "claude-desktop", "cursor", "codex"}]
        )

    if getattr(args, "with_systemd", False):
        results["__systemd__"] = install_systemd_user_unit()

    print(json.dumps(results, indent=2))
    return 0


# ---------------------------------------------------------------------------
# --with-systemd: install mind-nerve-routed.service as a long-lived user unit
# ---------------------------------------------------------------------------


def install_systemd_user_unit() -> dict:
    """Install the long-lived `mind-nerve-routed.service` user unit so the
    route daemon runs in its own cgroup and survives parent-CLI restarts.

    Without this unit, every CLI invocation that trips `ensure.py`
    inherits whatever cgroup spawned it. Concurrent invocations during
    the daemon's 5 s weight-load window are guarded by the flock from
    0.3.0-beta.2 onward, but a long-running parent process restart
    still orphans the daemon. This unit lifts the daemon out entirely:
    own cgroup, real Restart= semantics, memory cap.

    Idempotent: re-running over an existing unit is a no-op apart from
    `daemon-reload` + `restart`.
    """
    if sys.platform != "linux":
        return {
            "installed": False,
            "reason": f"systemd user units only supported on Linux (got {sys.platform})",
        }
    if shutil.which("systemctl") is None:
        return {"installed": False, "reason": "systemctl not found on PATH"}

    unit_src = Path(__file__).parent / "templates" / "mind-nerve-routed.service"
    if not unit_src.exists():
        return {
            "installed": False,
            "reason": f"unit template missing: {unit_src} — package install incomplete",
        }

    unit_dst_dir = Path.home() / ".config" / "systemd" / "user"
    unit_dst = unit_dst_dir / "mind-nerve-routed.service"
    log_dir = Path.home() / ".mind-nerve"

    try:
        unit_dst_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        unit_dst.write_text(unit_src.read_text())
    except OSError as exc:
        return {"installed": False, "reason": f"could not write unit: {exc}"}

    try:
        subprocess.run(
            ["systemctl", "--user", "daemon-reload"],
            check=True,
            capture_output=True,
            timeout=15,
        )
        subprocess.run(
            ["systemctl", "--user", "enable", "--now", "mind-nerve-routed.service"],
            check=True,
            capture_output=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as exc:
        return {
            "installed": False,
            "unit_path": str(unit_dst),
            "reason": f"systemctl returned {exc.returncode}",
            "stderr": exc.stderr.decode("utf-8", "replace")[:400],
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {
            "installed": False,
            "unit_path": str(unit_dst),
            "reason": f"systemctl invocation failed: {exc}",
        }

    return {
        "installed": True,
        "unit_path": str(unit_dst),
        "log_path": str(log_dir / "daemon.log"),
        "note": "service is enabled and started; survives parent-CLI restarts",
    }


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
    p_ins.add_argument(
        "--with-preselect",
        action="store_true",
        help="Also wire SessionStart + UserPromptSubmit hooks for top-K skill projection (claude-code only)",
    )
    p_ins.add_argument(
        "--with-mind-mem",
        action="store_true",
        help="Also register the mind-mem-mcp server next to mind-nerve (requires mind-mem installed)",
    )
    p_ins.add_argument(
        "--with-systemd",
        action="store_true",
        help="Also install mind-nerve-routed.service as a long-lived systemd user unit "
        "so the route daemon owns its own cgroup and survives parent-CLI restarts (Linux only)",
    )
    p_ins.set_defaults(func=cmd_install)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
