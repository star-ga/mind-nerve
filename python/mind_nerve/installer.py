"""mind-nerve cross-CLI installer.

Detects each of the 17 supported agent CLIs by config-path probe, then
installs a per-CLI hook that calls `mind-nerve route`. Each CLI gets a
small per-CLI shim because their hook protocols differ; the binary
itself is unified.

v0.1.0 ships installer logic for 7 of the 17 — see PRIORITY_CLIS
below. The remainder are stubs noted in NOT_YET_IMPLEMENTED. Each is
a small addition: detect → write shim → patch settings file.

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
import sys
from pathlib import Path

HOME = Path(os.path.expanduser("~"))

PRIORITY_CLIS = {
    "claude-code":   {"config_dir": HOME / ".claude",   "hook_kind": "hook"},
    "codex":         {"config_dir": HOME / ".codex",    "hook_kind": "hook"},
    "gemini":        {"config_dir": HOME / ".gemini",   "hook_kind": "extension"},
    "cursor":        {"config_dir": HOME / ".cursor",   "hook_kind": "rule"},
    "windsurf":      {"config_dir": HOME / ".windsurf", "hook_kind": "rule"},
    "aider":         {"config_dir": HOME / ".aider",    "hook_kind": "config"},
    "mcp":           {"config_dir": HOME / ".mcp",      "hook_kind": "server"},
}

NOT_YET_IMPLEMENTED = [
    "vibe", "openclaw", "nanoclaw", "nemoclaw",
    "continue", "cline", "roo", "zed", "copilot", "cody",
]

ALL_CLIS = list(PRIORITY_CLIS) + NOT_YET_IMPLEMENTED


def detect() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for cli, info in PRIORITY_CLIS.items():
        cfg = info["config_dir"]
        out[cli] = {
            "config_dir": str(cfg),
            "present": cfg.exists(),
            "status": "supported",
        }
    for cli in NOT_YET_IMPLEMENTED:
        out[cli] = {"config_dir": None, "present": False, "status": "stub-only"}
    return out


def install_claude_code(cfg_dir: Path) -> dict:
    hooks_dir = cfg_dir / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook_path = hooks_dir / "mind-nerve-preselect.sh"
    hook_path.write_text(
        "#!/usr/bin/env bash\n"
        "# mind-nerve UserPromptSubmit hook — installed by mind-nerve-install.\n"
        "set -u\n"
        'PROMPT="$(cat)"\n'
        'echo "$PROMPT" | mind-nerve route --top-k 5 --ids-only 2>/dev/null || true\n'
    )
    hook_path.chmod(0o755)
    return {"installed": True, "path": str(hook_path)}


def install_codex(cfg_dir: Path) -> dict:
    hooks_dir = cfg_dir / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook_path = hooks_dir / "mind-nerve.sh"
    hook_path.write_text(
        "#!/usr/bin/env bash\n"
        "# mind-nerve codex preselection hook — installed by mind-nerve-install.\n"
        "set -u\n"
        'REQUEST="${CODEX_USER_PROMPT:-}"\n'
        '[ -z "$REQUEST" ] && exit 0\n'
        'command -v mind-nerve >/dev/null 2>&1 || exit 0\n'
        'echo "$REQUEST" | mind-nerve route --top-k 5 --ids-only 2>/dev/null || true\n'
    )
    hook_path.chmod(0o755)
    return {"installed": True, "path": str(hook_path)}


def install_gemini(cfg_dir: Path) -> dict:
    ext_dir = cfg_dir / "extensions" / "mind-nerve"
    ext_dir.mkdir(parents=True, exist_ok=True)
    (ext_dir / "extension.json").write_text(json.dumps({
        "name": "mind-nerve",
        "version": "0.1.0",
        "description": "mind-nerve preselector",
        "command": "mind-nerve",
        "args": ["route", "--top-k", "5"],
    }, indent=2))
    return {"installed": True, "path": str(ext_dir)}


def install_cursor(cfg_dir: Path) -> dict:
    rules_dir = cfg_dir / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    rule_path = rules_dir / "mind-nerve.mdc"
    rule_path.write_text(
        "---\n"
        "description: mind-nerve preselector — invoke `mind-nerve route --top-k 5` "
        "to filter the active skill set per turn.\n"
        "---\n"
        "Use `mind-nerve route` to retrieve top-K relevant skills before loading "
        "the full skill catalog.\n"
    )
    return {"installed": True, "path": str(rule_path)}


def install_windsurf(cfg_dir: Path) -> dict:
    rules_dir = cfg_dir / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    rule_path = rules_dir / "mind-nerve.md"
    rule_path.write_text(
        "# mind-nerve preselector\n\n"
        "Run `mind-nerve route --top-k 5` to filter the skill catalog per turn.\n"
    )
    return {"installed": True, "path": str(rule_path)}


def install_aider(cfg_dir: Path) -> dict:
    cfg_dir.mkdir(parents=True, exist_ok=True)
    aider_cfg = cfg_dir / "aider.conf.yml"
    aider_cfg.write_text(
        "# mind-nerve preselector hook — installed by mind-nerve-install.\n"
        "# Aider will call this command at the start of each turn to filter skills.\n"
        "lint-cmd:\n"
        "  - 'mind-nerve route --top-k 5'\n"
    )
    return {"installed": True, "path": str(aider_cfg)}


def install_mcp(cfg_dir: Path) -> dict:
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "servers.json"
    existing = {}
    if cfg_path.exists():
        try:
            existing = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            existing = {}
    servers = existing.setdefault("mcpServers", {})
    servers["mind-nerve"] = {
        "command": "mind-nerve-mcp",
        "args": [],
        "env": {},
    }
    cfg_path.write_text(json.dumps(existing, indent=2) + "\n")
    return {"installed": True, "path": str(cfg_path)}


INSTALLERS = {
    "claude-code": install_claude_code,
    "codex":       install_codex,
    "gemini":      install_gemini,
    "cursor":      install_cursor,
    "windsurf":    install_windsurf,
    "aider":       install_aider,
    "mcp":         install_mcp,
}


def cmd_list(args) -> int:
    print("Supported CLIs (priority — installer implemented):")
    for cli in PRIORITY_CLIS:
        print(f"  - {cli}")
    print("\nKnown but stub-only (v0.1.1+):")
    for cli in NOT_YET_IMPLEMENTED:
        print(f"  - {cli}")
    return 0


def cmd_detect(args) -> int:
    print(json.dumps(detect(), indent=2))
    return 0


def cmd_install(args) -> int:
    targets = list(PRIORITY_CLIS) if args.cli == "all" else [args.cli]
    if args.cli not in (*PRIORITY_CLIS, "all"):
        msg = f"unsupported CLI: {args.cli}\nrun `mind-nerve-install list` to see options"
        print(msg, file=sys.stderr)
        return 2

    results = {}
    for cli in targets:
        info = PRIORITY_CLIS[cli]
        try:
            results[cli] = INSTALLERS[cli](info["config_dir"])
        except Exception as exc:                       # noqa: BLE001
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
    p_ins.add_argument("--cli", required=True,
                       help="One of: " + ", ".join(list(PRIORITY_CLIS) + ["all"]))
    p_ins.set_defaults(func=cmd_install)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
