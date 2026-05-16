"""mind-nerve-preselect — Claude Code UserPromptSubmit hook.

Reads the hook's JSON input from stdin, asks the mind-nerve-routed
daemon for the top-K most relevant skills for this prompt, then
atomically rewrites the Claude Code skills directory as a projection
of symlinks pointing into the real skill catalog.

The effect: Claude only ever sees the top-K relevant skill descriptions
in its system prompt, not the full library. On a 440-skill catalog
that's a ~95% reduction in skill-listing tokens per turn.

Fail-open at every step. The hook prints `{}` and exits 0 even on
errors — never blocks Claude Code, never returns a non-zero status.

Layout auto-detection
---------------------

Without any env vars set, the hook detects two install patterns:

  - **Regular**: real skill catalog at `~/.claude/skills.full/`
    (installer renamed the user's original `~/.claude/skills/`
    to `.full/` and started projecting into `~/.claude/skills/`).
  - **Shared catalog**: catalog at `~/.agents/skills/`, used by
    setups that point multiple agent CLIs at one skill directory.

If neither exists the hook short-circuits to pass-through and
leaves the current state alone.

Env knobs:

    MIND_NERVE_SOCKET         UNIX socket for the daemon
                              (default: $XDG_RUNTIME_DIR/mind-nerve.sock
                                        or /tmp/mind-nerve-<uid>.sock)
    MIND_NERVE_SOURCE_DIR     real skill catalog
                              (default: auto-detected, see above)
    MIND_NERVE_PROJECTED_DIR  projection dir              (default ~/.claude/skills)
    MIND_NERVE_TOP_K          top-K after dedup           (default 20)
    MIND_NERVE_OVERFETCH      ask daemon for this many    (default 300)
    MIND_NERVE_SOCKET_TIMEOUT seconds                     (default 2.0)
    MIND_NERVE_LOG            jsonl log                   (default ~/.mind-nerve/hook.log)
    MIND_NERVE_CORE_ALWAYS_ON colon-separated names that
                              are always added to the projection
                              (default: a small built-in list)
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import socket
import sys
import time
from pathlib import Path


def _default_socket() -> str:
    uid = os.getuid()
    xdg = f"/run/user/{uid}/mind-nerve.sock"
    if os.path.isdir(os.path.dirname(xdg)):
        return xdg
    return f"/tmp/mind-nerve-{uid}.sock"


def _default_source_dir() -> Path:
    """Auto-detect the real skill catalog.

    Priority:
      1. ``~/.claude/skills.full/`` — created by ``mind-nerve-install``
         after renaming the user's original ``~/.claude/skills/``.
      2. ``~/.agents/skills/`` — optional cross-CLI shared catalog.
      3. Fallback to ``~/.agents/skills/`` even if absent; the hook will
         simply find no skills and exit pass-through.
    """
    full = Path.home() / ".claude" / "skills.full"
    if full.is_dir():
        return full
    return Path.home() / ".agents" / "skills"


SOCKET_PATH = os.environ.get("MIND_NERVE_SOCKET", _default_socket())
SOURCE_DIR = Path(os.environ.get("MIND_NERVE_SOURCE_DIR", str(_default_source_dir())))
PROJECTED_DIR = Path(
    os.environ.get("MIND_NERVE_PROJECTED_DIR", str(Path.home() / ".claude" / "skills"))
)
TOP_K = int(os.environ.get("MIND_NERVE_TOP_K", "20"))
OVERFETCH = int(os.environ.get("MIND_NERVE_OVERFETCH", "300"))
SOCKET_TIMEOUT = float(os.environ.get("MIND_NERVE_SOCKET_TIMEOUT", "2.0"))
LOG_PATH = Path(os.environ.get("MIND_NERVE_LOG", str(Path.home() / ".mind-nerve" / "hook.log")))

_DEFAULT_CORE_ALWAYS_ON = (
    "diagnose",
    "code-review",
    "git-workflow",
    "git-advanced-workflows",
    "debugging-strategies",
    "skill-creator",
)

CORE_ALWAYS_ON: tuple[str, ...] = tuple(
    s.strip()
    for s in os.environ.get("MIND_NERVE_CORE_ALWAYS_ON", ":".join(_DEFAULT_CORE_ALWAYS_ON)).split(
        ":"
    )
    if s.strip()
)

_slug_re = re.compile(r"[^a-z0-9]+")


def _log(record: dict) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record["ts"] = time.time()
        with LOG_PATH.open("a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass


def _read_prompt() -> str:
    raw = sys.stdin.read()
    if not raw:
        return ""
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return ""
    return str(data.get("prompt") or data.get("user_prompt") or "").strip()


def _list_source_skills() -> dict[str, Path]:
    skills: dict[str, Path] = {}
    if not SOURCE_DIR.is_dir():
        return skills
    for child in SOURCE_DIR.iterdir():
        if child.is_dir() and (child / "SKILL.md").is_file():
            skills[child.name] = child
    return skills


def _query_daemon(prompt: str, k: int) -> list[str] | None:
    if not prompt or not os.path.exists(SOCKET_PATH):
        return None
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(SOCKET_TIMEOUT)
        s.connect(SOCKET_PATH)
        s.sendall((json.dumps({"prompt": prompt, "top_k": k}) + "\n").encode())
        s.shutdown(socket.SHUT_WR)
        buf = b""
        while True:
            chunk = s.recv(8192)
            if not chunk:
                break
            buf += chunk
            if len(buf) > 1_000_000:
                break
        s.close()
        reply = json.loads(buf.decode("utf-8", errors="replace"))
    except (OSError, ValueError) as e:
        _log({"event": "socket_err", "err": str(e)})
        return None
    if "error" in reply:
        _log({"event": "daemon_err", "err": reply["error"]})
        return None
    return [str(r.get("name", "")).strip() for r in reply.get("routes", []) if r.get("name")]


def _resolve_name(name: str, source: dict[str, Path]) -> str | None:
    if name in source:
        return name
    slug = _slug_re.sub("-", name.lower()).strip("-")
    if slug in source:
        return slug
    for src_name in source:
        if _slug_re.sub("-", src_name.lower()).strip("-") == slug:
            return src_name
    return None


def _write_projection(routed_names: list[str], source: dict[str, Path]) -> int:
    seen: set[str] = set()
    selected: list[tuple[str, Path]] = []
    for name in list(routed_names) + list(CORE_ALWAYS_ON):
        match = _resolve_name(name, source)
        if match is None or match in seen:
            continue
        seen.add(match)
        selected.append((match, source[match]))

    parent = PROJECTED_DIR.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp = parent / f".skills.tmp.{secrets.token_hex(4)}"
    try:
        tmp.mkdir()
        for name, src in selected:
            try:
                (tmp / name).symlink_to(src)
            except OSError:
                pass

        old = parent / f".skills.old.{secrets.token_hex(4)}"
        if PROJECTED_DIR.is_symlink() or PROJECTED_DIR.exists():
            os.rename(PROJECTED_DIR, old)
        os.rename(tmp, PROJECTED_DIR)
        if old.is_symlink():
            old.unlink(missing_ok=True)
        elif old.exists():
            shutil.rmtree(old, ignore_errors=True)
        return len(selected)
    except OSError as e:
        _log({"event": "projection_fail", "err": str(e)})
        shutil.rmtree(tmp, ignore_errors=True)
        return -1


def main() -> int:
    t0 = time.time()
    try:
        prompt = _read_prompt()
        source = _list_source_skills()
        if not source:
            _log({"event": "no_source", "dir": str(SOURCE_DIR)})
            print("{}")
            return 0

        overfetched = _query_daemon(prompt, OVERFETCH)
        if overfetched is None:
            _log({"event": "passthrough", "prompt_head": prompt[:80]})
            print("{}")
            return 0

        seen_local: set[str] = set()
        routed: list[str] = []
        for name in overfetched:
            match = _resolve_name(name, source)
            if match and match not in seen_local:
                seen_local.add(match)
                routed.append(match)
                if len(routed) >= TOP_K:
                    break

        n = _write_projection(routed, source)
        _log(
            {
                "event": "projected" if n > 0 else "projection_failed",
                "selected": n,
                "routed_top_5": routed[:5],
                "prompt_head": prompt[:80],
                "ms": int((time.time() - t0) * 1000),
            }
        )
        print("{}")
        return 0
    except Exception as e:  # noqa: BLE001  fail-open
        _log({"event": "fatal", "err": str(e)})
        print("{}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
