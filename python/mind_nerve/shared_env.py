"""Shared env file — one place for the pins every CLI config repeats.

``~/.mind-nerve/env`` holds ``KEY=VALUE`` lines (``#`` comments, blank
lines allowed) for the environment variables every host CLI would
otherwise repeat in its own config: ``MIND_NERVE_RUNTIME_DIR``,
``MIND_NERVE_BACKEND``, ``TRANSFORMERS_NO_TORCHVISION``, ...

Precedence rule (load-bearing): an explicitly exported variable or a
CLI-config pin ALWAYS wins — the shared file only fills variables that
are unset. Consumers call :func:`apply_shared_env` before reading any
``MIND_NERVE_*`` configuration.

The file is read, never written, by this module. It is parsed
line-by-line with no shell semantics: no quoting, no expansion, no
``export`` prefix. Malformed lines (no ``=``, invalid key) are ignored
silently — a broken env file must never break session start.
"""

from __future__ import annotations

import os
import re
from collections.abc import MutableMapping
from pathlib import Path

_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Lines longer than this are malformed by inspection (a real pin is tens
# of bytes); refusing them bounds parse cost on a hostile/garbage file.
_MAX_LINE = 4096


def shared_env_path() -> Path:
    """``$MIND_NERVE_ENV_FILE`` if set, else ``~/.mind-nerve/env``."""
    override = os.environ.get("MIND_NERVE_ENV_FILE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".mind-nerve" / "env"


def load_shared_env(path: str | Path | None = None) -> dict[str, str]:
    """Parse the shared env file. Missing file is fine (returns ``{}``)."""
    p = Path(path).expanduser() if path is not None else shared_env_path()
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line or len(line) > _MAX_LINE:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not _KEY_RE.fullmatch(key):
            continue
        out[key] = value.strip()
    return out


def apply_shared_env(
    env: MutableMapping[str, str] | None = None,
    path: str | Path | None = None,
) -> dict[str, str]:
    """Fill UNSET variables in *env* (default ``os.environ``) from the file.

    Returns the dict of variables that were applied (empty when the file
    is absent or every key was already set).
    """
    target = os.environ if env is None else env
    applied: dict[str, str] = {}
    for key, value in load_shared_env(path).items():
        if key not in target:
            target[key] = value
            applied[key] = value
    return applied
