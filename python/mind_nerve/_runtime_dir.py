from __future__ import annotations

import os
from pathlib import Path


def runtime_socket_dir() -> Path:
    """Return a 0700 directory suitable for sockets and lockfiles.

    Preference order:
      1. ``$XDG_RUNTIME_DIR`` — already 0700 by spec on systemd systems.
      2. ``~/.cache/mind-nerve/run`` — created at mode 0700 if absent.
      3. ``/tmp/mind-nerve-<uid>`` — last resort; caller gets a plain
         directory (symlink-attack hardening is the caller's concern
         when operating in a shared /tmp).
    """
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg and os.access(xdg, os.W_OK):
        return Path(xdg)

    cache = Path.home() / ".cache" / "mind-nerve" / "run"
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(cache, 0o700)
    except OSError:
        pass
    return cache
