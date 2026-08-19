from __future__ import annotations

import os
from pathlib import Path

# ``os.geteuid`` is POSIX-only and absent on Windows Python builds. This helper
# is reached at MODULE LOAD (via ``preselect_hook``'s top-level ``SOCKET_PATH``),
# so a bare ``os.geteuid()`` crashes the import on Windows with AttributeError.
# The same-UID ownership check it guards is a POSIX multi-user concern; Windows
# has no uid concept and per-user profile dirs (``~/.cache``) are already
# OS-scoped, so the check is simply skipped there. On POSIX this stays True and
# the code path below is byte-identical to before.
_HAS_EUID = hasattr(os, "geteuid")


def runtime_socket_dir() -> Path:
    """Return a 0700 directory suitable for sockets and lockfiles.

    Preference order:
      1. ``$XDG_RUNTIME_DIR`` — already 0700 by spec on systemd systems,
         and ONLY when owned by this euid: a writable-but-foreign runtime
         dir is another user's socket namespace (SECURITY.md's same-UID
         claim is enforced HERE; qwen Q13).
      2. ``~/.cache/mind-nerve/run`` — created at mode 0700 if absent and
         verified owned by this euid.
      3. ``/tmp/mind-nerve-<uid>`` — last resort; caller gets a plain
         directory (symlink-attack hardening is the caller's concern
         when operating in a shared /tmp).

    On platforms without ``os.geteuid`` (Windows) the same-UID ownership
    checks are skipped — there is no uid to compare against and the user's
    profile dir is already private to the account.
    """
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg and os.access(xdg, os.W_OK):
        try:
            if not _HAS_EUID or os.stat(xdg).st_uid == os.geteuid():
                return Path(xdg)
        except OSError:
            pass

    cache = Path.home() / ".cache" / "mind-nerve" / "run"
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    # POSIX same-UID ownership check (SECURITY.md). On Windows there is no uid
    # concept and the dir lives under the account's private profile (the OS
    # equivalent of the 0700 guarantee), so the check is skipped.
    if _HAS_EUID:
        try:
            os.chmod(cache, 0o700)
            foreign = os.stat(cache).st_uid != os.geteuid()
        except OSError:
            foreign = False
        if foreign:
            raise PermissionError(
                f"runtime dir {cache} is not owned by uid {os.geteuid()}; refusing to use it"
            )
    return cache
