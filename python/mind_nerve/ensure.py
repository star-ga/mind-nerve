"""mind-nerve-routed-ensure — idempotent starter for the route daemon.

Designed to be called from a Claude Code SessionStart hook (or any
similar "session warmup" trigger). Probes the daemon's UNIX socket;
if it's already responsive, returns immediately. Otherwise spawns
`mind-nerve-routed` detached. Always exits 0 — the daemon is a
performance optimisation, not a correctness requirement, so this
script must never block session start.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def default_socket_path() -> Path:
    uid = os.getuid()
    xdg = Path(f"/run/user/{uid}")
    if xdg.is_dir() and os.access(xdg, os.W_OK):
        return xdg / "mind-nerve.sock"
    return Path(f"/tmp/mind-nerve-{uid}.sock")


def _socket_responsive(sock_path: Path, timeout: float = 1.0) -> bool:
    if not sock_path.exists():
        return False
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(str(sock_path))
        s.sendall(b'{"prompt":"ping","top_k":1}\n')
        s.recv(4096)
        s.close()
        return True
    except OSError:
        return False


def _resolve_daemon_binary() -> str | None:
    """Find mind-nerve-routed. Prefer the package's installed console script."""
    from shutil import which

    found = which("mind-nerve-routed")
    if found:
        return found
    # Last-resort: look next to the current Python (venv install)
    here = Path(sys.executable).parent / "mind-nerve-routed"
    if here.exists():
        return str(here)
    return None


def main() -> int:
    sock_path = Path(os.environ.get("MIND_NERVE_SOCKET", str(default_socket_path())))

    if _socket_responsive(sock_path):
        return 0

    daemon = _resolve_daemon_binary()
    if daemon is None:
        # No daemon binary on PATH — silently skip. Caller falls back to
        # cold-subprocess `mind-nerve route`.
        return 0

    log_path = Path(
        os.environ.get("MIND_NERVE_DAEMON_LOG", str(Path.home() / ".mind-nerve" / "daemon.log"))
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_handle = log_path.open("ab", buffering=0)
    log_handle.write(
        f"\n--- mind-nerve-routed-ensure spawn @ {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n".encode()
    )
    try:
        subprocess.Popen(
            [daemon],
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=log_handle,
            start_new_session=True,
        )
    except OSError as e:
        # Daemon failed to spawn — log and move on. Session still loads.
        try:
            log_handle.write(f"spawn failed: {e}\n".encode())
        finally:
            log_handle.close()
        return 0
    # Don't wait for the daemon to come up — fail-open.
    return 0


if __name__ == "__main__":
    sys.exit(main())
