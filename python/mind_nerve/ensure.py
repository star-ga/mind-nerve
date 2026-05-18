"""mind-nerve-routed-ensure — idempotent starter for the route daemon.

Designed to be called from a Claude Code SessionStart hook (or any
similar "session warmup" trigger). Probes the daemon's UNIX socket;
if it's already responsive, returns immediately. Otherwise spawns
`mind-nerve-routed` detached. Always exits 0 — the daemon is a
performance optimisation, not a correctness requirement, so this
script must never block session start.

Concurrency contract (added 2026-05-18): parallel invocations during
the daemon's 5 s weight-load window must not all decide to spawn —
that produced a multi-spawn race in prior releases where each caller
saw an unresponsive socket and forked its own daemon. A non-blocking
`flock` on a sibling lock file serialises the "should I spawn?"
decision; losers poll the socket for up to `WAIT_SECONDS` instead of
spawning, then exit fail-open. Net effect: at most one daemon process
per socket, ever, across any number of parallel `ensure` invocations.
"""

from __future__ import annotations

import fcntl
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

# Max time a "loser" of the flock race will wait for the winner's daemon
# to come up before falling through fail-open. Sized at 4x the observed
# cold-load (~5 s) to give comfortable headroom on slow disks / boxes.
WAIT_SECONDS = 20.0


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


def _lock_path_for(sock_path: Path) -> Path:
    """Lockfile lives next to the socket so it shares lifecycle with the
    runtime-dir cleanup. `.lock` suffix avoids accidental confusion with
    the socket itself (mode `srw-` vs `-rw-`)."""
    return sock_path.with_name(sock_path.name + ".lock")


def _spawn_daemon(daemon: str, log_path: Path) -> None:
    """Spawn `mind-nerve-routed` detached. Caller MUST hold the spawn lock."""
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
        try:
            log_handle.write(f"spawn failed: {e}\n".encode())
        finally:
            log_handle.close()


def main() -> int:
    sock_path = Path(os.environ.get("MIND_NERVE_SOCKET", str(default_socket_path())))

    # Fast path: socket already responsive → exit before touching the lock.
    if _socket_responsive(sock_path):
        return 0

    daemon = _resolve_daemon_binary()
    if daemon is None:
        # No daemon binary on PATH — silently skip. Caller falls back to
        # cold-subprocess `mind-nerve route`. No need to take the lock.
        return 0

    log_path = Path(
        os.environ.get("MIND_NERVE_DAEMON_LOG", str(Path.home() / ".mind-nerve" / "daemon.log"))
    )

    # Concurrency guard. Non-blocking flock serialises the spawn decision
    # so parallel ensure() invocations during the daemon's ~5 s weight-load
    # window do not all spawn (the original bug — see module docstring).
    # Lockfile is opened for *append* (mode "a") so we never truncate any
    # state another holder might be writing; we only need a file descriptor
    # to flock against.
    lock_path = _lock_path_for(sock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_fp = lock_path.open("a")
    except OSError:
        # Can't even open the lock — fall through fail-open without spawning
        # to avoid the broken behaviour where ensure spams the system.
        return 0

    try:
        try:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # Another ensure() is already spawning. Poll the socket up to
            # WAIT_SECONDS so a CLI invocation that lost the race still
            # benefits from the *winner's* daemon if it comes up in time.
            deadline = time.monotonic() + WAIT_SECONDS
            while time.monotonic() < deadline:
                if _socket_responsive(sock_path, timeout=0.5):
                    return 0
                time.sleep(0.25)
            # Daemon still not up — fail-open. Caller falls back to a
            # cold subprocess, the daemon will be ready for the next call.
            return 0

        # We hold the lock. Re-check the socket: the previous holder may
        # have completed a spawn between our fast-path probe and now.
        if _socket_responsive(sock_path, timeout=0.5):
            return 0
        _spawn_daemon(daemon, log_path)
        # Hold the lock until the daemon's socket binds (or WAIT_SECONDS
        # elapses, fail-open). The previous version released as soon as
        # Popen returned — but Popen returns *before* the daemon binds
        # its socket (the 5 s weight-load happens first). Subsequent
        # ensure() callers that acquired the lock during that window saw
        # a still-down socket and spawned AGAIN, defeating the guard.
        # By waiting here, every later flock acquirer either sees a
        # responsive socket on re-check or hits the bounded timeout —
        # at most one spawn per WAIT_SECONDS window.
        deadline = time.monotonic() + WAIT_SECONDS
        while time.monotonic() < deadline:
            if _socket_responsive(sock_path, timeout=0.5):
                return 0
            time.sleep(0.25)
        return 0
    finally:
        try:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
        finally:
            lock_fp.close()


if __name__ == "__main__":
    sys.exit(main())
