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

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ._runtime_dir import runtime_socket_dir

# `fcntl` is POSIX-only. On Windows it is absent; the spawn-serialisation
# flock is then skipped (the daemon is an optional optimisation, and the
# re-check after a would-be lock plus the socket fast-path keep multi-spawn
# rare). Importing must never crash on Windows — guard at module load.
try:
    import fcntl
except ImportError:  # pragma: no cover - exercised only on Windows
    fcntl = None  # type: ignore[assignment]

# Max time a "loser" of the flock race will wait for the winner's daemon
# to come up before falling through fail-open. Sized at 4x the observed
# cold-load (~5 s) to give comfortable headroom on slow disks / boxes.
WAIT_SECONDS = 20.0


def default_socket_path() -> Path:
    return runtime_socket_dir() / "mind-nerve.sock"


def _socket_responsive(sock_path: Path, timeout: float = 1.0) -> bool:
    if not sock_path.exists() or not hasattr(socket, "AF_UNIX"):
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


def _detach_kwargs() -> dict[str, Any]:
    """Return the Popen kwargs that fully detach the daemon from this process.

    POSIX: ``start_new_session=True`` runs setsid so the daemon survives the
    caller's session. Windows has no sessions; instead set the
    DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP creation flags so the child
    does not share the console or process group and is not killed when the
    launching shell exits.
    """
    if os.name == "nt":  # pragma: no cover - exercised only on Windows
        detached = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
        new_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        return {"creationflags": detached | new_group}
    return {"start_new_session": True}


def _spawn_daemon(daemon: str, log_path: Path) -> None:
    """Spawn `mind-nerve-routed` detached. Caller MUST hold the spawn lock."""
    from .shared_env import load_shared_env

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("ab", buffering=0)
    log_handle.write(
        f"\n--- mind-nerve-routed-ensure spawn @ {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n".encode()
    )
    # The daemon inherits this process's environment, with UNSET pins filled
    # from the shared env file — explicit exports always win.
    env = {**load_shared_env(), **os.environ}
    try:
        subprocess.Popen(
            [daemon],
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=log_handle,
            env=env,
            **_detach_kwargs(),
        )
    except OSError as e:
        try:
            log_handle.write(f"spawn failed: {e}\n".encode())
        finally:
            log_handle.close()


def _running_daemon_pids() -> list[int]:
    """PIDs of live ``mind-nerve-routed`` processes (best-effort).

    Linux ``/proc`` scan matching the console-script basename in argv; on
    platforms without ``/proc`` returns ``[]`` (restart is then a no-op and
    the table reload simply takes effect on the next natural daemon
    restart). Never raises.
    """
    proc = Path("/proc")
    if not proc.is_dir():
        return []
    me = os.getpid()
    pids: list[int] = []
    try:
        entries = list(proc.iterdir())
    except OSError:
        return []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == me:
            continue
        # Same-UID only: without this, one project's `acquire install` would
        # SIGTERM/SIGKILL every mind-nerve-routed on the box — including
        # daemons owned by other runtime dirs (2026-08-10 codex audit).
        # Cross-user kills already fail EPERM and are caught below; this
        # filter keeps us from even considering them, and (with hidepid=2)
        # their cmdlines may not be readable anyway.
        try:
            if (entry / "status").read_bytes().split(b"Uid:")[1].split()[0] != str(
                os.geteuid()
            ).encode():
                continue
        except (OSError, IndexError):
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if _is_daemon_cmdline(raw):
            pids.append(pid)
    return sorted(pids)


def _is_daemon_cmdline(raw: bytes) -> bool:
    """True only for the daemon's own invocation shapes (qwen Q15).

    Matching any argv TOKEN's basename would SIGTERM an innocent
    `vim mind-nerve-routed` — only argv[0] (console script) or
    `python* -m mind_nerve.daemon` count.
    """
    args = [a for a in raw.split(b"\0") if a]
    if not args:
        return False
    base = os.path.basename(args[0].decode("utf-8", "replace"))
    if base == "mind-nerve-routed":
        return True
    rest = [a.decode("utf-8", "replace") for a in args[1:3]]
    return base.startswith("python") and rest == ["-m", "mind_nerve.daemon"]


def _pid_alive(pid: int) -> bool:
    stat = Path(f"/proc/{pid}/stat")
    try:
        # A zombie is as good as dead for our purposes.
        return stat.read_text().rsplit(")", 1)[-1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def restart_daemon(grace_seconds: float = 2.0) -> dict[str, Any]:
    """Terminate any running ``mind-nerve-routed`` so it reloads the table.

    The daemon loads the route table into memory once at startup, so after
    ANY table mutation (``acquire install`` / ``acquire remove``) the live
    process serves a stale catalog until restarted. There is deliberately
    no respawn here: the next CLI invocation goes through ``ensure.main()``
    which re-spawns the daemon under its flock guard. Best-effort and
    never raises — a missed restart degrades to stale routing, not an
    error.
    """
    pids = _running_daemon_pids()
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue
    deadline = time.monotonic() + grace_seconds
    survivors: list[int] = []
    while time.monotonic() < deadline:
        survivors = [p for p in pids if _pid_alive(p)]
        if not survivors:
            break
        time.sleep(0.05)
    for pid in survivors:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            continue
    return {"restarted": bool(pids), "pids": pids}


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

    # Platforms without fcntl (Windows) can't flock; skip the concurrency
    # guard and spawn directly after the socket re-check. The daemon is an
    # optional optimisation, so a rare double-spawn there is harmless (the
    # second loses the socket bind and exits); correctness is unaffected.
    if fcntl is None:  # pragma: no cover - exercised only on Windows
        if _socket_responsive(sock_path, timeout=0.5):
            return 0
        _spawn_daemon(daemon, log_path)
        return 0

    # Concurrency guard. Non-blocking flock serialises the spawn decision
    # so parallel ensure() invocations during the daemon's ~5 s weight-load
    # window do not all spawn (the original bug — see module docstring).
    # Lockfile is opened for *append* (mode "a") so we never truncate any
    # state another holder might be writing; we only need a file descriptor
    # to flock against.
    lock_path = _lock_path_for(sock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_fp = lock_path.open("a", encoding="utf-8")
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
