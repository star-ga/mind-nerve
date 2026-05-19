"""Regression tests for mind_nerve.ensure flock-guarded spawn (2026-05-18).

Prior to f4d0c83+ the ensure script spawned a fresh `mind-nerve-routed`
daemon every time `_socket_responsive()` returned False — which during
the daemon's 5 s weight-load window was *every parallel CLI invocation*.
A single bot-thrash incident left 9 zombie daemons holding ~1.3 GB each
under the same systemd cgroup.

The flock guard makes the spawn decision exclusive: only one of N
concurrent ensure() callers spawns, the rest poll the socket and exit
fail-open. These tests prove that contract holds without relying on
the real daemon (we mock `subprocess.Popen` and `_socket_responsive`).
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import patch

import pytest
from mind_nerve import ensure


@pytest.fixture
def isolated_paths(tmp_path: Path, monkeypatch):
    """Point ensure at a tmpdir socket + log so concurrent test cases
    don't collide with each other or with a real running daemon."""
    sock = tmp_path / "mind-nerve.sock"
    log = tmp_path / "daemon.log"
    monkeypatch.setenv("MIND_NERVE_SOCKET", str(sock))
    monkeypatch.setenv("MIND_NERVE_DAEMON_LOG", str(log))
    yield sock, log


def test_lock_path_is_sibling_of_socket(tmp_path: Path):
    sock = tmp_path / "mind-nerve.sock"
    lock = ensure._lock_path_for(sock)
    assert lock.parent == sock.parent
    assert lock.name == "mind-nerve.sock.lock"


def test_single_call_spawns_when_socket_dead(isolated_paths):
    """Sanity: with no daemon running, one ensure call results in one spawn."""
    sock, _ = isolated_paths
    spawn_count = 0

    def fake_spawn(daemon, log_path):
        nonlocal spawn_count
        spawn_count += 1

    with (
        patch.object(ensure, "_socket_responsive", return_value=False),
        patch.object(ensure, "_resolve_daemon_binary", return_value="/fake/daemon"),
        patch.object(ensure, "_spawn_daemon", side_effect=fake_spawn),
    ):
        rc = ensure.main()
    assert rc == 0
    assert spawn_count == 1


def test_responsive_socket_skips_spawn(isolated_paths):
    """Fast path: if the socket already answers, we exit before locking."""
    spawn_count = 0

    def fake_spawn(daemon, log_path):
        nonlocal spawn_count
        spawn_count += 1

    with (
        patch.object(ensure, "_socket_responsive", return_value=True),
        patch.object(ensure, "_resolve_daemon_binary", return_value="/fake/daemon"),
        patch.object(ensure, "_spawn_daemon", side_effect=fake_spawn),
    ):
        rc = ensure.main()
    assert rc == 0
    assert spawn_count == 0


def test_concurrent_callers_spawn_exactly_once(isolated_paths):
    """**The bug-fix contract.** 16 threads call ensure.main() concurrently,
    all see an unresponsive socket, but only one wins the flock and spawns.
    The other 15 poll the socket (which the test makes "come up" after a
    brief delay) and exit cleanly.

    NB: the `patch.object` context manager mutates module-level attributes
    and is **not** thread-safe — entering it concurrently from each thread
    will race and intermittently leak real functions through. We lift the
    patch context up so all 16 threads share one mock state for the full
    duration of the race.
    """
    sock, _ = isolated_paths
    spawn_count = 0
    spawn_count_lock = threading.Lock()
    # Toggle the "is socket responsive?" probe: returns False until the
    # spawning thread has run, then True for everyone else's wait loop.
    socket_up = threading.Event()

    def fake_socket_responsive(path, timeout=1.0):
        return socket_up.is_set()

    def fake_spawn(daemon, log_path):
        nonlocal spawn_count
        with spawn_count_lock:
            spawn_count += 1
        # Simulate the daemon needing some time to come up, then signal
        # "responsive" so the polling losers can exit.
        socket_up.set()

    def runner():
        ensure.main()

    with (
        patch.object(ensure, "_socket_responsive", side_effect=fake_socket_responsive),
        patch.object(ensure, "_resolve_daemon_binary", return_value="/fake/daemon"),
        patch.object(ensure, "_spawn_daemon", side_effect=fake_spawn),
    ):
        threads = [threading.Thread(target=runner) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive(), "ensure() must never block longer than WAIT_SECONDS"

    assert spawn_count == 1, (
        f"expected exactly one spawn under 16-way concurrency, got {spawn_count} "
        "— the flock guard is not serialising the spawn decision"
    )


def test_lock_loser_falls_through_fail_open_on_timeout(isolated_paths, monkeypatch):
    """If the winner's daemon never comes up within WAIT_SECONDS, every
    caller must still exit 0 (fail-open). The script doc-contract is
    'never block session start' — we honour it even when the daemon is
    broken. Note: retry is intentional — a *permanently* broken daemon
    must eventually get re-spawned on subsequent calls, not stay dead
    forever; the flock just ensures at most one spawn per WAIT_SECONDS
    window, not at most one spawn ever.
    """
    monkeypatch.setattr(ensure, "WAIT_SECONDS", 0.5)

    def fake_spawn(daemon, log_path):
        # Intentionally never signal "socket up" — simulate broken daemon.
        pass

    def fake_socket_responsive(path, timeout=1.0):
        return False

    rcs: list[int] = []
    rcs_lock = threading.Lock()

    def runner():
        rc = ensure.main()
        with rcs_lock:
            rcs.append(rc)

    with (
        patch.object(ensure, "_socket_responsive", side_effect=fake_socket_responsive),
        patch.object(ensure, "_resolve_daemon_binary", return_value="/fake/daemon"),
        patch.object(ensure, "_spawn_daemon", side_effect=fake_spawn),
    ):
        threads = [threading.Thread(target=runner) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            # WAIT_SECONDS=0.5 so each caller bounds at ~0.5s, with 4 callers
            # serialised through the flock the total is < 3s — set a comfortable
            # upper bound so a slow CI box doesn't false-fail.
            t.join(timeout=15)
            assert not t.is_alive(), (
                "ensure() must never block longer than ~WAIT_SECONDS per acquirer"
            )

    assert all(rc == 0 for rc in rcs), (
        "fail-open: every caller must return 0 even when daemon is broken"
    )


def test_no_daemon_binary_returns_zero_without_locking(isolated_paths):
    """If `mind-nerve-routed` isn't on PATH at all, ensure must short-circuit
    *before* touching the lock — keeps the fail-open behaviour intact on a
    machine that never installed the daemon (the package's optional path)."""
    spawn_count = 0

    def fake_spawn(daemon, log_path):
        nonlocal spawn_count
        spawn_count += 1

    with (
        patch.object(ensure, "_socket_responsive", return_value=False),
        patch.object(ensure, "_resolve_daemon_binary", return_value=None),
        patch.object(ensure, "_spawn_daemon", side_effect=fake_spawn),
    ):
        rc = ensure.main()
    assert rc == 0
    assert spawn_count == 0
    # And critically: no lockfile should have been created on this path.
    sock, _ = isolated_paths
    lock = ensure._lock_path_for(sock)
    assert not lock.exists(), "no-daemon path must not touch the lock filesystem"
