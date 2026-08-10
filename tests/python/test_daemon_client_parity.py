"""The daemon socket is the ONLY catalog.

The MCP server used to rank in-process, which made it a SECOND catalog: it
resolved the runtime independently of the daemon's systemd pin and, at one
measured moment, answered from 2475 rows while the daemon answered from 1437 —
different rows AND a different score scale. The router skill points agents at
the MCP tool, so agents were routed by the stale catalog while every
measurement was taken against the daemon.

These tests pin the two properties that keep it collapsed to one catalog:
  * the MCP tool answers from the socket, verbatim;
  * when the socket is down it FAILS CLOSED — no routes, and it says so —
    rather than quietly answering from a second brain.
"""

from __future__ import annotations

import json
import os
import socket
import threading
from pathlib import Path

import pytest
from mind_nerve import mcp_server
from mind_nerve.daemon_client import DaemonUnavailable, route_via_daemon

pytestmark = pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"), reason="AF_UNIX sockets unavailable"
)

CATALOG_SIZE = 1426

REPLY = {
    "routes": [
        {
            "name": "diagnose",
            "score": 0.4576,
            "id": "abc",
            "kind": "skill",
            "source_repo": "starga",
            "source_path": "/hub/diagnose/SKILL.md",
            "url": None,
        },
        {
            "name": "gh-fix-ci",
            "score": 0.4012,
            "id": "def",
            "kind": "skill",
            "source_repo": "starga",
            "source_path": "/hub/gh-fix-ci/SKILL.md",
            "url": None,
        },
    ],
    "catalog_size": CATALOG_SIZE,
    "catalog_version": "v-test",
    "model_version": "m-test",
    "top_k": 2,
    "ms": 7,
}


class FakeDaemon:
    """Serves one canned line-delimited JSON reply per connection."""

    def __init__(self, path: Path, reply: dict) -> None:
        self.path = str(path)
        self.reply = reply
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(self.path)
        self.srv.listen(4)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            try:
                conn.recv(65536)
                conn.sendall((json.dumps(self.reply) + "\n").encode())
            except OSError:
                pass
            finally:
                conn.close()

    def close(self) -> None:
        try:
            self.srv.close()
        finally:
            try:
                os.unlink(self.path)
            except OSError:
                pass


@pytest.fixture()
def daemon(tmp_path: Path):
    d = FakeDaemon(tmp_path / "n.sock", REPLY)
    yield d
    d.close()


class TestRouteViaDaemon:
    def test_returns_the_daemon_rows_verbatim(self, daemon: FakeDaemon) -> None:
        out = route_via_daemon("fix my build", top_k=2, socket_path=daemon.path)
        assert out["catalog_size"] == CATALOG_SIZE
        assert [r["name"] for r in out["routes"]] == ["diagnose", "gh-fix-ci"]
        assert [r["source_path"] for r in out["routes"]] == [
            "/hub/diagnose/SKILL.md",
            "/hub/gh-fix-ci/SKILL.md",
        ]
        assert out["served_by"] == "daemon"

    def test_missing_socket_raises_rather_than_degrading(self, tmp_path: Path) -> None:
        with pytest.raises(DaemonUnavailable):
            route_via_daemon("q", socket_path=str(tmp_path / "nope.sock"))

    def test_daemon_error_field_raises(self, tmp_path: Path) -> None:
        d = FakeDaemon(tmp_path / "e.sock", {"error": "boom", "routes": []})
        try:
            with pytest.raises(DaemonUnavailable):
                route_via_daemon("q", socket_path=d.path)
        finally:
            d.close()

    def test_tolerates_an_older_name_score_only_daemon(self, tmp_path: Path) -> None:
        d = FakeDaemon(tmp_path / "o.sock", {"routes": [{"name": "x", "score": 0.5}]})
        try:
            out = route_via_daemon("q", socket_path=d.path)
            assert out["routes"][0]["name"] == "x"
            # Fields the old daemon did not send are absent, not invented.
            assert "source_path" not in out["routes"][0]
            assert out["catalog_size"] == -1
        finally:
            d.close()


class TestMcpServerIsAThinClient:
    def _call(self, query: str) -> dict:
        return mcp_server.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "mind_nerve_route",
                    "arguments": {"query": query, "top_k": 2},
                },
            }
        )

    def test_serves_the_daemon_answer(
        self, daemon: FakeDaemon, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MIND_NERVE_SOCKET", daemon.path)
        payload = json.loads(self._call("fix my build")["result"]["content"][0]["text"])
        assert payload["catalog_size"] == CATALOG_SIZE
        assert payload["served_by"] == "daemon"
        assert [r["name"] for r in payload["routes"]] == ["diagnose", "gh-fix-ci"]

    def test_fails_closed_when_the_socket_is_down(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MIND_NERVE_SOCKET", str(tmp_path / "absent.sock"))
        text = self._call("fix my build")["result"]["content"][0]["text"]
        assert "router unavailable" in text.lower()
        # No routes, and the reason is visible to the caller.
        assert '"routes": []' in text
        assert '"served_by": "unavailable"' in text

    def test_has_no_in_process_catalog_left(self) -> None:
        # The in-process path is DELETED, not kept as a fallback: a fallback
        # re-creates the split silently, at the one moment nobody is watching.
        src = Path(mcp_server.__file__).read_text(encoding="utf-8")
        assert "load_default_runtime" not in src
        assert "from .inference import" not in src
        assert not hasattr(mcp_server, "_ensure_loaded")
