"""Client for the mind-nerve-routed UNIX socket.

WHY THIS MODULE EXISTS
----------------------
mind-nerve had **two independent catalogs** answering the same question.

The route daemon (``mind-nerve-routed``) loads the runtime pinned by its
systemd drop-in -- the curated STARGA table. The MCP server
(``mind-nerve-mcp``) called ``inference.route()`` in its own process, which
resolves the runtime independently and, without the same env pin, lands on a
different (larger, stale) catalog. Measured at one moment on the same box:
daemon ``catalog_size`` 1437, MCP ``catalog_size`` 2475.

That is worse than a stale cache. The two paths have different catalogs *and
different score scales*, so any threshold calibrated on one is meaningless on
the other -- and the router skill instructs agents to call the MCP tool, i.e.
agents were routed by the stale catalog while every measurement was taken
against the daemon.

This module makes the socket the single ranking authority. Callers that can
reach the daemon get the daemon's answer; callers that cannot fall back to
in-process ranking *and say so loudly*, because a silent fallback recreates
exactly the two-catalog split this replaces.
"""

from __future__ import annotations

import json
import os
import socket
from typing import Any

from ._runtime_dir import runtime_socket_dir

__all__ = ["DaemonUnavailable", "default_socket_path", "route_via_daemon"]

_MAX_REPLY_BYTES = 4_000_000


class DaemonUnavailable(RuntimeError):
    """The route daemon could not answer (absent, timed out, or malformed)."""


def default_socket_path() -> str:
    """Socket path, honouring ``MIND_NERVE_SOCKET``.

    Shares ``runtime_socket_dir()`` with the daemon and the hooks so all three
    agree on one path across platforms (XDG -> ~/.cache -> /tmp).
    """
    env = os.environ.get("MIND_NERVE_SOCKET")
    if env:
        return env
    return str(runtime_socket_dir() / "mind-nerve.sock")


def route_via_daemon(
    query: str,
    top_k: int = 5,
    *,
    socket_path: str | None = None,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Ask the daemon to rank *query*; return a RouteResult-shaped dict.

    Raises:
        DaemonUnavailable: socket missing, connection/timeout failure, a
            malformed reply, or an error field in the reply. Callers decide
            whether to fail open -- this function never silently degrades.
    """
    path = socket_path or default_socket_path()
    if not hasattr(socket, "AF_UNIX"):
        raise DaemonUnavailable("AF_UNIX sockets unavailable on this platform")
    if not os.path.exists(path):
        raise DaemonUnavailable(f"socket not present: {path}")

    s = None
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(path)
        s.sendall((json.dumps({"prompt": query, "top_k": top_k}) + "\n").encode("utf-8"))
        s.shutdown(socket.SHUT_WR)
        buf = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
            if len(buf) > _MAX_REPLY_BYTES:
                raise DaemonUnavailable("daemon reply exceeded size cap")
    except DaemonUnavailable:
        raise
    except (OSError, ValueError) as exc:
        raise DaemonUnavailable(f"daemon transport error: {exc}") from exc
    finally:
        if s is not None:
            try:
                s.close()
            except OSError:
                pass

    try:
        reply = json.loads(buf.decode("utf-8", errors="replace"))
    except ValueError as exc:
        raise DaemonUnavailable(f"daemon reply not JSON: {exc}") from exc
    if not isinstance(reply, dict):
        raise DaemonUnavailable("daemon reply was not a JSON object")
    if reply.get("error"):
        raise DaemonUnavailable(f"daemon error: {reply['error']}")
    routes = reply.get("routes")
    if not isinstance(routes, list):
        raise DaemonUnavailable("daemon reply had no routes list")

    # An older daemon build answers with name+score only. That is still a valid
    # ranking -- surface it, but do not invent the fields it did not send.
    normalised: list[dict[str, Any]] = []
    for r in routes:
        if not isinstance(r, dict):
            continue
        row: dict[str, Any] = {
            "id": str(r.get("id", "")),
            "name": str(r.get("name", "")),
            "kind": str(r.get("kind", "")),
            "score": round(float(r.get("score", 0.0)), 6),
            "source_repo": str(r.get("source_repo", "")),
        }
        if r.get("url"):
            row["url"] = r["url"]
        if r.get("source_path"):
            row["source_path"] = r["source_path"]
        normalised.append(row)

    return {
        "query": query,
        "top_k": int(reply.get("top_k", top_k)),
        "routes": normalised,
        "encode_ms": 0.0,
        "rank_ms": float(reply.get("ms", 0)),
        "catalog_size": int(reply.get("catalog_size", -1)),
        "catalog_version": str(reply.get("catalog_version", "unknown")),
        "model_version": str(reply.get("model_version", "unknown")),
        "served_by": "daemon",
    }
