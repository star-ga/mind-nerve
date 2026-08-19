"""mind-nerve MCP façade — stdio JSON-RPC proxy.

Sits between an MCP client (Claude Code, OpenClaw, etc.) and an
upstream MCP server. Intercepts `tools/list` to filter the catalog
down to top-K via mind-nerve; byte-forwards everything else.

The simplest implementation: this server is itself an MCP server
that exposes a single tool `mind_nerve_route`. Clients call it
directly with the user request and receive top-K route IDs.

Future: full transparent stdio proxy per integrations/mcp/DESIGN.md
(intercepting *upstream*'s tools/list). That is more invasive; this
stdio-direct version is sufficient for the OSS v0.1.0 release.

Usage:

    mind-nerve-mcp
    # listens on stdin/stdout JSON-RPC. Pass `tools/list` to discover
    # the route tool; pass `tools/call` with name=mind_nerve_route.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from . import __version__
from .daemon_client import DaemonUnavailable, default_socket_path, route_via_daemon

# THE DAEMON IS THE ONLY CATALOG.
#
# This server used to rank in-process via ``inference.route()``. That made it a
# SECOND catalog: it resolved the runtime independently of the daemon's systemd
# pin and, at one measured moment, answered from 2475 rows while the daemon
# answered from 1437 -- different rows AND a different score scale, so any
# threshold calibrated against one was meaningless against the other. The
# router skill points agents at THIS tool, so agents were being routed by the
# stale catalog while every measurement was taken against the daemon.
#
# The in-process path is deleted rather than kept as a fallback. A fallback
# would re-institutionalise exactly that split, silently, at the worst possible
# moment (daemon down = the one time nobody is watching). When the socket is
# unavailable this server FAILS CLOSED: no routes, one visible line saying the
# router is unavailable. No second brain.


def _ok(req_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _err(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def handle(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Dispatch one JSON-RPC message."""
    method = msg.get("method")
    req_id = msg.get("id")
    params = msg.get("params") or {}

    if method == "initialize":
        return _ok(
            req_id,
            {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "mind-nerve-mcp", "version": __version__},
                "capabilities": {"tools": {"listChanged": False}},
            },
        )

    if method == "notifications/initialized":
        return None  # notifications are not replied to

    if method == "tools/list":
        return _ok(
            req_id,
            {
                "tools": [
                    {
                        "name": "mind_nerve_route",
                        "description": "Return the top-K most relevant skill/tool/agent routes for a "
                        "query. Ranked by the mind-nerve-routed daemon — the SAME "
                        "catalog and the SAME score scale the CLI hooks use, so a score "
                        "seen here means the same thing there. If the daemon is down "
                        "this returns no routes and says so; it never answers from a "
                        "second, unpinned catalog.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The user request / intent.",
                                },
                                "top_k": {
                                    "type": "integer",
                                    "default": 5,
                                    "minimum": 1,
                                    "maximum": 50,
                                },
                            },
                            "required": ["query"],
                        },
                    }
                ]
            },
        )

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        if name != "mind_nerve_route":
            return _err(req_id, -32601, f"unknown tool: {name}")
        query = args.get("query", "").strip()
        if not query:
            return _err(req_id, -32602, "missing query")
        if len(query) > 100_000:
            return _err(req_id, -32602, "query too long")
        try:
            top_k = int(args.get("top_k", 5))
        except (ValueError, TypeError):
            return _err(req_id, -32602, "top_k must be an integer")
        top_k = max(1, min(top_k, 64))
        try:
            payload = route_via_daemon(query, top_k=top_k)
        except DaemonUnavailable as exc:
            # Windows fallback: when AF_UNIX is structurally unavailable there is
            # NO daemon on this box, hence no second catalog for an in-process
            # answer to disagree with. The two-catalog split this tool guards
            # against is a Linux-only failure (daemon pinned to one table, an
            # unpinned in-process resolver landing on another); on Windows the
            # in-process table IS the only catalog, and the fleet pins it via
            # MIND_NERVE_RUNTIME_DIR (a missing pin now warns loudly — see
            # inference._resolve_runtime_dir). Gated on AF_UNIX absence so the
            # Linux path below is byte-identical (Linux always has AF_UNIX and
            # never takes this branch).
            # deferred: a 127.0.0.1 AF_INET daemon would restore the warm-socket
            # latency win on Windows too — upgrade path if per-call cold ranking
            # proves too slow on the Windows fleet nodes.
            import socket as _socket

            if not hasattr(_socket, "AF_UNIX"):
                from .inference import route as _inproc_route

                sys.stderr.write(
                    "[mind-nerve-mcp] no AF_UNIX on this platform; ranking "
                    "in-process (the daemon optimisation is Linux-only)\n"
                )
                result = _inproc_route(query, top_k=top_k)
                payload = result.as_dict()
                payload["served_by"] = "in-process (no-daemon platform)"
                body = json.dumps(payload, indent=2)
                return _ok(req_id, {"content": [{"type": "text", "text": body}]})

            # Fail CLOSED and LOUD. Returning zero routes with a visible reason
            # is strictly better than returning plausible routes from a
            # different catalog: the caller can see the router is down, whereas
            # a silent second catalog is indistinguishable from a working one.
            sys.stderr.write(f"[mind-nerve-mcp] route daemon unavailable: {exc}\n")
            body = json.dumps(
                {
                    "query": query,
                    "top_k": top_k,
                    "routes": [],
                    "catalog_size": 0,
                    "served_by": "unavailable",
                    "error": str(exc),
                },
                indent=2,
            )
            text = (
                "**mind-nerve router unavailable — no routes returned.**\n\n"
                f"The route daemon did not answer ({exc}). This tool deliberately "
                "does NOT fall back to its own catalog: a second, unpinned catalog "
                "is how stale routes got served silently before.\n\n"
                f"Socket: `{default_socket_path()}`\n"
                "Start it with: `systemctl --user start mind-nerve-routed`\n\n"
                f"```json\n{body}\n```"
            )
            return _ok(req_id, {"content": [{"type": "text", "text": text}]})
        body = json.dumps(payload, indent=2)
        return _ok(req_id, {"content": [{"type": "text", "text": body}]})

    return _err(req_id, -32601, f"method not found: {method}")


def main(argv: list[str] | None = None) -> int:
    # Fill any unset MIND_NERVE_* pins from the shared env file BEFORE the
    # daemon socket path is resolved — an explicit export always wins.
    from .shared_env import apply_shared_env

    apply_shared_env()
    # No warmup: this process never loads a model. Ranking belongs to the
    # daemon, so ``initialize`` and ``tools/list`` answer instantly and the
    # ~860 MB encoder is resident exactly once on the box instead of twice.
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            sys.stdout.write(json.dumps(_err(None, -32700, f"parse error: {exc}")) + "\n")
            sys.stdout.flush()
            continue
        try:
            resp = handle(msg)
        except Exception as exc:  # noqa: BLE001 — never let one bad request kill the loop
            req_id = msg.get("id") if isinstance(msg, dict) else None
            resp = _err(req_id, -32603, f"internal error: {exc}")
        if resp is not None:
            sys.stdout.write(json.dumps(resp, separators=(",", ":")) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
