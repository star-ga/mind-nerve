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
import threading
from typing import Any

from . import __version__
from .inference import load_default_runtime
from .inference import route as _route

# Model warmup runs off the stdin loop (see ``main``) so the JSON-RPC
# ``initialize`` handshake is never blocked by the multi-second model load.
# Strict MCP clients otherwise mark the server "failed" before it answers.
# ``_ensure_loaded`` is idempotent and thread-safe: the background warmup thread
# and any early ``tools/call`` converge on a single one-time load.
_warm_lock = threading.Lock()
_warmed = False


def _ensure_loaded() -> None:
    """Load the default runtime once (thread-safe). Blocks until warm."""
    global _warmed
    if _warmed:
        return
    with _warm_lock:
        if not _warmed:
            load_default_runtime()
            _warmed = True


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
                        "description": "Return the top-K most relevant skill/tool/agent routes for a query, "
                        "from a catalog of ~12k entries (catalog v1.0).",
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
        _ensure_loaded()
        result = _route(query, top_k=top_k)
        body = json.dumps(result.as_dict(), indent=2)
        return _ok(req_id, {"content": [{"type": "text", "text": body}]})

    return _err(req_id, -32601, f"method not found: {method}")


def main(argv: list[str] | None = None) -> int:
    # Warm the model in a background thread (not inline) so ``initialize`` and
    # ``tools/list`` answer immediately; the model still starts loading right
    # away, so the first ``tools/call`` is rarely slow. A ``tools/call`` that
    # arrives before warmup finishes blocks in ``_ensure_loaded`` until ready.
    def _warm() -> None:
        try:
            _ensure_loaded()
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"[mind-nerve-mcp] background warmup failed: {exc}\n")

    threading.Thread(target=_warm, name="mind-nerve-warmup", daemon=True).start()

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
