"""mind-nerve-routed — long-lived route daemon over a UNIX socket.

Loads the mind-nerve runtime once at startup, then answers JSON-line route
queries forever. Hook clients send a single-line JSON request, daemon
replies with a single-line JSON response. Eliminates the ~250 ms encoder
load + ~280 ms cold encode cost on every CLI invocation; typical
round-trip after warmup is sub-30 ms — inside the Phase 2 p95 ≤ 30 ms
target even on the Phase 1 PyTorch path.

Protocol (line-delimited JSON over UNIX socket):

    request : {"prompt": "...", "top_k": 20}
    reply   : {"routes": [{"name": "...", "score": 0.81}, ...], "ms": 12}

Defaults:
    socket  $MIND_NERVE_SOCKET (default: $XDG_RUNTIME_DIR/mind-nerve.sock,
            falling back to /tmp/mind-nerve-<uid>.sock)
    runtime resolved by mind_nerve.inference._resolve_runtime_dir
            (i.e. respects MIND_NERVE_RUNTIME_DIR, auto-downloads from
             Hugging Face if neither is set)
"""

from __future__ import annotations

import json
import os
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any

from ._runtime_dir import runtime_socket_dir


def default_socket_path() -> Path:
    """Return the preferred socket path using the shared runtime-dir helper."""
    return runtime_socket_dir() / "mind-nerve.sock"


def main() -> int:
    if not hasattr(socket, "AF_UNIX"):
        # The long-lived daemon serves over a UNIX-domain socket. A few
        # Windows Python builds lack AF_UNIX entirely; there the daemon
        # optimisation is unavailable and callers use the one-shot
        # `mind-nerve route` path instead. Exit with a clear message rather
        # than an opaque AttributeError.
        print(
            "mind-nerve-routed: AF_UNIX sockets are unavailable on this platform; "
            "the daemon is optional — use `mind-nerve route` for one-shot queries.",
            file=sys.stderr,
        )
        return 1
    sock_path = Path(os.environ.get("MIND_NERVE_SOCKET", str(default_socket_path())))
    try:
        if sock_path.exists():
            sock_path.unlink()
    except OSError as e:
        print(f"mind-nerve-routed: could not clear stale socket: {e}", file=sys.stderr)
        return 1

    from .inference import load_default_runtime
    from .inference import route as _route

    t0 = time.time()
    runtime = load_default_runtime()
    _route("warmup", top_k=1)
    print(
        f"mind-nerve-routed: runtime loaded in {time.time() - t0:.2f}s "
        f"({len(runtime.routes)} routes), socket={sock_path}",
        file=sys.stderr,
    )

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    # Bind under a restrictive umask so the socket inode is 0600 from creation.
    # bind()-then-chmod() leaves a brief window where the socket carries
    # umask-default perms and is world-connectable — only mitigated when it lives
    # under XDG_RUNTIME_DIR's 0700, not for the /tmp fallback. The umask closes
    # that gap without depending on the parent dir's mode; the chmod stays as a
    # belt-and-suspenders exact-mode set regardless of the caller's prior umask.
    _old_umask = os.umask(0o077)
    try:
        srv.bind(str(sock_path))
    finally:
        os.umask(_old_umask)
    os.chmod(sock_path, 0o600)
    srv.listen(8)

    def shutdown(*_: object) -> None:
        try:
            srv.close()
        finally:
            try:
                sock_path.unlink()
            except OSError:
                pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    while True:
        conn, _ = srv.accept()
        try:
            conn.settimeout(2.0)
            data = b""
            while b"\n" not in data:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
                if len(data) > 65536:
                    break
            line = data.split(b"\n", 1)[0]
            req = json.loads(line.decode("utf-8", errors="replace"))
            prompt = str(req.get("prompt") or "").strip()
            k = int(req.get("top_k") or 20)
            k = max(1, min(k, 64))  # clamp to route()'s [1,64] bounds (matches mcp_server)
            t_q = time.time()
            result = _route(prompt, top_k=k)
            # The reply carries the FULL route record, not just name+score.
            # Every other surface (the MCP tool, the CLI) must be able to answer
            # a query from this socket and be byte-comparable with it -- two
            # catalogs answering the same question with different rows is the
            # failure this protocol widening exists to make detectable.
            # `name` and `score` keep their historical position and types, so
            # older hook builds keep working unchanged.
            reply: dict[str, Any] = {
                "routes": [
                    {
                        "name": r.name,
                        "score": float(r.score),
                        "id": r.id,
                        "kind": r.kind,
                        "source_repo": r.source_repo,
                        "source_path": r.source_path,
                        "url": r.url,
                    }
                    for r in result.routes
                ],
                "catalog_size": result.catalog_size,
                "catalog_version": result.catalog_version,
                "model_version": result.model_version,
                "top_k": result.top_k,
                "ms": int((time.time() - t_q) * 1000),
            }
        except Exception as e:  # noqa: BLE001  daemon must keep serving
            reply = {"error": str(e), "routes": []}
        try:
            conn.sendall((json.dumps(reply) + "\n").encode("utf-8"))
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
