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

from ._runtime_dir import runtime_socket_dir


def default_socket_path() -> Path:
    """Return the preferred socket path using the shared runtime-dir helper."""
    return runtime_socket_dir() / "mind-nerve.sock"


def main() -> int:
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
    srv.bind(str(sock_path))
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
            t_q = time.time()
            result = _route(prompt, top_k=k)
            reply: dict = {
                "routes": [{"name": r.name, "score": float(r.score)} for r in result.routes],
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
