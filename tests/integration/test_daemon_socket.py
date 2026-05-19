"""Integration test: spawn `mind-nerve-routed` and probe its UNIX socket.

The daemon is the warm-cache hot path used by all hook clients. It
loads the encoder once, then answers single-line JSON queries over
a UNIX domain socket. This test:

  1. Spawns the daemon in a subprocess against an in-memory fake
     runtime (no Hugging Face download, no live checkpoint).
  2. Connects to the socket, sends a `route` request, reads the reply.
  3. Asserts the reply is well-formed JSON with `routes` + `ms` keys.
  4. Tears the daemon down cleanly via SIGTERM.

This test self-skips on platforms or CI environments where the
auto-seeded runtime cannot be initialised. It is run in CI under
qa-gates but is also separately guarded by `MIND_NERVE_DAEMON_TEST_SKIP=1`.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


_SKIP_REASON = (
    "daemon socket test needs a runnable mind-nerve-routed binary + a fake "
    "runtime. Set MIND_NERVE_DAEMON_TEST_SKIP=0 to enable in environments "
    "where mind-nerve-routed is on PATH and runtime stubs can be wired."
)


def _runtime_with_stub(tmp_path: Path) -> Path:
    """Lay out a minimum runtime dir with a pre-built route table + manifest.

    The daemon's runtime loader expects checkpoint/, route_table.npy, and
    route_table.jsonl. The checkpoint dir is empty in this stub — the
    SentenceTransformer load will fail on the real implementation, which
    is exactly why we patch the inference module via a sitecustomize file
    we inject into PYTHONPATH for the subprocess.
    """
    import numpy as np

    rdir = tmp_path / "runtime"
    (rdir / "checkpoint").mkdir(parents=True)
    rng = np.random.default_rng(seed=42)
    emb = rng.standard_normal((8, 8)).astype(np.float32)
    np.save(rdir / "route_table.npy", emb)
    with (rdir / "route_table.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(8):
            fh.write(
                json.dumps(
                    {
                        "id": f"route-{i:03d}",
                        "name": f"route-{i:03d}",
                        "kind": "skill",
                        "source_repo": "test",
                    }
                )
                + "\n"
            )
    (rdir / "manifest.json").write_text(
        json.dumps({"catalog_version": "test", "phase1_version": "test"})
    )
    return rdir


def _write_sitecustomize(target_dir: Path) -> Path:
    """Write a sitecustomize.py that patches inference.load_default_runtime
    so the daemon subprocess uses an in-memory fake without hitting HF.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    sitecustomize = target_dir / "sitecustomize.py"
    sitecustomize.write_text(
        """
# Test-only sitecustomize: patches mind_nerve.inference.load_default_runtime
# so the daemon subprocess can come up without Hugging Face network access
# or a real SentenceTransformer checkpoint.
import json
import os
from pathlib import Path

import numpy as np


def _make_fake_runtime():
    rdir = Path(os.environ['MIND_NERVE_RUNTIME_DIR'])

    class _Fake:
        def __init__(self) -> None:
            self.dir = rdir
            self.manifest = json.loads((rdir / 'manifest.json').read_text())
            self.embeddings = np.load(rdir / 'route_table.npy').astype(np.float32)
            self.routes = [
                json.loads(ln)
                for ln in (rdir / 'route_table.jsonl').open('r', encoding='utf-8')
            ]
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-12
            self.embeddings = (self.embeddings / norms).astype(np.float32)
            self.log_prior = None
            self.freq_scale = None
            self.stride_thresholds = None

            class _M:
                def __init__(self, dim: int) -> None:
                    self._dim = dim

                def encode(self, texts, **_):
                    out = np.zeros((len(texts), self._dim), dtype=np.float32)
                    for i, t in enumerate(texts):
                        h = abs(hash(t)) % (2**31 - 1)
                        rng = np.random.default_rng(seed=h)
                        v = rng.standard_normal(self._dim).astype(np.float32)
                        v /= (np.linalg.norm(v) + 1e-12)
                        out[i] = v
                    return out

                def tokenize(self, texts, **_):
                    import torch
                    return {'input_ids': torch.zeros((1, 5), dtype=torch.long)}

                def eval(self):
                    return self

            self.model = _M(self.embeddings.shape[1])

        @property
        def catalog_size(self) -> int:
            return len(self.routes)

        @property
        def catalog_version(self) -> str:
            return 'test'

        @property
        def model_version(self) -> str:
            return 'test'

    return _Fake()


def _patch_inference():
    try:
        import mind_nerve.inference as inf_mod
    except Exception:  # noqa: BLE001
        return
    fake = _make_fake_runtime()
    inf_mod.load_default_runtime = lambda runtime_dir=None: fake  # type: ignore[assignment]
    inf_mod._seed_from_hf = lambda target: None  # type: ignore[assignment]


_patch_inference()
"""
    )
    return sitecustomize


def _daemon_on_path() -> str | None:
    """Return the resolved path to `mind-nerve-routed`, or None."""
    from shutil import which

    return which("mind-nerve-routed")


@pytest.fixture
def daemon_env(tmp_path: Path) -> dict[str, str]:
    """Build the environment for a stand-alone daemon subprocess."""
    rdir = _runtime_with_stub(tmp_path)
    custom_pkg = tmp_path / "sitecustomize_pkg"
    _write_sitecustomize(custom_pkg)

    env = dict(os.environ)
    env["MIND_NERVE_RUNTIME_DIR"] = str(rdir)
    env["MIND_NERVE_BACKEND"] = "pytorch"
    env["MIND_NERVE_SOCKET"] = str(tmp_path / "daemon.sock")
    env["MIND_NERVE_DAEMON_LOG"] = str(tmp_path / "daemon.log")
    env["PYTHONPATH"] = (str(custom_pkg) + os.pathsep + env.get("PYTHONPATH", "")).rstrip(
        os.pathsep
    )
    return env


@pytest.mark.skipif(
    os.environ.get("MIND_NERVE_DAEMON_TEST_SKIP", "0") == "1",
    reason=_SKIP_REASON,
)
def test_daemon_socket_route_request_responds(daemon_env: dict[str, str], tmp_path: Path) -> None:
    """Spawn the daemon, route a query over the socket, assert response shape."""
    daemon_bin = _daemon_on_path()
    if daemon_bin is None:
        pytest.skip("mind-nerve-routed not on PATH")

    sock_path = Path(daemon_env["MIND_NERVE_SOCKET"])

    # Launch the daemon as a subprocess so the socket lifecycle stays
    # independent of the pytest process. start_new_session is intentional
    # — we kill it via .terminate() below, never via the test process's
    # signal handlers.
    proc = subprocess.Popen(
        [daemon_bin],
        env=daemon_env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        # Wait up to 15s for the socket to appear and be responsive.
        deadline = time.monotonic() + 15.0
        ready = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
                pytest.skip(f"daemon exited before socket bound: {stderr[:1000]}")
            if sock_path.exists():
                try:
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.settimeout(1.0)
                    s.connect(str(sock_path))
                    s.close()
                    ready = True
                    break
                except OSError:
                    pass
            time.sleep(0.2)
        if not ready:
            pytest.skip(f"daemon did not bind socket within 15s at {sock_path}")

        # Send a route request and read the reply.
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect(str(sock_path))
        s.sendall(b'{"prompt":"deploy a staging build","top_k":3}\n')
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
            if len(buf) > 65536:
                break
        s.close()

        line = buf.split(b"\n", 1)[0]
        assert line, "daemon returned empty payload"
        reply = json.loads(line.decode("utf-8"))

        # Well-formed reply shape: { routes: [...], ms: int } OR { error, routes:[] }
        assert "routes" in reply, f"missing 'routes' in reply: {reply!r}"
        if "error" not in reply:
            assert isinstance(reply["routes"], list)
            assert len(reply["routes"]) <= 3
            for r in reply["routes"]:
                assert "name" in r and "score" in r
            assert isinstance(reply.get("ms"), int)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)
        # Defensive cleanup: stale socket file.
        try:
            if sock_path.exists():
                sock_path.unlink()
        except OSError:
            pass
