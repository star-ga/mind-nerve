"""Native-disabled fallback: a missing/unloadable native encoder must never
crash — the router degrades to the pure-Python backend with a one-line notice.

These tests pin the cross-platform contract: on Windows/macOS (and any Linux
box without the .so) `MIND_NERVE_BACKEND=native` (the default) silently falls
back to `_Runtime`, and `route()` dispatches on the resolved instance type so
ranking still works.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _make_minimal_runtime(tmp: Path) -> Path:
    """Build a 3-route fake runtime directory (checkpoint dir + catalog)."""
    runtime = tmp / "runtime"
    (runtime / "checkpoint").mkdir(parents=True)
    (runtime / "manifest.json").write_text(
        json.dumps({"catalog_version": "test-v0", "phase1_version": "test-v0"})
    )
    np.save(runtime / "route_table.npy", np.eye(3, 8, dtype=np.float32))
    with (runtime / "route_table.jsonl").open("w") as fh:
        for i in range(3):
            fh.write(
                json.dumps(
                    {"id": f"r{i}", "name": f"route{i}", "kind": "skill", "source_repo": "test"}
                )
                + "\n"
            )
    return runtime


class _FakeModel:
    def encode(self, texts: list, **kwargs: object) -> np.ndarray:
        return np.ones((len(texts), 8), dtype=np.float32) / 8.0

    def tokenize(self, texts: list, **kwargs: object) -> dict:
        import torch

        return {"input_ids": torch.zeros((1, 5), dtype=torch.long)}

    def eval(self) -> "_FakeModel":
        return self


def _install_fake_pytorch_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, inf_mod: object
) -> Path:
    """Make _Runtime(...) construct without loading real weights."""
    runtime = _make_minimal_runtime(tmp_path)
    monkeypatch.setenv("MIND_NERVE_RUNTIME_DIR", str(runtime))

    class _FakeRuntime:
        def __init__(self, runtime_dir: Path) -> None:
            self.model = _FakeModel()
            embeddings = np.eye(3, 8, dtype=np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
            self.embeddings = (embeddings / norms).astype(np.float32)
            self.routes = [
                {"id": f"r{i}", "name": f"route{i}", "kind": "skill", "source_repo": "t"}
                for i in range(3)
            ]
            self.log_prior = None
            self.manifest = {"catalog_version": "t", "phase1_version": "t"}

        @property
        def catalog_size(self) -> int:
            return 3

        @property
        def catalog_version(self) -> str:
            return "test"

        @property
        def model_version(self) -> str:
            return "test"

    monkeypatch.setattr(inf_mod, "_Runtime", _FakeRuntime)
    # _load_cached is LRU-cached; clear so each test re-resolves the runtime.
    inf_mod.load_default_runtime.cache_clear()
    return runtime


def _disable_native(monkeypatch: pytest.MonkeyPatch, inf_mod: object, exc: Exception) -> None:
    """Make _NativeEncoderRuntime construction fail while keeping it a class.

    Patching ``__init__`` (rather than rebinding the whole name) preserves the
    ``isinstance(rt, _NativeEncoderRuntime)`` dispatch in route() — exactly how
    production behaves when the .so is missing: construction raises, the loader
    catches it and returns _Runtime, and route() sees a non-native instance.
    """

    def _boom(self: object, runtime_dir: Path) -> None:
        raise exc

    monkeypatch.setattr(inf_mod._NativeEncoderRuntime, "__init__", _boom)


def test_native_absent_falls_back_to_pytorch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Default (native) backend + missing .so => _Runtime, with a stderr notice."""
    import mind_nerve.inference as inf_mod

    monkeypatch.setenv("MIND_NERVE_BACKEND", "native")
    _install_fake_pytorch_runtime(tmp_path, monkeypatch, inf_mod)
    _disable_native(
        monkeypatch, inf_mod, FileNotFoundError("libmind_nerve_encoder.so not found (simulated)")
    )

    rt = inf_mod.load_default_runtime()
    assert type(rt).__name__ == "_FakeRuntime"  # not the native runtime

    captured = capsys.readouterr()
    assert "native encoder unavailable" in captured.err
    assert "pure-Python" in captured.err


def test_route_works_with_native_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """route() must dispatch on instance type and return correct top-K."""
    import mind_nerve.inference as inf_mod

    monkeypatch.setenv("MIND_NERVE_BACKEND", "native")
    _install_fake_pytorch_runtime(tmp_path, monkeypatch, inf_mod)
    _disable_native(monkeypatch, inf_mod, FileNotFoundError("no .so"))

    result = inf_mod.route("query", top_k=2)
    assert len(result.routes) == 2
    assert all(r.name.startswith("route") for r in result.routes)


def test_unloadable_native_oserror_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A present-but-unloadable .so (OSError, e.g. wrong arch) also falls back."""
    import mind_nerve.inference as inf_mod

    monkeypatch.setenv("MIND_NERVE_BACKEND", "native")
    _install_fake_pytorch_runtime(tmp_path, monkeypatch, inf_mod)
    _disable_native(monkeypatch, inf_mod, OSError("wrong ELF class"))

    rt = inf_mod.load_default_runtime()
    assert type(rt).__name__ == "_FakeRuntime"


def test_native_del_safe_when_init_failed() -> None:
    """_NativeEncoderRuntime.__del__ must not raise when __init__ bailed early
    (no _handle / _native attributes bound)."""
    import mind_nerve.inference as inf_mod

    obj = inf_mod._NativeEncoderRuntime.__new__(inf_mod._NativeEncoderRuntime)
    # No attributes set — emulate a constructor that raised before assigning.
    obj.__del__()  # must be a no-op, not AttributeError


def test_ensure_imports_without_fcntl(monkeypatch: pytest.MonkeyPatch) -> None:
    """mind_nerve.ensure must import on platforms without fcntl (Windows)."""
    import builtins
    import importlib
    import sys

    real_import = builtins.__import__

    def _no_fcntl(name: str, *args: object, **kwargs: object):
        if name == "fcntl":
            raise ImportError("no fcntl on this platform (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "mind_nerve.ensure", raising=False)
    monkeypatch.setattr(builtins, "__import__", _no_fcntl)
    mod = importlib.import_module("mind_nerve.ensure")
    assert mod.fcntl is None
    # Restore a clean module for other tests.
    monkeypatch.setattr(builtins, "__import__", real_import)
    importlib.reload(mod)


def test_daemon_exits_cleanly_without_af_unix(monkeypatch: pytest.MonkeyPatch) -> None:
    """daemon.main() must not raise AttributeError when AF_UNIX is absent."""
    import socket as _socket

    import mind_nerve.daemon as daemon_mod

    monkeypatch.delattr(_socket, "AF_UNIX", raising=False)
    rc = daemon_mod.main()
    assert rc == 1  # clean refusal, not a crash
