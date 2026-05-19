"""Regression tests for route() bounds: top_k range and request-length guard.

spec/architecture.md mandates:
  - top_k in [1, 64]; values outside raise ValueError.
  - Queries longer than 1024 BPE tokens raise ValueError("RequestTooLong: ...").
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# top_k bounds tests — no runtime required, validated before load
# ---------------------------------------------------------------------------


def _make_minimal_runtime(tmp: Path) -> Path:
    """Build a 3-route fake runtime directory."""
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


class TestTopKBounds:
    """top_k must be in [1, 64]; validated before any model load."""

    def test_top_k_zero_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import mind_nerve.inference as inf_mod

        self._patch_runtime(tmp_path, monkeypatch, inf_mod)
        with pytest.raises(ValueError, match=r"top_k must be in \[1, 64\]"):
            inf_mod.route("query", top_k=0)

    def test_top_k_negative_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import mind_nerve.inference as inf_mod

        self._patch_runtime(tmp_path, monkeypatch, inf_mod)
        with pytest.raises(ValueError, match=r"top_k must be in \[1, 64\]"):
            inf_mod.route("query", top_k=-5)

    def test_top_k_65_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import mind_nerve.inference as inf_mod

        self._patch_runtime(tmp_path, monkeypatch, inf_mod)
        with pytest.raises(ValueError, match=r"top_k must be in \[1, 64\]"):
            inf_mod.route("query", top_k=65)

    def test_top_k_large_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import mind_nerve.inference as inf_mod

        self._patch_runtime(tmp_path, monkeypatch, inf_mod)
        with pytest.raises(ValueError, match=r"top_k must be in \[1, 64\]"):
            inf_mod.route("query", top_k=1000)

    def test_top_k_1_is_valid(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import mind_nerve.inference as inf_mod

        self._patch_runtime(tmp_path, monkeypatch, inf_mod)
        result = inf_mod.route("query", top_k=1)
        assert len(result.routes) == 1

    def test_top_k_64_is_valid(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """top_k=64 is valid even if there are fewer than 64 routes."""
        import mind_nerve.inference as inf_mod

        self._patch_runtime(tmp_path, monkeypatch, inf_mod)
        result = inf_mod.route("query", top_k=64)
        # 3-route runtime: returns min(64, 3) = 3 routes.
        assert len(result.routes) == 3

    def test_top_k_default_5_is_valid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mind_nerve.inference as inf_mod

        self._patch_runtime(tmp_path, monkeypatch, inf_mod)
        result = inf_mod.route("query")
        assert len(result.routes) <= 5

    @staticmethod
    def _patch_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, inf_mod: object) -> None:
        """Wire up a fake pytorch runtime to avoid loading real weights."""
        monkeypatch.setenv("MIND_NERVE_BACKEND", "pytorch")
        runtime = _make_minimal_runtime(tmp_path)
        monkeypatch.setenv("MIND_NERVE_RUNTIME_DIR", str(runtime))

        dim = 8

        class _FakeModel:
            def encode(self, texts: list, **kwargs: object) -> np.ndarray:
                return np.ones((len(texts), dim), dtype=np.float32) / dim

            def tokenize(self, texts: list, **kwargs: object) -> dict:
                import torch

                return {"input_ids": torch.zeros((1, 5), dtype=torch.long)}

            def eval(self) -> "_FakeModel":
                return self

        class _FakeRuntime:
            def __init__(self) -> None:
                self.model = _FakeModel()
                embeddings = np.eye(3, dim, dtype=np.float32)
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
                self.embeddings = (embeddings / norms).astype(np.float32)
                self.routes = [
                    {"id": f"r{i}", "name": f"r{i}", "kind": "skill", "source_repo": "t"}
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

        fake_rt = _FakeRuntime()
        import mind_nerve.inference as inf

        inf._load_cached.cache_clear()
        monkeypatch.setattr(inf, "load_default_runtime", lambda runtime_dir=None: fake_rt)


# ---------------------------------------------------------------------------
# Request-length guard tests
# ---------------------------------------------------------------------------


class TestRequestTooLong:
    """Queries exceeding 1024 BPE tokens must raise ValueError("RequestTooLong: ...")."""

    def test_long_query_raises_request_too_long(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mind_nerve.inference as inf_mod

        self._patch_runtime_with_token_count(tmp_path, monkeypatch, inf_mod, token_count=1025)
        with pytest.raises(ValueError, match=r"RequestTooLong"):
            inf_mod.route("long query", top_k=5)

    def test_exactly_1024_tokens_is_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mind_nerve.inference as inf_mod

        self._patch_runtime_with_token_count(tmp_path, monkeypatch, inf_mod, token_count=1024)
        result = inf_mod.route("max length query", top_k=1)
        assert len(result.routes) >= 0  # accepted, no ValueError

    def test_short_query_is_accepted(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import mind_nerve.inference as inf_mod

        self._patch_runtime_with_token_count(tmp_path, monkeypatch, inf_mod, token_count=12)
        result = inf_mod.route("short", top_k=1)
        assert len(result.routes) >= 0

    def test_error_message_contains_token_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mind_nerve.inference as inf_mod

        self._patch_runtime_with_token_count(tmp_path, monkeypatch, inf_mod, token_count=2000)
        with pytest.raises(ValueError, match=r"2000"):
            inf_mod.route("query", top_k=5)

    @staticmethod
    def _patch_runtime_with_token_count(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        inf_mod: object,
        token_count: int,
    ) -> None:
        """Fake runtime where tokenize() returns exactly token_count tokens."""
        monkeypatch.setenv("MIND_NERVE_BACKEND", "pytorch")
        runtime = _make_minimal_runtime(tmp_path)
        monkeypatch.setenv("MIND_NERVE_RUNTIME_DIR", str(runtime))

        dim = 8

        class _FakeModel:
            def encode(self, texts: list, **kwargs: object) -> np.ndarray:
                return np.ones((len(texts), dim), dtype=np.float32) / dim

            def tokenize(self, texts: list, **kwargs: object) -> dict:
                import torch

                # Return a tensor with exactly token_count columns.
                return {"input_ids": torch.zeros((1, token_count), dtype=torch.long)}

            def eval(self) -> "_FakeModel":
                return self

        class _FakeRuntime:
            def __init__(self) -> None:
                self.model = _FakeModel()
                embeddings = np.eye(3, dim, dtype=np.float32)
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
                self.embeddings = (embeddings / norms).astype(np.float32)
                self.routes = [
                    {"id": f"r{i}", "name": f"r{i}", "kind": "skill", "source_repo": "t"}
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

        fake_rt = _FakeRuntime()
        import mind_nerve.inference as inf

        inf._load_cached.cache_clear()
        monkeypatch.setattr(inf, "load_default_runtime", lambda runtime_dir=None: fake_rt)
