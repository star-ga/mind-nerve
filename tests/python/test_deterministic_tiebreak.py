"""Regression tests for the SHA-256 tie-break contract in route().

The spec (spec/architecture.md) mandates that equal-score routes are
ordered by ascending SHA-256(route_id) so that the same input produces
the same top-K ranking on every architecture.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


def _expected_order_by_sha256(scores: np.ndarray, route_ids: list[str]) -> list[int]:
    """Reference implementation: primary key -score, tie-break SHA-256 digest."""
    return sorted(
        range(len(scores)),
        key=lambda i: (
            -float(scores[i]),
            hashlib.sha256(route_ids[i].encode("utf-8")).digest(),
        ),
    )


def _make_minimal_runtime(tmp: Path, routes: list[dict], embeddings: np.ndarray) -> Path:
    """Build a fake runtime directory with the given routes and embeddings."""
    runtime = tmp / "runtime"
    (runtime / "checkpoint").mkdir(parents=True)
    (runtime / "manifest.json").write_text(
        json.dumps({"catalog_version": "test-v0", "phase1_version": "test-v0"})
    )
    np.save(runtime / "route_table.npy", embeddings)
    with (runtime / "route_table.jsonl").open("w") as fh:
        for route in routes:
            fh.write(json.dumps(route) + "\n")
    return runtime


# ---------------------------------------------------------------------------
# Unit tests for the helper functions
# ---------------------------------------------------------------------------


class TestTieKey:
    def test_returns_bytes(self) -> None:
        from mind_nerve.inference import _tie_key

        result = _tie_key("some-route")
        assert isinstance(result, bytes)
        assert len(result) == 32  # SHA-256 is 32 bytes

    def test_deterministic_across_calls(self) -> None:
        from mind_nerve.inference import _tie_key

        assert _tie_key("route-alpha") == _tie_key("route-alpha")

    def test_distinct_ids_produce_distinct_digests(self) -> None:
        from mind_nerve.inference import _tie_key

        assert _tie_key("route-a") != _tie_key("route-b")

    def test_empty_string(self) -> None:
        from mind_nerve.inference import _tie_key

        result = _tie_key("")
        assert len(result) == 32


class TestDeterministicTopk:
    def test_returns_descending_score_order(self) -> None:
        from mind_nerve.inference import _deterministic_topk

        scores = np.array([0.3, 0.9, 0.5], dtype=np.float32)
        route_ids = ["r0", "r1", "r2"]
        indices = _deterministic_topk(scores, route_ids, k=3)
        assert list(indices) == [1, 2, 0]

    def test_tie_break_by_sha256_ascending(self) -> None:
        """Two equal-score routes must be ordered by ascending SHA-256(route_id)."""
        from mind_nerve.inference import _deterministic_topk

        # Both routes have identical scores.
        scores = np.array([0.8, 0.8, 0.1], dtype=np.float32)
        route_ids = ["route-beta", "route-alpha", "route-low"]
        indices = _deterministic_topk(scores, route_ids, k=2)

        # The expected order is determined purely by SHA-256 digest comparison.
        expected = _expected_order_by_sha256(
            np.array([0.8, 0.8], dtype=np.float32), ["route-beta", "route-alpha"]
        )
        # Map local positions back to original indices (0=beta, 1=alpha).
        expected_original = [expected[0], expected[1]]
        assert list(indices) == expected_original

    def test_sha256_tiebreak_is_stable_across_runs(self) -> None:
        """The same call must return the same order every time (no random seed)."""
        from mind_nerve.inference import _deterministic_topk

        scores = np.array([1.0, 1.0, 1.0, 0.5], dtype=np.float32)
        route_ids = ["zz-route", "aa-route", "mm-route", "low-route"]
        first = list(_deterministic_topk(scores, route_ids, k=3))
        for _ in range(10):
            again = list(_deterministic_topk(scores, route_ids, k=3))
            assert again == first, "tie-break result changed across calls"

    def test_returns_int64_array(self) -> None:
        from mind_nerve.inference import _deterministic_topk

        scores = np.array([0.5, 0.7], dtype=np.float32)
        indices = _deterministic_topk(scores, ["r0", "r1"], k=2)
        assert indices.dtype == np.int64

    def test_k_equals_one(self) -> None:
        from mind_nerve.inference import _deterministic_topk

        scores = np.array([0.2, 0.9, 0.5], dtype=np.float32)
        indices = _deterministic_topk(scores, ["r0", "r1", "r2"], k=1)
        assert list(indices) == [1]


# ---------------------------------------------------------------------------
# Integration test: sha256_tiebreak_two_equal_score_routes_ordered_deterministically
# (grep handle: test_sha256_tiebreak_two_equal_score_routes_ordered_deterministically)
# ---------------------------------------------------------------------------


def test_sha256_tiebreak_two_equal_score_routes_ordered_deterministically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two routes with identical cosine-similarity scores must be ordered by
    ascending SHA-256(route_id) and that order must be stable across calls.

    This is the load-bearing regression test for the cross-arch bit-identity
    contract described in spec/architecture.md.
    """
    # Force pytorch backend so we can control embeddings exactly.
    monkeypatch.setenv("MIND_NERVE_BACKEND", "pytorch")

    # Build a fake runtime directory.
    # All embeddings are identical unit vectors — dot product = 1.0 for any query.
    dim = 8
    embeddings = np.ones((3, dim), dtype=np.float32) / np.sqrt(dim)
    # Third route gets a different embedding so top-2 is stable.
    embeddings[2] = np.zeros(dim, dtype=np.float32)
    embeddings[2, 0] = 0.1

    routes = [
        {"id": "route-beta", "name": "beta", "kind": "skill", "source_repo": "test"},
        {"id": "route-alpha", "name": "alpha", "kind": "skill", "source_repo": "test"},
        {"id": "route-low", "name": "low", "kind": "skill", "source_repo": "test"},
    ]
    runtime = _make_minimal_runtime(tmp_path, routes, embeddings)

    # Patch _seed_from_hf so no network call is made.
    import mind_nerve.inference as inf_mod

    monkeypatch.setattr(inf_mod, "_seed_from_hf", lambda target: None)
    # Clear the LRU cache so our fake runtime is used.
    inf_mod._load_cached.cache_clear()

    monkeypatch.setenv("MIND_NERVE_RUNTIME_DIR", str(runtime))

    # Stub out the pytorch model to avoid loading real weights.
    class _FakeModel:
        def encode(self, texts: list, **kwargs: object) -> np.ndarray:
            # Return a unit vector in the same direction as all catalog rows.
            v = np.ones((len(texts), dim), dtype=np.float32) / np.sqrt(dim)
            return v

        def tokenize(self, texts: list, **kwargs: object) -> dict:
            import torch

            # Return a small token-count so the 1024-token guard is not hit.
            return {"input_ids": torch.zeros((1, 5), dtype=torch.long)}

        def eval(self) -> "_FakeModel":
            return self

    class _FakeRuntime:
        """Minimal _Runtime surface expected by _route_pytorch."""

        def __init__(self) -> None:
            self.model = _FakeModel()
            self.embeddings = embeddings.copy()
            # L2-normalise (already unit vectors, just in case).
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-12
            self.embeddings = (self.embeddings / norms).astype(np.float32)
            self.routes = routes
            self.log_prior = None
            self.manifest = {"catalog_version": "test", "phase1_version": "test"}

        @property
        def catalog_size(self) -> int:
            return len(self.routes)

        @property
        def catalog_version(self) -> str:
            return "test"

        @property
        def model_version(self) -> str:
            return "test"

    fake_rt = _FakeRuntime()
    monkeypatch.setattr(inf_mod, "load_default_runtime", lambda runtime_dir=None: fake_rt)

    result = inf_mod.route("any query", top_k=2)
    got_ids = [r.id for r in result.routes]

    # The first two routes have identical scores (1.0 after norm); they must
    # be ordered by ascending SHA-256(route_id).
    sha_beta = hashlib.sha256(b"route-beta").digest()
    sha_alpha = hashlib.sha256(b"route-alpha").digest()
    if sha_alpha < sha_beta:
        expected_ids = ["route-alpha", "route-beta"]
    else:
        expected_ids = ["route-beta", "route-alpha"]

    assert got_ids == expected_ids, (
        f"tie-break order mismatch: got {got_ids}, expected {expected_ids}"
    )

    # Call again — must be identical.
    result2 = inf_mod.route("any query", top_k=2)
    assert [r.id for r in result2.routes] == got_ids, "order changed on second call"
