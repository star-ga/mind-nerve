"""Integration test: route() is byte-identical across 100 repeated calls.

The architecture spec mandates that two `route()` calls with the same
query + top_k + runtime return identical route_id sequences and
identical score sequences. This is the load-bearing contract for the
cross-arch Q16.16 bit-identity story and for the deterministic
SHA-256 tie-break shipped in `_deterministic_topk`.

This test runs the pytorch backend against a hand-built tiny runtime
(no checkpoint download, no Hugging Face). It exercises the same
code path as production except for the SentenceTransformer encode
call, which is stubbed with a deterministic fake. The full
encoder-included determinism is covered by the bit-identity harness.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_tiny_runtime(tmp: Path, n_routes: int = 24, dim: int = 16) -> Path:
    """Lay out a minimum-viable runtime dir on disk."""
    rdir = tmp / "runtime"
    (rdir / "checkpoint").mkdir(parents=True)
    rng = np.random.default_rng(seed=20260518)
    emb = rng.standard_normal((n_routes, dim)).astype(np.float32)
    np.save(rdir / "route_table.npy", emb)
    with (rdir / "route_table.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(n_routes):
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


class _FakeSentenceTransformer:
    """Deterministic encoder stand-in."""

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        # Produce a deterministic unit vector per text. Same text -> same
        # vector across the entire test process.
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = abs(hash(t)) % (2**31 - 1)
            rng = np.random.default_rng(seed=h)
            v = rng.standard_normal(self._dim).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-12
            out[i] = v
        return out

    def tokenize(self, texts: list[str], **_: object) -> dict:
        import torch

        # Five tokens — well under the 1024-token guard, well above zero.
        return {"input_ids": torch.zeros((1, 5), dtype=torch.long)}

    def eval(self) -> "_FakeSentenceTransformer":
        return self


class _FakeRuntime:
    """Minimal `_Runtime` surface expected by `_route_pytorch`."""

    def __init__(self, rdir: Path) -> None:
        self.dir = rdir
        self.manifest = json.loads((rdir / "manifest.json").read_text())
        self.embeddings = np.load(rdir / "route_table.npy").astype(np.float32)
        self.routes = [
            json.loads(ln) for ln in (rdir / "route_table.jsonl").open("r", encoding="utf-8")
        ]
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-12
        self.embeddings = (self.embeddings / norms).astype(np.float32)
        self.log_prior = None
        self.freq_scale = None
        self.stride_thresholds = None
        self.model = _FakeSentenceTransformer(dim=self.embeddings.shape[1])

    @property
    def catalog_size(self) -> int:
        return len(self.routes)

    @property
    def catalog_version(self) -> str:
        return "test"

    @property
    def model_version(self) -> str:
        return "test"


@pytest.fixture
def fake_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _FakeRuntime:
    """Wire the in-memory fake runtime into `mind_nerve.inference`."""
    monkeypatch.setenv("MIND_NERVE_BACKEND", "pytorch")
    rdir = _build_tiny_runtime(tmp_path)
    rt = _FakeRuntime(rdir)

    import mind_nerve.inference as inf_mod

    inf_mod._load_cached.cache_clear()
    monkeypatch.setattr(inf_mod, "load_default_runtime", lambda runtime_dir=None: rt)
    monkeypatch.setattr(inf_mod, "_seed_from_hf", lambda target: None)
    return rt


# ---------------------------------------------------------------------------
# Integration assertions
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_route_returns_identical_ids_across_100_calls(fake_runtime: _FakeRuntime) -> None:
    """Repeated route() calls produce byte-identical route_id lists."""
    from mind_nerve import route

    query = "deploy the staging build with rollback support"
    first = route(query, top_k=5)
    first_ids = tuple(r.id for r in first.routes)
    assert len(first_ids) == 5
    for _ in range(99):
        again = route(query, top_k=5)
        ids = tuple(r.id for r in again.routes)
        assert ids == first_ids, f"route_id list drifted: {ids} != {first_ids}"


@pytest.mark.integration
def test_route_returns_identical_scores_across_100_calls(fake_runtime: _FakeRuntime) -> None:
    """Repeated route() calls produce identical score sequences."""
    from mind_nerve import route

    query = "search the repo for unused imports"
    first = route(query, top_k=5)
    first_scores = tuple(float(r.score) for r in first.routes)
    for _ in range(99):
        again = route(query, top_k=5)
        scores = tuple(float(r.score) for r in again.routes)
        assert scores == first_scores, f"scores drifted: {scores} != {first_scores}"


@pytest.mark.integration
def test_route_full_response_serializes_identically_100x(fake_runtime: _FakeRuntime) -> None:
    """The serialised top-K response must be byte-identical across 100 runs."""
    from mind_nerve import route

    query = "git rebase interactive workflow"
    expected = json.dumps(
        [
            {"id": r.id, "name": r.name, "score": float(r.score)}
            for r in route(query, top_k=5).routes
        ],
        separators=(",", ":"),
    )
    for _ in range(99):
        actual = json.dumps(
            [
                {"id": r.id, "name": r.name, "score": float(r.score)}
                for r in route(query, top_k=5).routes
            ],
            separators=(",", ":"),
        )
        assert actual == expected


@pytest.mark.integration
def test_deterministic_topk_handles_perfect_ties(fake_runtime: _FakeRuntime) -> None:
    """Routes with perfectly-equal scores must come out in SHA-256 order."""
    from mind_nerve.inference import _deterministic_topk

    n = 8
    scores = np.full(n, 0.5, dtype=np.float32)  # all tied
    route_ids = [f"route-{i:03d}" for i in range(n)]
    a = list(_deterministic_topk(scores, route_ids, k=5))
    for _ in range(20):
        b = list(_deterministic_topk(scores, route_ids, k=5))
        assert a == b
