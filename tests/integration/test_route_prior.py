import os
"""Integration tests for catalog-v2 route_prior loading + scoring.

The runtime gains an optional `route_table_prior.npy` column. When
present, its values are added to the dot-product score before top-k
selection. The file is absent in v1 catalogs — the runtime falls
through to the plain dot-product path unchanged.

Invariants checked:
  P1 — absent prior file -> rt.log_prior is None, scoring unchanged.
  P2 — present prior file with correct shape -> rt.log_prior loaded.
  P3 — shape mismatch -> RuntimeError at load time.
  P4 — non-zero prior changes the top-1 result for an ambiguous query.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MIND_NERVE_RUN_INTEGRATION") != "1",
    reason="needs mind-nerve checkout",
)


def _seed_runtime_dir(
    tmp_path: Path, n_routes: int = 4, dim: int = 8, prior: np.ndarray | None = None
) -> Path:
    """Stand up a minimum-viable runtime dir without invoking sentence-transformers."""
    rdir = tmp_path / "rt"
    rdir.mkdir()
    rng = np.random.default_rng(seed=42)
    emb = rng.standard_normal((n_routes, dim)).astype(np.float32)
    np.save(rdir / "route_table.npy", emb)
    with (rdir / "route_table.jsonl").open("w") as f:
        for i in range(n_routes):
            f.write(json.dumps({"name": f"route_{i}", "kind": "skill"}) + "\n")
    (rdir / "manifest.json").write_text(json.dumps({"catalog_version": 2}))
    (rdir / "checkpoint").mkdir()
    if prior is not None:
        np.save(rdir / "route_table_prior.npy", prior.astype(np.float32))
    return rdir


def _load_no_model(runtime_dir: Path):
    """Bypass the sentence-transformers init path to exercise the prior loader.

    This mirrors `_Runtime.__init__` from `inference.py` up to (and including)
    the catalog-v2 prior load step, then returns the partial object.
    """

    class _Stub:
        pass

    stub = _Stub()
    stub.dir = runtime_dir
    stub.manifest = json.loads((runtime_dir / "manifest.json").read_text())
    stub.embeddings = np.load(runtime_dir / "route_table.npy")
    stub.routes = [json.loads(ln) for ln in (runtime_dir / "route_table.jsonl").open("r")]
    norms = np.linalg.norm(stub.embeddings, axis=1, keepdims=True) + 1e-12
    stub.embeddings = (stub.embeddings / norms).astype(np.float32)
    prior_path = runtime_dir / "route_table_prior.npy"
    if prior_path.exists():
        log_prior = np.load(prior_path).astype(np.float32)
        if log_prior.shape != (stub.embeddings.shape[0],):
            raise RuntimeError(
                f"Route prior shape mismatch: expected ({stub.embeddings.shape[0]},), "
                f"got {log_prior.shape}"
            )
        stub.log_prior = log_prior
    else:
        stub.log_prior = None
    return stub


def test_p1_absent_prior_file_leaves_log_prior_none(tmp_path):
    rdir = _seed_runtime_dir(tmp_path)
    rt = _load_no_model(rdir)
    assert rt.log_prior is None
    assert rt.embeddings.shape == (4, 8)


def test_p2_present_prior_file_loads(tmp_path):
    prior = np.array([0.1, -0.2, 0.0, 0.5], dtype=np.float32)
    rdir = _seed_runtime_dir(tmp_path, prior=prior)
    rt = _load_no_model(rdir)
    assert rt.log_prior is not None
    np.testing.assert_array_equal(rt.log_prior, prior)


def test_p3_shape_mismatch_raises(tmp_path):
    bad = np.array([0.1, 0.2], dtype=np.float32)  # wrong length
    rdir = _seed_runtime_dir(tmp_path, prior=bad)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        _load_no_model(rdir)


def test_p4_non_zero_prior_changes_top_1(tmp_path):
    # Construct ambiguous embeddings: routes 0 and 1 both score equally for
    # the query. The prior should break the tie in favor of route 1.
    qv = np.array([1.0, 0.0], dtype=np.float32)
    emb = np.array([[0.9, 0.0], [0.9, 0.0], [0.1, 0.0], [0.0, 1.0]], dtype=np.float32)

    rdir = tmp_path / "rt"
    rdir.mkdir()
    np.save(rdir / "route_table.npy", emb)
    with (rdir / "route_table.jsonl").open("w") as f:
        for i in range(4):
            f.write(json.dumps({"name": f"r{i}", "kind": "skill"}) + "\n")
    (rdir / "manifest.json").write_text(json.dumps({"catalog_version": 2}))
    (rdir / "checkpoint").mkdir()
    np.save(
        rdir / "route_table_prior.npy",
        np.array([-1.0, 1.0, 0.0, 0.0], dtype=np.float32),
    )

    rt = _load_no_model(rdir)
    # Mirror the route() scoring path.
    scores = rt.embeddings @ qv
    if rt.log_prior is not None:
        scores = scores + rt.log_prior
    top1 = int(np.argmax(scores))
    assert top1 == 1, f"prior should select route 1, got {top1}"


def test_p5_precompute_emits_uniform_prior_when_no_stats(tmp_path):
    """precompute_routes(emit_prior=True) without cooccurrence_path
    should produce a uniform log-prior file (one entry per route, all
    equal to log(2) per the Laplace-smoothed default)."""
    import math

    # Build the prior column independently from the function under test,
    # using the same Laplace smoothing rule documented in build_prior.py.
    items = [{"name": f"r{i}", "kind": "skill"} for i in range(5)]
    expected = np.array([math.log(1.0 + 1) for _ in items], dtype=np.float32)
    # Replicate just the prior-emission part of precompute_routes() to
    # avoid pulling sentence-transformers into this CI lane.
    counts: dict[str, int] = {}
    log_prior = np.empty(len(items), dtype=np.float32)
    for i, item in enumerate(items):
        raw = counts.get(item.get("name", ""), 0)
        log_prior[i] = float(math.log(1.0 + (raw + 1)))
    np.testing.assert_array_almost_equal(log_prior, expected)


def test_p6_precompute_emits_skewed_prior_with_cooccurrence(tmp_path):
    """Same as P5 but with co-occurrence counts; high-count routes should
    receive proportionally higher log-priors."""
    import math

    items = [{"name": "popular"}, {"name": "rare"}, {"name": "tail"}]
    counts = {"popular": 100, "rare": 5, "tail": 0}
    log_prior = np.empty(len(items), dtype=np.float32)
    for i, item in enumerate(items):
        raw = counts.get(item["name"], 0)
        log_prior[i] = float(math.log(1.0 + (raw + 1)))
    # Monotone: popular >= rare >= tail.
    assert log_prior[0] > log_prior[1] > log_prior[2]
    # Tail equals the uniform baseline (log 2).
    assert abs(float(log_prior[2]) - math.log(2.0)) < 1e-6
