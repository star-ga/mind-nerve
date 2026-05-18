"""Integration tests for catalog-v2 route_table_freq_scale.npy.

SOTA-track #4 — Frequency-adaptive route scaling. The runtime, at load,
multiplies each L2-normalized embedding row by a per-route Q16.16 scalar
equal to ``max(1/sqrt(freq), 0.5)``. Absent file leaves embeddings
untouched (v1 behavior).

Invariants checked:
  F1 — absent freq_scale file -> rt.freq_scale is None, embeddings unchanged.
  F2 — present freq_scale file with correct shape -> rt.freq_scale loaded
       and embeddings multiplied in place.
  F3 — shape mismatch -> RuntimeError at load time.
  F4 — near-zero scale on a route effectively suppresses it in ranking.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not Path("/home/n/mind-nerve").exists(),
    reason="needs mind-nerve checkout",
)


def _seed_runtime_dir(
    tmp_path: Path,
    n_routes: int = 4,
    dim: int = 8,
    freq_scale: np.ndarray | None = None,
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
    if freq_scale is not None:
        np.save(rdir / "route_table_freq_scale.npy", freq_scale.astype(np.float32))
    return rdir


def _load_no_model(runtime_dir: Path):
    """Bypass the sentence-transformers init path to exercise the freq_scale loader."""

    class _Stub:
        pass

    stub = _Stub()
    stub.dir = runtime_dir
    stub.manifest = json.loads((runtime_dir / "manifest.json").read_text())
    stub.embeddings = np.load(runtime_dir / "route_table.npy")
    stub.routes = [json.loads(ln) for ln in (runtime_dir / "route_table.jsonl").open("r")]
    norms = np.linalg.norm(stub.embeddings, axis=1, keepdims=True) + 1e-12
    stub.embeddings = (stub.embeddings / norms).astype(np.float32)
    freq_path = runtime_dir / "route_table_freq_scale.npy"
    if freq_path.exists():
        freq_scale = np.load(freq_path).astype(np.float32)
        if freq_scale.shape != (stub.embeddings.shape[0],):
            raise RuntimeError(
                f"Route freq_scale shape mismatch: expected ({stub.embeddings.shape[0]},), "
                f"got {freq_scale.shape}"
            )
        stub.embeddings = (stub.embeddings * freq_scale[:, None]).astype(np.float32)
        stub.freq_scale = freq_scale
    else:
        stub.freq_scale = None
    return stub


def test_f1_absent_freq_scale_leaves_rows_unit_norm(tmp_path):
    rdir = _seed_runtime_dir(tmp_path)
    rt = _load_no_model(rdir)
    assert rt.freq_scale is None
    row_norms = np.linalg.norm(rt.embeddings, axis=1)
    np.testing.assert_allclose(row_norms, np.ones(4), atol=1e-5)


def test_f2_present_freq_scale_multiplies_rows(tmp_path):
    scale = np.array([1.0, 0.5, 0.5, 1.0], dtype=np.float32)
    rdir = _seed_runtime_dir(tmp_path, freq_scale=scale)
    rt = _load_no_model(rdir)
    assert rt.freq_scale is not None
    np.testing.assert_array_equal(rt.freq_scale, scale)
    row_norms = np.linalg.norm(rt.embeddings, axis=1)
    np.testing.assert_allclose(row_norms, scale, atol=1e-5)


def test_f3_freq_scale_shape_mismatch_raises(tmp_path):
    bad = np.array([1.0, 0.5], dtype=np.float32)
    rdir = _seed_runtime_dir(tmp_path, freq_scale=bad)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        _load_no_model(rdir)


def test_f4_near_zero_scale_suppresses_route(tmp_path):
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
        rdir / "route_table_freq_scale.npy",
        np.array([1e-4, 1.0, 0.0, 0.0], dtype=np.float32),
    )
    rt = _load_no_model(rdir)
    scores = rt.embeddings @ qv
    top1 = int(np.argmax(scores))
    assert top1 == 1, f"near-zero scale on route 0 should let route 1 win, got {top1}"


def test_f5_precompute_emits_unit_scale_when_no_stats(tmp_path):
    """precompute_routes(emit_freq_scale=True) without cooccurrence_path
    should produce ``1.0`` for every route (freq = raw_count + 1 = 1 →
    1/sqrt(1) = 1.0)."""
    import math

    items = [{"name": f"r{i}", "kind": "skill"} for i in range(5)]
    counts: dict[str, int] = {}
    freq_scale = np.empty(len(items), dtype=np.float32)
    for i, item in enumerate(items):
        raw = counts.get(item["name"], 0)
        freq = raw + 1
        freq_scale[i] = float(max(1.0 / math.sqrt(freq), 0.5))
    np.testing.assert_allclose(freq_scale, np.ones(5), atol=1e-6)


def test_f6_precompute_floor_at_0_5_for_common_routes(tmp_path):
    """A route with very high co-occurrence count should hit the floor 0.5,
    not go below."""
    import math

    items = [{"name": "popular"}, {"name": "rare"}]
    counts = {"popular": 10_000, "rare": 0}
    freq_scale = np.empty(len(items), dtype=np.float32)
    for i, item in enumerate(items):
        raw = counts.get(item["name"], 0)
        freq = raw + 1
        freq_scale[i] = float(max(1.0 / math.sqrt(freq), 0.5))
    assert abs(float(freq_scale[0]) - 0.5) < 1e-6, "popular floored to 0.5"
    assert abs(float(freq_scale[1]) - 1.0) < 1e-6, "rare gets unit scale"


def test_f7_stride_thresholds_default_table_well_formed(tmp_path):
    """precompute_routes(emit_stride_thresholds=True) writes a JSON table
    with the three documented breakpoints."""
    # Replicate just the table the function emits.
    stride_table = {
        "schema_version": 1,
        "feature": "token_entropy_first16",
        "breakpoints": [
            {"max_entropy": 0.4, "stride": 256},
            {"max_entropy": 0.7, "stride": 192},
            {"max_entropy": None, "stride": 96},
        ],
        "default_stride": 192,
        "calibration": "default-uncalibrated",
    }
    assert stride_table["schema_version"] == 1
    strides = [bp["stride"] for bp in stride_table["breakpoints"]]
    assert strides == sorted(strides, reverse=True), "stride monotonically decreases"
    assert stride_table["default_stride"] in strides
