"""Perf gate: score-only p50/p95/p99 on the 11,922-route catalog.

A1.5 verdict: the tail-recursive scalar matmul in mind/kernels/matmul_q16.mind
saturated at ~15 ms p95 on a single i7-5930K core. This test measures the
BLAS-backed score path (mind/runtime/blas_shims_i64.c + mind/kernels/matmul_blas.mind)
under the same conditions.

Hard gate:
  * p95 < 2.0 ms — the expected post-A1.5 floor is ~1.5 – 2.0 ms on the i64
    stride-8 layout (memory-bandwidth limited at ~36 MB / 14 GB/s), well under
    the original 15 ms scalar baseline. Anything above 2 ms indicates the AVX2
    path is not engaged or a regression hit the matmul kernel.

Encode-path note:
  This test deliberately bypasses ``encode_query`` (which depends on the
  Phase-6.2 quantizer) and feeds synthetic Q16.16 query vectors directly
  into the native score path via the low-level ``_NativeRuntime`` — the
  same self-contained construction the byte-identity gate uses. The score
  path is the entire A1.5 scope; encode is tracked separately.

Skip semantics:
  * MIND_NERVE_PERF_SKIP=1     -> skip cleanly (CI without a built .so).
  * native .so unavailable     -> skip with a clear reason.

Emits a JSON-friendly report block to stdout for the orchestrator to absorb
into the commit message.
"""

from __future__ import annotations

import json
import os
import statistics
import time

import numpy as np
import pytest
from mind_nerve._native import _f32_to_q16, _NativeRuntime

# Hard gate for the score-only path. Honest budget: AVX2 FMA at 16 ops/cycle
# on a single core gives ~0.3-0.5 ms on dim=384, n_rows=11922 — but the i64
# stride-8 layout (8 bytes per Q16.16 element) puts the catalog at ~36 MB
# which hits the single-channel DDR4 bandwidth ceiling around 1.5-2.0 ms.
# Re-packing the catalog to i32 stride-4 lazily on the first call halves the
# bandwidth pressure but does not eliminate it. 2 ms is therefore a generous
# regression-detector ceiling; expected steady-state is ~1.4-1.6 ms p95.
_SCORE_P95_HARD_MS = 2.0

# Reference budget reported for context (pre-A1.5 scalar floor).
_PRE_A1_5_REF_P95_MS = 15.0

_N_WARMUP = 5
_N_MEASURE = 1000

# Synthetic catalog geometry — identical to the byte-identity gate so the
# perf path exercises the same accumulation regime as the production catalog.
_N_ROWS = 11922
_DIM = 384
_SEED = 0xA1_5_BEEF

pytestmark = pytest.mark.perf


def _make_catalog(rng: np.random.Generator) -> np.ndarray:
    """Synthetic Q16.16 catalog with shape (_N_ROWS, _DIM), seeded + L2-norm."""
    f = rng.standard_normal((_N_ROWS, _DIM)).astype(np.float32)
    norms = np.linalg.norm(f, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    f = f / norms
    return np.ascontiguousarray(_f32_to_q16(f), dtype=np.int64)


def _make_queries(rng: np.random.Generator, n: int) -> list[np.ndarray]:
    """n synthetic Q16.16 query vectors of shape (_DIM,)."""
    queries: list[np.ndarray] = []
    for _ in range(n):
        f = rng.standard_normal(_DIM).astype(np.float32)
        f = f / max(float(np.linalg.norm(f)), 1e-8)
        queries.append(np.ascontiguousarray(_f32_to_q16(f), dtype=np.int64))
    return queries


def _resolve_native_runtime() -> _NativeRuntime:
    """Return a live _NativeRuntime or skip cleanly."""
    if os.environ.get("MIND_NERVE_PERF_SKIP") == "1":
        pytest.skip("MIND_NERVE_PERF_SKIP=1 set — perf gate intentionally skipped")
    try:
        return _NativeRuntime()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"native runtime unavailable: {exc.__class__.__name__}: {exc}")


@pytest.mark.perf
def test_score_path_warm_p95_under_hard_gate() -> None:
    """Measure score-only p50/p95/p99 over 1000 queries; gate p95 < 2 ms."""
    rt = _resolve_native_runtime()
    handle = rt.init(0, 0)
    if handle == 0:
        pytest.skip("native runtime init() returned 0 — encoder handle unavailable")

    rng_cat = np.random.default_rng(_SEED)
    rng_q = np.random.default_rng(_SEED + 1)
    catalog = _make_catalog(rng_cat)
    # 64 distinct synthetic queries, cycled across the measurement window so
    # the score path sees varied operands without re-encoding cost.
    queries = _make_queries(rng_q, 64)

    # Warmup score path.
    for i in range(_N_WARMUP):
        _ = rt.score(handle, queries[i % len(queries)], catalog)

    # Measure score-only.
    samples_ms: list[float] = []
    for i in range(_N_MEASURE):
        qv = queries[i % len(queries)]
        t0 = time.perf_counter()
        _ = rt.score(handle, qv, catalog)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)

    samples_sorted = sorted(samples_ms)
    p50 = samples_sorted[len(samples_sorted) // 2]
    p95 = samples_sorted[int(len(samples_sorted) * 0.95)]
    p99 = samples_sorted[int(len(samples_sorted) * 0.99)]
    mean = statistics.mean(samples_ms)

    report = {
        "samples": _N_MEASURE,
        "catalog_rows": _N_ROWS,
        "dim": _DIM,
        "mean_ms": round(mean, 4),
        "p50_ms": round(p50, 4),
        "p95_ms": round(p95, 4),
        "p99_ms": round(p99, 4),
        "ref_pre_a1_5_p95_ms": _PRE_A1_5_REF_P95_MS,
        "hard_gate_p95_ms": _SCORE_P95_HARD_MS,
    }

    print()
    print("===== mind-nerve score-only perf report =====")
    print(json.dumps(report, indent=2))
    print()
    print(f"  pre-A1.5 reference p95 = {_PRE_A1_5_REF_P95_MS:.2f} ms (scalar tail-rec)")
    print(f"  speedup vs reference   = {_PRE_A1_5_REF_P95_MS / p95:.1f}x")
    print()

    assert p95 < _SCORE_P95_HARD_MS, (
        f"score-only p95 = {p95:.3f} ms exceeds hard gate {_SCORE_P95_HARD_MS:.2f} ms — "
        f"the AVX2 BLAS path is not engaged or a regression has landed."
    )
