"""Shared seeded synthetic catalog/query builders for the perf benches.

Single source of truth for the synthetic Q16.16 workload geometry used by
``bench_criterion.py`` (speed bench), ``bench_efficiency.py`` (efficiency
bench), ``test_score_latency.py`` and ``test_blas_byte_identity.py``.

The geometry mirrors the live STARGA v1.0 catalog (11,922 routes × 384 dim,
Q16.16, i64 stride-8 heap layout) closely enough that the reduction exercises
the same accumulation regime — without depending on any externally
unavailable checkpoint artifact.

Determinism contract:
  * ``_make_catalog`` / ``_make_queries`` are pure functions of the RNG state.
  * The catalog RNG is seeded with ``_SEED``; the query RNG with ``_SEED + 1``.
  * Values are L2-normalised in f32 then quantised to Q16.16 int64.

This is the exact ``_make_catalog`` / ``_make_queries`` pattern previously
duplicated in the byte-identity and score-latency tests, factored out so the
three perf files share one definition.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np
import pytest
from mind_nerve._native import _f32_to_q16, _NativeRuntime

# ---------------------------------------------------------------------------
# Synthetic workload geometry — frozen; changing any of these legitimately
# invalidates the cross-arch reference hash in test_blas_byte_identity.py.
# ---------------------------------------------------------------------------
_N_ROWS = 11922
_DIM = 384
_TOP_K = 5
_SEED = 0xA1_5_BEEF

# Hash corpus size (cross-arch Q16.16 bit-identity oracle, task #57).
_N_HASH_QUERIES = 100


def _make_catalog(rng: np.random.Generator) -> np.ndarray:
    """Synthetic Q16.16 catalog with shape (_N_ROWS, _DIM).

    Generated from a seeded RNG so every host produces identical bytes.
    Values are L2-normalised in f32 then quantised — this matches the live
    catalog's value distribution closely enough for the reduction to exercise
    the same accumulation regime.
    """
    f = rng.standard_normal((_N_ROWS, _DIM)).astype(np.float32)
    norms = np.linalg.norm(f, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    f = f / norms
    return np.ascontiguousarray(_f32_to_q16(f), dtype=np.int64)


def _make_queries(rng: np.random.Generator, n: int) -> list[np.ndarray]:
    """``n`` synthetic Q16.16 query vectors of shape (_DIM,)."""
    queries: list[np.ndarray] = []
    for _ in range(n):
        f = rng.standard_normal(_DIM).astype(np.float32)
        f = f / max(float(np.linalg.norm(f)), 1e-8)
        queries.append(np.ascontiguousarray(_f32_to_q16(f), dtype=np.int64))
    return queries


def _catalog_rng() -> np.random.Generator:
    """Seeded RNG for the catalog (stable across hosts)."""
    return np.random.default_rng(_SEED)


def _query_rng() -> np.random.Generator:
    """Seeded RNG for the query stream (stable across hosts)."""
    return np.random.default_rng(_SEED + 1)


def _resolve_native_runtime() -> _NativeRuntime:
    """Return a live ``_NativeRuntime`` or skip cleanly.

    Honours ``MIND_NERVE_PERF_SKIP=1`` (CI without a built .so) and skips
    with a clear reason if the native library is unavailable.
    """
    if os.environ.get("MIND_NERVE_PERF_SKIP") == "1":
        pytest.skip("MIND_NERVE_PERF_SKIP=1 set — perf bench intentionally skipped")
    try:
        return _NativeRuntime()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"native runtime unavailable: {exc.__class__.__name__}: {exc}")


def _try_native_runtime() -> _NativeRuntime | None:
    """Return a live ``_NativeRuntime`` or ``None`` (no pytest.skip).

    Used by the standalone ``python tests/perf/bench_*.py`` entry points
    where pytest skip semantics are unavailable.
    """
    if os.environ.get("MIND_NERVE_PERF_SKIP") == "1":
        return None
    try:
        return _NativeRuntime()
    except Exception:  # noqa: BLE001
        return None


def _bind_blas_dispatch(rt: _NativeRuntime) -> dict[str, ctypes._FuncPointer]:
    """Bind the BLAS dispatcher get/set hooks exported by the .so."""
    lib = rt._lib
    get_fn = lib.__mind_nerve_blas_get_use_avx2
    get_fn.restype = ctypes.c_int
    get_fn.argtypes = []
    set_fn = lib.__mind_nerve_blas_set_use_avx2
    set_fn.restype = ctypes.c_int
    set_fn.argtypes = [ctypes.c_int]
    return {"get": get_fn, "set": set_fn}


def _topk_from_scores(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Stable top-K: argpartition + descending stable sort on score."""
    k = min(k, scores.shape[0])
    cand = np.argpartition(-scores, k - 1)[:k]
    order = np.argsort(-scores[cand], kind="stable")
    idx = cand[order]
    return idx.astype(np.int64), scores[idx]


def _percentiles(samples_ms: list[float]) -> dict[str, float]:
    """p50/p95/p99/mean from a list of millisecond samples."""
    s = sorted(samples_ms)
    n = len(s)
    return {
        "mean_ms": round(sum(s) / n, 4),
        "p50_ms": round(s[n // 2], 4),
        "p95_ms": round(s[min(int(n * 0.95), n - 1)], 4),
        "p99_ms": round(s[min(int(n * 0.99), n - 1)], 4),
    }
