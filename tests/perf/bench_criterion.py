"""Criterion speed bench — score-only, head-to-head against BLAS.

Single-thread, warm, synthetic 11,922 × 384 Q16.16 catalog + 1000 queries
(64 distinct, cycled). Measures the score-only path for every backend that
is runnable today:

  * MIND + mind-blas-A (AVX2)  — the live path (``MIND_NERVE_BLAS=1``)
  * MIND + scalar (oracle)     — forced via the test-only dispatch hook
  * numpy + BLAS reference     — idealised f32 lower bound
  * pytorch (if importable)    — optional; skipped cleanly if unavailable

Encode-path and end-to-end are **deliberately out of scope** here: they are
blocked on the Phase 6.2 full-catalog run with the real Phase 1 checkpoint,
which is externally unavailable. The JSON and the human table both carry an
explicit ``encode_path_pending`` stanza so the gap is honest, not hidden.

Outputs:
  * ``bench_criterion.json`` next to this file (machine-readable)
  * a human-readable table to stdout

Gate (pytest entry point only): hard-fail iff mind-blas-A p95 > 2.0 ms.
This is a regression detector — the expected steady state is ~1.6 ms.

Run modes:
  * ``pytest tests/perf/bench_criterion.py``        (gated; self-skips under
    ``MIND_NERVE_PERF_SKIP=1`` or when the native .so is unavailable)
  * ``python tests/perf/bench_criterion.py``        (standalone; prints the
    table + writes the JSON, no gate)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Make ``_bench_common`` importable both under pytest (no tests-package
# __init__) and as a standalone ``python tests/perf/bench_criterion.py``.
sys.path.insert(0, str(Path(__file__).parent))

from _bench_common import (  # noqa: E402
    _DIM,
    _N_ROWS,
    _bind_blas_dispatch,
    _catalog_rng,
    _make_catalog,
    _make_queries,
    _percentiles,
    _query_rng,
    _resolve_native_runtime,
    _try_native_runtime,
)

# Hard regression gate for the mind-blas-A score-only path.
_MIND_BLAS_A_P95_HARD_MS = 2.0
# Pre-A1.5 scalar tail-recursive reference (context only).
_PRE_A1_5_REF_P95_MS = 15.0

_N_WARMUP = 5
_N_MEASURE = 1000
_N_DISTINCT_QUERIES = 64
_TOP_K = 5

_JSON_PATH = Path(__file__).parent / "bench_criterion.json"

# Honest pending stanza — encode path is blocked on the Phase 6.2 full-catalog
# run with the real Phase 1 checkpoint (externally unavailable).
_ENCODE_PENDING = {
    "status": "PENDING",
    "scope": "encode-only + end-to-end",
    "reason": (
        "blocked on the Phase 6.2 full-catalog run with the real Phase 1 "
        "checkpoint (externally unavailable). Score-only is the entire "
        "measurable scope today; encode is tracked separately."
    ),
}


def _peak_rss_mb() -> float | None:
    """Best-effort peak RSS in MiB (ru_maxrss is KiB on Linux)."""
    try:
        import resource

        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except Exception:  # noqa: BLE001
        return None


def _measure_native(
    rt: Any,
    handle: int,
    catalog: np.ndarray,
    queries: list[np.ndarray],
    set_avx2: int,
    dispatch: dict[str, Any],
) -> dict[str, Any] | None:
    """Measure score-only latency for a native dispatch path.

    ``set_avx2`` selects the path: 1 = AVX2 (mind-blas-A), 0 = scalar oracle.
    Returns ``None`` if the requested path is unavailable on this host.
    """
    prev = dispatch["set"](set_avx2)
    if dispatch["get"]() != set_avx2:
        dispatch["set"](prev)
        return None
    try:
        for i in range(_N_WARMUP):
            rt.score(handle, queries[i % len(queries)], catalog)

        samples_ms: list[float] = []
        t_start = time.perf_counter()
        for i in range(_N_MEASURE):
            qv = queries[i % len(queries)]
            t0 = time.perf_counter()
            rt.score(handle, qv, catalog)
            samples_ms.append((time.perf_counter() - t0) * 1000.0)
        wall_s = time.perf_counter() - t_start
    finally:
        dispatch["set"](prev)

    stats = _percentiles(samples_ms)
    stats["qps"] = round(_N_MEASURE / wall_s, 1)
    stats["samples"] = _N_MEASURE
    return stats


def _measure_numpy(
    catalog: np.ndarray,
    queries: list[np.ndarray],
) -> dict[str, Any]:
    """Idealised numpy+BLAS f32 score-only lower bound."""
    cat_f = catalog.astype(np.float32)
    qs_f = [q.astype(np.float32) for q in queries]

    for i in range(_N_WARMUP):
        _ = cat_f @ qs_f[i % len(qs_f)]

    samples_ms: list[float] = []
    t_start = time.perf_counter()
    for i in range(_N_MEASURE):
        qv = qs_f[i % len(qs_f)]
        t0 = time.perf_counter()
        _ = cat_f @ qv
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    wall_s = time.perf_counter() - t_start

    stats = _percentiles(samples_ms)
    stats["qps"] = round(_N_MEASURE / wall_s, 1)
    stats["samples"] = _N_MEASURE
    return stats


def _measure_pytorch(
    catalog: np.ndarray,
    queries: list[np.ndarray],
) -> dict[str, Any] | None:
    """Optional pytorch CPU score-only path. Returns None if unavailable."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    try:
        torch.set_num_threads(1)
        cat_t = torch.from_numpy(catalog.astype(np.float32))
        qs_t = [torch.from_numpy(q.astype(np.float32)) for q in queries]

        for i in range(_N_WARMUP):
            _ = torch.mv(cat_t, qs_t[i % len(qs_t)])

        samples_ms: list[float] = []
        t_start = time.perf_counter()
        for i in range(_N_MEASURE):
            qv = qs_t[i % len(qs_t)]
            t0 = time.perf_counter()
            _ = torch.mv(cat_t, qv)
            samples_ms.append((time.perf_counter() - t0) * 1000.0)
        wall_s = time.perf_counter() - t_start
    except Exception:  # noqa: BLE001
        return None

    stats = _percentiles(samples_ms)
    stats["qps"] = round(_N_MEASURE / wall_s, 1)
    stats["samples"] = _N_MEASURE
    return stats


def _run(rt: Any) -> dict[str, Any]:
    """Run the full speed-bench matrix and return the report dict."""
    handle = rt.init(0, 0)
    if handle == 0:
        raise RuntimeError("native runtime init() returned 0")

    try:
        catalog = _make_catalog(_catalog_rng())
        queries = _make_queries(_query_rng(), _N_DISTINCT_QUERIES)
        dispatch = _bind_blas_dispatch(rt)

        backends: dict[str, Any] = {}
        backends["mind_blas_a_avx2"] = _measure_native(rt, handle, catalog, queries, 1, dispatch)
        backends["mind_scalar_oracle"] = _measure_native(rt, handle, catalog, queries, 0, dispatch)
        backends["numpy_blas_ref"] = _measure_numpy(catalog, queries)
        backends["pytorch_cpu"] = _measure_pytorch(catalog, queries)

        report: dict[str, Any] = {
            "bench": "criterion-speed",
            "workload": {
                "catalog_rows": _N_ROWS,
                "dim": _DIM,
                "dtype": "q16.16-i64-stride8",
                "queries": _N_MEASURE,
                "distinct_queries": _N_DISTINCT_QUERIES,
                "top_k": _TOP_K,
                "threads": 1,
                "scope": "score-only",
            },
            "backends": backends,
            "rss_peak_mb": _peak_rss_mb(),
            "pre_a1_5_ref_p95_ms": _PRE_A1_5_REF_P95_MS,
            "hard_gate_mind_blas_a_p95_ms": _MIND_BLAS_A_P95_HARD_MS,
            "encode_path_pending": _ENCODE_PENDING,
        }

        mba = backends.get("mind_blas_a_avx2")
        npr = backends.get("numpy_blas_ref")
        if mba and npr:
            report["mind_blas_a_vs_numpy_blas_p95_ratio"] = round(mba["p95_ms"] / npr["p95_ms"], 2)
            report["mind_blas_a_speedup_vs_pre_a1_5"] = round(
                _PRE_A1_5_REF_P95_MS / mba["p95_ms"], 1
            )
        return report
    finally:
        rt.free(handle)


def _print_table(report: dict[str, Any]) -> None:
    """Human-readable table to stdout."""
    print()
    print("===== mind-nerve criterion speed bench (score-only, 1-thread, warm) =====")
    w = report["workload"]
    print(
        f"  catalog {w['catalog_rows']}x{w['dim']} {w['dtype']} · "
        f"{w['queries']} queries ({w['distinct_queries']} distinct) · top_k={w['top_k']}"
    )
    print()
    header = f"  {'backend':<22} {'p50 ms':>9} {'p95 ms':>9} {'p99 ms':>9} {'QPS':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    labels = {
        "mind_blas_a_avx2": "MIND+mind-blas-A",
        "mind_scalar_oracle": "MIND+scalar(oracle)",
        "numpy_blas_ref": "numpy+BLAS(ref)",
        "pytorch_cpu": "pytorch(cpu)",
    }
    for key, label in labels.items():
        b = report["backends"].get(key)
        if b is None:
            print(f"  {label:<22} {'—  (unavailable / skipped)':>40}")
            continue
        print(
            f"  {label:<22} {b['p50_ms']:>9.4f} {b['p95_ms']:>9.4f} "
            f"{b['p99_ms']:>9.4f} {b['qps']:>9.1f}"
        )
    print()
    if "mind_blas_a_vs_numpy_blas_p95_ratio" in report:
        print(
            f"  mind-blas-A p95 / numpy+BLAS p95 = {report['mind_blas_a_vs_numpy_blas_p95_ratio']}x"
        )
        print(
            f"  mind-blas-A speedup vs pre-A1.5 scalar = "
            f"{report['mind_blas_a_speedup_vs_pre_a1_5']}x"
        )
    rss = report.get("rss_peak_mb")
    print(f"  peak RSS = {rss} MiB" if rss is not None else "  peak RSS = unavailable")
    ep = report["encode_path_pending"]
    print()
    print(f"  encode path: {ep['status']} ({ep['scope']}) — {ep['reason']}")
    print()


def _emit_json(report: dict[str, Any]) -> None:
    _JSON_PATH.write_text(json.dumps(report, indent=2) + "\n")


@pytest.mark.perf
def test_criterion_speed_bench() -> None:
    """Run the speed bench; gate mind-blas-A p95 < 2 ms."""
    rt = _resolve_native_runtime()
    report = _run(rt)
    _print_table(report)
    _emit_json(report)

    mba = report["backends"].get("mind_blas_a_avx2")
    if mba is None:
        pytest.skip("mind-blas-A AVX2 path unavailable on this host")
    assert mba["p95_ms"] < _MIND_BLAS_A_P95_HARD_MS, (
        f"mind-blas-A score-only p95 = {mba['p95_ms']:.3f} ms exceeds the "
        f"{_MIND_BLAS_A_P95_HARD_MS:.2f} ms hard gate — the AVX2 path is not "
        f"engaged or a regression has landed."
    )


def main() -> int:
    """Standalone entry point — no gate, just measure + emit."""
    rt = _try_native_runtime()
    if rt is None:
        print("native runtime unavailable (or MIND_NERVE_PERF_SKIP=1) — nothing to bench")
        return 0
    report = _run(rt)
    _print_table(report)
    _emit_json(report)
    print(f"  wrote {_JSON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
