"""Efficiency bench — the substrate-aware bench MIND uniquely wins.

Three measurements, none of which a BLAS-backed routing stack can offer:

  1. Cross-arch Q16.16 bit-identity (task #57): SHA-256 of the concatenated
     top-5 ``(idx, q16_score)`` stream over the 100-query deterministic
     corpus, computed on BOTH dispatch paths (``MIND_NERVE_BLAS=1`` AVX2 and
     ``=0`` scalar). Both must equal the x86 reference hash pinned in
     ``tests/python/test_blas_byte_identity.py``. This run records the x86
     reference; the printed hash is the cross-arch oracle for future
     ARM / CUDA / photonic comparison.

  2. L1 / L2 / L∞ metric matrix: top-5 under each reduction (L2 = dot, the
     current cosine flavor; L1 = sum-abs; L∞ = max-abs), computed in numpy
     from the same Q16.16 catalog. Reports L1-vs-L2 and L∞-vs-L2 top-5
     Jaccard + rank overlap. This feeds the substrate-metric story; it is a
     *measurement* of metric-flavor behaviour, not a new MIND kernel.

  3. Joules/query estimate (best-effort): x86 RAPL via
     ``/sys/class/powercap/intel-rapl:0/energy_uj`` delta across the
     1000-query mind-blas-A run ÷ 1000. If RAPL is unreadable (perms) the
     value is ``null`` with ``reason: rapl_unreadable`` — never fabricated.
     The nvidia-smi GPU path is PENDING (no GPU score path yet).

Outputs:
  * ``bench_efficiency.json`` next to this file (machine-readable)
  * a human-readable table to stdout

Run modes mirror ``bench_criterion.py``: gated pytest entry point +
standalone ``python tests/perf/bench_efficiency.py``.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Make ``_bench_common`` and the byte-identity reference importable both
# under pytest (no tests-package __init__) and standalone.
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from _bench_common import (  # noqa: E402
    _DIM,
    _N_HASH_QUERIES,
    _N_ROWS,
    _bind_blas_dispatch,
    _catalog_rng,
    _make_catalog,
    _make_queries,
    _query_rng,
    _resolve_native_runtime,
    _topk_from_scores,
    _try_native_runtime,
)


def _reference_hash() -> str:
    """The pinned x86/AVX2 cross-arch reference hash.

    Loaded from ``tests/python/test_blas_byte_identity.py`` so there is a
    single source of truth. Falls back to a direct text scrape when the
    test module is not importable (standalone run without a tests package).
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
    try:
        from test_blas_byte_identity import REFERENCE_HASH_X86_AVX2  # noqa: PLC0415

        return REFERENCE_HASH_X86_AVX2
    except Exception:  # noqa: BLE001
        pass

    src = (Path(__file__).parent.parent / "python" / "test_blas_byte_identity.py").read_text()
    for line in src.splitlines():
        if line.startswith("REFERENCE_HASH_X86_AVX2"):
            return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("could not resolve REFERENCE_HASH_X86_AVX2")


REFERENCE_HASH_X86_AVX2 = _reference_hash()

_TOP_K = 5
_N_OVERLAP_QUERIES = 100
_N_RAPL_QUERIES = 1000
_N_DISTINCT_QUERIES = 64
_RAPL_ENERGY_PATH = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
# RAPL max_energy_range wraps; the i7-5930K package domain wraps near this.
_RAPL_WRAP_PATH = Path("/sys/class/powercap/intel-rapl:0/max_energy_range_uj")

_JSON_PATH = Path(__file__).parent / "bench_efficiency.json"


# ---------------------------------------------------------------------------
# 1. Cross-arch Q16.16 bit-identity (task #57)
# ---------------------------------------------------------------------------


def _hash_topk_stream(
    rt: Any,
    handle: int,
    catalog: np.ndarray,
    queries: list[np.ndarray],
    dispatch: dict[str, Any],
    set_avx2: int,
) -> str | None:
    """SHA-256 of the concatenated top-5 (idx, score) stream on one path."""
    prev = dispatch["set"](set_avx2)
    if dispatch["get"]() != set_avx2:
        dispatch["set"](prev)
        return None
    try:
        hasher = hashlib.sha256()
        for qv in queries:
            scores = rt.score(handle, qv, catalog)
            idx, top_sc = _topk_from_scores(scores, _TOP_K)
            hasher.update(idx.astype(np.int64).tobytes())
            hasher.update(top_sc.astype(np.int64).tobytes())
        return hasher.hexdigest()
    finally:
        dispatch["set"](prev)


# ---------------------------------------------------------------------------
# 2. L1 / L2 / L∞ metric matrix (numpy, from the same Q16.16 catalog)
# ---------------------------------------------------------------------------


def _topk_idx(scores: np.ndarray, k: int) -> np.ndarray:
    cand = np.argpartition(-scores, k - 1)[:k]
    return cand[np.argsort(-scores[cand], kind="stable")].astype(np.int64)


def _metric_matrix(
    catalog: np.ndarray,
    queries: list[np.ndarray],
) -> dict[str, Any]:
    """Top-5 overlap of L1-vs-L2 and L∞-vs-L2 across the query corpus.

    L2 = dot product (the current cosine flavor on L2-normalised vectors).
    L1 = -sum|cat - q|  (Manhattan similarity; larger = closer).
    L∞ = -max|cat - q|  (Chebyshev similarity; larger = closer).
    """
    cat = catalog.astype(np.float64)
    l1_jacc: list[float] = []
    linf_jacc: list[float] = []
    l1_rank: list[float] = []
    linf_rank: list[float] = []

    for q in queries:
        qf = q.astype(np.float64)
        s_l2 = cat @ qf
        s_l1 = -np.sum(np.abs(cat - qf), axis=1)
        s_linf = -np.max(np.abs(cat - qf), axis=1)

        t_l2 = _topk_idx(s_l2, _TOP_K)
        t_l1 = _topk_idx(s_l1, _TOP_K)
        t_linf = _topk_idx(s_linf, _TOP_K)

        set_l2 = set(t_l2.tolist())
        set_l1 = set(t_l1.tolist())
        set_linf = set(t_linf.tolist())

        def _jaccard(a: set[int], b: set[int]) -> float:
            return len(a & b) / len(a | b) if (a | b) else 1.0

        l1_jacc.append(_jaccard(set_l1, set_l2))
        linf_jacc.append(_jaccard(set_linf, set_l2))
        l1_rank.append(len(set_l1 & set_l2) / _TOP_K)
        linf_rank.append(len(set_linf & set_l2) / _TOP_K)

    return {
        "n_queries": len(queries),
        "top_k": _TOP_K,
        "l2_reference": "dot-product (cosine on L2-normalised vectors)",
        "l1_vs_l2": {
            "mean_jaccard": round(float(np.mean(l1_jacc)), 4),
            "mean_rank_overlap_pct": round(float(np.mean(l1_rank)) * 100.0, 2),
        },
        "linf_vs_l2": {
            "mean_jaccard": round(float(np.mean(linf_jacc)), 4),
            "mean_rank_overlap_pct": round(float(np.mean(linf_rank)) * 100.0, 2),
        },
    }


# ---------------------------------------------------------------------------
# 3. Joules/query (best-effort x86 RAPL)
# ---------------------------------------------------------------------------


def _read_rapl_uj() -> int | None:
    try:
        return int(_RAPL_ENERGY_PATH.read_text().strip())
    except Exception:  # noqa: BLE001
        return None


def _rapl_joules_per_query(
    rt: Any,
    handle: int,
    catalog: np.ndarray,
    queries: list[np.ndarray],
    dispatch: dict[str, Any],
) -> dict[str, Any]:
    """RAPL energy delta over a 1000-query mind-blas-A run ÷ 1000."""
    e0 = _read_rapl_uj()
    if e0 is None:
        return {"joules_per_query": None, "reason": "rapl_unreadable"}

    prev = dispatch["set"](1)
    if dispatch["get"]() != 1:
        dispatch["set"](prev)
        return {"joules_per_query": None, "reason": "avx2_path_unavailable"}
    try:
        t0 = time.perf_counter()
        for i in range(_N_RAPL_QUERIES):
            rt.score(handle, queries[i % len(queries)], catalog)
        wall_s = time.perf_counter() - t0
        e1 = _read_rapl_uj()
    finally:
        dispatch["set"](prev)

    if e1 is None:
        return {"joules_per_query": None, "reason": "rapl_unreadable"}

    delta_uj = e1 - e0
    if delta_uj < 0:
        try:
            wrap = int(_RAPL_WRAP_PATH.read_text().strip())
            delta_uj += wrap
        except Exception:  # noqa: BLE001
            return {"joules_per_query": None, "reason": "rapl_counter_wrap"}

    joules = delta_uj / 1e6
    return {
        "joules_per_query": round(joules / _N_RAPL_QUERIES, 6),
        "domain": "intel-rapl:0 (package)",
        "queries": _N_RAPL_QUERIES,
        "total_joules": round(joules, 3),
        "wall_s": round(wall_s, 4),
        "reason": None,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _run(rt: Any) -> dict[str, Any]:
    handle = rt.init(0, 0)
    if handle == 0:
        raise RuntimeError("native runtime init() returned 0")

    try:
        catalog = _make_catalog(_catalog_rng())
        dispatch = _bind_blas_dispatch(rt)

        # 1. cross-arch bit-identity over the 100-query hash corpus.
        hash_queries = _make_queries(_query_rng(), _N_HASH_QUERIES)
        h_avx2 = _hash_topk_stream(rt, handle, catalog, hash_queries, dispatch, 1)
        h_scalar = _hash_topk_stream(rt, handle, catalog, hash_queries, dispatch, 0)

        bit_identity = {
            "task": "#57 cross-arch Q16.16 bit-identity",
            "n_queries": _N_HASH_QUERIES,
            "sha256_avx2": h_avx2,
            "sha256_scalar": h_scalar,
            "reference_x86_avx2": REFERENCE_HASH_X86_AVX2,
            "avx2_matches_reference": h_avx2 == REFERENCE_HASH_X86_AVX2,
            "scalar_matches_reference": h_scalar == REFERENCE_HASH_X86_AVX2,
            "avx2_matches_scalar": h_avx2 == h_scalar,
            "note": (
                "this hash is the cross-arch oracle for future ARM / CUDA / "
                "photonic backends — they must reproduce it byte-for-byte"
            ),
        }

        # 2. L1/L2/L∞ metric matrix over the 100-query overlap corpus.
        overlap_queries = _make_queries(_query_rng(), _N_OVERLAP_QUERIES)
        metrics = _metric_matrix(catalog, overlap_queries)

        # 3. joules/query (best-effort RAPL) on the mind-blas-A path.
        rapl_queries = _make_queries(_query_rng(), _N_DISTINCT_QUERIES)
        joules = _rapl_joules_per_query(rt, handle, catalog, rapl_queries, dispatch)
        joules_gpu = {
            "joules_per_query": None,
            "reason": "no_gpu_score_path",
            "status": "PENDING",
        }

        return {
            "bench": "efficiency",
            "workload": {
                "catalog_rows": _N_ROWS,
                "dim": _DIM,
                "dtype": "q16.16-i64-stride8",
                "top_k": _TOP_K,
            },
            "cross_arch_bit_identity": bit_identity,
            "metric_matrix": metrics,
            "joules_per_query_cpu_rapl": joules,
            "joules_per_query_gpu": joules_gpu,
        }
    finally:
        rt.free(handle)


def _print_table(report: dict[str, Any]) -> None:
    print()
    print("===== mind-nerve efficiency bench =====")
    bi = report["cross_arch_bit_identity"]
    print()
    print("  [1] cross-arch Q16.16 bit-identity (task #57)")
    print(f"      reference x86/AVX2 : {bi['reference_x86_avx2']}")
    print(f"      AVX2 path          : {bi['sha256_avx2']}")
    print(f"      scalar path        : {bi['sha256_scalar']}")
    print(
        f"      AVX2==scalar={bi['avx2_matches_scalar']}  "
        f"AVX2==ref={bi['avx2_matches_reference']}  "
        f"scalar==ref={bi['scalar_matches_reference']}"
    )
    mm = report["metric_matrix"]
    print()
    print(f"  [2] metric matrix (top-{mm['top_k']}, {mm['n_queries']} queries, L2=dot)")
    print(
        f"      L1  vs L2: jaccard={mm['l1_vs_l2']['mean_jaccard']}  "
        f"rank-overlap={mm['l1_vs_l2']['mean_rank_overlap_pct']}%"
    )
    print(
        f"      L∞ vs L2: jaccard={mm['linf_vs_l2']['mean_jaccard']}  "
        f"rank-overlap={mm['linf_vs_l2']['mean_rank_overlap_pct']}%"
    )
    jp = report["joules_per_query_cpu_rapl"]
    print()
    print("  [3] joules/query")
    if jp.get("joules_per_query") is None:
        print(f"      CPU (RAPL): null  (reason: {jp.get('reason')})")
    else:
        print(
            f"      CPU (RAPL): {jp['joules_per_query']} J/query "
            f"(total {jp['total_joules']} J over {jp['queries']} queries)"
        )
    jg = report["joules_per_query_gpu"]
    print(f"      GPU       : {jg['status']} (reason: {jg['reason']})")
    print()


def _emit_json(report: dict[str, Any]) -> None:
    _JSON_PATH.write_text(json.dumps(report, indent=2) + "\n")


@pytest.mark.perf
def test_efficiency_bench() -> None:
    """Run the efficiency bench; assert both dispatch paths hit the reference hash."""
    rt = _resolve_native_runtime()
    report = _run(rt)
    _print_table(report)
    _emit_json(report)

    bi = report["cross_arch_bit_identity"]
    if bi["sha256_avx2"] is None or bi["sha256_scalar"] is None:
        pytest.skip("one dispatch path unavailable — bit-identity needs both")
    assert bi["avx2_matches_scalar"], (
        f"AVX2 vs scalar top-5 hash diverged: {bi['sha256_avx2']} != {bi['sha256_scalar']}"
    )
    assert bi["avx2_matches_reference"], (
        f"AVX2 top-5 hash drifted from x86 reference: "
        f"{bi['sha256_avx2']} != {bi['reference_x86_avx2']}"
    )
    assert bi["scalar_matches_reference"], (
        f"scalar top-5 hash drifted from x86 reference: "
        f"{bi['sha256_scalar']} != {bi['reference_x86_avx2']}"
    )


def main() -> int:
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
