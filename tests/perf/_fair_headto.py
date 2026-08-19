"""Fair head-to-head: MIND MT-gemv score path vs numpy+BLAS, thread counts pinned.

Thread policy is set by the CALLER via env BEFORE numpy imports:
  OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / MKL_NUM_THREADS.
MIND's score path spawns sysconf(_SC_NPROCESSORS_ONLN) threads (=12 here,
not affinity- or env-configurable), so the only apples-to-apples pinning is
numpy=12 (full machine, same as MIND). numpy=1 is reported for context only
and is explicitly NOT the headline.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _bench_common import (  # noqa: E402
    _catalog_rng,
    _make_catalog,
    _make_queries,
    _percentiles,
    _query_rng,
    _try_native_runtime,
)

_N_WARMUP = 20
_N_MEASURE = 2000
_N_DISTINCT = 64


def _blas_threads() -> str:
    return (
        f"OMP={os.environ.get('OMP_NUM_THREADS', '?')} "
        f"OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS', '?')} "
        f"MKL={os.environ.get('MKL_NUM_THREADS', '?')}"
    )


def _measure_mind(rt, handle, catalog, queries):
    for i in range(_N_WARMUP):
        rt.score(handle, queries[i % len(queries)], catalog)
    s = []
    t0all = time.perf_counter()
    for i in range(_N_MEASURE):
        qv = queries[i % len(queries)]
        t0 = time.perf_counter()
        rt.score(handle, qv, catalog)
        s.append((time.perf_counter() - t0) * 1000.0)
    wall = time.perf_counter() - t0all
    st = _percentiles(s)
    st["qps"] = round(_N_MEASURE / wall, 1)
    return st


def _measure_numpy(catalog, queries):
    cat_f = catalog.astype(np.float32)
    qs = [q.astype(np.float32) for q in queries]
    for i in range(_N_WARMUP):
        _ = cat_f @ qs[i % len(qs)]
    s = []
    t0all = time.perf_counter()
    for i in range(_N_MEASURE):
        qv = qs[i % len(qs)]
        t0 = time.perf_counter()
        _ = cat_f @ qv
        s.append((time.perf_counter() - t0) * 1000.0)
    wall = time.perf_counter() - t0all
    st = _percentiles(s)
    st["qps"] = round(_N_MEASURE / wall, 1)
    return st


def main() -> int:
    rt = _try_native_runtime()
    if rt is None:
        print("native runtime unavailable — cannot bench")
        return 1
    handle = rt.init(0, 0)
    if handle == 0:
        print("init() returned 0")
        return 1
    try:
        catalog = _make_catalog(_catalog_rng())
        queries = _make_queries(_query_rng(), _N_DISTINCT)
        print(
            f"catalog {catalog.shape} dtype={catalog.dtype}  queries={_N_MEASURE} ({_N_DISTINCT} distinct)"
        )
        print(
            f"MIND threads = sysconf(_SC_NPROCESSORS_ONLN) = {os.sysconf('SC_NPROCESSORS_ONLN')} (fixed)"
        )
        print(f"numpy BLAS thread env: {_blas_threads()}")
        print()
        m = _measure_mind(rt, handle, catalog, queries)
        n = _measure_numpy(catalog, queries)
        hdr = f"  {'backend':<28}{'p50':>9}{'p95':>9}{'p99':>9}{'mean':>9}{'QPS':>10}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        print(
            f"  {'MIND MT-gemv Q16.16 (12t)':<28}{m['p50_ms']:>9.4f}{m['p95_ms']:>9.4f}{m['p99_ms']:>9.4f}{m['mean_ms']:>9.4f}{m['qps']:>10.1f}"
        )
        print(
            f"  {'numpy+BLAS f32 matvec':<28}{n['p50_ms']:>9.4f}{n['p95_ms']:>9.4f}{n['p99_ms']:>9.4f}{n['mean_ms']:>9.4f}{n['qps']:>10.1f}"
        )
        print()
        print(f"  p50 ratio  MIND/numpy = {m['p50_ms'] / n['p50_ms']:.2f}x   (>1 = MIND slower)")
        print(f"  p95 ratio  MIND/numpy = {m['p95_ms'] / n['p95_ms']:.2f}x")
        print(f"  p50 speedup numpy/MIND = {n['p50_ms'] / m['p50_ms']:.2f}x  (>1 = MIND faster)")
        return 0
    finally:
        rt.free(handle)


if __name__ == "__main__":
    raise SystemExit(main())
