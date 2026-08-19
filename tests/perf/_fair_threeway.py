"""Fair three-way head-to-head: MIND MT-gemv vs numpy+BLAS vs PyTorch, threads matched.

MIND's score path spawns sysconf(_SC_NPROCESSORS_ONLN) owner-computes threads
and is NOT env- or affinity-configurable. So the only apples-to-apples policy
is to let numpy and torch use the SAME full-machine thread count. The caller
sets OMP/OPENBLAS/MKL_NUM_THREADS before numpy imports; torch is pinned via
torch.set_num_threads() to the same value.

Reports p50/p95/p99/mean/QPS per backend plus MIND-relative speedups, so a
"we beat both" claim is either supported by the table or it is not.
"""

from __future__ import annotations

import json
import os
import platform
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


def _nthreads() -> int:
    return int(os.sysconf("SC_NPROCESSORS_ONLN"))


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


def _measure_torch(catalog, queries, threads):
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        print(f"  (torch unavailable: {e})")
        return None
    torch.set_num_threads(threads)
    cat_t = torch.from_numpy(np.ascontiguousarray(catalog.astype(np.float32)))
    qs_t = [torch.from_numpy(np.ascontiguousarray(q.astype(np.float32))) for q in queries]
    with torch.no_grad():
        for i in range(_N_WARMUP):
            _ = torch.mv(cat_t, qs_t[i % len(qs_t)])
        s = []
        t0all = time.perf_counter()
        for i in range(_N_MEASURE):
            qv = qs_t[i % len(qs_t)]
            t0 = time.perf_counter()
            _ = torch.mv(cat_t, qv)
            s.append((time.perf_counter() - t0) * 1000.0)
        wall = time.perf_counter() - t0all
    st = _percentiles(s)
    st["qps"] = round(_N_MEASURE / wall, 1)
    st["torch_version"] = torch.__version__
    st["torch_threads"] = torch.get_num_threads()
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
    nt = _nthreads()
    try:
        catalog = _make_catalog(_catalog_rng())
        queries = _make_queries(_query_rng(), _N_DISTINCT)
        host = {
            "node": os.environ.get("BENCH_NODE", platform.node()),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "online_cpus": nt,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "blas_env": {
                k: os.environ.get(k, "?")
                for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            },
        }
        print(f"host: {host['node']} {host['machine']} cpus={nt} py={host['python']} numpy={host['numpy']}")
        print(f"catalog {catalog.shape} dtype={catalog.dtype}  iters={_N_MEASURE} ({_N_DISTINCT} distinct)")
        print(f"MIND threads = sysconf(_SC_NPROCESSORS_ONLN) = {nt} (fixed)")
        print(f"numpy BLAS env: {host['blas_env']}")
        print()

        res = {}
        res["mind"] = _measure_mind(rt, handle, catalog, queries)
        res["numpy"] = _measure_numpy(catalog, queries)
        res["torch"] = _measure_torch(catalog, queries, nt)

        hdr = f"  {'backend':<32}{'p50':>9}{'p95':>9}{'p99':>9}{'mean':>9}{'QPS':>10}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        labels = {
            "mind": f"MIND MT-gemv Q16.16 ({nt}t)",
            "numpy": f"numpy+BLAS f32 matvec ({nt}t)",
            "torch": f"pytorch mv f32 ({nt}t)",
        }
        for k in ("mind", "numpy", "torch"):
            v = res.get(k)
            if not v:
                continue
            print(
                f"  {labels[k]:<32}{v['p50_ms']:>9.4f}{v['p95_ms']:>9.4f}"
                f"{v['p99_ms']:>9.4f}{v['mean_ms']:>9.4f}{v['qps']:>10.1f}"
            )
        print()
        m = res["mind"]
        for k in ("numpy", "torch"):
            v = res.get(k)
            if not v:
                continue
            print(
                f"  vs {k:<8} p50 speedup = {v['p50_ms'] / m['p50_ms']:>6.2f}x   "
                f"p95 speedup = {v['p95_ms'] / m['p95_ms']:>6.2f}x   (>1 = MIND faster)"
            )

        out = {"host": host, "iters": _N_MEASURE, "results": res}
        p = Path(__file__).parent / "bench_threeway.json"
        p.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {p}")
        return 0
    finally:
        rt.free(handle)


if __name__ == "__main__":
    raise SystemExit(main())
