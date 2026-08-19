"""Determinism demo: sha256 of the MIND score stream over the seeded corpus."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _bench_common import (  # noqa: E402
    _catalog_rng,
    _make_catalog,
    _make_queries,
    _query_rng,
    _topk_from_scores,
    _try_native_runtime,
)


def main() -> int:
    rt = _try_native_runtime()
    if rt is None:
        print("native runtime unavailable")
        return 1
    handle = rt.init(0, 0)
    catalog = _make_catalog(_catalog_rng())
    queries = _make_queries(_query_rng(), 100)
    h = hashlib.sha256()
    for qv in queries:
        scores = rt.score(handle, qv, catalog)
        idx, sc = _topk_from_scores(np.asarray(scores, dtype=np.int64), 5)
        for i, s in zip(idx.tolist(), sc.tolist(), strict=False):
            h.update(int(i).to_bytes(8, "little", signed=True))
            h.update(int(s).to_bytes(8, "little", signed=True))
    rt.free(handle)
    print(h.hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
