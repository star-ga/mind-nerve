"""Perf budget gate: warm route() p95 over 100 queries.

PHASE 1 budget contract:
  * Spec target  : ≤ 30 ms warm p95 on a 4-core CPU (architecture.md).
  * README floor : ~90 ms today on the Phase-1 sentence-transformers
                   path. Phase-2 native Q16.16 encoder is in flight.

This test is a **report**, not a hard fail. It:
  1. Skips cleanly if MIND_NERVE_PERF_SKIP=1 OR the auto-seeded runtime
     can't be loaded (no HF access, no checkpoint on disk).
  2. Warms the route() cache.
  3. Measures p50/p95/p99 over 100 sequential queries.
  4. Prints the latency table with budget comparisons.
  5. Fails ONLY if p95 > 200 ms (regression detector — anything past
     that ceiling means something materially worse than the README
     baseline shipped).

The 30ms spec gate is reported, not enforced, until Phase-2 native
lands. Until then the published numbers (warm daemon ~23 ms p95 on
GPU, ~90 ms on 4-core CPU) are the honest comparison points.
"""

from __future__ import annotations

import os
import statistics
import time

import pytest

# Phase-1 spec budgets — reported, not enforced.
_SPEC_P95_TARGET_MS = 30.0
_README_PHASE1_CPU_P95_MS = 90.0

# Regression-detector ceiling — failure means something is materially
# worse than the README baseline (2x slower).
_REGRESSION_P95_CEILING_MS = 200.0

_N_WARMUP = 5
_N_MEASURE = 100

pytestmark = pytest.mark.perf


def _skip_if_runtime_unavailable() -> None:
    """Skip the test if the auto-seeded runtime cannot be reached.

    `MIND_NERVE_PERF_SKIP=1` is the explicit opt-out for environments
    without the Phase-1 checkpoint.
    """
    if os.environ.get("MIND_NERVE_PERF_SKIP") == "1":
        pytest.skip("MIND_NERVE_PERF_SKIP=1 set — perf gate intentionally skipped")

    # Try to load the runtime. If it fails (no checkpoint on disk, no HF
    # network access, ctypes binding missing), skip rather than fail.
    try:
        from mind_nerve import load_default_runtime  # noqa: PLC0415

        rt = load_default_runtime()
        # Touch the runtime so a lazy proxy resolves now.
        _ = rt.catalog_size
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"runtime unavailable for perf test: {exc.__class__.__name__}: {exc}")


@pytest.mark.perf
def test_route_warm_p95_within_regression_ceiling() -> None:
    """Measure warm p50/p95/p99 over 100 queries; report budget comparisons."""
    _skip_if_runtime_unavailable()

    from mind_nerve import route  # noqa: PLC0415

    queries = [
        "search for unused imports in the repo",
        "deploy the staging build with rollback support",
        "run the test suite and produce coverage report",
        "git rebase interactive workflow",
        "fix the linting errors in the python package",
        "generate an SBOM for the wheel",
        "audit production dependencies for CVEs",
        "format the codebase with ruff and pyright",
        "trace the daemon's startup sequence with strace",
        "investigate the failing CI job on main",
    ]

    # Warmup — exclude these samples from the percentile calculation.
    for i in range(_N_WARMUP):
        route(queries[i % len(queries)], top_k=5)

    # Measure.
    samples_ms: list[float] = []
    for i in range(_N_MEASURE):
        q = queries[i % len(queries)]
        t0 = time.perf_counter()
        route(q, top_k=5)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)

    samples_sorted = sorted(samples_ms)
    p50 = samples_sorted[len(samples_sorted) // 2]
    p95 = samples_sorted[int(len(samples_sorted) * 0.95)]
    p99 = samples_sorted[int(len(samples_sorted) * 0.99)]
    mean = statistics.mean(samples_ms)

    print()
    print("===== mind-nerve perf budget report =====")
    print(f"  samples            : {_N_MEASURE} warm route() calls")
    print(f"  mean               : {mean:7.2f} ms")
    print(f"  p50                : {p50:7.2f} ms")
    print(f"  p95                : {p95:7.2f} ms")
    print(f"  p99                : {p99:7.2f} ms")
    print()
    print("  budget comparisons (reported, not enforced for spec gates)")
    print(
        f"    spec p95 target  : {_SPEC_P95_TARGET_MS:7.2f} ms"
        f"  -> p95 is {'OK' if p95 <= _SPEC_P95_TARGET_MS else 'OVER'}"
    )
    print(
        f"    Phase-1 README   : {_README_PHASE1_CPU_P95_MS:7.2f} ms"
        f"  -> p95 is {'OK' if p95 <= _README_PHASE1_CPU_P95_MS else 'OVER'}"
    )
    print()
    print(f"  regression ceiling : {_REGRESSION_P95_CEILING_MS:7.2f} ms  (HARD FAIL above)")
    print("=========================================")
    print()

    assert p95 < _REGRESSION_P95_CEILING_MS, (
        f"route() warm p95={p95:.2f} ms exceeds regression ceiling "
        f"{_REGRESSION_P95_CEILING_MS:.2f} ms — "
        f"something is materially slower than the Phase-1 baseline."
    )
