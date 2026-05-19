# mind-nerve benchmarks

Two reproducible benches ship with mind-nerve under `tests/perf/`:

| Bench | File | What it answers |
|---|---|---|
| **Speed** | `tests/perf/bench_criterion.py` | How fast is the score-only path against an idealised BLAS lower bound? |
| **Efficiency** | `tests/perf/bench_efficiency.py` | The substrate properties a BLAS stack cannot offer: cross-arch bit-identity, metric-flavor behaviour, energy. |

Both are runnable two ways:

```bash
# standalone — prints a table, writes bench_{criterion,efficiency}.json
python tests/perf/bench_criterion.py
python tests/perf/bench_efficiency.py

# under the test runner — adds a regression gate, self-skips with
# MIND_NERVE_PERF_SKIP=1 or when the native library is absent
pytest tests/perf/bench_criterion.py tests/perf/bench_efficiency.py
```

All numbers below were measured on an i7-5930K (6-core / 12-thread,
single-channel DDR4-2400, 64 GiB), single thread, warm, on a seeded synthetic
catalog of 11,922 rows × 384 dims in Q16.16 fixed-point (i64 stride-8 heap
layout, ≈36 MB), 1000 queries (64 distinct, cycled). The synthetic geometry
mirrors the live catalog's value distribution; no externally unavailable
checkpoint is required.

## Speed bench (score-only)

| Backend | p50 (ms) | p95 (ms) | p99 (ms) | QPS |
|---|---:|---:|---:|---:|
| MIND + mind-blas-A (AVX2) | 1.42 | 1.61 | 1.73 | ~696 |
| MIND + scalar (oracle) | 1.69 | 1.94 | 2.13 | ~583 |
| numpy + BLAS (idealised f32 reference) | 0.24 | 0.37–0.66¹ | ~3.2¹ | ~3000 |
| pytorch CPU (single thread) | 1.03 | 1.24 | 1.33 | ~934 |

Peak RSS over the run: ≈460–475 MiB.

¹ The idealised BLAS reference is a tiny `(11922,384) @ (384,)` mat-vec; its
upper-percentile latency is dominated by scheduler/allocator jitter and varies
2–4× run to run. The stable comparison point is **p50**.

### Honest headline

> **mind-blas-A reaches roughly 1/6 of the idealised numpy+BLAS score-only
> path (p50 1.42 ms vs 0.24 ms) while preserving cross-arch Q16.16
> bit-identity — a determinism property BLAS does not offer — and is 9.3×
> faster than the prior pure-scalar reduction (15 ms → 1.61 ms p95).**

This is **not** a "we beat BLAS" claim. We do not. The relevant facts are:

- **Regime.** The Q16.16 catalog uses an i64 stride-8 layout: ≈36 MB, which
  saturates single-channel DDR4 bandwidth at roughly 1.6 ms p95 regardless of
  how fast the inner reduction is. mind-blas-A is **memory-bandwidth-limited
  in this layout**, not compute-limited. A future i32 stride-4 repack halves
  the resident catalog and is expected to approach the ≈0.4 ms compute floor.
- **The trade we actually make.** mind-blas-A gives up raw throughput against
  a float BLAS GEMV in exchange for an integer-domain reduction whose result
  is byte-identical across dispatch paths (and, by construction, across
  architectures). The efficiency bench measures that property directly.
- **No naive-vs-vectorized framing.** Every row above varies exactly one
  axis (the backend) on the same workload, same vectorisation assumptions.
  No scalar-vs-SIMD speedup is presented as a substrate finding.

### Encode path: PENDING

The speed bench measures **score-only**. Encode-only and end-to-end routing
are deliberately out of scope here and are reported as `PENDING` in
`bench_criterion.json`:

> blocked on the Phase 6.2 full-catalog run with the real Phase 1 checkpoint
> (externally unavailable). Score-only is the entire measurable scope today;
> encode is tracked separately.

The gap is surfaced, not hidden. Any end-to-end number will be published only
once the encode path is measurable on the real checkpoint.

### Regression gate

Under pytest, the speed bench hard-fails iff mind-blas-A score-only
**p95 > 2.0 ms**. This is a regression detector — the expected steady state
is ≈1.6 ms. A failure means the AVX2 path is not engaged or a regression
landed in the matmul shim.

## Efficiency bench

The bench that exercises properties a BLAS-backed routing stack structurally
cannot provide.

### 1. Cross-arch Q16.16 bit-identity (task #57)

SHA-256 of the concatenated top-5 `(idx, q16_score)` stream over the
100-query deterministic corpus, computed on **both** dispatch paths:

| Path | SHA-256 |
|---|---|
| AVX2 (mind-blas-A) | `f4524bd56fd74e9dfbfb17b5b1f56fafda0e7e99321ef75ebce777219cda45fc` |
| scalar oracle | `f4524bd56fd74e9dfbfb17b5b1f56fafda0e7e99321ef75ebce777219cda45fc` |
| pinned x86 reference | `f4524bd56fd74e9dfbfb17b5b1f56fafda0e7e99321ef75ebce777219cda45fc` |

All three are identical. The integer-domain SIMD reduction with explicit
per-lane i64 widening is associative, so the AVX2 path is byte-identical to
the scalar oracle. **This hash is the cross-arch oracle**: a future ARM,
CUDA, or photonic Q16.16 backend must reproduce it byte-for-byte to pass the
task #57 gate. A float BLAS GEMV cannot make this guarantee — its reduction
order is implementation- and architecture-dependent.

### 2. Metric-flavor matrix (L1 / L2 / L∞)

Top-5 sets under three reductions over the same Q16.16 catalog (100 queries),
measured in numpy — L2 (dot product, the current cosine flavor) as the
reference, L1 (Manhattan, sqrt-free), L∞ (Chebyshev, max-abs):

| Flavor vs L2 | Mean Jaccard | Mean rank-overlap |
|---|---:|---:|
| L1 vs L2 | 0.24 | 37.4% |
| L∞ vs L2 | 0.01 | 2.4% |

On this synthetic Gaussian catalog, L1 and L∞ top-5 sets diverge sharply
from L2 — they are **different metrics, not approximations** of cosine here.
This is consistent with the prior observation that L1-cosine fell well below
the adoption gate on real embedding blocks. The substrate-metric story is
therefore: L1/L∞ are attractive on substrates without a native sqrt (the
sqrt-free reduction is thermodynamically cheaper), but a metric swap must be
re-validated for top-5 agreement on real traffic before adoption — synthetic
data does not by itself justify it. The bench *measures* this; it does not
assert L1 is a drop-in for L2.

### 3. Joules / query

Best-effort, directional, never fabricated.

| Source | Result | Reason |
|---|---|---|
| CPU (Intel RAPL package domain) | `null` | `rapl_unreadable` — `/sys/class/powercap/intel-rapl:0/energy_uj` is root-readable only on this host |
| GPU (nvidia-smi) | `null` | `no_gpu_score_path` — PENDING; there is no GPU score path yet |

When RAPL is readable, the bench reports the energy delta over a 1000-query
mind-blas-A run divided by 1000, with the domain and counter-wrap handling
recorded in `bench_efficiency.json`. On a host where the RAPL sysfs node is
root-only (the default on most distributions), the field is `null` with an
explicit reason rather than an invented figure.

## Reproducing

```bash
pip install -e .                 # builds / links the native library
python tests/perf/bench_criterion.py
python tests/perf/bench_efficiency.py
cat tests/perf/bench_criterion.json tests/perf/bench_efficiency.json
```

The JSON artefacts are git-ignored (machine-specific timings); regenerate
them locally. The cross-arch reference hash and the regression gate are the
two values that travel — everything else is host-dependent.
