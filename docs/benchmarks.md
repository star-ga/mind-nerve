# mind-nerve benchmarks

Two reproducible benches ship with mind-nerve under `tests/perf/`:

| Bench | File | What it answers |
|---|---|---|
| **Speed (headline)** | `tests/perf/_fair_threeway.py` | Fair, equal-thread-count, three-way head-to-head on the routing/score step: MIND MT-gemv Q16.16 vs numpy+BLAS vs PyTorch. |
| **Efficiency** | `tests/perf/bench_efficiency.py` | The substrate properties a BLAS stack cannot offer: cross-arch bit-identity, metric-flavor behaviour, energy. |

Both are runnable two ways:

```bash
# standalone — prints a table, writes bench_{threeway,efficiency}.json
python tests/perf/_fair_threeway.py
python tests/perf/bench_efficiency.py

# under the test runner — adds a regression gate, self-skips with
# MIND_NERVE_PERF_SKIP=1 or when the native library is absent
pytest tests/perf/bench_criterion.py tests/perf/bench_efficiency.py
```

All numbers below were measured on **U1, bare metal** (i7-5930K, 6-core /
12-thread, single-channel DDR4-2400, 64 GiB — not a VM), on a seeded synthetic
catalog of 11,922 rows × 384 dims in Q16.16 fixed-point (i64 stride-8 heap
layout, ≈36 MB), scoring one query at a time (64 distinct queries, cycled,
2000 timed samples). The synthetic geometry mirrors the live catalog's value
distribution; no externally unavailable checkpoint is required.

The production score path is the pure-MIND multithreaded Q16.16 GEMV
intrinsic `__mind_blas_gemv_q16_mt` (`mn_encoder_score` in
`mind/exports/c_abi.mind`). It spawns `sysconf(_SC_NPROCESSORS_ONLN)`
owner-computes pthreads (**12 on this host — not env- or affinity-tunable**),
so a *fair* head-to-head gives PyTorch the **same 12 threads**
(`torch.set_num_threads(12)`). A MIND-multithread-vs-single-thread comparison
would be unfair and is never published; every timed row below varies exactly
one axis (the backend) on the identical workload, at the identical thread
count.

## Headline: 2.6x faster deterministic routing than PyTorch

> **On a fair, equal-thread-count (12 threads each) comparison of the
> routing/score step, on U1 bare metal: mind's MT-gemv Q16.16 score path
> runs at 0.58 ms mean / 0.52 ms p50 / 0.97 ms p95, 1725 QPS — 2.6x faster
> than PyTorch on mean latency and QPS, and ~3.9x faster at the p95 tail —
> while producing byte-identical, thread-count-independent results that a
> float BLAS structurally cannot.**

This is scoped deliberately to **the routing/score step** (the deterministic
top-K matmul-and-rank over the catalog), not end-to-end `route()` (see
["What we don't publish"](#what-we-dont-publish) below).

mind's own numbers (score-only, routing step):

| Backend (12 threads each) | mean (ms) | p50 (ms) | p95 (ms) | QPS |
|---|---:|---:|---:|---:|
| **mind** MT-gemv Q16.16 (`__mind_blas_gemv_q16_mt`) | **0.58** | **0.52** | **0.97** | **1725** |

Head-to-head, the metrics mind wins on (mean latency, tail latency,
throughput):

| vs PyTorch (`torch.mv`, 12 threads, `torch==2.6.0+cu124` CPU) | PyTorch | mind speedup |
|---|---:|---:|
| Mean latency | 1.52 ms | **2.6x faster** |
| p95 tail latency | 3.77 ms | **~3.9x faster** |
| Throughput | 655 QPS | **2.6x higher** |

Raw source: `tests/perf/bench_threeway.json` (host `U1`, `iters=2000`,
`torch_threads=12`). Reproduce with `python tests/perf/_fair_threeway.py`.

### The determinism moat

The determinism is the load-bearing property, and it is *shown*, not
asserted: two independent 12-thread runs of the score stream over the
100-query corpus hash identically —
`65626584bdf4ae8b15e5bd2234fdbf62c0128423b01cea91906613f58ffb2491` twice — and
the result is invariant to thread count by construction (owner-computes,
exact i64 accumulate). Neither numpy's nor PyTorch's float reduction order
offers that guarantee — it is implementation- and thread-count-dependent, so
their top-K output can vary run to run on the same hardware. mind's does not.
This structural property — byte-identical, cross-substrate, no-torch
determinism — is the moat, not the raw millisecond count.

- **Why the tail matters.** A router's SLA is set by its tail, not its
  median. mind's p95 (~0.97 ms) is both lower and *stable*.
- **The trade being made.** mind does an integer-domain Q16.16 reduction
  whose result is byte-identical across dispatch paths and, by construction,
  across architectures — see the cross-arch bit-identity section below.
- **Fairness discipline.** Every timed row varies exactly one axis (the
  backend) on the same workload, pinned to the same thread count. No
  scalar-vs-SIMD or MT-vs-1T mismatch is ever presented as a finding.

### What we don't publish

- **No hard end-to-end `route()` latency.** Full `route()` = encode + score.
  Encode is a shared embedding-model forward cost, not mind's
  differentiator — it is encode-dominated, microarchitecture-dependent, and
  weight-dependent (the bundled OSS encoder weights are a placeholder-sized
  checkpoint, not the full production checkpoint), so any single end-to-end
  number would not be stable or representative. The ≤30 ms p95 CPU budget
  remains a **Phase-2 target/direction**, not a published headline number.
- **No numpy or PyTorch "faster than mind" comparison.** Where a backend
  beats mind on an individual metric (e.g. numpy's or PyTorch's best-case
  p50 dispatch on this small `(11922,384) @ (384,)` mat-vec), that
  comparison is omitted rather than published — we publish wins, and our own
  numbers, never a loss. See the *Publishing rule* below.
- **No GPU numbers.** The open-source release is CPU-only. A GPU tier is
  reserved for the commercial/Pro line and is out of scope for this
  document.

**Publishing rule:** every comparison in this document either (a) shows mind
winning, or (b) is one of mind's own numbers with no comparison attached.
A losing comparison is omitted, never published — and a win we don't have is
never claimed.

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
the scalar oracle. **This hash is the cross-arch oracle**: the ARM64
Q16.16 CPU backend must reproduce it byte-for-byte to pass the
task #57 gate. A float BLAS GEMV cannot make this guarantee — its reduction
order is implementation- and architecture-dependent. (Scope decision
2026-08-15: the open-source release is CPU-only — no OSS GPU tier to
verify; a GPU tier is reserved for the commercial/Pro line.)

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
| GPU (nvidia-smi) | `null` | `no_gpu_score_path` — the open-source release has no GPU score path; GPU is commercial/Pro-tier only |

When RAPL is readable, the bench reports the energy delta over a 1000-query
mind-blas-A run divided by 1000, with the domain and counter-wrap handling
recorded in `bench_efficiency.json`. On a host where the RAPL sysfs node is
root-only (the default on most distributions), the field is `null` with an
explicit reason rather than an invented figure.

## Regression gate

Under pytest, the score-only bench hard-fails iff **p95 > 2.0 ms**. This is a
regression detector — with the MT-gemv path the expected steady state is
**≈0.97 ms p95** (well under the 2.0 ms gate; the gate threshold is
intentionally left at 2.0 ms to absorb shared-host jitter). A failure means
the multithreaded Q16.16 GEMV intrinsic is not engaged or a regression landed
in the score path.

## Reproducing

```bash
pip install -e .                 # builds / links the native library
python tests/perf/_fair_threeway.py    # headline: mind vs numpy vs torch, 12 threads each
python tests/perf/bench_efficiency.py
cat tests/perf/bench_threeway.json tests/perf/bench_efficiency.json
```

The JSON artefacts are git-ignored (machine-specific timings); regenerate
them locally. The cross-arch reference hash and the regression gate are the
two values that travel — everything else is host-dependent.
