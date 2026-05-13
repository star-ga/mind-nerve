# Quality Targets

The bar mind-nerve must clear before any Phase 1 release.

## Why these specific numbers

The targets below are not chosen to look ambitious. They are the floors below
which mind-nerve is not useful — either because it would mis-route too often,
or because it would be too slow to call on every turn.

A preselector that ships at 88% top-5 accuracy is a regression vs naive
top-K-on-vector-similarity on most realistic catalogs; that floor sets the
accuracy target.

A preselector that adds 80 ms p95 to every turn is unusable in interactive
agent loops where total budget is ~2 seconds; that ceiling sets the latency
target.

## Accuracy

| Metric | Phase 1 target (EN) | Phase 2 target (EN + RU) |
|---|---|---|
| Top-1 accuracy | ≥ 78% | ≥ 80% |
| Top-5 accuracy | ≥ 92% | ≥ 93% |
| Top-10 accuracy | ≥ 97% | ≥ 97% |
| Mean Reciprocal Rank | ≥ 0.84 | ≥ 0.85 |

Evaluated on a held-out corpus of intent-labelled requests against the
production STARGA agent skill catalog (440 skills as of 2026-05).

**Stretch targets** for Phase 2:

- Top-1 accuracy ≥ 84% on English
- Top-5 accuracy ≥ 95% on English
- Per-route precision ≥ 0.80 on the 50 most-frequently-used skills

Russian-only metrics are tracked separately; cross-lingual transfer is not
the goal. mind-nerve is bilingual from Phase 2, not multilingual through
English.

## Latency

| Backend | p50 | p95 | p99 |
|---|---|---|---|
| x86 CPU (4-core, 1024 tok) | ≤ 18 ms | ≤ 30 ms | ≤ 45 ms |
| ARM CPU (Apple silicon M-series, 1024 tok) | ≤ 14 ms | ≤ 24 ms | ≤ 36 ms |
| CUDA (any, batch 32) | ≤ 4 ms/req | ≤ 7 ms/req | ≤ 12 ms/req |
| WebGPU (Chrome desktop, single request) | ≤ 22 ms | ≤ 38 ms | ≤ 60 ms |

Phase 1 ships x86 CPU + CUDA budgets only. ARM and WebGPU budgets are
Phase 2.

## Cross-architecture bit-identity

Mandatory across all backends in scope for the relevant phase. For every
`(model_hash, catalog_hash, request_hash)` triple, every backend MUST emit
the same `result_hash`.

CI gate: `tests/bit_identity/run.sh` must produce zero divergence across
{x86-cpu, cuda} in Phase 1; across {x86-cpu, arm-cpu, cuda, webgpu, npu} in
Phase 2.

One byte of divergence fails the build. There is no "approximately
bit-identical."

## Catalog size scaling

mind-nerve MUST scale at least to:

- **Phase 1**: 500 routes without latency regression
- **Phase 2**: 5,000 routes with p95 ≤ 50 ms on x86 CPU
- **Phase 3**: 50,000 routes (federated routing across multiple mind-nerve
  instances)

Linear scaling of inference latency with `|RouteCatalog|` is acceptable up
to 500 routes; logarithmic scaling becomes mandatory above 5,000. The route
embedding table is the dominant cost above 1,000 routes, and quantizing the
table to Q8.8 (rather than Q16.16) is the planned mitigation if needed.

## Attestation overhead

The attestation envelope adds latency. Budget:

- Envelope construction: ≤ 5 ms p95 on x86 CPU
- Envelope verification (replay): ≤ 3 ms p95 on x86 CPU

If attestation overhead exceeds 15% of total inference latency, the envelope
hashing schedule is the problem, not the model. Pre-Phase-1 work includes
benchmarking the SHA-256 cost on the smallest viable catalog (50 routes) to
confirm the budget.

## Failure modes that are NOT acceptable

These are not "low-priority bugs." A release that exhibits any of these is
not Phase-1-complete.

1. **Deterministic top-K returns different orderings on the same request
   across runs.** Tie-breaking by `SHA-256(route_id) ascending` is the
   load-bearing contract. Failure here invalidates attestation.
2. **A request that does not exist in the training corpus produces a
   high-confidence single match.** Confidence calibration must be honest;
   uncertain inputs must produce uncertain output distributions.
3. **The attestation envelope is correct but the chain link to the previous
   envelope is missing or stale.** Cleared envelopes from prior catalogs must
   either be retained in the chain or explicitly marked as a chain reset
   with a documented reason.
4. **Latency p99 is 5× p95 or worse.** Heavy-tail latency makes the
   preselector unusable in agent loops; we either fix the tail or we have not
   shipped.

## Comparison reference points

For reviewer calibration, these are public approximate baselines on similar
intent-classification tasks (not directly comparable; reported here as
context, not as benchmark targets to beat):

- Function-calling-specialised small models in the 25–50M parameter range
  report top-1 accuracy in the high-70s to low-80s on single-shot function
  selection benchmarks
- Sentence-transformer + ANN baselines report top-5 in the 88–92% range on
  comparable skill-selection tasks
- LLM-based routing (full inference call) reports top-1 in the high 80s to
  low 90s but at 100-1000× the latency budget

mind-nerve's design intent is to match the sentence-transformer top-5 floor
while improving top-1 toward LLM-routing quality, all within the
sentence-transformer latency envelope. That's the sweet spot the architecture
targets.
