# Model Card — mind-nerve Phase 1

This card describes the Phase-1 shipped backend. The Phase-2 native MIND
Q16.16 backend is in progress and not yet end-to-end (see
[`README.md`](../README.md#phase-2-backend--in-progress-a15-partial)).

## Model details

| Field | Value |
| --- | --- |
| Model name | `mind-nerve-phase1` |
| Hugging Face repo | [`star-ga/mind-nerve-phase1`](https://huggingface.co/star-ga/mind-nerve-phase1) |
| Maintainer | STARGA, Inc. — <info@star.ga> |
| Type | Sentence-encoder fine-tune for intent classification over a route catalog |
| Base model | `BAAI/bge-small-en-v1.5` |
| Backend | PyTorch + `sentence-transformers` |
| Training objective | `MultipleNegativesRankingLoss` over `(name, body)` positive pairs |
| Catalog | `route_table.jsonl` v1.0 (11,922 routes, frozen and signed) |
| License (code) | Apache-2.0 |
| License (weights) | Apache-2.0 |
| Released | 2026 |

## Intended use

mind-nerve Phase 1 is intended for **local intent routing** in agent
runtimes. The encoder takes a free-text user prompt and produces a pooled
query vector that is matched against a precomputed catalog of skills,
tools, MCP servers, and agent capabilities. The runtime returns a
deterministic top-K list, which the host CLI uses to decide which subset
of its library to expose to the downstream LLM on the next turn.

Typical deployment surfaces:

- The `mind-nerve-routed` UNIX-socket daemon, fronted by a CLI hook
  (e.g. `mind-nerve-preselect`) or an MCP server (`mind-nerve-mcp`).
- The synchronous Python API (`from mind_nerve import route`).
- The one-shot CLI (`mind-nerve route "<prompt>" --top-k 5`).

## Out-of-scope use

mind-nerve is **not**:

- a content classifier, safety classifier, or moderation system;
- a general-purpose embedding model for downstream tasks beyond route
  retrieval over the trained catalog;
- a generative model — it returns route IDs, never free text;
- a recommendation system trained on user behaviour — there is no
  per-user fine-tune in the public Phase-1 weights;
- a multilingual model — Phase 1 is English-only.

Do **not** use mind-nerve to make automated decisions about access,
eligibility, safety, or any other high-stakes outcome. It is a retrieval
optimiser, not a classifier.

## Training data

See [`docs/dataset.md`](dataset.md) for the full schema, source
collection rules, deduplication, filtering, and split procedure. The
Phase-1 v1.1-oss catalog is curated to be **public-clean** — no
STARGA-private content. License posture is documented in
[`docs/data_governance.md`](data_governance.md).

## Evaluation

Headline numbers on the v1.1-oss catalog:

| Metric | Value |
| --- | --- |
| Routes evaluated against | 11,922 |
| Top-5 accuracy | **96.06%** |

The eval set is the held-out fraction (default `eval_frac = 0.1`) of the
deduplicated catalog. Top-1, top-5, and top-10 accuracy are reported on
the eval queries against the full positives pool (see
`mind_train.train` → `eval.json`).

MRR, nDCG, per-slice calibration, and ECE are planned for Phase 2 (see
the audit "Model and algorithm correctness" finding and
[`ROADMAP.md`](../ROADMAP.md)).

## Performance

Phase 1 (shipped):

- Warm-daemon p95 ~**23 ms** on GPU.
- Warm-daemon p95 ~**90 ms** on a 4-core CPU.
- Cold start: ~250 ms (model load) + ~7 s warmup for the daemon.

The ≤30 ms-on-CPU target documented in
[`spec/architecture.md`](../spec/architecture.md) is the **Phase 2** target,
not a Phase 1 result.

Phase 2 (A1.5 PARTIAL):

- Native MIND Q16.16 **score path** (matmul against the 11,922-row route
  table) measures **p50 14.4 ms / p95 15.1 ms** on a 4-core CPU at
  commit
  [`b9b6401`](https://github.com/star-ga/mind-nerve/commit/b9b6401).
- The full end-to-end native encoder forward is blocked on the `mindc`
  Phase 6.2 quantizer + SIMD lowering. Until that lands, the wheel
  routes through the Phase 1 PyTorch backend.

## Known limitations

- **English-only at Phase 1.** The encoder vocabulary is the
  English-only BPE inherited from the base model. Multilingual support
  is on the roadmap (see [`spec/quality_targets.md`](../spec/quality_targets.md)).
- **CPU latency lags spec target.** Phase 1 warm-daemon p95 on a 4-core
  CPU is ~90 ms; the ≤30 ms-on-CPU budget closes with Phase 2.
- **Catalog-conditioned.** Accuracy numbers are reported on the v1.1-oss
  catalog. Routes added at runtime via `mind-nerve learn` are encoded
  with the same base model but are not re-evaluated against the eval
  pool.
- **No native training backend yet.** `mind-nerve train --backend native`
  currently raises `NotImplementedError`; Phase 1 trains in PyTorch.
- **No confidence calibration.** Phase 1 emits raw dot-product scores,
  not calibrated probabilities. Do not threshold the score as if it
  were a probability.

## Reproducibility

Each training run produces a `manifest.json` next to the checkpoint
containing the base model, hyperparameters, seed, `model_hash`,
`corpus_hash`, eval metrics, elapsed time, and platform details. See
[`docs/dataset.md`](dataset.md#provenance-fields-in-manifestjson) for the
full field list.

The runtime download is revision-pinned (the wheel records the pinned
Hugging Face revision; override with `MIND_NERVE_HF_REVISION`).

## Ethical considerations

- **Local-only by default.** No prompt content leaves the host; see
  [`docs/privacy.md`](privacy.md).
- **License-aware ingestion.** `mind_nerve.discovery.scan` defaults to
  OSS-compatible licenses only and rejects unlabelled content; see
  [`docs/data_governance.md`](data_governance.md).
- **No tamper-friendly path.** Every inference can emit an attestation
  envelope tying the request, model, and result hashes together.

## Citation

```bibtex
@software{mind_nerve_2026,
  author  = {STARGA, Inc.},
  title   = {mind-nerve: Intent-classification preselector for agent runtimes},
  year    = {2026},
  url     = {https://github.com/star-ga/mind-nerve}
}
```
