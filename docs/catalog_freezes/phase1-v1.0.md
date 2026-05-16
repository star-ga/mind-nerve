# Phase 1 training run v1.0

- **Trained:** 2026-05-16
- **Path:** PyTorch internal (per `docs/catalog_and_training_plan.md`)
- **Location:** `/data/datasets/mind-nerve-catalog/phase1/v1.0/`
- **Bound to:** catalog [`v1.0`](./v1.0.md) + tokenizer
  [`v1.0`](./tokenizer-v1.0.md) (the latter is locked in for the
  Phase 2 native-MIND retrain; Phase 1 uses BGE's bundled tokenizer).

## Setup

| Item | Value |
|---|---|
| Base model | `BAAI/bge-small-en-v1.5` (33M params, 384-dim) |
| Loss | `MultipleNegativesRankingLoss` (HF ST's InfoNCE) |
| Pairs | `(query=item name, positive=item body)` — 11,221 train / 1,247 eval |
| Optimizer | AdamW, lr=2e-5, warmup=10% of steps |
| Batch | 32 |
| Epochs | 3 |
| Seed | 1337 |
| Device | RTX 3080 (CUDA), AMP enabled |
| Train wall-time | 126.5 s |

## Quality (12,468-pool candidate set)

| Metric | Baseline (BGE-small, off-the-shelf) | Phase 1 final | Δ |
|---|---|---|---|
| top-1 | 75.38% | **85.49%** | +10.11 |
| top-5 | 91.50% | **96.55%** | +5.05 |
| top-10 | 94.07% | **97.27%** | +3.20 |

**Phase 1 exit criterion** (`ROADMAP.md` line: "≥ 92% top-5 accuracy
on the held-out STARGA agent skill catalog") — **MET** at 96.55%
(4.55 points above threshold). Closes task #58.

Training loss curve: 0.27 → 0.013 over 3 epochs (well-conditioned).

## Latency (CPU-only, 20-query benchmark)

| Stage | mean | p50 | p95 | p99 |
|---|---|---|---|---|
| encode | 75.6 ms | 81.8 ms | 86.2 ms | 86.2 ms |
| rank | 6.5 ms | 5.7 ms | 9.6 ms | 14.3 ms |
| total | 82.1 ms | 87.1 ms | **90.9 ms** | 95.8 ms |

`ROADMAP.md` target: `p95 ≤ 30 ms on 4-core CPU at single-batch
(4096 tokens)`. **NOT MET by the PyTorch Phase 1 path** (p95 = 90.9 ms,
3× the budget). Expected — PyTorch FP32 BGE-small is not the target
runtime; that target requires:

- INT8 quantization
- Sliding-window attention (window=256, stride=192)
- Native MIND inference path (mindc-compiled `.so`)

All three land in Phase 2 with the native MIND retrain. The Phase 1
PyTorch run is the *reference* model; production latency requires
the architecture freeze in `spec/architecture.md` to be implemented.

Task #59 stays open until the native MIND inference path is
operational.

## Cross-arch bit-identity (task #57)

**Not applicable to Phase 1.** Cross-arch bit-identity is a
Q16.16-determinism property of the native MIND inference path. The
PyTorch float-path will *not* produce bit-identical outputs on x86
vs. CUDA — that's the point of the Phase 2 retrain.

Task #57 stays open until Phase 2.

## Smoke-test query examples (CPU only, encoded by the trained model)

```
'git status'                         → status / Managing Git / Git Workflow
'create a python file'               → python (rule) / Python Development / create
'review my code for security issues' → Secure Code Review / Security Review / 🛡️ Prompt: Secure Code Review
'how do I deploy to vercel'          → Deploy to Vercel (0.969) / Vercel Deploy
'explain this regex'                 → explain / regex-issues-finder / extract-text
```

Sensible top results across diverse query shapes. Score gaps between
the right answer and the next-best are healthy (typical 0.1+).

## Reproducibility

```bash
cd /home/n/mind-nerve/catalog-builder
python3 train_phase1.py --epochs 3 --batch 32 --lr 2e-5 --out-version v1.0
# To precompute the route table the Python wheel needs:
PYTHONPATH=../python python3 -m mind_nerve.cli precompute-routes
```

Deterministic given fixed seed=1337 + identical corpus + identical
hyperparameters. AMP (mixed-precision) on CUDA introduces minor
non-determinism in absolute weights but does not affect the metric
band ±0.5 points.

## What's bound into `model_hash`

The Phase 1 manifest binds:

- `corpus_hash` = catalog-v1.0 `freeze_id` (`a63b55d728492fee…fa59bb51`)
- `tokenizer_hash` = tokenizer-v1.0 `sha256` (`1b9ebc24b712e10f…fad71f3`)
- `model_hash` = SHA-256 over the saved `checkpoint/` directory bytes
- `phase1_version` = `v1.0`

Any change in (corpus, tokenizer, training config) re-derives a
different `model_hash`. Future Phase 1 retrains (v1.1, v1.2, …) get
their own subdirectory and bind their own hashes.
