# `mind-train` — native MIND training infrastructure (new repo spec)

## Why a new repo

`rfn-mind` is busy with the recurrent-field architecture, Cerebras pipeline, and Phase 6 work. Its `v3_native_mind_training.md` is a design doc, not code. Shoehorning the universal training primitives into rfn-mind couples concerns and slows both.

A standalone `mind-train` repo gives us:

- One clean home for MIND-native autograd, optimizers, dataloaders, loss functions
- A dependency that **any** STARGA model project consumes (mind-nerve, rfn-mind v3, MindLLM, future)
- Cleaner versioning — training-infra has its own release cadence
- Open-sourcable independently of model code (this is the toolchain, not a product)

## Naming options

| Name | Pro | Con |
|---|---|---|
| **`mind-train`** | Direct, matches family (`mind-mem`, `mind-nerve`, `mind-flow`) | Slightly generic |
| `mind-grad` | Focuses on the hard part (gradient computation) | Less obvious |
| `mind-forge` | STARGA-flavored, smith metaphor | Cute, less descriptive |
| `forge.md` | Public-brand parallel to `nerve.md` | Premature branding for infra |

**Recommendation: `mind-train`** for the GitHub repo (`star-ga/mind-train`). A public brand domain can come later if we ever make it user-facing.

## Scope ladder

| Version | What's in | Consumer |
|---|---|---|
| **v0.1** | Autograd + AdamW + dataloader + InfoNCE loss + Q16.16 checkpoint export — enough to train **mind-nerve's encoder** (50M params, contrastive setup) | mind-nerve |
| v0.2 | Cross-entropy + per-head learned masks (RFC-005 from mind-nerve catalog) + grad accumulation | mind-nerve hard-negative iteration |
| v0.3 | Distributed primitives (DP / ZeRO-1) + checkpoint sharding | larger STARGA models |
| v0.4 | rfn-mind v3 reductions (Polyak-Ruppert EMA per RFC-028, QAT per RFC-026, layer-wise LR decay per RFC-029) | rfn-mind v3 / MindLLM |
| v1.0 | All canonical features, stable API, Cerebras backend | rfn-mind v3 Phase 6-C |

v0.1 is the only blocker for mind-nerve Phase 1. **Everything above v0.1 is ladder, not gate.**

## v0.1 surface (the minimum that lets mind-nerve train end-to-end native)

### Components

| Component | Source file | Status today | Effort to v0.1 |
|---|---|---|---|
| Compute-graph representation | `src/graph.mind` | New | 1 week — typed DAG over Q16.16 ops |
| Backward pass / autograd | `src/backward.mind` | New | 2-3 weeks — manual rule per Q16.16 op |
| AdamW optimizer | `src/optimizer/adamw.mind` | New | 3-5 days — `m`, `v`, bias correction, decoupled weight decay |
| Cosine LR scheduler | `src/optimizer/lr.mind` | New | 1-2 days |
| Dataloader (streaming, batched, shuffled) | `src/data/loader.mind` | New | 1 week — leverages rfn-mind bundle format for caching |
| Contrastive InfoNCE loss | `src/loss/infonce.mind` | New | 2-3 days — pairwise dot-product over Q16.16 |
| Hard-negative mining iterator | `src/data/hard_negs.mind` | New | 3-5 days — BM25 + cosine threshold |
| Checkpoint emitter (rfn-mind bundle v3 compatible) | `src/checkpoint.mind` | Reuse from rfn-mind | 2-3 days — port |
| Training loop driver | `src/train.mind` | New | 1 week — assembles the pieces |

**Total v0.1 effort: 6-10 weeks.** This is the realistic native-MIND-trainer ship time. It's **longer than mind-nerve's Phase 0 (catalog mining) plus Phase 1 (PyTorch training)**.

That's why the recommendation was: **mind-nerve Phase 1 trains in PyTorch internally; mind-train v0.1 lands in parallel and is consumed by mind-nerve v0.2 retrain.** Two paths converging at v0.2.

### Hard problems (real engineering risk)

1. **Q16.16 gradients are noisy.** Standard FP32 autograd assumes high-precision storage. Q16.16 (16 fractional bits) loses information on small gradients. Mitigations: gradient accumulation in i64, master weights at higher precision, gradient scaling.
2. **Backward-pass derivation must be done by hand** per Q16.16 op. PyTorch's autograd works because it has an FP32 chain rule library. We'd need to write the chain rule for Q16.16 mul, Q16.16 div, fixed-point softmax, Q16.16 RMSNorm, etc. — every op the encoder uses.
3. **Bit-identical training across architectures** is harder than bit-identical inference. Reduction order in backward pass must be deterministic.
4. **No PyTorch ecosystem leverage** — no `torch.compile`, no FlashAttention, no FSDP, no DeepSpeed. We build it ourselves.

### Cross-references

- Bundle format: reuse rfn-mind `src/bundle.mind` (operational today)
- Q16.16 primitives: reuse rfn-mind `src/fixed_point.mind` (operational today)
- Tensor reductions: reuse rfn-mind `src/reduce.mind` (experimental but compiled)
- The arch-mind v0.2.0 catalog of 84 IMPLEMENTED RFCs has some training-side RFCs (RFC-026 QAT, RFC-028 EMA, RFC-029 LLRD, etc.) — those become the v0.2-v0.4 feature backlog for mind-train

## How `mind-train` fits with `mind-nerve` (the consumer)

```
                  mind-train (training infra)
                  ───────────────────────────
                  autograd · optimizers · dataloaders · losses
                                ▲
                                │ depends on
                                │
mind-nerve (product)            │
─────────────────────           │
catalog · encoder kernels ──────┘
       (Q16.16 inference; trained
        with mind-train v0.2+)
```

- mind-nerve v0.1 ships with PyTorch-trained weights converted via `torch_to_bundle.py`
- mind-train v0.1 ships in parallel (separate ship date)
- mind-nerve v0.2 retrains using mind-train v0.1 — first product to use native MIND training end-to-end
- rfn-mind v3 retrains using mind-train v0.2-v0.4 features — second consumer
- MindLLM uses mind-train v1.0 — third consumer

This makes `mind-train` the **load-bearing training-infrastructure dependency** for the whole STARGA model fleet. Worth investing in cleanly from day one.

## Repo bootstrap (proposed structure)

```
star-ga/mind-train/
├── README.md
├── LICENSE.md           ← STARGA Commercial (same as arch-mind, rfn-mind, mind-runtime)
├── Mind.toml
├── ROADMAP.md
├── CHANGELOG.md
├── docs/
│   ├── architecture.md
│   ├── q16_gradient_calculus.md     ← the "how to do backward pass on every Q16.16 op" reference
│   ├── reduction_order_contract.md   ← the cross-arch bit-identity gate spec
│   └── consumers.md                  ← who depends on us (mind-nerve, rfn-mind, MindLLM)
├── src/
│   ├── lib.mind
│   ├── graph.mind
│   ├── backward.mind
│   ├── optimizer/
│   ├── data/
│   ├── loss/
│   ├── checkpoint.mind  (port from rfn-mind)
│   └── train.mind
├── tests/
│   ├── test_gradient_correctness.mind   ← finite-difference cross-check
│   ├── test_optimizer_step.mind
│   └── test_cross_arch_determinism.mind  ← x86 vs CUDA gradient bit-identity
├── bench/
│   └── bench_step_time.mind
├── tools/
│   └── torch_to_bundle.py    (port from rfn-mind, the bridge tool)
└── .arch-mind/
    └── rules.mind
```

## License & visibility

- **Private repo** initially (same as rfn-mind, mind-runtime, mind-internal)
- STARGA Commercial license
- Open-source candidate at v0.4+ once it stabilizes — same path mind-runtime is on
- Public artifacts: trained-checkpoint releases (open weights), but training-infra binary stays protected

## Open decisions before bootstrap

1. **Repo creation**: now (lock the name + scaffold) or after mind-nerve Phase 0 finishes (catalog work doesn't depend on it)?
2. **Initial commit**: just README + ROADMAP, or a working `src/graph.mind` stub?
3. **License**: STARGA Commercial (mirrors rfn-mind / mind-runtime) — or Apache-2.0 for adoption (mirrors public arch-mind plan)?
4. **rfn-mind coordination**: rfn-mind v3 design doc references native MIND training as its own roadmap item. Do we (a) carve it out into mind-train and let rfn-mind v3 become a thin consumer, or (b) build mind-train as a separate parallel project rfn-mind eventually adopts? **(a) is the cleaner architecture; needs rfn-mind owner buy-in.**

## Recommended next move

If you authorize creating the repo this session, I can:

1. `mkdir /home/n/mind-train && git init`
2. Scaffold the README, ROADMAP, Mind.toml, LICENSE, .gitignore, .arch-mind/rules.mind
3. Add the v0.1-only file skeleton (empty modules with TODOs and the cross-arch determinism gate stub)
4. Write the `q16_gradient_calculus.md` reference doc with backward-pass rules for the ops mind-nerve's encoder uses (sliding-window attention, RMSNorm, MLP, softmax)
5. Commit the scaffold

That's ~1-2h of autonomous work, gets the repo to "first sketch committable" state, doesn't burn GPU or compute.

Defers to next session: actually implementing autograd. That's 6-10 weeks of focused engineering.
