# mind-nerve final-ship plan (locked 2026-05-17)

Authoritative scoping output for the FINAL mind-nerve ship — Phase 2 + Phase 3 — produced by the autoscoping pass against `ROADMAP.md`, `CHANGELOG.md`, `spec/architecture.md`, `src/lib.mind`, `python/mind_nerve/installer.py`, and `RFCs/INDEX.md` as of v0.1.0a13.

## TL;DR

`v0.1.0a13` → cadence converges on `v1.0.0`. **Phase 1 deferred items + Phase 2 production path + Phase 3 ecosystem all land in this plan**, with three hard external gates explicitly called out so the v1.0.0 CHANGELOG can be transparent about what's "deferred behind mindc 0.3.0" if those gates slip.

## Block-status matrix

### Phase 2

| Item | Status |
|---|---|
| 18-backend cross-arch bit-identity | **BLOCKED** — needs mindc 0.3.0 cdylib emit |
| Native MIND inference replacing PyTorch | **BLOCKED** — needs mindc 0.3.0 cdylib emit |
| p95 ≤ 30 ms on 4-core CPU (native) | **BLOCKED** — needs mindc 0.3.0 |
| p95 ≤ 30 ms on ARM | **BLOCKED** — needs mindc 0.3.0 + ARM CI runner |
| Russian intent classification ≥ 90% top-5 | **SHIPPABLE NOW** — compute-bound training run |
| Native-MIND training pipeline (`mind-train`) | **PARTIAL** — standalone bring-up shippable; deep work |
| codex shim | **DONE** |
| gemini shim | **SHIPPABLE NOW** |
| vibe shim | **SHIPPABLE NOW** |
| Per-neuron weight attestation w/ MindLLM | **PARTIAL** — per-tensor manifest exists, needs cross-binding handshake |
| SOTA-track #1: learnable per-route prior | **PARTIAL** — switch landed, needs catalog v2 emit |
| SOTA-track #2: input-fingerprinted attestation | **DONE** — promote `request_hash != 0` to verifier invariant |
| SOTA-track #3: adaptive window stride | **PARTIAL** — switch landed, needs calibrated thresholds |
| SOTA-track #4: frequency-adaptive route scaling | **PARTIAL** — pure catalog-builder change |
| SOTA-track #5: per-head learned drop masks | **PARTIAL** — switch landed, needs native trainer first |

### Phase 3

| Item | Status |
|---|---|
| Skill marketplace adapter | **SHIPPABLE NOW** as design + stub; functional ship awaits Phase 2 |
| Federated routing | **SHIPPABLE NOW** as design + stub; functional ship awaits Phase 2 + mind-flow typed-edges |
| mind-mem v4 cognitive-kernel integration | **BLOCKED** — needs mind-mem v4 |

## Final ship version path

```
v0.1.0a13   (current, shipped 2026-05-16)
  ↓
v0.1.0-beta.1   — Tier 1 (Russian weights, gemini/vibe/claw installers,
                  evidence verifier hardening, Phase 3 scaffolds)
  ↓
v0.2.0-beta.1   — Tier 2 catalog-builder v2
                  (route prior, freq-adaptive scaling, stride thresholds)
                  model_hash bump #1
  ↓
v0.2.0          — Tier 3 attestation cross-binding + Russian ≥ 90% verified
                  All Phase 2 SHIPPABLE / PARTIAL items landed
  ↓
v0.3.0-beta.1   — Native-MIND training pipeline (mind-train)
                  First locally-reproducible reference checkpoint
  ↓
v0.9.0-rc.1     — Switch flip wave: per-head drop masks, L2-cosine,
                  RMSNorm, ALiBi — each behind a model_hash bump
                  Phase 3 stubs flip to functional
  ↓
v1.0.0          — Native cdylib inference path (gated mindc 0.3.0)
                  Cross-arch bit-identity gate passes
                  p95 ≤ 30 ms on CPU + ARM
                  Skill marketplace + federated routing functional
                  mind-mem v4 cognitive-kernel integration
```

## External dependencies that gate "fully shipped"

- **mindc 0.3.0** — `--lib` cdylib emit, AOT codegen, MIC profile-locked headers
- **mind-mem v4 cognitive kernel** — Phase 3 route-history-as-memory-class binding
- **ARM CI runner** — Apple Silicon or `macos-14-arm64` GitHub runner
- **rfn-mind Phase 6** — only if `mind-train` defers to rfn-mind's trainer (decoupled by Tier 3 item #10)
- **Russian training corpus** — 1-2 d GPU pod (terminate after eval)

## Implementation tiers

(Full ranked list, files, commit-subject lines, effort estimates: see N1 planner output in conversation transcript.)
