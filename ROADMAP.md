# Roadmap

Phased delivery plan. Bandwidth gates each phase against ARC-AGI-3 close
(2026-11-01) and mind-mem v4 retry2e completion; revise dates against those
anchors as they slip.

## Guiding constraints

These hold across every phase. If a phase violates one, the phase has not
shipped.

1. **Pure MIND.** No PyTorch, no ONNX runtime, no third-party ML framework in
   the inference path. Reference training pipeline may use external tooling
   in Phase 1, must port to native MIND by Phase 2.
2. **Q16.16 throughout the inference path.** No IEEE-754 fallback. Cross-arch
   bit-identity is non-negotiable.
3. **Single binary, all backends.** One `mind-nerve` CLI runs on x86, ARM,
   CUDA, WebGPU, NPU without rebuild.
4. **Latency p95 ≤ 30 ms on CPU.** Architecture decisions that cannot meet
   this on commodity CPU are rejected.
5. **Attestation on every inference.** Request hash, model hash, result hash
   into the evidence envelope. No opt-out.
6. **Compile speed never regresses.** mind-runtime frontend compile times
   stay within the published envelope; module-level gating only.

## Hard prerequisites — status (revised 2026-05-16)

Three blockers were raised 2026-05-14. Status after the Phase 1 alpha sprint:

1. **Catalog freeze — DONE.** `catalog-v1.0` (12,468 items, draft-unsigned
   `freeze_id a63b55d7…`) shipped 2026-05-14, refined to `v1.1-oss` (11,922
   items, `freeze_id 1cd130fa…`) 2026-05-15 after license-gate filtering.
2. **Native MIND training pipeline — DEFERRED TO PHASE 2.** The Phase 1
   alpha trains with PyTorch + `sentence-transformers` per the *Pure MIND*
   guideline above ("Reference training pipeline may use external tooling
   in Phase 1, must port to native MIND by Phase 2"). The shipped
   *inference* path is what must move to native MIND; the *training* path
   is allowed to remain external until Phase 2's native-MIND trainer
   (`mind-train`) is built.
3. **Wire MIND with protected libs — DONE.** `libmindnerve.so` is the
   native runtime bundled inside the wheel. The Phase-1 PyTorch
   inference path published in this repository works without it.

**Release status (2026-05-18):** `v0.3.0-beta.2` is the current PyPI
public — wheel + sdist live at
[pypi.org/project/mind-nerve/0.3.0b2](https://pypi.org/project/mind-nerve/0.3.0b2/),
GitHub release at
[v0.3.0-beta.2](https://github.com/star-ga/mind-nerve/releases/tag/v0.3.0-beta.2).
Single change on top of beta.1 is the flock-guarded `ensure()` daemon
spawn (closes the multi-spawn race that pinned memory under
high-concurrency CLI use). Weights on Hugging Face under Apache-2.0
(`star-ga/mind-nerve`). The PyTorch-based inference path remains
the trial surface that drives adoption; Phase-2 native MIND inference +
cross-arch bit-identity + p95 ≤ 30 ms remain on the deferred list below,
but the upstream `mindc` blockers underneath that list have moved — see
the §"Deferred to Phase 2 — gated on `mindc` toolchain" table for current
status.

## Phase 1 — Reference implementation (public on PyPI as v0.2.0, 2026-05-16)

**Architecture finalised (scaffold landed 2026-05-13):** encoder + direct
scoring head, no decoder. Sliding-window self-attention (window=256,
stride=192). Drop-the-decoder removed 33% of FLOPs vs the initial design
without measurable accuracy impact on agent-CLI request distributions.

**Exit criteria**:

- Architecture compiles in pure MIND on at least the CUDA and CPU backends
- Forward pass produces bit-identical Q16.16 outputs on x86-CPU and CUDA
- Reference weights (English intent corpus) achieve ≥ 92% top-5 accuracy on
  the held-out STARGA agent skill catalog
- p95 inference latency ≤ 30 ms on 4-core CPU at single-batch (4096 tokens)
- Claude Code shim ships with installable hook
- MCP server façade routes `mind-mem` tool calls
- Cross-CLI installer covers all 17 supported runtimes (claude-code, codex,
  vibe, gemini, cursor, windsurf, aider, openclaw, nanoclaw, nemoclaw,
  continue, cline, roo, zed, copilot, cody, qodo)

**Deferred to Phase 2**:

- ARM, WebGPU, NPU backends (CUDA + CPU only in Phase 1)
- Native-MIND training pipeline (Phase 1 uses external training framework,
  reads weights into MIND inference)
- Per-CLI hook-surface stabilisation for runtimes whose hook protocol is
  still in flux upstream

**Deferred to Phase 2 — gated on `mindc` toolchain** (status 2026-05-18):

These exit criteria from the original Phase-1 list are sequenced behind
the `mindc` 0.2.6 / 0.3.0 / **0.4.0** milestones in the STARGA ecosystem
roadmap. The Phase-1 alpha ships without them by design; PyTorch covers
the inference path until the native cdylib path lands.

**`mindc` 0.4.0 landed 2026-05-18** — RFC 0005 Phase 2 fully shipped:
`std/vec.mind`, `std/string.mind`, `std/map.mind`, `std/io.mind` (the
four pure-MIND collections + I/O on the seven `__mind_*` intrinsics)
plus `use std.foo` cross-module resolution.  This unblocks pure-MIND
runtime libraries that mind-nerve's Phase-2 Q16.16 native encoder will
build on top of.  Compile-speed gate clean (small_matmul -2.24% /
medium_mlp -1.95% / large_network -0.61%) — the IP moat held.

**`mindc` 0.4.1 landed 2026-05-18** — RFC 0005 **Phase B** (per-arg
signature matching) closes the v0.4.0 deferred loose-end.  Imported
`pub fn` calls under `use std.foo` now validate arity + per-arg types
against the imported declaration and return the declared return type
(falling back to Phase-A loose typing for `export { ... }`-block
donors).  Bench gate clean against the new
`.bench-baseline-2026-05-18-rfc0005.txt` floor.

**`mindc` 0.4.2 landed 2026-05-18** — RFC 0005 **Phase C** (stdlib
auto-bundled into mindc).  Closes the RFC 0005 stdlib-discovery loop:
v0.4.0 (Phase A) wired the resolver, v0.4.1 (Phase B) added per-arg
signature matching + return-type fidelity, v0.4.2 (Phase C) bakes
`std/{vec,string,map,io}.mind` into the mindc binary via
`include_str!` and seeds the project loader's module table with them
before walking the user's src tree.  A downstream `mind build` of a
project that says `use std.vec` now resolves with no external file
dependency.  Phase C also lit a new gated-feature CI step (commit
`996553e`) that runs the std-surface + cross-module-imports test
suites separately + combined, closing a real CI blind spot — 65+
gated tests are now under cloud guard against silent regression.

**`mindc` 0.4.3 landed 2026-05-18** — RFC 0005 **Phase D₁**
(`$MIND_STDLIB_PATH` env-var override).  Lets downstream users fork
the pure-MIND stdlib without rebuilding mindc — pointing the env
var at a directory containing all four `.mind` files swaps the
bundled blobs at project-load time.  Same fork-without-recompile
escape hatch Rust's `RUSTC_BOOTSTRAP` provides; same module-level
feature gate as Phase C so the default-build hot path stays
branchless.  Bench gate clean (small +1.8% / medium -0.8% / large
+1.9%, all inside +5%).

**`mindc` 0.4.4 landed 2026-05-18** — RFC 0005 **Phase D₂a**
(Named-struct parameter names preserved in error messages).
Cold-path-only diagnostic refinement: when a call to an imported
`pub fn` has an arity or type mismatch on a parameter whose
declared annotation is a Named struct (e.g. `vec_set(v: Vec, …)`),
the error now reads `expects Vec (heap-record i64 addr), got
tensor<f32[3]>` rather than collapsing the param to the lowered
`ScalarI64` ABI surface.  The compatibility check itself stays
permissive (i64 values still accept into Named struct params under
the Option-C heap-record ABI) — purely an error-message-clarity
fidelity contract.  Bench gate clean (hot path untouched; bench
threshold loosened to +7% to absorb GitHub-hosted runner variance
without weakening the moat).  Phase D₂b (cross-arg Named-struct
identity matching) deferred until first user-visible need.

| Task | Blocker | mindc milestone | Status |
|---|---|---|---|
| Cross-arch bit-identity (x86-CPU vs CUDA) | `pub fn` → C symbol export so the native MIND inference kernel is callable as a `cdylib` | **0.2.6** — `pub fn`-to-C, `[exports]`, `--profile` flag | **mindc-side SHIPPED** (RFC 0002 D2–D5 in `0a408e3`, `_v1` ABI lock in `de6cf18`, RFC 0003 cdylib seam). mind-nerve-side validation (mindc CUDA build + bit-identical SHA) still pending hardware — task #57 stays open. |
| p95 ≤ 30 ms on 4-core CPU | Native `cdylib` emit so the PyTorch encode-cost (~270 ms today) can be replaced by a Q16.16 native kernel | **0.3.0** — `--lib` cdylib, AOT codegen, MIC profile-locked headers | **mindc-side SHIPPED in mindc v0.3.0** (2026-05-18). Tagged at `star-ga/mind@v0.3.0` with all of: `--emit-shared` cdylib flag (`c444c77`), Phase 0/1/1.5 std-surface intrinsics, P0d `Instr::FnDef`→`func.func` (`aacebe1`), RFC 0005 P0e Step 1 struct heap-record write (`2f98a4f`), P0f Step 1 `FieldAccess` read for local-`Ident` receivers (`c706a3e`), and P0f Step 2 (`b458932`) covering chained access, fn-return receivers, and struct-typed parameters via the `struct_resolver` Span-keyed side-table. 16 std-surface integration tests + 145 lib tests (each feature config) all green; bench gate `<17.3 µs` on the largest tracked network. mind-nerve-side wheel/native-encoder integration still pending (#59 stays open until the Q16.16 mindc-emitted `.so` is linked into the published wheel sibling-style). |

Both tasks remain tracked (#57 and #59 in the work queue). `#57` re-opens
the moment mind-nerve has a CUDA host to run the bit-identical-SHA harness
against the now-emittable C-ABI library; `#59` re-opens once the mindc
0.3.0 struct-ABI lowering ships and the Q16.16 native encoder kernel can
be linked into the wheel as a sibling `.so`.

## Phase 2 — Production path (target: Q2 2027)

**Exit criteria**:

- All 18 mind-runtime backends pass cross-arch bit-identity tests
- Native-MIND training pipeline; reproducibility on identical
  (corpus, config, seed) tuples
- codex, gemini, vibe shims merged
- Per-neuron weight attestation integrated with MindLLM evidence chain
- Latency p95 ≤ 30 ms on ARM (Apple silicon, Snapdragon)
- **Tier-1 multilingual coverage** — twelve languages (English, Spanish,
  Mandarin Simplified, Hindi, Arabic, Portuguese, Russian, Japanese,
  French, German, Bengali, Korean) each clear the per-language
  accuracy gates in
  [`spec/quality_targets.md`](spec/quality_targets.md)
  §"Multilingual language policy". One language failing fails the ship.
- **Tier-2 monitoring dashboard** — next 20 languages by speaker count
  + remaining UN official languages have published per-release eval
  numbers (no gate; regressions logged).
- **Tier-3 script floor** — every BPE-encodable language survives a
  `tokenizer.encode().decode()` round-trip CI gate over the FLORES-200
  dev set. No language silently breaks.

### Phase 2 multilingual workstream

Splitting the language deliverables from the SOTA-track items so the
compute budget is honest about what each ship gate costs.

**Tier 1 (gated, twelve languages).** Per-language deliverables:
1. Held-out intent-labelled eval set (≥ 5,000 requests per language).
2. Trained reference checkpoint at the current `model_hash` cadence.
3. CI gate against the language's eval set in `tests/multilingual/`.
4. Published quality numbers on the model card.

Twelve languages × five-thousand requests is a multi-pod compute
spend. The expected shape is a single multilingual training run over
the merged corpus, followed by per-language eval passes; not twelve
isolated trainings.

**Tier 2 (monitored, ~20 additional languages).** Eval sets exist;
results are published per release as a dashboard. Regressions logged
but not blocking.

**Tier 3 (script floor, all other languages).** Tokenizer round-trip
CI gate over FLORES-200 dev. No model eval; the contract is "no
language silently breaks at the tokenizer layer."

The 32k BPE vocab from Phase 1 is almost certainly insufficient for
proper CJK + Devanagari + Arabic-script coverage; expansion to 48-64k
is a tracked decision in the Phase 2 catalog-builder workstream.

### Phase 2 accuracy & latency enhancements (SOTA-track)

**Update 2026-05-14:** the autoresearch IMPLEMENT phase landed
backwards-soft compile-time switches for items 1, 3, 4, and the
group-wise INT8 / matryoshka / cosine / RMSNorm / ALiBi / sink-token
/ multi-query-pooling / prefix-conditioning research vectors —
together covering ~80% of the Phase 2 inference-surface roadmap.
Each switch is **off by default** (binary byte-identical to today)
and flips on once the offline catalog-builder pipeline emits a
matching reference checkpoint. See
`spec/architecture.md` § "Backwards-soft architecture switches"
and `RFCs/INDEX.md` for the full set with source-paper citations.

What's still ahead for Phase 2:

The following improvements landed against fleet consensus on 2026-05-13 as
the highest-leverage moves to differentiate mind-nerve from sentence-
transformer retrievers, LLM-based routers, and ColBERT-style late-
interaction models. Each preserves the Phase 1 non-negotiables (pure MIND,
Q16.16 in flight, INT8 weights, cross-arch bit-identity, 30 ms p95).

1. **Learnable per-route prior** (must-have). Adds a Q16.16 prior vector
   over `|RouteCatalog|` routes derived offline from catalog-load
   co-occurrence statistics. Logit-level addition before top-K extraction.
   Encodes usage patterns (e.g. `git status → git diff` affinity) that no
   general-purpose retriever can capture. Zero-latency win.

2. **Input-fingerprinted attestation** (must-have). Bind each envelope's
   nonce slot to `SHA-256(request_bytes[..32])` so every inference's
   envelope is verifiably unique to its input — critical for regulated
   agent deployments. Already implied by `request_hash` field in v2
   envelope; promote to a strict non-zero invariant in Phase 2.

3. **Adaptive window stride via input-entropy gating** (must-have).
   Compute token-level entropy from the first 16 tokens' Q16.16
   activations; select `ATTN_WINDOW_STRIDE ∈ {96, 192, 256}` from a
   compile-baked table. Wider stride for low-entropy CLI commands
   ("list files"), tighter overlap for high-entropy multi-clause queries.
   Latency win on the realistic-workload median.

4. **Frequency-adaptive route scaling** (nice-to-have). Multiply each row
   of the route embedding table by a precomputed `1/sqrt(freq)` Q16.16
   scalar at catalog-load time, floored at 0.5. Addresses the long-tail
   problem of rare-but-critical routes being drowned by frequent ones.
   Zero runtime cost (table is pre-scaled).

5. **Per-head learned drop masks** (experimental). During training, learn
   a binary mask per attention head; at inference, skip masked heads
   entirely. Up to 50% compute reduction; gated on validation accuracy
   not regressing more than 0.5 points top-5. Adds a training-pipeline
   change; landing depends on Phase 2 native-MIND training reaching
   stability first.

6. **L1-substrate encoder window similarity** (experimental, added
   2026-05-18). Today the sliding-window encoder's window-ranking
   similarity is L2-cosine (`(a·b) / (‖a‖₂‖b‖₂)`). For Q16.16
   in-flight ranking the L2 step costs a fixed-point `sqrt` that
   becomes a cross-substrate approximation contract under the bit-
   identity gate. An L1-cosine variant (`(a·b) / (‖a‖₁‖b‖₁)`) or
   raw L1 ranking (`−‖a − b‖₁`) is exact in Q16.16 by construction,
   eliminating the `sqrt` and any fixed-point approximation contract
   on the ranking path. Gate: leave-one-out top-K overlap and rank
   correlation vs the L2 baseline on the held-out intent corpus
   must clear `Kendall τ ≥ 0.85` and `top-5 overlap ≥ 90%` before
   adoption. Throughput win (~5–15% on CPU encode, larger on
   accelerators without native `sqrt`) is secondary to the
   cross-substrate determinism win on Q16.16. Landing depends on
   Phase 2 native-MIND inference path reaching stability first
   (see [`spec/architecture.md`](spec/architecture.md) §
   "Backwards-soft architecture switches" — this would be wired in
   as one more off-by-default compile-time switch).

## Phase 3 — Ecosystem (target: Q3 2027)

**Exit criteria**:

- Skill marketplace adapter — third-party skill libraries can register
  themselves at runtime, mind-nerve incorporates them without retraining the
  base classifier
- Federated routing — multiple mind-nerve instances collaborate across hosts
  with cryptographic evidence chain reconciliation
- mind-mem v4 cognitive-kernel integration so route history becomes a
  first-class memory class

### Federated trust-rating — mind-nerve's slice (design, 2026-06-03, revised)

**Ownership correction (2026-06-03):** the federated trust-rating *system*
— node federation, consent/governance, DRD-derived scoring, Ed25519
signing, Q16.16 aggregation, evidence-chained collective evolution — is an
**OS-level concern and lives in naestro** (`ROADMAP.md` R19), next to the
existing `AI Agent Governance` module. See naestro for the full three-leg
design. Putting reputation scoring + badge tiers inside a skill-router was
too much product in a router.

**What mind-nerve keeps is the thin consumer slice: rating-as-routing-input.**
mind-nerve *reads* the trust score / badge tier that naestro's federation
produces and uses it as a routing signal — prefer higher-tier
skills/agents/MCPs, break ties by node-local fit. mind-nerve does not own
the federation, the consent model, the governance gate, or the scoring
math; it consumes their output.

Boundary (three layers):

| Concern | Home |
|---------|------|
| Federation transport (vclock, conflict log, propose/approve) | **mind-mem v4 (Group D)** — generic primitive |
| Node federation + consent + governance + DRD scoring + badge tiers + Ed25519 + Q16.16 aggregation + collective-evolution gate | **naestro** (R19) — OS-level |
| Rating consumed as a routing signal (tier-preference, tie-break) | **mind-nerve** — this slice |
| Trust-score dashboards / badge distributions / swarm-health views | observability (read/telemetry, on top) |

Dependency arrows: `naestro federation → mind-mem v4 transport` (product
on primitive); `mind-nerve routing → naestro rating output` (router
consumes governance). mind-nerve never reaches into mind-mem internals or
reimplements scoring.

Deferred decision (gated on second consumer): if naestro's use of
mind-mem v4 Group-D transport keeps a clean import boundary, extract the
transport into a standalone domain-agnostic `mind-federation` package both
import. Extract on the second *real* consumer, not the anticipated one; a
leaky boundary is the signal to leave it in mind-mem. Not observability:
federation is a write/consistency layer; observability is the read layer
on top.

Layer note (per-node vs cross-node cost): the federation/rating work is the
*cross-node* half (signed deltas, not re-shipped context). The *per-node*
prefill-explosion half (O(b^d × T), KV-cache eviction) is answered
structurally by **rfn-mind** — no KV cache by construction, deterministic
carried state via explicit memory slots — not by mind-nerve. Different layer;
mind-nerve neither owns nor needs it. Claim scoped to "rfn-mind removes the
KV-cache problem by construction" (rfn-mind empirical Stages A–E unstarted).
Note: rfn-mind is the roadmap target, not the live substrate — naestro runs
frontier models today, so the per-node prefill cost is mitigated at the ops/
routing layer for now. The rating signal mind-nerve consumes is substrate-agnostic
and does not depend on rfn-mind being live.

Sequencing: builds on the existing Phase 3 federated-routing spec
(`spec/federated_routing.md`), gated behind Phase 2 + the typed-edges
composition layer. The CLI surfaces the rating signal in routing; the
federation/scoring it consumes is built in naestro.

**Open decisions — defer to roadmap review (do not bank as decided):**

1. **Operator CLI ownership.** The federation/observability workstream needs
   an operator control surface (enroll node, view swarm trust, approve rating
   deltas, inspect telemetry). Candidates: `starga` umbrella CLI (shim
   currently dangling — `~/starga-cli` missing), `mind` (rejected — domain-locked
   compiler), a new `naestro` CLI (naestro is API-driven, has no CLI by design).
   Working lean: `starga` as the single umbrella that drives naestro's API.
   **Decide at review.**
2. **mind-nerve as a CLI.** Working assumption: mind-nerve stays a
   library + daemon + MCP tool, *not* a product CLI (keeps only the small
   `learn`/runtime-dir admin binary). Don't re-fatten the router with
   operator commands. **Confirm at review.**
3. **Decentralized vs central LLM-model observability.** Lean: decentralized
   signed-telemetry gossip over the mind-mem append-only path (no conflict
   resolution — telemetry is monotonic), aggregated-summaries-only for privacy,
   feeding the trust-rating scorer in naestro. No central collector. This is the
   *input layer* to the rating system, not a separate system. **Decide at review.**

Build-time framing: these are architecture/ownership calls, settled when we
review roadmaps before writing the federation code — not pre-committed here.

### Phase 3 design landables (status: 2026-05-18)

| Item | Status |
|---|---|
| Skill marketplace adapter spec (`spec/skill_marketplace.md`) | **DONE** (design-only) |
| Skill marketplace typed-interface stub (`python/mind_nerve/marketplace.py`) | **DONE** (design-only) |
| Skill marketplace contract tests (`tests/integration/test_marketplace_registration.py`) | **DONE** (design-only) |
| Federated routing spec (`spec/federated_routing.md`) | **DONE** (design-only) |
| Federated routing typed-interface stub (`python/mind_nerve/federation.py`) | **DONE** (design-only) |
| Federated routing contract tests (`tests/integration/test_federation_reconcile.py`) | **DONE** (design-only) |
| mind-mem v4 cognitive-kernel binding spec (`spec/mind_mem_v4_integration.md`) | **DONE** (design-only) |
| Skill marketplace functional ship | **BLOCKED** — requires Phase 2 completion |
| Federated routing functional ship | **BLOCKED** — requires Phase 2 + the typed-edges composition layer |
| mind-mem v4 cognitive-kernel integration functional ship | **BLOCKED** — requires mind-mem v4 |

## Comprehensive Malicious-Code Detection (Phase 3 — design, 2026-06-25)

> Goal stated by the operator: a **very comprehensive security scanner that
> detects malicious code other scanners (Cisco/AV/SAST vendors, etc.) cannot
> understand.** Tracked here because the routing layer is mind-nerve's; the
> heavy detection engine is a separate subsystem (see ownership boundary).

### Honest scoping first

A general malicious-code detector is a **large new subsystem**, not a router
feature. mind-nerve is a deterministic intent→skill classifier (Q16.16,
cross-arch bit-identity, route IDs out — see *Non-goals*). It must not become
a scanner: that would re-fatten the router, the exact mistake the federated
trust-rating §already corrected. So the work splits into two layers with a
clean boundary:

| Concern | Home |
|---------|------|
| **Detection engine** — static + semantic + behavioral analysis of code/artifacts; the actual "understands what AV can't" capability | **new subsystem** (working name `mind-scan`) + the existing security-skill corpus (`analyzing-*`, `detecting-*`, `hunting-*`, `reverse-engineering-*`, hexstrike toolchain) |
| **Routing layer** — recognize a "scan this for malicious code" intent and route to the right analyzer/skill/agent, rank by confidence, chain multi-stage | **mind-nerve** — this slice |

mind-nerve's slice is **routing-to-detection**, not detection. That is the
part that legitimately lives in this repo and belongs on this roadmap.

### The differentiator — why "other scanners can't understand it"

The thesis that makes this worth building (vs. wrapping ClamAV/Semgrep):
signature/rule scanners match *known* patterns. The MIND-ecosystem edge is
**deterministic semantic analysis with a signed evidence chain** — analyze
the code's actual computed behavior (data-flow, capability use, obfuscation
unfolding) reproducibly and bit-identically across substrates, and emit an
**attested verdict** (request hash → artifact hash → verdict hash into the
evidence envelope). A reproducible, signed, semantics-level verdict is
something signature engines structurally cannot produce. This reuses the
exact wedge mind-nerve already carries (Q16.16 determinism + attestation),
applied to a detection target instead of a routing target.

> Note (no public attribution): frame any external surface as "deterministic
> semantic malware analysis," never name the inspiration sources.

### mind-nerve deliverables (the routing slice)

1. **Malicious-intent route class.** Add a first-class intent category for
   "analyze/scan this code or artifact for malicious behavior" so a host CLI
   request routes deterministically to the correct analyzer in the security
   corpus (per-language reverse-engineering skills, behavioral-analysis
   skills, the hexstrike families) rather than a generic match.
2. **Multi-stage detection chaining.** Route a single scan request through an
   ordered analyzer pipeline (triage → static → semantic/behavioral →
   verdict) using the typed-edges composition layer, surfacing each stage's
   route + the calibrated-confidence sidecar (§*Calibrated Routing
   Confidence*) so low-confidence findings escalate to a deeper analyzer.
3. **Verdict-as-routing-input (consumer slice).** mind-nerve *reads* the
   `mind-scan` engine's attested verdict and uses it as a routing/escalation
   signal — exactly the thin-consumer pattern used for the naestro
   trust-rating (mind-nerve consumes the verdict; it does not own the
   detection math or the evidence chain).

### Exit criteria (Phase 3 item)

- A malicious-analysis intent routes bit-identically to the right analyzer on
  x86-CPU and at least one other backend (the routing path, not the engine).
- Multi-stage chain demonstrated end-to-end against a labelled sample set,
  with the confidence sidecar driving escalation.
- The verdict the router consumes is attested (engine-side), and the route
  decision stays **outside** any identity hash (probabilistic verdict must
  never become load-bearing for the deterministic route — same invariant as
  the calibrated-confidence sidecar).

### Status & dependencies

- **Status: design / not started.** Gated behind Phase 2 + the typed-edges
  composition layer (same gate as federated routing).
- **Engine dependency:** the `mind-scan` detection subsystem must exist for
  deliverable 3; until then mind-nerve can ship deliverables 1–2 routing to
  the *existing* security-skill corpus (which is already large and learnable
  into the route table today).
- **Open decision (defer to roadmap review):** does the detection engine live
  as a standalone `mind-scan` repo, inside naestro's governance layer (it is
  an OS-level safety concern), or as a MIND-compiled analyzer kernel? Decide
  at review — do not pre-commit. This roadmap entry only commits mind-nerve's
  routing slice.

## Non-goals (now and ever)

- Generative output. mind-nerve emits route IDs and relevance scores, never
  natural language.
- Conversational state. Stateless per call; calling host owns history.
- Tool execution. Routing only; the host calls the chosen tool itself.
- Multi-modal input. Text in, route IDs out. Audio / image routing is a
  separate question handled by separate models (mind-voice, future).

## Coordination with sibling repositories

| Repo | Role |
|---|---|
| `mind-runtime` | Provides the 18-backend lowering matrix mind-nerve compiles against |
| `mind-mem` | v4.1 §7 tool-routing preselector consumes mind-nerve at the MCP layer |
| `MindLLM` | Per-neuron weight attestation discipline applies symmetrically to mind-nerve |
| native-MIND training pipeline | v3.x deterministic fine-tuning path applies to mind-nerve training in Phase 2 |

## Calibrated Routing Confidence (sidecar)

Surface a **calibrated confidence/utility score** on each intent→skill route — a
usable signal for how strongly the selected skill matches intent, beyond the raw
route rank.

- **Outside the deterministic path.** The confidence is a sidecar field on the
  routing result, not an input to the deterministic route table lookup. Route
  selection stays deterministic and reproducible; the confidence rides beside it
  and is **excluded from any identity hash**. A probabilistic estimate must never
  become load-bearing for the deterministic route.
- **Calibrated, not raw.** Report a calibrated score (reliability-curve fit over
  the existing route scoring), not a raw distance, so a caller can threshold on it
  to fall back to a broader catalog or ask for clarification.
- **Use.** Low-confidence routes can defer to a wider search or a clarifying turn
  instead of committing to a weak match.
- **Status:** Planned. Composes over the existing route scoring; no change to the
  deterministic route table.

## Pure-MIND Self-Hosting Migration

> Ecosystem-wide milestone — gated on the `mind` compiler reaching self-host completeness.

Once the `mind` toolchain self-hosts (the open-core compiler builds itself byte-identically),
this repository's **TypeScript + Python** implementation is migrated to **pure, executing MIND**, so the whole
MIND ecosystem runs on its own deterministic, byte-identical, evidence-carrying toolchain — the
wedge applied to ourselves.

- **Gate:** `mind` self-host keystone complete (see the `mind` roadmap self-host track).
- **Approach:** port via the `mind-migrator` path — to the executable MIND subset, verifying every
  emitted symbol actually runs and reusing `std` primitives; no silent AOT-only stubs.
- **Invariant:** migration preserves behavior and the cross-substrate byte-identity gate — no
  regression in determinism or the signed evidence chain.
- **Status:** Planned — sequenced after `mind` self-host; tracked here so the endgame is explicit.
