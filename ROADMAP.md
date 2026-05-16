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
3. **Wire MIND with protected libs — DONE.** `libmindnerve.so` ships
   bundled inside the wheel at 51,280 bytes with 8 FORTRESS C-side
   primitives. The build pipeline + 846-line `protection.mind` + 1199-line
   `protection.c` live in the private `star-ga/mind-nerve-protected`
   sibling repo. Public surface passes a 7-check leak verifier.

**Public alpha (`v0.1.0-alpha.3`) shipped 2026-05-16.** Python wheel on PyPI,
weights on Hugging Face under Apache-2.0 (`star-ga/mind-nerve-phase1`). The
PyTorch-based Phase-1 inference path is the trial surface that drives
adoption; Phase-2 native MIND inference + cross-arch bit-identity + p95 ≤ 30 ms
remain on the deferred list below.

## Phase 1 — Reference implementation (private alpha shipped 2026-05-16)

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

- Russian intent classification (English-only in Phase 1)
- ARM, WebGPU, NPU backends (CUDA + CPU only in Phase 1)
- Native-MIND training pipeline (Phase 1 uses external training framework,
  reads weights into MIND inference)
- Per-CLI hook-surface stabilisation for runtimes whose hook protocol is
  still in flux upstream

**Deferred to Phase 2 — gated on `mindc` toolchain**:

These exit criteria from the original Phase-1 list are now sequenced behind
the `mindc` 0.2.6 / 0.3.0 milestones in the STARGA ecosystem roadmap. The
Phase-1 alpha ships without them by design; PyTorch covers the inference
path until the native cdylib path lands.

| Task | Blocker | mindc milestone |
|---|---|---|
| Cross-arch bit-identity (x86-CPU vs CUDA) | `pub fn` → C symbol export so the native MIND inference kernel is callable as a `cdylib` | **0.2.6** — `pub fn`-to-C, `[exports]`, `--profile` flag |
| p95 ≤ 30 ms on 4-core CPU | Native `cdylib` emit so the PyTorch encode-cost (~270 ms today) can be replaced by a Q16.16 native kernel | **0.3.0** — `--lib` cdylib, AOT codegen, MIC profile-locked headers |

Both tasks remain tracked (#57 and #59 in the work queue) and re-open the
moment the matching `mindc` release ships.

## Phase 2 — Production path (target: Q2 2027)

**Exit criteria**:

- All 18 mind-runtime backends pass cross-arch bit-identity tests
- Russian intent classification at ≥ 90% top-5 accuracy
- Native-MIND training pipeline; reproducibility on identical
  (corpus, config, seed) tuples
- codex, gemini, vibe shims merged
- Per-neuron weight attestation integrated with MindLLM evidence chain
- Latency p95 ≤ 30 ms on ARM (Apple silicon, Snapdragon)

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

## Phase 3 — Ecosystem (target: Q3 2027)

**Exit criteria**:

- Skill marketplace adapter — third-party skill libraries can register
  themselves at runtime, mind-nerve incorporates them without retraining the
  base classifier
- Federated routing — multiple mind-nerve instances collaborate across hosts
  with cryptographic evidence chain reconciliation
- mind-mem v4 cognitive-kernel integration so route history becomes a
  first-class memory class

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
| `rfn-mind` | v3.x deterministic FT path applies to mind-nerve training in Phase 2 |
