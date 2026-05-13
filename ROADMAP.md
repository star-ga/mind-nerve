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

## Phase 1 — Reference implementation (target: Q1 2027)

**Exit criteria**:

- Architecture compiles in pure MIND on at least the CUDA and CPU backends
- Forward pass produces bit-identical Q16.16 outputs on x86-CPU and CUDA
- Reference weights (English intent corpus) achieve ≥ 92% top-5 accuracy on
  the held-out STARGA agent skill catalog
- p95 inference latency ≤ 30 ms on 4-core CPU at single-batch
- Claude Code shim ships with installable hook
- MCP server façade routes `mind-mem` tool calls

**Deferred to Phase 2**:

- Russian intent classification (English-only in Phase 1)
- ARM, WebGPU, NPU backends (CUDA + CPU only in Phase 1)
- Native-MIND training pipeline (Phase 1 uses external training framework,
  reads weights into MIND inference)
- codex, gemini, vibe shims

## Phase 2 — Production path (target: Q2 2027)

**Exit criteria**:

- All 18 mind-runtime backends pass cross-arch bit-identity tests
- Russian intent classification at ≥ 90% top-5 accuracy
- Native-MIND training pipeline; reproducibility on identical
  (corpus, config, seed) tuples
- codex, gemini, vibe shims merged
- Per-neuron weight attestation integrated with MindLLM evidence chain
- Latency p95 ≤ 30 ms on ARM (Apple silicon, Snapdragon)

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
