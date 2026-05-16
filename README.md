# mind-nerve

**Intent-classification preselector for agent runtimes.**

A small, fast classifier that sits between a user request and the host
runtime. It reads the request, decides which subset of available
tools/skills/agents is relevant, and hands the host a short list — so
the downstream LLM never sees the full library in its system prompt.

The result: library size decouples from token cost. Hosting 4,400 skills costs
the same prompt budget as hosting 44, because only the top-K are ever loaded
per turn.

## Status

**Phase 1 — private alpha (`v0.1.0-alpha.2`, 2026-05-16).** Python wheel ships;
the FORTRESS-protected `libmindnerve.so` is bundled inside it. The router runs
end-to-end on PyTorch via `BAAI/bge-small-en-v1.5` fine-tuned with
MultipleNegativesRankingLoss; top-5 accuracy is 96.06 % against the full
v1.1-oss catalog of 11,922 routing candidates.

Phase 2 (Q3 2027 target) replaces the PyTorch path with a native MIND Q16.16
inference loop and adds the cross-architecture bit-identity gate + 4-core CPU
p95 ≤ 30 ms latency budget. Until Phase 2 closes, the inference path uses
external ML tooling — explicitly permitted by the
[ROADMAP](./ROADMAP.md) Phase 1 exception.

## Why this exists

Agent runtimes today load entire skill/tool/MCP libraries into the LLM's system
prompt on every turn. At small scale this is fine. At hundreds of skills, the
prompt-cache and per-call token cost become the binding constraint on library
growth.

Standard responses to this problem all degrade either correctness or latency:

- Vector-only retrieval over skill descriptions loses precise intent matching
- LLM-based routing pays full inference cost just to decide what to load
- Manual skill grouping shifts the problem onto the operator

mind-nerve takes the third option: a purpose-built sub-50M-parameter classifier
that runs in tens of milliseconds on CPU, returns top-K relevant routes, and is
small enough to call on every turn without paying real cost.

## Integration surface

mind-nerve exposes a single contract across two host classes:

- **Claude Code, codex, gemini, vibe, and 13 other CLIs** — preselects which
  agent skills load into the system prompt for a given turn
- **MCP servers** — preselects which tools are surfaced as candidates before
  the calling LLM sees the full registry

Same model, same binary, same evidence chain — both host targets.

## Design constraints (non-negotiable)

- **Latency p95 ≤ 30 ms** on CPU. If we miss this, the preselector becomes the
  bottleneck instead of relieving it.
- **Cross-architecture bit-identity**. Same request on x86, ARM, CUDA,
  WebGPU returns the same top-K. Q16.16 fixed-point throughout, no IEEE-754
  fallback in the inference path.
- **No training data leakage at inference.** The classifier reveals only
  route names, never the training corpora content.
- **Tamper detection.** Every inference emits an attestation envelope tying
  the request hash, model hash, and result hash into the evidence chain.

## Architecture (one paragraph)

Asymmetric encoder/decoder with a classifier head. Encoder reads the request,
no feed-forward blocks (attention + gated residuals only) for compact
representation. Decoder cross-attends to the encoder output and to a fixed
embedding of every available route (skills/tools/agents). Classifier head
emits per-route relevance scores. Top-K extraction is deterministic
tie-breaking by route ID hash. Full spec in
[`spec/architecture.md`](spec/architecture.md).

## Repository structure

```
mind-nerve/
  README.md                       this file
  ROADMAP.md                      phased delivery plan
  LICENSE.md                      Apache-2.0 architecture, weights separate
  spec/                           authoritative design documents
    architecture.md
    quality_targets.md
    integration_surface.md
  src/                            pure MIND implementation
    lib.mind
    model.mind
    inference.mind
    evidence.mind
  cli/
    main.mind                     single-binary entrypoint
  integrations/
    claude-code/                  TypeScript hook shim
    codex/                        shell hook wrapper
    mcp/                          MCP server façade
  tests/
    bit_identity/                 cross-architecture reproducibility
    accuracy/                     classification benchmarks
```

## License

The architecture, integration shims, and reference inference loop are
Apache-2.0. Trained weights are not distributed under this license; weight
distribution is handled separately under STARGA terms.

See [`LICENSE.md`](LICENSE.md) for the full split.

## Dependencies

- `mind-runtime` — provides the 18-backend lowering matrix
- `mind-mem` (optional) — consumes mind-nerve preselection for tool routing

No third-party machine learning framework dependency. No PyTorch, no ONNX,
no TensorFlow in the inference path. Pure MIND end-to-end.
