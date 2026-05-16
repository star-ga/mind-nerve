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

**Phase 1 — public alpha (`v0.1.0-alpha.3`, 2026-05-16).** Python wheel on PyPI;
weights on Hugging Face. The router runs end-to-end on PyTorch via
`BAAI/bge-small-en-v1.5` fine-tuned with MultipleNegativesRankingLoss; top-5
accuracy is 96.06 % against the v1.1-oss catalog of 11,922 routing candidates.

MIND Language Profile target: `default` (full tensor stdlib + Q16.16 + heap)
— see [`mind` Phase 10.6](https://github.com/star-ga/mind/blob/main/docs/roadmap.md#phase-106--library-output--c-abi-mindc-026--030)
for the `--profile` flag landing in `mindc` 0.2.6. <!-- mind-profile: default -->

- PyPI: <https://pypi.org/project/mind-nerve/>
- Weights: <https://huggingface.co/star-ga/mind-nerve-phase1>

Phase 2 replaces the PyTorch path with a native MIND Q16.16 inference loop
and adds the cross-architecture bit-identity gate + 4-core CPU p95 ≤ 30 ms
latency budget. Phase 2 is gated on `mindc` 0.2.6 (`pub fn` → C symbol
export) and 0.3.0 (cdylib emit). Until Phase 2 closes, the inference path
uses external ML tooling — explicitly permitted by the
[ROADMAP](./ROADMAP.md) Phase 1 exception.

## Quickstart

```bash
pip install mind-nerve==0.1.0a3
```

```python
from mind_nerve import route
result = route("git status", top_k=5)
for r in result.routes:
    print(r.score, r.name, r.kind)
```

Pre-download the runtime once and point `MIND_NERVE_RUNTIME_DIR` at it:

```bash
huggingface-cli download star-ga/mind-nerve-phase1 --local-dir ~/.mind-nerve/phase1
export MIND_NERVE_RUNTIME_DIR=~/.mind-nerve/phase1
```

Auto-download on first `route()` call is on the 0.1.0a4 backlog.

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

mind-nerve ships under **Apache-2.0** — repository, Python wheel, and the
Phase-1 trained weights on Hugging Face all carry the same license. The
wheel additionally bundles `libmindnerve.so`, a FORTRESS-protected runtime
component whose source remains private under STARGA Commercial terms. The
protected binary is the future Phase-2 native inference layer; the Phase-1
PyTorch path does not depend on it.

For commercial deployments needing per-customer FORTRESS-locked builds of
the runtime layer, contact `license@star.ga`. See [`LICENSE.md`](LICENSE.md)
for the full split.

## Dependencies

- `numpy`, `sentence-transformers`, `torch` — Phase-1 inference path
- `mind-runtime` — Phase-2 native inference (gated on `mindc` 0.3.0)
- `mind-mem` (optional) — consumes mind-nerve preselection for tool routing

The "no third-party ML framework" goal applies to **Phase 2**. Phase 1
(this release) deliberately uses sentence-transformers + PyTorch to ship
the API, evaluation harness, and integration surface before the native
runtime lands.
