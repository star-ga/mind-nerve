# autoresearch program — mind-nerve architecture evolution

This file is the research direction for the autoresearch loop. The agent
running each iteration reads this verbatim before deciding what to search,
what to read, and what to propose.

## What mind-nerve is

mind-nerve is a CPU-first intent-classification preselector. A request +
catalog → top-K route IDs in ≤30 ms p95 on commodity 4-core x86 CPU.
Pure MIND inference path. Q16.16 activations × INT8 weights. Cross-arch
bit-identity gate. 212-byte tamper-evident attestation envelope per
inference. The architecture is locked at v3 (2 sliding-window encoder
layers, hidden=256, no FFN sublayer, no decoder, direct dot scoring
against route embedding table). See spec/architecture.md for the
authoritative design.

## The goal of this loop

State-of-the-art. mind-nerve must be the preselector that every agent
runtime reaches for. The way you get there is not by inventing things in
isolation — it is by reading what the rest of the field has shipped in
the last six months and incorporating the techniques that fit our
non-negotiables.

Each iteration of this loop is judged by ONE question: did you surface
a recent piece of research that mind-nerve can adopt, draft an RFC for
it, and survive the arch-mind + skill-improver validation gates?

## Where to look (priority order)

1. **arxiv.org/list/cs.CL** (recent, last 6 months). Filter for:
   - intent classification with small models (< 100M params)
   - tool-routing / function-routing for LLM agents
   - sparse / sliding-window / linear attention variants
   - small-model retrieval / sentence-transformer alternatives
   - quantization (INT4/INT8) with deterministic inference
2. **Hugging Face papers** (huggingface.co/papers, sorted by trending)
   for the same topics, since HF surfaces practitioner-validated work.
3. **STARGA's own ecosystem** — mind-mem, MindLLM, rfn-mind, 512-mind
   sometimes ship techniques mind-nerve can borrow. Don't reinvent.
4. **Industry blogs** — Anthropic, OpenAI, Google DeepMind, Mistral,
   DeepSeek occasionally publish architecture notes that fit our
   constraints.

Do NOT search:
- Papers older than 2024 unless they are foundational and you can show
  why a 2026-era reimplementation would beat a 2023-era one.
- Generic benchmarks (MMLU, HellaSwag) — mind-nerve is a router, not
  a question-answering model; benchmark inflation is irrelevant.
- Closed-source / paywall content — RFCs must reference openly
  readable sources so future reviewers can verify the citation.

## Constraints any proposal MUST respect

Non-negotiable. An RFC that violates any of these is rejected without
further validation.

1. **Pure MIND inference path.** No PyTorch / ONNX / TF on the hot path.
2. **Q16.16 activations + INT8 weights.** No FP16 / BF16 / FP32. No new
   numerical types without amending spec/numerics.md (which is a
   separate workstream, not part of this loop).
3. **Cross-arch bit-identity.** Same bytes on x86, ARM, CUDA, WebGPU,
   NPU. Any technique that requires randomness, non-deterministic
   reduction order, or atomic-RMW with arbitrary ordering is out.
4. **≤30 ms p95 on 4-core x86 at 1024-token cap.** Latency analysis
   must show the proposed change fits the budget (see Mind.toml for
   the breakdown).
5. **Single static binary.** No external ML framework dependency, no
   GPU requirement, no BLAS, no autotuned GEMM.
6. **Tamper-evident envelope chain.** Every inference still emits a
   212-byte envelope contributing to the chain. The proposed change
   cannot bypass this.

If a paper's technique requires breaking one of these, the RFC should
say so explicitly in the "non-negotiable conflict" section and stop —
do not propose half-adoption that quietly relaxes a constraint.

## RFC format

Save proposed RFCs to `research/RFC-NNN-<short-slug>.md`. NNN is a
three-digit incrementing index (look at existing files and use the
next number). The file is gitignored by /research/ in .gitignore, so
RFCs accumulate locally; commit + push them only after human review.

Required sections:

```markdown
# RFC-NNN — <Short Title>

**Source paper:** <full citation + arxiv/HF URL>
**Date discovered:** YYYY-MM-DD
**Iteration:** <autoresearch iteration number>

## One-sentence summary
What the paper proposes, in one sentence.

## Why it fits mind-nerve
Two-paragraph case. Tie it to a specific open question in
spec/architecture.md "Open questions, Phase 1" or to a Phase 2
enhancement listed in ROADMAP.md.

## Adoption plan
Concrete steps:
  1. Module(s) touched
  2. Spec changes required (if any)
  3. Test additions
  4. Expected latency delta (estimate)
  5. Expected accuracy delta (estimate, with the data scope)

## Non-negotiable conflict
Either "None — proposal respects all six non-negotiables" or a
specific section explaining the conflict and stopping there.

## Validation gates run
- arch-mind score before / after
- skill-improver mean before / after
- (latency / accuracy actual numbers if code changes landed)

## Decision
Accepted | Rejected | Needs-human-review
```

## Things to avoid

- Don't propose changing the architecture frame (e.g. "switch back to
  having a decoder", "use FP16 activations"). The architecture is
  FROZEN at v3 for Phase 1. Phase 2 is where bigger reshapes might
  land; this loop targets refinements that survive within v3.
- Don't propose external dependencies. Even excellent libraries are
  out of scope; we ship a single static binary.
- Don't propose changes whose only justification is "the paper has X
  citations." Citations are a popularity signal, not a fit signal.
- Don't propose more than one RFC per iteration. Quality over volume.

## Pinned open questions (highest leverage)

These are the four questions where a good paper today would have the
biggest payoff:

1. **Learnable per-route prior from catalog co-occurrence.** ROADMAP
   Phase 2 #1. We chose this from a 2026 fleet consensus; the
   research question is which exact parametrization (log-frequency
   vs. PMI vs. shrinkage estimator) the best published work uses.
2. **Adaptive window stride.** ROADMAP Phase 2 #3. Entropy-gated
   stride selection is our current sketch; the open question is what
   features of the input actually predict the optimal stride.
3. **Frequency-adaptive route scaling.** ROADMAP Phase 2 #4. Inverse
   square-root scaling is the placeholder; tighter estimators exist
   in recent retrieval literature.
4. **Bit-identical INT8 weight quantization.** Our current scheme is
   per-output-channel with Q16.16 scales. Recent quantization papers
   may show that group-wise (e.g. per-32-element) quantization with
   the same Q16.16 scale type recovers accuracy without breaking the
   bit-identity contract.

A proposal that closes one of these is more valuable than a
proposal addressing a question we have not yet asked.
