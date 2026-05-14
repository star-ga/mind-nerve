# autoresearch program — mind-nerve IMPLEMENT phase

This file is the research direction for the autoresearch loop. The agent
running each iteration reads this verbatim before deciding what to do.

The DRAFT phase (iterations #1–#38) surfaced 34 RFCs in `RFCs/INDEX.md`.
This file now drives the IMPLEMENT phase: each iteration picks one
unprocessed RFC, lands the code change in `src/*.mind` (or marks it
SKIPPED with a clear reason), adds or extends a test, and verifies
that arch-mind + skill-improver gates don't regress.

## What mind-nerve is (recap)

CPU-first intent-classification preselector. Request + catalog → top-K
route IDs in ≤30 ms p95 on commodity 4-core x86 CPU. Pure MIND
inference path. Q16.16 activations × INT8 weights. Cross-arch
bit-identity gate. 212-byte tamper-evident attestation envelope per
inference. Architecture is FROZEN at v3 (2 sliding-window encoder
layers, hidden=256, no FFN sublayer, no decoder, direct dot scoring
against route embedding table). See `spec/architecture.md`.

## The goal of this loop (CHANGED)

State-of-the-art mind-nerve, code-complete. The previous loop drafted
RFCs from recent literature. **This loop ships them**. Each iteration
is judged by ONE question: did you make a real, tested, gates-passing
code change that advances mind-nerve toward its quality and latency
targets, OR did you correctly identify an RFC that does not belong on
the inference path and mark it SKIPPED with a clean reason?

When no unprocessed RFCs remain (every entry in `RFCs/INDEX.md` has
a `**Status:** IMPLEMENTED` or `**Status:** SKIPPED` line), respond
with `CHANGE: All RFCs processed. No further improvement possible
within current scope.` — the loop will register no-improvement,
hit the crash budget, and pause.

## What the agent does each iteration

1. **Read `RFCs/INDEX.md` end-to-end** (it is in `target_files`).
   Identify the first RFC whose section does NOT contain a
   `**Status:** IMPLEMENTED` or `**Status:** SKIPPED` line. Call this
   the *target RFC*.

2. **Classify the target RFC**:
   - **Inference-side**: the change touches files in `src/*.mind`
     (model, kernels, scoring, attention, normalization, tokenizer,
     loader, evidence, etc.). These are implementable here.
   - **Training-side**: the change is about how the model is trained
     before shipping — synthetic queries, hard-negative mining,
     curriculum, AnglE/InfoNCE loss, distillation, EMA, SAM, R-Drop,
     ANCE refresh, GradCache, LLRD, RetroMAE pretraining, multi-task
     instruction tuning, etc. These belong in a separate training
     repo, not mind-nerve. **Mark SKIPPED** with a one-line reason
     pointing to where they belong.
   - **Spec-only**: clarifies a constraint or contract but doesn't
     change inference code. Mark SKIPPED with a reason.

3. **For inference-side RFCs — implement**:
   a. Read the source files the RFC names. If the RFC was vague about
      which files, infer from the technique (e.g. attention bias →
      `src/encoder_kernels.mind` + `src/model.mind`; normalization
      change → `src/encoder_kernels.mind`).
   b. Make the minimum change that realizes the RFC and preserves
      every non-negotiable in the section below. Prefer surgical
      patches; do not rewrite modules.
   c. Add or extend ONE test in `src/*_test.mind` (or new
      `tests/*_test.mind` if cleaner). The test must demonstrate the
      new behavior produces the expected Q16.16 result on a fixed
      input.
   d. If the RFC introduces tunable constants, register them in
      `src/lib.mind` alongside the other config constants
      (`ENCODER_LAYERS`, `ATTN_WINDOW_SIZE`, etc.) so the
      bit-identity contract pins them at compile time.
   e. **Backwards-soft default**: pick the default value such that
      the binary stays byte-identical to today when the RFC is
      disabled. This lets us land code now and turn it on after
      trained weights catch up.

4. **For SKIPPED RFCs — mark and move on**:
   a. Append a `**Status:** SKIPPED — <one-line reason>` line to the
      RFC's section in `RFCs/INDEX.md`.
   b. No `src/` changes. The composite metric does not reward SKIPs;
      it only rewards IMPLEMENTED. This is intentional — skipping is
      progress (it shortens the queue) but isn't worth the same as
      shipping.

5. **For IMPLEMENTED RFCs — mark and let the gates speak**:
   a. Append `**Status:** IMPLEMENTED at exp<N>` to the RFC's
      section in `RFCs/INDEX.md` (autorun.py will replace `exp<N>`
      with the commit hash after the commit; for now the marker
      itself is what the metric counts).
   b. The `run_command` then runs:
      - arch-mind structural scan (must not regress vs the anchor)
      - skill-improver report (must not regress)
      - composite includes `+5000` per IMPLEMENTED RFC.

## Constraints any implementation MUST respect

Non-negotiable. A code change that violates any of these is rejected
without further validation.

1. **Pure MIND inference path.** No PyTorch / ONNX / TF on the hot path.
2. **Q16.16 activations + INT8 weights.** No FP16 / BF16 / FP32.
3. **Cross-arch bit-identity.** Same bytes on x86, ARM, CUDA, WebGPU,
   NPU. Reductions stay in fixed order; no atomic-RMW with arbitrary
   ordering; no clock/randomness reads inside pure kernels.
4. **≤30 ms p95 on 4-core x86 at 1024-token cap.** Estimate the
   latency delta in the RFC's IMPLEMENTED note. If the delta + budget
   used so far exceeds 30 ms, push back to the RFC author (in the
   note) before implementing.
5. **Single static binary.** No external ML framework dependency.
6. **Tamper-evident envelope chain.** Every inference still emits a
   212-byte envelope contributing to the chain.

If an RFC can't satisfy all six, mark it SKIPPED with the conflict in
the reason field. Do not implement a "watered-down" version that
quietly relaxes a constraint.

## The composite metric (CHANGED)

```
composite = arch_mind_purity
          + skill_improver_mean * 100
          + rfcs_implemented   * 5000
```

`rfcs_implemented` = `grep -c '^\*\*Status:\*\* IMPLEMENTED' RFCs/INDEX.md`.

Each shipped RFC is worth far more than any structural metric — that
is the point. arch-mind + skill-improver act as guard rails (they
gate against regressions) rather than as the main optimization
target. The +5000 weight keeps the loop biased toward shipping while
the structural metrics still get vetoed on regression.

## Termination

When the agent responds with
`CHANGE: All RFCs processed. No further improvement possible within current scope.`
the metric will not move, the loop will accumulate no-improvement
discards, hit `max_consecutive_failures`, and pause for human review.
That is the natural end state of this phase.
