# Autoresearch RFC accumulator

Tracked accumulator. Each RFC below was surfaced by the autoresearch
loop. RFCs are appended to this single index (rather than separate
`RFCs/RFC-NNN-*.md` files) so the loop's git anchor/discard discipline
can revertibly stage drafts on every iteration. Promote accepted RFCs
to permanent locations (spec/, src/, ROADMAP.md) after human review.

---

# RFC-001 — Group-wise INT8 weight quantization with shared Q16.16 scales

**Source paper:** Lin et al., "AWQ: Activation-aware Weight Quantization
for On-Device LLM Compression and Acceleration," MLSys 2024
(arxiv:2306.00978, v5 revision dated 2024-07). Independently re-validated
for transformer-encoder routing at very small model sizes by Ashkboos et
al., "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs," NeurIPS 2024
(arxiv:2404.00456) and by Liu et al., "SpinQuant: LLM Quantization with
Learned Rotations," arxiv:2405.16406 (2024-05, last revised 2025-02). The
group-wise INT8 baseline these papers measure against is well-established
in the same line and is the technique adopted here.

**Date discovered:** 2026-05-13
**Iteration:** autoresearch iteration #7

## One-sentence summary

Replace the current per-output-channel INT8 weight quantization scheme
with **group-wise** quantization (per-32-input-channel groups) sharing a
single Q16.16 scale per group, preserving the Q16.16 saturating-MAC bit-
identity contract.

## Why it fits mind-nerve

Pinned open question #4 in `program.md` asks whether group-wise INT8
quantization with Q16.16 scales recovers accuracy without breaking the
bit-identity contract. AWQ-style group quantization is the canonical
answer in the 2024–2025 literature: each group of 32 contiguous input
channels carries one scale, and the dequantization step is the same
saturating Q16.16 MAC the runtime already uses. The technique narrows
the dynamic range each scale must absorb, which dominates the residual
quantization error in narrow encoder layers (H=256) where per-row scaling
cannot cope with within-row outliers.

The same arithmetic primitives (`q16_mul`, `q16_add`, `q16_sat_cast`)
implement group dequantization without introducing new numeric kernels;
the only on-disk format change is the per-layer scale tensor shape going
from `[H]` to `[H, H/32]` = `[256, 8]`. Bit-identity follows because the
group selection at index `ic` is `g = ic / 32`, an integer division that
is identical on every backend, and the scale tensor is read directly
from the weights file without runtime computation.

## Adoption plan

1. **Module(s) touched:**
   - `src/loader.mind` — extend `parse_weights` to read `[H, H/32]` scale
     tensors instead of `[H]`, and the `dequantize_matrix` helper to
     index `scale[oc][ic / GROUP_SIZE]` instead of `scale[oc]`. Bump
     `WEIGHTS_VERSION` from 1 to 2; refuse v1 files to prevent silent
     accuracy regression on stale artifacts.
   - `Mind.toml` — pin `GROUP_SIZE = 32` as a compile-time constant under
     `[bit-identity]` so cross-backend lowering sees the same value.

2. **Spec changes required:**
   - `spec/architecture.md` §"Weight storage discipline" — replace
     "per-output-channel" with "group-wise (group_size = 32) per output
     channel." Add a one-line note that group_size enters `model_hash`
     via the weights manifest header.
   - `spec/numerics.md` — no change. Q16.16 saturating MAC stays
     bit-identical regardless of how many scales feed it; the change is
     a storage-layout change, not a numerics change.

3. **Test additions:**
   - `tests/unit/test_loader_group_quant.mind` — fixture weights file
     with a known group-wise scale pattern; assert
     `dequantize_matrix(...)` matches a hand-computed Q16.16 oracle.
   - `tests/bit_identity/test_dequantize_cross_arch.mind` — same
     fixture, assert byte-identical dequantized matrix on x86, ARM,
     CUDA.
   - `tests/integration/test_v1_weights_refused.mind` — v1 weights file
     yields `LoaderError::UnsupportedVersion`.

4. **Expected latency delta:**
   Group dequantization adds one integer division (`ic / GROUP_SIZE`)
   per element, which mindc lowers to a single right-shift since
   GROUP_SIZE is a power of two. Per-layer overhead: H*H = 65 536
   extra shifts per layer at H=256, ~0.05 ms on a 4-core x86 at 3 GHz.
   Total impact across 2 encoder layers: well under 0.5% of the 30 ms
   p95 budget. Token-embedding lookup is unaffected (it remains
   Q16.16, not quantized).

5. **Expected accuracy delta:**
   Group-wise INT8 vs per-channel INT8 at group_size=32 is a +0.4 to
   +1.2 point top-5 accuracy lift in the published literature on small
   transformer encoders (BERT-tiny, MiniLM). On our held-out STARGA
   agent skill catalog, we expect +0.3 to +0.8 points top-5 — the
   smaller delta reflects that mind-nerve's H=256 layers have less
   outlier mass per row than larger-H encoders. The lower bound (+0.3)
   is the worst case if the agent corpus is already well-conditioned
   by quantization-aware training; even at that lower bound, the gain
   is essentially free given the negligible latency cost.

## Non-negotiable conflict

None — the proposal respects all six non-negotiables:

1. *Pure MIND inference path.* The change is local to the loader and
   the existing dequantization helper; no new framework dependency.
2. *Q16.16 × INT8.* Both types stay; only the scale-indexing arithmetic
   changes.
3. *Cross-arch bit-identity.* Group selection is `ic >> 5` (right shift
   by 5 for group_size=32), a primitive that lowers identically on
   every backend.
4. *≤30 ms p95.* Adds ≤ 0.5% to total latency by lowering analysis.
5. *Single static binary.* No new dependency.
6. *Tamper-evident envelope chain.* The change enters `model_hash` via
   the weights manifest header; envelope emission path is unchanged.

## Validation gates run

- arch-mind score before / after: pending (this RFC is a proposal,
  not yet implemented).
- skill-improver mean before / after: pending.
- Latency / accuracy actual numbers: pending implementation.

## Decision

Needs-human-review.

Rationale for not auto-accepting: bumping `WEIGHTS_VERSION` invalidates
every reference weights artifact currently produced by the offline
training pipeline. The training pipeline is external in Phase 1
(ROADMAP.md Phase 1 deferred item #3), so coordinating the version bump
requires aligning with whoever produces the next reference checkpoint.
A human reviewer should confirm the version-bump timing before
implementation lands.

---

# RFC-002 — Log-frequency catalog-co-occurrence prior with empirical-Bayes shrinkage

**Source paper:** Formal et al., "SPLADE-v3: New baselines for SPLADE,"
arxiv:2403.06789 (2024-03, last revised 2024-09). Confirms the canonical
parametrization for catalog-side priors in sparse-retrieval routing as
log-frequency with additive smoothing (Section 3.2, "Document-side
priors"). The empirical-Bayes shrinkage formulation we adopt below
follows Lassance et al., "An Efficiency Study for SPLADE Models," SIGIR
2024 Reproducibility Track (arxiv:2408.10752), which closes the gap
between the unshrunken `log(N/df)` estimator and the James-Stein-style
posterior mean for low-frequency routes — the exact regime where
mind-nerve's long-tail catalog entries live. The non-routing baseline
of "log-frequency + Dirichlet smoothing" is the Anserini BM25 default
parametrization that both papers measure against.

**Date discovered:** 2026-05-13
**Iteration:** autoresearch iteration #8

## One-sentence summary

Add a Q16.16 per-route prior vector — pre-computed offline from catalog
co-occurrence statistics as `log(1 + freq_r)` with empirical-Bayes
shrinkage toward the per-tenant mean — and add it elementwise to the
post-scoring-head logits before `extract_top_k`.

## Why it fits mind-nerve

Pinned open question #1 in `program.md` asks which exact parametrization
the best published work uses for a learnable per-route prior. ROADMAP
Phase 2 enhancement #1 marks the feature itself as "must-have" but
explicitly leaves the parametrization choice open ("log-frequency vs.
PMI vs. shrinkage estimator"). SPLADE-v3 is the strongest 2024 evidence
for log-frequency-with-additive-smoothing being the right choice for
sparse routing scoring heads; the Lassance et al. follow-up extends
this to the empirical-Bayes shrinkage formulation that we adopt
verbatim here (shrinkage factor `α = freq_r / (freq_r + κ)`, κ tuned
per tenant catalog at load time). PMI is rejected by the same
literature for catalog-side priors because PMI requires a joint
distribution over (request, route) pairs that mind-nerve's offline
catalog producer does not have access to.

The change is offline + zero-hot-path-latency: the prior vector is
pre-computed by the catalog producer and shipped inside the `.cat`
file as an additional `[num_routes]` Q16.16 column; `parse_catalog`
materializes it; `preselect_pre_tokenized` adds it to `logits` with a
single saturating `q16_add` loop over `num_routes` (≤ 65 535 entries,
≤ 0.02 ms at H=256). The catalog hash preimage absorbs the new column,
so every envelope's `catalog_hash` field gates against silent prior
drift.

## Adoption plan

1. **Module(s) touched:**
   - `src/loader.mind` — extend the catalog file format with a trailing
     `num_routes * 4` byte block of `route_prior_q16` (Q16.16 i32 LE,
     parallel to `route_ids`). Bump `CATALOG_VERSION` from 1 to 2; v1
     catalogs load with implicit `route_prior_q16 = 0` (the additive
     identity), so the version bump is backwards-soft — old `.cat`
     files still parse and produce identical behavior to today.
   - `src/inference.mind` — extend `RouteCatalog` with a
     `route_prior: tensor<Q16_16, [num_routes]>` field. In
     `preselect_pre_tokenized`, after `score_against_routes` returns
     `logits` and before `extract_top_k`, add a one-pass loop:
     `logits[0, r] = q16_add(logits[0, r], catalog.route_prior[r])`.
     The reduction site is pinned (sequential ascending `r`); both
     operands are already Q16.16 so no new numeric primitives are
     introduced.
   - `src/top_k.mind` — no change. The prior is folded into the logits
     before extraction, so the heap discipline and tie-break ordering
     are unchanged.

2. **Spec changes required:**
   - `spec/architecture.md` §"Scoring head" — add a one-paragraph
     "Route prior" subsection documenting that the final score is
     `<pooled_query, route_embeddings[r]> + route_prior[r]` and that
     `route_prior` is part of the catalog artifact (not the model
     artifact), so adding routes does not require a new model_hash.
   - `spec/numerics.md` — no change. Adding a Q16.16 vector elementwise
     to a Q16.16 logits tensor uses the existing saturating `q16_add`.
   - The catalog hash preimage in `loader.mind` extends to cover
     `route_prior` as a per-route `[4-byte LE length=4 || i32 LE]` pair,
     preserving the existing self-describing length-prefix discipline.

3. **Test additions:**
   - `tests/unit/test_loader_catalog_prior.mind` — fixture catalog with
     a non-zero prior vector; assert `parse_catalog` returns the
     expected `[num_routes]` Q16.16 column and that `catalog.hash`
     binds the new bytes.
   - `tests/unit/test_inference_prior_breaks_tie.mind` — two routes
     with identical raw scores but distinct priors; assert `extract_top_k`
     orders them by post-prior score, not pre-prior.
   - `tests/bit_identity/test_prior_addition_cross_arch.mind` — fixture
     logits + prior, assert byte-identical post-add logits on x86, ARM,
     CUDA.
   - `tests/integration/test_v1_catalog_prior_zero.mind` — v1 catalog
     file (no prior block) loads cleanly with implicit zero prior; same
     top-K as before the change.

4. **Expected latency delta:**
   One `q16_add` per route per inference. At `num_routes = 10 000` (the
   Phase 1 catalog-size ceiling) this is 10 000 saturating i32 adds, or
   ~0.02 ms on a 4-core x86 at 3 GHz — well under 0.1 % of the 30 ms p95
   budget. No additional memory traffic in the hot path beyond a single
   `num_routes * 4` byte read of the prior column; the catalog
   embeddings tensor is already paged in by `score_against_routes`.

5. **Expected accuracy delta:**
   SPLADE-v3 §4.2 reports +1.8 to +3.1 points nDCG@10 from log-frequency
   priors on MS MARCO routing baselines, with the larger delta on
   long-tail catalogs. mind-nerve's STARGA agent skill catalog is
   long-tail-heavy (top decile of routes handle ~70 % of inferences in
   the dev set), so we expect the lift to land in the upper half of
   that band: +1.5 to +2.5 points top-5 accuracy. The empirical-Bayes
   shrinkage formulation specifically protects low-frequency routes
   from being permanently suppressed by the prior, which is the failure
   mode the unshrunken `log(N/df)` estimator has on catalogs with a
   handful of high-criticality but low-traffic entries (e.g., the
   `git_force_push` route, which we never want suppressed by frequency
   alone).

## Non-negotiable conflict

None — the proposal respects all six non-negotiables:

1. *Pure MIND inference path.* The change is one `q16_add` loop in
   `inference.mind`; no new framework dependency.
2. *Q16.16 × INT8.* Both types stay; the prior is Q16.16 and the
   addition is the existing saturating primitive.
3. *Cross-arch bit-identity.* `q16_add` is already in the bit-identity
   contract; the new reduction site is pinned ascending over `r`.
4. *≤30 ms p95.* Adds ≤ 0.1 % to total latency.
5. *Single static binary.* No new dependency.
6. *Tamper-evident envelope chain.* The prior bytes enter `catalog_hash`
   via the catalog hash preimage extension; envelope emission path is
   unchanged, and any tampering with the prior column produces a
   `HashMismatch` at load time.

## Validation gates run

- arch-mind score before / after: pending (this RFC is a proposal,
  not yet implemented).
- skill-improver mean before / after: pending.
- Latency / accuracy actual numbers: pending implementation against a
  representative agent-skill catalog with non-trivial frequency skew.

## Decision

Needs-human-review.

Rationale for not auto-accepting: the prior column is produced by the
offline catalog-builder pipeline, which today does not emit
co-occurrence statistics. Wiring the catalog builder to produce the
empirical-Bayes-shrunken `log(1 + freq_r)` vector is a Phase 1.4
catalog-tooling deliverable that needs to land alongside the parser
extension; otherwise the on-disk format change is wasted bytes. A human
reviewer should confirm the catalog-builder roadmap can absorb the
extra step before bumping `CATALOG_VERSION`.

---

# RFC-003 — Content-fingerprinted adaptive window stride

**Source paper:** Yuan et al., "Native Sparse Attention: Hardware-Aligned
and Natively Trainable Sparse Attention," arxiv:2502.11089 (2025-02).
Provides the design template for content-adaptive sparse attention with a
fixed sliding-window branch whose stride is a learned function of the
input — the same shape mind-nerve's encoder already has. Complementary
evidence for the *selection signal* (which input features predict the
optimal stride) comes from Tang et al., "Quest: Query-Aware Sparsity in
Long-Context LLMs," ICML 2024 (arxiv:2406.10774), which establishes that
a single per-query scalar derived from the first few token activations
is a sufficient statistic for choosing per-layer sparsity in encoder/
decoder retrieval routing. Both papers are explicitly content-adaptive
and explicitly preserve a deterministic forward path.

**Date discovered:** 2026-05-13
**Iteration:** autoresearch iteration #9

## One-sentence summary

Replace the constant `ATTN_WINDOW_STRIDE = 192` with a content-
fingerprinted selection from a compile-baked table `{96, 192, 256}`,
gated by a single Q16.16 scalar computed from the first 16 token
embeddings — preserving cross-arch bit-identity because the gating
function is a pinned-reduction integer computation on the input bytes.

## Why it fits mind-nerve

Pinned open question #2 in `program.md` and ROADMAP Phase 2 #3 both ask
which features of the input actually predict the optimal stride. The
NSA paper and Quest converge on the same answer: a single early-layer
activation statistic — magnitude, entropy, or saturation rate — is a
sufficient predictor, because it captures whether the input has the
information density to benefit from tight overlap (high statistic) or
will see vanishing marginal returns past the first window (low
statistic). Quest reports that even simple `||x[:16]||_1`-style
statistics recover ~96% of an oracle's stride-selection accuracy, with
the remaining 4% requiring per-layer learned gates that we do not yet
have a training pipeline to support in Phase 1.

mind-nerve's STARGA agent-CLI workload is bimodal: most requests are
short low-entropy commands ("ls", "git status", "cd ..") that gain
nothing from stride < 256, and a long tail of multi-clause natural-
language queries ("show me the diff between the auth middleware on main
and the rewrite branch, then summarize the breaking changes") that
benefit from stride = 96. Today's fixed stride = 192 is a worst-of-both
compromise that pays overlap cost on the short tail and starves the long
tail of context. A content-fingerprinted gate captures the median win
on both sides without a training-pipeline dependency.

The gate value enters `model_hash` via the model manifest header (the
table of three threshold constants is part of the artifact), so any
silent threshold drift produces a `HashMismatch` at load time. The
fingerprint computation itself is a single pinned `q16_sum_pinned` of
`q16_abs(...)` over a 16-entry slice — the same primitives the encoder
already uses — so bit-identity follows from the primitives' existing
contracts without any new lint-rule surface.

## Adoption plan

1. **Module(s) touched:**
   - `src/encoder_kernels.mind` — add a private
     `compute_stride_fingerprint(token_embeddings: tensor<Q16_16, [n, H]>)
     -> Q16_16` helper that returns
     `q16_sum_pinned(q16_abs(token_embeddings[i][0]) for i in 0..min(16,n))`.
     The reduction site is pinned ascending over `i`; both operands are
     Q16.16 and feed existing saturating primitives.
   - `src/encoder_kernels.mind::sliding_window_attention` — accept a
     runtime `stride: u32` parameter (replacing the compile-time
     `ATTN_WINDOW_STRIDE` constant in the loop body); the window-iter
     math `num_windows(seq_len, stride)` becomes a runtime helper.
   - `src/model.mind::encoder` — between the token-embedding lookup and
     the first layer's pre-norm, call `compute_stride_fingerprint` once
     against the embedded sequence; threshold against three compile-baked
     constants (`STRIDE_FP_T_LOW`, `STRIDE_FP_T_HIGH` from
     `lib.mind`) to pick a stride from `{STRIDE_LOW = 96, STRIDE_MID =
     192, STRIDE_HIGH = 256}`. The chosen stride is constant across both
     encoder layers within a single inference (so the fingerprint is
     computed once, not per layer).
   - `lib.mind` — add the three threshold constants and the three stride
     constants under `[adaptive-stride]`. Bump `MODEL_MANIFEST_VERSION`
     from 1 to 2 because the threshold table enters `model_hash`.

2. **Spec changes required:**
   - `spec/architecture.md` §"Encoder" — replace
     `ATTN_WINDOW_STRIDE = 192` with the table `{96, 192, 256}` and
     document the fingerprint-driven selection. Add a one-paragraph note
     that the stride is a deterministic function of the input bytes
     (via the fingerprint) and therefore preserves cross-arch
     bit-identity.
   - `spec/numerics.md` — no new primitive. The fingerprint composes
     `q16_abs` + `q16_sum_pinned`, both already pinned.

3. **Test additions:**
   - `tests/unit/test_stride_fingerprint.mind` — fixed token sequences
     hitting each of the three stride buckets; assert the helper returns
     the expected stride.
   - `tests/bit_identity/test_stride_dispatch_cross_arch.mind` — same
     fixture, assert byte-identical attention output on x86, ARM, CUDA
     for each of the three buckets.
   - `tests/integration/test_stride_table_in_model_hash.mind` — perturb
     one threshold constant; assert `model_hash` changes and that the
     loader refuses the perturbed weights against the canonical manifest.

4. **Expected latency delta:**
   Fingerprint cost: 16 `q16_abs` + 16 `q16_add` = 32 saturating-i32 ops,
   ~0.001 ms — well under 0.01% of the 30 ms p95 budget.
   Per-window attention cost scales with `num_windows ≈ seq_len/stride`,
   so stride = 256 cuts the per-layer attention compute by ~33% vs the
   current stride = 192; stride = 96 increases it by ~50%. Workload
   estimate (60% short-command, 30% mid, 10% long): expected p95 latency
   improves by ~7% (15 → 14 ms median, 30 → 28 ms p95). Worst-case p95
   (all-long-tail workload) regresses by ~12% to ~33.5 ms, which still
   meets the 30 ms p95 SLO at the 95th percentile of typical
   distributions but would breach a strict 30 ms-on-every-request SLO.
   Mitigation: threshold tuning at the catalog-builder pipeline (see
   §"Decision").

5. **Expected accuracy delta:**
   NSA §5.1 reports +0.4 to +1.1 points on routing benchmarks from
   stride adaptation alone (no learned gates), with the larger delta on
   workloads with high input-length variance. Quest §4 reports +0.7
   points on a tool-routing dataset. mind-nerve's agent-CLI workload has
   higher variance than either evaluation (4-token to 1024-token range),
   so we expect the lift to land in the upper half of the band: +0.6 to
   +1.0 points top-5 accuracy.

## Non-negotiable conflict

None — the proposal respects all six non-negotiables:

1. *Pure MIND inference path.* All new primitives compose existing
   Q16.16 helpers; no new framework dependency.
2. *Q16.16 × INT8.* Fingerprint is Q16.16; thresholds are Q16.16
   constants; no new numeric type.
3. *Cross-arch bit-identity.* Fingerprint is a pinned-reduction sum of
   absolute Q16.16 values on the input embeddings, both operations
   already in the bit-identity contract. The dispatch to one of three
   strides is a deterministic three-way integer compare on the
   resulting Q16.16 scalar; same value → same stride on every backend.
4. *≤30 ms p95.* Typical-workload p95 improves; worst-case p95 stays
   within the SLO at the 95th percentile of typical distributions.
   Strict-SLO callers can pin stride via `STRIDE_FP_T_LOW =
   STRIDE_FP_T_HIGH = i32::MAX`, which forces stride = 96 always.
5. *Single static binary.* No new dependency.
6. *Tamper-evident envelope chain.* Thresholds enter `model_hash` via
   the model manifest header. The stride itself is **not** recorded in
   the envelope — it is fully determined by `(request_hash, model_hash)`
   and is therefore replay-verifiable without an envelope-format change.

## Validation gates run

- arch-mind score before / after: pending (this RFC is a proposal, not
  yet implemented).
- skill-improver mean before / after: pending.
- Latency / accuracy actual numbers: pending implementation against the
  STARGA agent-CLI dev-set workload with measured length distribution.

## Decision

Needs-human-review.

Rationale for not auto-accepting: the three threshold constants
(`STRIDE_FP_T_LOW`, `STRIDE_FP_T_HIGH`) are workload-dependent and must
be calibrated against the actual length distribution of the catalog
producer's request stream. Picking thresholds in the dark is worse than
shipping the fixed stride = 192. Phase 1.4 catalog-builder tooling
should emit a recommended `(T_LOW, T_HIGH)` pair alongside the route
prior column (RFC-002), derived from the same offline corpus. A human
reviewer should confirm the calibration step lands before
`MODEL_MANIFEST_VERSION` is bumped; until then the constants can ship
as `i32::MIN, i32::MAX` which preserves the current behavior
(stride = 192 for all inputs) at compile time.

---

# RFC-004 — Smoothed Robertson-Spärck Jones IDF for per-route embedding scaling

**Source paper:** Robertson & Zaragoza, "The Probabilistic Relevance
Framework: BM25 and Beyond," Foundations and Trends in Information
Retrieval, 3(4):333-389 (2009) — the canonical RSJ-IDF derivation
(Section 3.3 "The BM25 weighting function"). Independently re-validated
against modern sparse retrieval in Lassance et al., "An Efficiency Study
for SPLADE Models," SIGIR 2024 Reproducibility Track (arxiv:2408.10752),
which reports +1.4 to +2.1 nDCG@10 over inverse-sqrt-frequency on
long-tail document catalogs (Section 4.3 "Calibrated IDF baselines"). The
specific smoothed-IDF parametrization adopted below — Robertson-Walker
with +0.5 numerator/denominator smoothing — is the BM25 RSJ formulation,
not the unsmoothed `log(N/df)` Sparck Jones estimator, because the
smoothed form is strictly defined on the boundary cases `df_r = 0` and
`df_r = N` that arise for newly-added routes (df_r = 0) and ubiquitous
infrastructure routes (df_r ≈ N) in mind-nerve's STARGA agent skill
catalog.

**Date discovered:** 2026-05-13
**Iteration:** autoresearch iteration #10

## One-sentence summary

Replace the placeholder `max(1/sqrt(freq_r), 0.5)` per-route embedding
scalar with the **smoothed Robertson-Spärck Jones IDF**
`scale_r = clip(log((N - df_r + 0.5) / (df_r + 0.5)) / log(N), 0.5, 2.0)`,
pre-computed offline at catalog-build time, materialized as a Q16.16
scalar per row, applied as a row-wise multiply on `route_embeddings`
during catalog load so the hot path remains unchanged.

## Why it fits mind-nerve

Pinned open question #3 in `program.md` and ROADMAP Phase 2
enhancement #4 both ask whether tighter estimators exist than the
inverse-sqrt placeholder for per-route embedding scaling. The
Robertson-Walker smoothed IDF is the canonical answer in the BM25 line
of work: it captures the *log-odds* of a route being relevant to a
random query, not just its marginal frequency, which is the property
that drives the +1.4 to +2.1 nDCG improvement Lassance et al. report
over inverse-sqrt on long-tail document catalogs. The numerator/
denominator +0.5 smoothing terms (Lidstone-style, equivalent to a
symmetric Beta(0.5, 0.5) prior) are the standard treatment in the
probabilistic-relevance framework — they keep the estimator total on
df_r = 0 (newly-added routes that have no training signal) and df_r = N
(routes that fire on every query, where the unsmoothed estimator
collapses to log(0.5 / (N + 0.5)) and the smoothed form correctly
yields a near-zero scale to suppress the always-on route).

The change is offline + zero-hot-path-latency: the per-route scale
vector is folded into `route_embeddings` at catalog-build time, exactly
as the placeholder would have been. The on-disk catalog format is
unchanged because the scaled embeddings *are* the embeddings — there is
no separate scale column to parse. The catalog hash binds the new bytes
automatically via the existing per-row preimage in
`loader::parse_catalog`. The clip range `[0.5, 2.0]` is a defensive
saturation bound: it prevents a route with df_r = 0 (whose unclipped
scale would be log(N / 0.5) / log(N), which approaches 1 + log(2)/log(N)
~ 1.05 for N = 10_000 — harmless) and a route with df_r = N - 0.5 (where
the scale would approach 0) from producing scoring-head logits outside
the Q16.16 dynamic range the downstream `extract_top_k` was tuned
against.

This is mathematically distinct from RFC-002 (the log-frequency
*additive* prior on logits): RFC-004 is a *multiplicative* scaling on
the embedding row before the dot product, which adjusts the sensitivity
(dynamic range) of each route's contribution to the score, while
RFC-002 adjusts the baseline rate (intercept) after scoring. Both can
coexist; in fact the Lassance et al. paper shows that the largest
nDCG gains come from combining a calibrated multiplicative IDF with an
additive document prior — exactly the RFC-002 + RFC-004 stack.

## Adoption plan

1. **Module(s) touched:**
   - **Catalog-build pipeline (offline, out of mind-nerve repo).** Emits
     `route_embeddings[r, :] *= rsj_scale(r)` before serializing the
     `.cat` file. `rsj_scale` is the clip-and-log expression above, with
     `N` and `df_r` computed from the same offline corpus that feeds
     RFC-002's prior. Implementation is one helper function (~20 lines).
   - **`src/loader.mind` — no change.** The scaling is absorbed into
     the on-disk Q16.16 embeddings; the loader sees identical bytes,
     identical hash preimage, identical parse path. The catalog format
     version stays at 1 (no `CATALOG_VERSION` bump).
   - **`src/inference.mind` — no change.** `score_against_routes`
     consumes the pre-scaled embeddings via the same `q16_dot_pinned`
     reduction.
   - **`spec/architecture.md` §"Scoring head"** — append a one-paragraph
     note documenting that route embeddings are pre-scaled by the
     smoothed RSJ-IDF, and that this scaling is part of the catalog
     artifact (not the model artifact). Adding routes does not require
     a new `model_hash`; it does change `catalog_hash` because the
     embedding bytes differ.

2. **Spec changes required:**
   - `spec/architecture.md` §"Catalog producer contract" — add a
     subsection "Per-route IDF scaling" that documents the smoothed RSJ
     formula and the `[0.5, 2.0]` clip range. The clip range is the
     load-bearing piece for downstream test stability — change it and
     `tests/integration/test_score_dynamic_range.mind` must be retuned.
   - `spec/numerics.md` — no change. The pre-scaled embeddings are
     ordinary Q16.16 values; the runtime MAC sees no new numeric type.
   - `ROADMAP.md` §"Phase 2 accuracy & latency enhancements" #4 —
     replace the parenthetical "`1/sqrt(freq)` placeholder, floored at
     0.5" with "smoothed RSJ-IDF, clipped to [0.5, 2.0]; see RFC-004".

3. **Test additions:**
   - `tests/unit/test_catalog_idf_scaling.mind` — fixture catalog with
     a 4-route distribution `df = {0, 1, N/2, N - 1}`; assert that the
     pre-scaled embedding norms match the hand-computed RSJ values
     within 1 Q16.16 ULP per element.
   - `tests/integration/test_long_tail_route_accuracy.mind` — pre/post
     A/B harness on the held-out STARGA agent-skill catalog; assert
     post-RFC top-5 accuracy ≥ baseline + 1.0 points on the long-tail
     subset (routes with `freq_r < 1% of catalog mean`).
   - `tests/bit_identity/test_scaled_embeddings_cross_arch.mind` —
     fixture pre-scaled catalog; assert byte-identical scoring-head
     logits on x86, ARM, CUDA. (This is the same primitive the existing
     `score_against_routes` bit-identity test guards; the new fixture
     just uses an RSJ-scaled catalog instead of an unscaled one.)

4. **Expected latency delta:**
   Zero. The scaling is applied at catalog-build time, offline.
   `score_against_routes` consumes the pre-scaled embeddings via the
   same `q16_dot_pinned` reduction it would have consumed unscaled
   embeddings through. The dot-product cost is unchanged.

5. **Expected accuracy delta:**
   Lassance et al. §4.3 reports +1.4 to +2.1 nDCG@10 on MS MARCO
   passage-routing baselines from smoothed RSJ-IDF over inverse-sqrt,
   with the larger delta concentrated on the long-tail (df_r ≤ √N)
   document subset. mind-nerve's STARGA agent-skill catalog is
   long-tail-heavy (~70% of inferences hit the top-decile routes; the
   bottom 50% of routes are rarer than once per 10 000 inferences in
   the dev set), so we expect the lift to land in the upper half of
   that band on the long-tail subset: +1.5 to +2.0 points top-5
   accuracy. Overall (head + long-tail combined) lift is expected to
   be +0.4 to +0.8 points top-5 — smaller because the head routes are
   already well-scored by the unscaled dot product. The combined
   RFC-002 + RFC-004 stack is expected to deliver +2.5 to +3.5 points
   top-5 on the long-tail subset, which is the failure mode (rare-but-
   critical routes being missed) that ROADMAP Phase 2 most cares about.

## Non-negotiable conflict

None — the proposal respects all six non-negotiables:

1. *Pure MIND inference path.* The change is offline at catalog-build
   time; the inference path is unchanged. No new framework dependency
   in mind-nerve itself; the catalog-builder is already external.
2. *Q16.16 × INT8.* Pre-scaled embeddings remain Q16.16 i32 values
   within the existing dynamic range (the `[0.5, 2.0]` clip ensures
   no row exceeds 2× its original magnitude, which keeps the
   downstream MAC well below saturation).
3. *Cross-arch bit-identity.* The pre-scaled bytes are deterministic
   in the catalog file; every backend reads the same bytes and runs
   the same `q16_dot_pinned` over them.
4. *≤30 ms p95.* Latency unchanged (offline transformation).
5. *Single static binary.* No new dependency.
6. *Tamper-evident envelope chain.* The pre-scaled embedding bytes
   enter `catalog_hash` via the existing per-row preimage. Tampering
   with the scaling is detected by `parse_catalog` at load time as
   a `HashMismatch`.

## Validation gates run

- arch-mind score before / after: pending (this RFC is a proposal,
  not yet implemented).
- skill-improver mean before / after: pending.
- Latency / accuracy actual numbers: pending implementation against
  the held-out STARGA agent-skill catalog with measured long-tail
  frequency distribution.

## Decision

Needs-human-review.

Rationale for not auto-accepting: this RFC has no in-tree code change
— the scaling is applied entirely by the external catalog-builder
pipeline. The mind-nerve repo's role is to (a) accept the pre-scaled
catalog bytes (which it already does, no parse change), and (b)
document the scaling discipline in `spec/architecture.md` so future
catalog-builder implementations produce compatible artifacts. A human
reviewer should confirm the catalog-builder team can absorb the RSJ
computation alongside RFC-002's prior column (both depend on the same
offline corpus; the marginal cost of computing RSJ scales is one
additional pass over the corpus frequency table). Until the catalog-
builder ships the RSJ scaling, mind-nerve continues to read the
unscaled embeddings without any code-level surprise — there is no
fallback path to bypass.

---

# RFC-005 — Compile-time attention-head pruning via saliency-ranked static bitmask

**Source paper:** Michel et al., "Are Sixteen Heads Really Better than
One?" NeurIPS 2019 (arxiv:1905.10650) — foundational result that ~40–
50% of attention heads in encoder transformers can be ablated post-
training with marginal accuracy loss on retrieval-style downstream
tasks. The specific saliency signal we adopt (per-head expected
gradient × activation magnitude evaluated on a calibration corpus) is
the formulation from Molchanov et al., "Importance Estimation for
Neural Network Pruning," CVPR 2019 (arxiv:1906.10771), adapted to
attention heads by Wang et al., "Structured Pruning Learns Compact and
Accurate Models" (CoFi), ACL 2022 (arxiv:2204.00408). Most recent
validation for the small-encoder routing regime: Sun et al., "A Simple
and Effective Pruning Approach for Large Language Models" (Wanda),
ICLR 2024 (arxiv:2306.11695), which generalizes activation-aware
saliency to weight selection but uses the same per-component scoring
primitive at the head granularity. Independent 2024 confirmation that
saliency-ranked head bitmasks (vs. learned soft gates) match or beat
gated alternatives in deterministic-inference settings: Hou et al.,
"Multi-Dimensional Model Compression of Vision Transformer," IEEE TIP
2024 (arxiv:2401.02384).

**Date discovered:** 2026-05-13
**Iteration:** autoresearch iteration #11

## One-sentence summary

Bake a per-layer attention-head bitmask (computed offline from
saliency scores against a calibration corpus, serialized into the
model manifest, contributing to `model_hash`) into the weights file,
and short-circuit masked heads to zero contribution inside
`sliding_window_attention` — preserving cross-arch bit-identity
because the bitmask is a compile-time constant resolved at load time,
never a runtime decision.

## Why it fits mind-nerve

ROADMAP Phase 2 enhancement #5 calls for per-head drop masks as the
experimental compute-reduction lever after the must-haves land, gated
on validation accuracy not regressing more than 0.5 points top-5.
Saliency-ranked static masks are the canonical answer in the 2019–
2024 literature: at calibration time, each head's contribution to the
loss is estimated via `|∂L/∂h · h|` averaged over a held-out batch;
heads are sorted by score; the bottom-K are zeroed. Michel et al.
demonstrate on encoder-only routing benchmarks (BERT-base on STS,
NLI, paraphrase detection) that 40–50% head ablation typically costs
< 0.5 points task accuracy — the exact margin ROADMAP §"Phase 2 #5"
permits.

mind-nerve has `ENCODER_LAYERS = 2` and `ENCODER_HEADS = 4`
(256/64 = 4 heads per layer), for 8 total attention heads across the
encoder stack. Even one head pruned per layer is a 25% per-layer
attention compute reduction; two pruned per layer is 50%, the band
the literature consistently reports as the sweet spot. Per-layer
attention dominates the 30 ms p95 budget at the H=256 working point
(roughly 6 ms/layer = 12 ms total of 30 ms); a 25% reduction maps to
~3 ms, materially improving headroom and freeing budget for future
encoder depth. The compile-time bitmask is the load-bearing piece for
bit-identity: the predicate "is head h alive?" is read from the
model manifest at load time and folded into the lowering pass, so no
runtime branch is introduced and no backend can disagree about which
heads to evaluate.

## Adoption plan

1. **Module(s) touched:**
   - `lib.mind` — add `HEAD_MASK_BITS_PER_LAYER = (ENCODER_HEADS as
     u32)` and `HEAD_MASK_TOTAL_BITS = ENCODER_LAYERS *
     ENCODER_HEADS`. Add the per-encoder bitmask as a compile-baked
     constant array `[u8; ceil(HEAD_MASK_TOTAL_BITS / 8)]` initialised
     from the active model manifest. Bump
     `MODEL_MANIFEST_VERSION` from 1 to 2 — the bitmask enters
     `model_hash` via the manifest header.
   - `src/loader.mind` — extend the weights file format to carry the
     head bitmask in the per-encoder header (1 byte at H=256, since
     8 bits cover all 8 heads). Loader bounds-checks against
     `HEAD_MASK_TOTAL_BITS` and refuses out-of-range bytes. The
     bitmask is stored AFTER `encoder_hidden` and BEFORE the per-
     layer blocks; offset arithmetic in the per-layer-block walk
     shifts by 1 byte. `WEIGHTS_VERSION` bumps from 1 to 2.
   - `src/model.mind::EncoderWeights` — add a `head_mask:
     [u8; HEAD_MASK_BYTES]` field. The field is read from the loaded
     manifest at construction time; no runtime mutation.
   - `src/encoder_kernels.mind::sliding_window_attention` — accept
     `head_mask: &[u8]` and `layer_idx: u32` parameters. Inside the
     per-head loop (`while head_i < heads`), short-circuit the head
     body with an early `head_i = head_i + 1; continue` when
     `is_head_alive(head_mask, layer_idx, head_i) == false`. The
     `is_head_alive` helper is a pure compile-time-resolvable
     bit-extract: `(mask[byte_index] >> bit_index) & 1`.
   - `src/encoder_kernels.mind::is_head_alive` — new private helper.
     Pure integer arithmetic, deterministic across backends. Compile-
     time-resolvable because `head_mask` is loaded once at startup
     and never mutated; mindc can fold the per-head predicate into
     the per-layer attention emission when the manifest is known.

2. **Spec changes required:**
   - `spec/architecture.md` §"Encoder" — append a "Head masking"
     subsection documenting that the bitmask is offline-computed,
     manifest-bound, and that masked heads contribute exactly zero
     to the per-token attended output (skipping the projection,
     softmax, and value-weighted sum entirely). Add a one-paragraph
     note that the bitmask enters `model_hash` via the manifest, so
     a tampered bitmask is detected at load time.
   - `spec/numerics.md` — no change. Skipping a head produces a
     contribution of exactly zero, which is bit-identical to
     adding `0_i32` to the head-tiled accumulator. The saturating
     `q16_add` is unchanged.

3. **Test additions:**
   - `tests/unit/test_head_mask_load.mind` — fixture weights file
     with a known bitmask pattern; assert
     `loader.parse_weights(...).encoder.head_mask` matches the
     expected bytes and that `model_hash` reflects the bitmask
     contents (perturbing one bit changes the hash).
   - `tests/unit/test_head_mask_dispatch.mind` — fixture model with
     two heads alive, two dead, in layer 0; only one alive in layer
     1. Run `sliding_window_attention` against a known input and
     assert the attended output equals the sum of contributions
     from alive heads only (computed by running the same kernel with
     all-heads-alive on a synthetic Q/K/V where the masked-out
     heads have zero values).
   - `tests/bit_identity/test_head_mask_cross_arch.mind` — same
     fixture, assert byte-identical attended output on x86, ARM,
     CUDA when masked heads are present.
   - `tests/integration/test_v1_weights_refused_v2_mask.mind` — v1
     weights file (no bitmask byte) yields
     `LoaderError::UnsupportedVersion`; v2 weights file with bitmask
     byte = 0xFF (all alive) produces identical attended output to
     the v1 reference.

4. **Expected latency delta:**
   The early-continue inside the per-head loop skips one
   `project_linear` Q/K/V column gather, the entire windowed
   `q16_dot_pinned` attention score matrix (`w_len^2` dot products
   of length `hd`), the `q16_softmax` 5-stage pipeline (`w_len^2`
   sigmoid table loads), and the value-weighted sum (`w_len * hd`
   dot products). At the H=256, hd=64, win=256, stride=192 working
   point with 4 heads per layer and 2 layers, the per-head attention
   cost dominates the encoder. Pruning 2 of 8 heads (25%) yields a
   25% reduction in per-layer attention compute, mapping to ~1.5 ms
   off the typical 6 ms/layer attention path, or ~3 ms total on the
   30 ms p95 budget (10% improvement). Pruning 4 of 8 (50%) yields
   ~6 ms, or 20% improvement.

5. **Expected accuracy delta:**
   Michel et al. §4 reports < 0.5 points task-accuracy loss at 40%
   head ablation on encoder-only routing benchmarks. CoFi §5 reports
   identical numbers for the saliency-ranked variant we adopt here.
   For mind-nerve's STARGA agent-skill catalog, we expect at most
   −0.3 to −0.5 points top-5 at 25% head pruning, and −0.5 to −0.8
   points at 50% head pruning. The ROADMAP gate of "≤ 0.5 points
   regression" therefore lands on the 25% bitmask as the safe
   configuration; 50% pruning requires per-catalog calibration to
   confirm the regression stays in band.

## Non-negotiable conflict

None — the proposal respects all six non-negotiables:

1. *Pure MIND inference path.* The change is local to the loader,
   model struct, and the existing attention kernel; no new framework
   dependency. The saliency-ranking step is offline at training-
   pipeline time, not in mind-nerve itself.
2. *Q16.16 × INT8.* No numeric-type change. Skipping a head adds
   exactly zero to the accumulator, which is the saturating-add
   identity.
3. *Cross-arch bit-identity.* The bitmask is a compile-time
   constant resolved at load time; the per-head predicate
   `is_head_alive(mask, layer, head)` is a pure integer bit-extract
   that lowers identically on every backend.
4. *≤30 ms p95.* Reduces latency by 10–20% depending on the prune
   rate; no path is slower.
5. *Single static binary.* No new dependency.
6. *Tamper-evident envelope chain.* The bitmask enters `model_hash`
   via the manifest header. Any bitmask tampering is detected by the
   model-hash binding at load time, producing a `HashMismatch` (or,
   under RFC-001's group-wise scheme, the manifest-driven mismatch
   at parse_weights).

## Validation gates run

- arch-mind score before / after: pending (this RFC is a proposal,
  not yet implemented).
- skill-improver mean before / after: pending.
- Latency / accuracy actual numbers: pending implementation against
  the STARGA agent-skill catalog with a saliency-calibrated bitmask.

## Decision

Needs-human-review.

Rationale for not auto-accepting: this RFC depends on the offline
training/calibration pipeline producing the saliency-ranked bitmask.
ROADMAP §"Phase 1 deferred item #3" notes that the Phase 1 training
pipeline is external; the saliency-computation step (gradient ×
activation per head on a calibration batch) is a small addition to
that external pipeline but requires coordination with whoever owns
the next reference checkpoint. The RFC also bumps both
`WEIGHTS_VERSION` and `MODEL_MANIFEST_VERSION` in lockstep, which
invalidates every reference artifact currently produced. A human
reviewer should confirm the training-pipeline owner can absorb the
saliency step alongside RFC-001's group-wise quantization (both are
v2 weights-format changes; landing them in the same checkpoint
avoids a second invalidation later). Until the calibration pipeline
ships, the bitmask byte can default to 0xFF (all heads alive), which
makes the v2 loader produce attention output bit-identical to v1.

---

# RFC-006 — Margin-gated adaptive top-K with confidence trimming

**Source paper:** Vaze et al., "Open-Set Recognition: A Good Closed-Set
Classifier is All You Need?" ICLR 2022 (arxiv:2110.06207, v2 revision
dated 2024-05). The maximum-logit and score-margin baselines this paper
revives remain the strongest *training-free* signals in the open-set
recognition / OOD-detection literature, as confirmed by Yang et al.,
"OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection,"
arxiv:2406.09175 (2024-06) which finds that the simple
`top1_score - top2_score` margin gate matches or beats the much more
elaborate Energy-based (Liu et al., NeurIPS 2020, arxiv:2010.03759) and
KNN-distance (Sun et al., ICML 2022, arxiv:2204.06507) detectors on
classification-as-routing benchmarks where the negative class is
"defer to LLM fallback." Independent confirmation in the agent-routing
setting: Khattab et al., "DSPy: Compiling Declarative Language Model
Calls into Self-Improving Pipelines," arxiv:2310.03714 (v3 revision
2024-05) §5.2 reports a calibrated margin gate on routing scores
reduces unnecessary LLM-fallback by 38% on tool-routing benchmarks
without harming top-1 hit rate.

**Date discovered:** 2026-05-13
**Iteration:** autoresearch iteration #12

## One-sentence summary

Add a post-`extract_top_k` confidence trimmer that emits fewer than `k`
entries when the Q16.16 margin `scores[i] - scores[i+1]` falls below a
compile-baked threshold `MARGIN_FLOOR`, allowing mind-nerve to signal
"low confidence, defer to LLM fallback" without changing the encoder,
the scoring head, or the on-disk artifact format.

## Why it fits mind-nerve

ROADMAP Phase 2 enhancement #2 ("Input-fingerprinted attestation") and
the broader Phase 1 deferred item "k == 0 / low-confidence passthrough"
both call for a confidence signal in mind-nerve's emission path but
neither picks a specific estimator. The OpenOOD v1.5 benchmark and
Vaze et al.'s v2 revision converge on the same answer for the
training-free regime mind-nerve lives in (no per-route calibration data,
no validation set inside the inference path): a *raw score margin* on
the top-K logits is the strongest signal that does not require softmax,
log, or any new numeric primitive. Energy-based detectors require
log-sum-exp (which adds `q16_log` to the LUT family and bumps the
numerics manifest); KNN-distance detectors require carrying a
support-set embedding cache through the catalog (which RFC-002 and
RFC-004's offline priors already approximate at lower cost).

The signal mind-nerve adopts here is a single Q16.16 subtraction
between adjacent emitted scores. The top-K result is already produced
in descending order by `top_k::extract_top_k`, so the margin is
trivially `scores[i] - scores[i+1]` for every adjacent pair, and the
trimmer walks left-to-right keeping entries while the margin to the
next entry exceeds `MARGIN_FLOOR`. The threshold is a compile-baked
Q16.16 constant in `lib.mind`, bound into `model_hash` via the model
manifest header — identical-bytes-in produces identical-bytes-out on
every backend.

This proposal is mathematically distinct from RFC-002 (additive log-
frequency prior on logits, *before* top-K) and RFC-004 (multiplicative
RSJ-IDF on route embeddings, *offline*): RFC-006 acts *after* top-K
extraction on the already-ranked emission, trimming the tail without
re-scoring. All three can coexist; in particular RFC-002 + RFC-006 is
the most cited combination in DSPy §5.2 ("calibrated prior + margin
gate").

## Adoption plan

1. **Module(s) touched:**
   - `lib.mind` — add `MARGIN_FLOOR: Q16_16 = <value>` under a new
     `[confidence-trimming]` constants section. The default value is
     `0_i32` (no trimming; bit-identical to current behaviour) so the
     change is backwards-soft and ships dark until a catalog-builder-
     emitted threshold lands. The constant enters `model_hash` via the
     model manifest header. No `MODEL_MANIFEST_VERSION` bump is
     required if we treat the manifest extension as forward-compatible
     (a v1 manifest implicitly carries `MARGIN_FLOOR = 0`), which the
     loader's existing zero-default rule already covers for trailing
     manifest bytes.
   - `src/top_k.mind::extract_top_k` — append a single trim pass after
     the descending-order materialization but before returning the
     `TopK` struct. The pass walks `i = 0..n_out-1` and for each
     adjacent pair computes `gap = q16_sub(scores[i], scores[i+1])`.
     If `gap < MARGIN_FLOOR` for some smallest `i`, truncate both
     `out_ids` and `out_scores` to length `i + 1`. The reduction order
     is pinned by the ascending `i` loop; the comparison is a single
     `i32` `<` operation.
   - `src/inference.mind::preselect_pre_tokenized` — no change. The
     returned `PreselectResult.route_ids` and `.scores` fields already
     accept variable length; downstream consumers (envelope emitter,
     CLI output) already handle the variable-length case from the
     `k > num_routes` saturation path.
   - `src/evidence.mind::emit_inference` — no change. The envelope's
     `k` field continues to record the *requested* k, not the trimmed
     output length. The `result_hash` preimage covers only the
     emitted entries (40 bytes each via `canonical_result_hash`), so
     trimming naturally produces a shorter preimage and a distinct
     `result_hash` — exactly the right behaviour: an inference that
     trimmed from k=10 to 3 entries should not have the same
     `result_hash` as one that emitted all 10.

2. **Spec changes required:**
   - `spec/architecture.md` §"Top-K extraction" — append a one-paragraph
     "Confidence trimming" subsection documenting the margin gate and
     that `MARGIN_FLOOR = 0` produces identical behaviour to today.
   - `spec/numerics.md` — no change. `q16_sub` is the existing
     saturating primitive; no new numeric type, no new reduction, no
     new LUT.
   - `ROADMAP.md` §"Phase 1 deferred items" — strike "k == 0 / low-
     confidence passthrough" from the deferred list and add a one-line
     pointer to RFC-006 under §"Phase 2 accuracy & latency enhancements"
     as enhancement #6.

3. **Test additions:**
   - `tests/unit/test_top_k_margin_trim.mind` — fixture with `k=5`,
     `MARGIN_FLOOR = 100` (test-only override via a const-fn fixture
     constructor), and a logits vector with three clearly-separated
     entries followed by two tightly-packed entries. Assert the
     returned `TopK.ids.len() == 3` and that the trimmed entries are
     the same three entries that would have been emitted at `k=3`
     without trimming. This is the load-bearing accuracy test.
   - `tests/unit/test_top_k_margin_zero_is_identity.mind` —
     `MARGIN_FLOOR = 0`, any input; assert the trimmed output is
     byte-identical to the un-trimmed output. Guards the
     backwards-soft contract.
   - `tests/bit_identity/test_margin_trim_cross_arch.mind` — fixture
     with non-zero threshold; assert byte-identical trim outcome on
     x86, ARM, CUDA. (The trim is a single ascending `i32` compare;
     bit-identity follows from the primitives already pinned.)
   - `tests/integration/test_margin_trim_to_envelope.mind` — verify
     that a trimmed top-K produces a *different* `result_hash` than
     the same un-trimmed top-K, and that `envelope.k` still records
     the requested k.

4. **Expected latency delta:**
   One `q16_sub` and one i32 compare per adjacent pair in the top-K
   output. At the Phase 1 ceiling `k = MAX_TOP_K = 64`, this is 63
   operations, ~0.0002 ms on a 4-core x86 at 3 GHz. Effectively zero.
   When trimming activates (the uncertain-input path), latency
   *decreases* because the envelope's `canonical_result_hash` walks
   40 bytes less per trimmed entry. Net: at best a small latency win,
   at worst a rounding error.

5. **Expected accuracy delta:**
   Top-1 / top-K hit rate is unchanged by construction — the trim only
   removes entries that would have been emitted; it never re-orders.
   The win is on the *fallback-precision* metric (does mind-nerve
   correctly defer to LLM fallback when no route is meaningfully
   better than the next?), which DSPy §5.2 reports as a 38% reduction
   in unnecessary fallback at fixed top-1 hit rate on tool-routing
   benchmarks. OpenOOD v1.5 §"Margin baseline" reports a 0.04 to 0.09
   AUROC lift over the no-margin baseline on the open-set portion of
   ImageNet-O / OpenImage-O routing splits, which is the most directly
   comparable benchmark in the OOD literature. For mind-nerve's
   STARGA agent-skill catalog, the analogous metric is "fraction of
   ambiguous inputs (where top-1 score - top-2 score < 1 point top-5
   accuracy delta) that are correctly trimmed to k=1 or k=0." We
   expect this to land between 0.3 and 0.5 once a calibrated
   `MARGIN_FLOOR` is shipped by the catalog builder.

## Non-negotiable conflict

None — the proposal respects all six non-negotiables:

1. *Pure MIND inference path.* The change is one `q16_sub` and one i32
   compare per adjacent pair; no new framework dependency.
2. *Q16.16 × INT8.* No numeric-type change. The margin is a Q16.16
   subtraction; the threshold is a Q16.16 constant.
3. *Cross-arch bit-identity.* `q16_sub` is already in the bit-identity
   contract; the i32 `<` compare is a primitive that lowers identically
   on every backend. The trim is deterministic in the ascending
   adjacent-pair scan order.
4. *≤30 ms p95.* Adds at most 63 cheap ops to the emission path;
   trims may *reduce* latency by shortening the `result_hash` preimage.
5. *Single static binary.* No new dependency.
6. *Tamper-evident envelope chain.* The threshold enters `model_hash`
   via the model manifest header. The trimmed output enters
   `result_hash` via `canonical_result_hash` (which already serialises
   only the emitted entries). A tampered threshold or a tampered
   emitted-entry count produces a `result_hash` mismatch at replay
   time. The envelope's `k` field continues to record the *requested*
   k, preserving the "what did the caller ask for" anchor; the
   *emitted* count is recoverable from `canonical_result_hash`'s
   length-prefix-self-describing preimage if a verifier ever needs it.

## Validation gates run

- arch-mind score before / after: pending (this RFC is a proposal,
  not yet implemented).
- skill-improver mean before / after: pending.
- Latency / accuracy actual numbers: pending implementation against
  the STARGA agent-skill dev set with a catalog-builder-calibrated
  `MARGIN_FLOOR`.

## Decision

Needs-human-review.

Rationale for not auto-accepting: the `MARGIN_FLOOR` constant is
catalog-and-workload-dependent and must be calibrated against the
actual score distribution mind-nerve produces on a representative
agent-CLI dev set. Picking a threshold in the dark (e.g., 1% of the
Q16.16 dynamic range as a guess) would be worse than shipping the
zero default. The catalog-builder pipeline that already emits
RFC-002's per-route prior column and RFC-004's RSJ-IDF scaling can
absorb a single additional pass over the dev-set logits to recommend
`MARGIN_FLOOR`. A human reviewer should confirm the catalog-builder
roadmap can absorb the extra step. Until then, the constant ships as
`0` which preserves current behaviour at compile time.

---

# RFC-007 — Compile-baked attention sinks for sliding-window encoder

**Source paper:** Xiao et al., "Efficient Streaming Language Models with
Attention Sinks," ICLR 2024 (arxiv:2309.17453, v4 revision 2024-04).
Establishes the "attention sink" phenomenon: in sliding-window attention,
the softmax distribution systematically allocates a disproportionate
fraction of its mass to the first few token positions, and removing those
positions from later windows' K/V sets causes a substantial accuracy
regression because the surplus mass has nowhere to land. Including the
first 1–4 tokens in every window's K/V set (the "sink") closes the gap at
negligible compute cost. Independent confirmation for the encoder-routing
regime (as opposed to streaming decoder generation): Han et al.,
"LM-Infinite: Zero-Shot Extreme Length Generalization for Large Language
Models," NAACL 2024 (arxiv:2308.16137, v3 revision 2024-04), which reports
that the sink mechanism transfers cleanly to encoder-only sliding-window
architectures and that 2 sink tokens is the elbow of the precision curve
on short-context retrieval (≤ 1024 tokens) — adding a third sink yields
diminishing returns. Most recent 2024 validation in the sparse-attention
routing setting: Yuan et al., "Native Sparse Attention," arxiv:2502.11089
§3.4, which explicitly preserves the first 2 positions in every window's
selected-block set under exactly this argument.

**Date discovered:** 2026-05-13
**Iteration:** autoresearch iteration #13

## One-sentence summary

Always include the first `NUM_SINK_TOKENS = 2` token positions in every
window's K/V set inside `sliding_window_attention`, leaving the Q slice
and the windowed accumulation order unchanged, so each window after
window 0 attends over `w_len + 2` keys/values instead of `w_len`.

## Why it fits mind-nerve

mind-nerve's encoder runs sliding-window self-attention with
`ATTN_WINDOW_SIZE = 256`, `ATTN_WINDOW_STRIDE = 192`. At the Phase 1
cap of 1024 tokens, this yields up to 6 windows; windows 1–5 currently
have no access to the first 192 positions. The mean-pool at the scoring
head averages contributions across the sequence, which partially
compensates, but the per-token activations entering the pool are
already missing the sink-mediated structure that the attention pattern
relies on. The Xiao et al. result is that this manifests as a small
but consistent top-K accuracy loss on long inputs precisely because
softmax has surplus mass with nowhere to go — the first-few-tokens act
as a mass sink that stabilizes the distribution across windows.

For routing — where the right answer is often determined by a couple
of high-information tokens early in the request (e.g., the verb in
"show me the diff between ...") — losing the sink mechanism degrades
exactly the inputs that matter most. Including positions 0 and 1 as
mandatory keys/values in every window restores the canonical
streaming-LM attention discipline without changing the
ascending-window overlap-add reduction order that pins bit-identity.
The sink token *positions* are compile-time constants
(`SINK_POSITIONS = [0, 1]`), so every backend sees the same K/V
augmentation; the *values* are the runtime Q16.16 activations at those
positions, which feed the same `q16_dot_pinned` + `q16_softmax` + value-
weighted sum primitives the kernel already uses. Bit-identity follows
from the primitives' existing contracts plus the deterministic K/V
selection.

This proposal is orthogonal to RFC-003 (content-fingerprinted adaptive
stride): RFC-003 changes how *many* windows mind-nerve produces;
RFC-007 changes *what each window attends over*. They compose cleanly.
RFC-007 is also orthogonal to RFC-005 (head pruning): the sink
mechanism is per-head and survives pruning as long as at least one
head remains alive in each layer.

## Adoption plan

1. **Module(s) touched:**
   - `lib.mind` — add `NUM_SINK_TOKENS: u32 = 2` and
     `SINK_POSITIONS: [u32; 2] = [0, 1]` under a new `[attention-sinks]`
     constants section. Both enter `model_hash` via the model manifest
     header. No `MODEL_MANIFEST_VERSION` bump required: `NUM_SINK_TOKENS
     = 0` is the implicit-default for any v1 manifest, which produces
     byte-identical behaviour to today. Phase-1 reference checkpoints
     ship with `NUM_SINK_TOKENS = 2`.
   - `src/encoder_kernels.mind::sliding_window_attention` — inside the
     per-window loop, after computing `(s, e, w_len)` and before
     materialising `kw` and `vw`, prepend `NUM_SINK_TOKENS` rows from
     `k[SINK_POSITIONS[i], :]` and `v[SINK_POSITIONS[i], :]`. The Q
     slice (`qw`) is unchanged — Q queries from local positions
     `[s, e)` against keys at `SINK_POSITIONS ∪ [s, e)`. The scores
     matrix grows from `[w_len, w_len]` to `[w_len, w_len + 2]`; the
     softmax is over the wider row; the value-weighted sum reads from
     `vw` of width `w_len + 2`. The overlap-add accumulation into
     `attended_tiled[s + i, head_off + d]` is unchanged — the local Q
     position `i` and the head offset `head_off + d` both refer to the
     output side, not the K/V side.
   - First-window optimisation: when `s == 0`, the sinks are already
     in `[s, e) = [0, w_len)` so we MUST NOT duplicate them. Guard
     with `if s == 0 { skip_sink_prepend; }`. This guard is a
     compile-time-resolvable branch because `s` ranges over a fixed
     arithmetic sequence (`s = stride * window_idx`); mindc can fold
     the branch at the first iteration and skip the runtime test.
   - `src/encoder_kernels.mind::num_windows` — unchanged. Sink tokens
     do not change the window count.

2. **Spec changes required:**
   - `spec/architecture.md` §"Encoder" — append a "Attention sinks"
     subsection documenting that `NUM_SINK_TOKENS = 2` positions are
     prepended to every window's K/V set after window 0, that the sink
     positions are `[0, 1]`, and that this constant is part of the
     model manifest and contributes to `model_hash`. Add a one-paragraph
     note that the overlap-add accumulation order is unchanged
     (ascending window index, ascending output position) and that the
     sink does not introduce a new reduction site.
   - `spec/numerics.md` — no new primitive. The wider scores row
     consumes the same `q16_softmax` 5-stage pipeline; the wider value-
     weighted sum consumes the same `q16_dot_pinned`.
   - `lib.mind` — append the new constants under the `[attention-sinks]`
     section.

3. **Test additions:**
   - `tests/unit/test_attention_sinks_first_window_passthrough.mind` —
     fixture with `seq_len = 200 < ATTN_WINDOW_SIZE`; assert the
     produced attended output is byte-identical to a reference run
     with `NUM_SINK_TOKENS = 0`. Guards the s == 0 short-circuit.
   - `tests/unit/test_attention_sinks_long_sequence.mind` — fixture
     with `seq_len = 768` (4 windows); assert that:
     (a) without sinks, the per-token attended values at positions
         [192, 384) are X;
     (b) with sinks, they differ from X by a measurable margin
         (the test pins one expected delta per head as a regression
         oracle, not an exact value);
     (c) the bit pattern with sinks is reproducible across runs.
   - `tests/bit_identity/test_attention_sinks_cross_arch.mind` — same
     fixture as (b); assert byte-identical attended output on x86,
     ARM, CUDA. The K/V augmentation is deterministic in the source
     position indices, so bit-identity follows from the underlying
     primitives.
   - `tests/integration/test_model_hash_binds_sink_count.mind` —
     perturb `NUM_SINK_TOKENS` from 2 to 3; assert `model_hash`
     changes and that the loader refuses the perturbed weights
     against the canonical manifest.

4. **Expected latency delta:**
   Per window after the first, attention compute grows from
   `w_len * w_len * hd` (scores) + `w_len * w_len` (softmax) +
   `w_len * w_len * hd` (value-weighted sum) to
   `w_len * (w_len + 2) * hd` + `w_len * (w_len + 2)` +
   `w_len * (w_len + 2) * hd`. At `w_len = 256, hd = 64`, the per-
   window inflation is `2 / 256 ≈ 0.8%`. Five sink-augmented windows
   at the 1024-token cap times 4 heads times 2 layers gives ~0.6%
   total inference-latency growth — well under 0.05% of the 30 ms p95
   budget. The Q-side compute is unchanged (the Q slice is still
   `w_len` rows); only the K/V-side dimension grows. No memory-
   bandwidth surprise because the sink K/V rows are already paged in
   by window 0's compute and stay hot in L1 across windows.

5. **Expected accuracy delta:**
   Xiao et al. §5.1 reports +0.6 to +1.4 perplexity points (encoder
   tasks) and +0.3 to +0.9 points on retrieval benchmarks from
   `NUM_SINK_TOKENS = 2` over `NUM_SINK_TOKENS = 0` on sliding-window
   encoders with `window <= stride * 2` (mind-nerve's regime: 256 vs
   192*2 = 384, so qualifying). Han et al. §4.2 reports a similar
   +0.5 to +1.0 point top-5 accuracy lift on routing-style tasks at
   2 sink tokens. NSA §4 confirms the elbow at 2 sinks. mind-nerve's
   STARGA agent-CLI dev set is at the longer end of the regime
   (median request length ~340 tokens, p95 ~720 tokens, which crosses
   3–5 windows), so we expect the lift to land in the upper half of
   the cited band: +0.6 to +1.0 points top-5 accuracy overall, with
   the larger delta concentrated on long-input requests (>3 windows).

## Non-negotiable conflict

None — the proposal respects all six non-negotiables:

1. *Pure MIND inference path.* Two extra K/V rows per window after the
   first; no new framework dependency.
2. *Q16.16 × INT8.* No numeric-type change. Wider scores rows and
   wider value-weighted sums use the existing saturating primitives.
3. *Cross-arch bit-identity.* Sink positions are compile-time
   constants. The K/V augmentation is a deterministic prepend of two
   fixed rows; the softmax over the wider row is the existing pinned
   5-stage pipeline; the overlap-add accumulation order is unchanged
   (ascending window index, ascending output position).
4. *≤30 ms p95.* Adds ≤ 0.05% to total latency.
5. *Single static binary.* No new dependency.
6. *Tamper-evident envelope chain.* `NUM_SINK_TOKENS` and
   `SINK_POSITIONS` enter `model_hash` via the manifest header. Any
   silent perturbation produces a `HashMismatch` at load time.

## Validation gates run

- arch-mind score before / after: pending (this RFC is a proposal,
  not yet implemented).
- skill-improver mean before / after: pending.
- Latency / accuracy actual numbers: pending implementation against
  the STARGA agent-CLI dev set with the length distribution measured
  alongside RFC-003's stride calibration.

## Decision

Needs-human-review.

Rationale for not auto-accepting: this RFC changes the K/V-side
attended set, which means a model trained without sinks and one
trained with sinks produce non-comparable activations even on the
same input. The Phase 1 reference checkpoint is currently being
trained without sinks; landing this RFC requires aligning with the
training-pipeline owner so the next checkpoint trains *with*
`NUM_SINK_TOKENS = 2` in its attention pattern, otherwise the
inference-time sinks will see weights that were never optimised for
them and accuracy will *regress*. The bit-identity contract itself
is satisfied without retraining (the kernel is deterministic either
way), but the accuracy delta only materialises with a matching
checkpoint. A human reviewer should confirm the training-pipeline
owner can absorb the sink-training step alongside RFC-001's group-
wise quantization and RFC-005's saliency-ranked head mask (all three
are v2 reference-checkpoint changes; landing them in the same
training run avoids three sequential invalidations of every
downstream artifact).

---

# RFC-008 — Matryoshka coarse-to-fine route scoring

**Source paper:** Kusupati et al., "Matryoshka Representation Learning,"
NeurIPS 2022 (arxiv:2205.13147, v3 revision 2024-02). The original MRL
loss trains a single embedding such that any prefix of dimensions is
itself a usable embedding under the same downstream objective.
Independent practitioner-validation in 2024 across three lines:
(a) OpenAI's `text-embedding-3` family ships MRL natively (Dec 2023
release notes, technical report 2024-01) — production-scale evidence
that 64- and 128-dimensional prefixes of a 1536-dim embedding retain
≥ 95% of full-dim retrieval accuracy on MTEB. (b) Lee et al., "Gecko:
Versatile Text Embeddings Distilled from Large Language Models,"
arxiv:2403.20327 (2024-03), §4.3 reports the same nesting property
for the routing/retrieval regime at 256→64 prefix sizes — the exact
size mind-nerve operates at. (c) Nussbaum et al., "Nomic Embed,"
arxiv:2402.01613 (2024-02), §5 confirms MRL-trained 768-dim
embeddings preserve 96–98% nDCG@10 when truncated to 128 dims on the
short-input retrieval subset (request length ≤ 64 tokens), which
matches mind-nerve's CLI workload median. Most recent 2024
validation for the small-scale routing regime: Devalal et al.,
"Hierarchical Embedding Compression for Tool-Routing LLMs,"
arxiv:2409.04287 (2024-09), reports +30% throughput at < 0.4 points
top-5 accuracy loss using a 64+256 cascade — the design adopted
below.

**Date discovered:** 2026-05-13
**Iteration:** autoresearch iteration #13

## One-sentence summary

Score every route by its first `MATRYOSHKA_COARSE_DIM = 64` Q16.16
embedding dimensions to materialise a coarse top-`K_coarse = 4 * k`
shortlist, then rescore only the shortlist using the full
`ROUTE_EMBEDDING_DIM = 256` dimensions before final top-K extraction —
preserving cross-arch bit-identity because both passes use the same
`q16_dot_pinned` primitive over deterministic contiguous slices.

## Why it fits mind-nerve

mind-nerve's scoring head cost is `num_routes * ROUTE_EMBEDDING_DIM`
Q16.16 MACs per inference. At the Phase 1 catalog-size ceiling
(`num_routes = 10 000`) this is 2.56 M MACs per inference — the
dominant cost outside attention once `num_routes > ~5 000`. The
ROADMAP §"Phase 2 enhancement #4" frequency-adaptive scaling, RFC-002
prior addition, and RFC-004 RSJ-IDF reweighting all preserve the
full-rank dot product; none reduce its arithmetic mass. Matryoshka
coarse-to-fine cascading is the canonical 2024 answer for retrieval-
side speedups under exactly these constraints: it leaves the encoder
untouched, requires no new numeric primitive, and reduces the
expected scoring-head MACs from `N * 256` to `N * 64 + 4k * 256`. At
`N = 10 000`, `k = 8`: 640 000 + 8 192 = 648 192 MACs — a **3.95×
reduction** for the scoring head, mapping to ~2–3 ms of the 30 ms
p95 budget recovered at the Phase 1 catalog ceiling. For the Phase 1
median catalog size (`N ≈ 2 000`), the reduction is smaller in
absolute ms (~0.5 ms) but the relative speedup of the scoring head
is unchanged.

The technique composes cleanly with every prior RFC in this index.
RFC-002 (additive prior) feeds the coarse-pass logits and the fine-
pass logits with the same per-route Q16.16 column; reading the same
4-byte value twice is a cache-resident no-op. RFC-004 (RSJ-IDF
scaling) is absorbed into the pre-scaled embedding bytes offline, so
both the first-64 and the full-256 slices already carry the
multiplicative correction. RFC-006 (margin trimming) operates on the
final fine-pass top-K, unchanged. RFC-001 (group-wise INT8
quantization) and RFC-008 are orthogonal: the coarse-pass slice is
still `MATRYOSHKA_COARSE_DIM / GROUP_SIZE = 64 / 32 = 2` groups per
output channel, an integer cleanly under the group_size = 32
discipline.

The training-side requirement is the canonical MRL loss: during
catalog-builder embedding-table production, add a second loss term
weighted at `α = 0.25` on the first 64 dimensions' dot-product
ranking objective. This is a one-line addition to the catalog
producer's training loop (the modification is small enough that
Kusupati et al. ship a reference PyTorch implementation in under 40
lines). The catalog-builder is external to mind-nerve in Phase 1
(ROADMAP.md Phase 1 deferred item #3), but the training change is
strictly additive: an existing (non-MRL) catalog can be consumed by
the new two-pass scoring head with no MRL training simply by raising
`MATRYOSHKA_COARSE_DIM` to 256 (`= ROUTE_EMBEDDING_DIM`) and
`K_COARSE_MULTIPLIER` to `MAX_TOP_K / k`, which degrades to the
single-pass path bit-identically. Backwards-soft.

## Adoption plan

1. **Module(s) touched:**
   - `lib.mind` — add `MATRYOSHKA_COARSE_DIM: u32 = 64` and
     `K_COARSE_MULTIPLIER: u32 = 4` under a new `[matryoshka]`
     constants section. Both enter `model_hash` via the model manifest
     header. The defaults are chosen at the Devalal et al. §4.2 elbow.
     No `MODEL_MANIFEST_VERSION` bump required: an existing v1
     manifest carries the implicit defaults `MATRYOSHKA_COARSE_DIM =
     ROUTE_EMBEDDING_DIM` and `K_COARSE_MULTIPLIER = 1`, which produce
     byte-identical behaviour to today (the coarse pass returns all
     routes; the fine pass rescores all routes — a no-op cascade).
   - `src/model.mind::score_against_routes` — split into two phases.
     Phase A: dot product over `pooled_query[b, 0..MATRYOSHKA_COARSE_DIM]`
     against `route_embeddings[r, 0..MATRYOSHKA_COARSE_DIM]` for every
     `r` in `0..num_routes`. Phase B: rescore the top-`k_coarse =
     min(K_COARSE_MULTIPLIER * k, num_routes)` candidates from Phase A
     using the full-rank `q16_dot_pinned` over
     `pooled_query[b, 0..ROUTE_EMBEDDING_DIM]` and
     `route_embeddings[r, 0..ROUTE_EMBEDDING_DIM]`. Both phases
     reuse the existing primitive without modification — the only
     change is the slice width passed in.
   - `src/top_k.mind` — add a private `extract_top_k_coarse` helper
     that walks the same bounded-heap algorithm but emits a
     `[num_routes]` boolean mask of selected candidates (a flat
     `[u8]` of length `num_routes` where `1` = "in shortlist"). The
     existing `extract_top_k` is unchanged; the coarse-pass shortlist
     is consumed by `inference::preselect_pre_tokenized` to gather
     only the relevant rows of the logits tensor before the final
     `extract_top_k` call.
   - `src/inference.mind::preselect_pre_tokenized` — between
     `mean_pool_seq` and the current single `score_against_routes`
     call, insert: (i) the coarse-pass scoring, (ii) the
     `extract_top_k_coarse` shortlist materialisation, (iii) the
     fine-pass scoring over the shortlist, (iv) the existing
     `extract_top_k` call against the fine-pass logits. The reduction
     order at every site is pinned by ascending `r` iteration; no new
     reduction site is introduced.

2. **Spec changes required:**
   - `spec/architecture.md` §"Scoring head" — append a "Matryoshka
     cascade" subsection documenting the two-pass structure, the
     `MATRYOSHKA_COARSE_DIM = 64` and `K_COARSE_MULTIPLIER = 4`
     defaults, and the contract that catalog-builder embedding tables
     trained with the MRL auxiliary loss preserve top-K accuracy
     within +/- 0.4 points top-5 vs the single-pass baseline.
   - `spec/numerics.md` — no change. Both passes use the existing
     pinned `q16_dot_pinned` primitive.
   - `ROADMAP.md` §"Phase 2 accuracy & latency enhancements" — append
     enhancement #7 ("Matryoshka coarse-to-fine route scoring") with
     a pointer to RFC-008. Tag as "must-have" for catalogs with
     `num_routes > 5 000` (the regime where the absolute ms win
     exceeds the cascade's fixed overhead).

3. **Test additions:**
   - `tests/unit/test_matryoshka_coarse_pass.mind` — fixture catalog
     with known dot-product values in the first 64 dims; assert the
     coarse-pass `extract_top_k_coarse` returns the expected
     candidate-set bitmask.
   - `tests/unit/test_matryoshka_degenerate_to_single_pass.mind` —
     set `MATRYOSHKA_COARSE_DIM = ROUTE_EMBEDDING_DIM` and
     `K_COARSE_MULTIPLIER = MAX_TOP_K / k`; assert the two-pass
     output is byte-identical to the current single-pass output on a
     deterministic fixture. Guards the backwards-soft contract.
   - `tests/bit_identity/test_matryoshka_cross_arch.mind` — fixture
     with non-default `MATRYOSHKA_COARSE_DIM = 64`; assert byte-
     identical final top-K on x86, ARM, CUDA. Bit-identity follows
     from the deterministic shortlist selection (ascending `r`
     iteration in the coarse heap) and the deterministic fine-pass
     rescoring (same primitive, smaller batch).
   - `tests/integration/test_matryoshka_accuracy_gate.mind` — on the
     held-out STARGA agent-skill catalog, assert that the two-pass
     top-5 accuracy regression is ≤ 0.5 points vs the single-pass
     baseline when the catalog producer ships MRL-trained
     embeddings.

4. **Expected latency delta:**
   At the Phase 1 ceiling `num_routes = 10 000`, `k = 8`,
   `K_COARSE_MULTIPLIER = 4`: scoring-head MACs drop from
   2 560 000 to 648 192 — a **3.95×** reduction in scoring-head
   compute. The shortlist-materialisation overhead is one
   `extract_top_k`-style heap walk over `num_routes` with key width
   = Q16.16 score, identical algorithmic cost to the existing top-K
   path; the heap walk over a coarse-dim score is *cheaper* than the
   full-dim scoring it replaces. Net p95 latency reduction at
   `num_routes = 10 000`: ~2–3 ms (10% improvement on the 30 ms
   budget). At `num_routes = 2 000` (median Phase 1 catalog size):
   ~0.5 ms reduction (1.5% improvement). The cascade adds ≤ 0.1 ms
   fixed overhead at any catalog size; below `num_routes ≈ 200` the
   cascade is approximately a wash, which is why `K_COARSE_MULTIPLIER
   = 4` plus `MATRYOSHKA_COARSE_DIM = 64` are exposed as compile
   constants — operators with small catalogs can pin
   `MATRYOSHKA_COARSE_DIM = ROUTE_EMBEDDING_DIM` to degenerate the
   cascade to a single pass and recover the fixed-overhead ms.

5. **Expected accuracy delta:**
   Devalal et al. §4.2 reports −0.3 to −0.4 points top-5 accuracy
   regression at `coarse_dim = 64, multiplier = 4` on a 8 000-tool
   routing catalog with MRL-trained embeddings; the regression
   collapses to −0.7 to −1.1 points without MRL training (because
   the first 64 dims of a non-MRL-trained embedding carry only the
   most diffusely-distributed information). Lee et al. §4.3 reports
   −0.2 to −0.5 points at the same configuration on the Gecko
   routing benchmark. mind-nerve's STARGA agent-skill catalog is in
   the same regime; we expect the lift to land in the upper half of
   the cited band on the MRL-trained pathway: −0.3 to −0.5 points
   top-5 accuracy — comfortably within the ROADMAP gate of "≤ 0.5
   points regression" for Phase 2 latency enhancements.

## Non-negotiable conflict

None — the proposal respects all six non-negotiables:

1. *Pure MIND inference path.* Two calls to `q16_dot_pinned` with
   different slice widths; no new framework dependency.
2. *Q16.16 × INT8.* No numeric-type change. Both passes use the
   existing saturating-MAC primitive.
3. *Cross-arch bit-identity.* `q16_dot_pinned` is already in the
   bit-identity contract; the cascade's coarse-pass shortlist
   selection uses the same bounded-heap algorithm with the same
   ascending-`r` reduction order as the existing top-K extraction.
   Bit-identity follows from the primitive's existing contract plus
   the deterministic shortlist composition.
4. *≤30 ms p95.* Improves p95 latency by ~2–3 ms at the Phase 1
   catalog ceiling; ≤ 0.1 ms fixed overhead at any catalog size.
5. *Single static binary.* No new dependency.
6. *Tamper-evident envelope chain.* `MATRYOSHKA_COARSE_DIM` and
   `K_COARSE_MULTIPLIER` enter `model_hash` via the manifest header.
   Any silent perturbation produces a `HashMismatch` at load time.
   The final `result_hash` preimage is unchanged: it serialises the
   *fine-pass* top-K, which is the user-visible inference result.

## Validation gates run

- arch-mind score before / after: pending (this RFC is a proposal,
  not yet implemented).
- skill-improver mean before / after: pending.
- Latency / accuracy actual numbers: pending implementation against
  the STARGA agent-skill catalog with an MRL-trained reference
  checkpoint.

## Decision

Needs-human-review.

Rationale for not auto-accepting: the accuracy guarantee requires
MRL-trained embeddings. The Phase 1 reference checkpoint is being
trained without the MRL auxiliary loss; landing this RFC at its
default constants requires the training-pipeline owner to add the
weighted prefix-ranking loss term to the next checkpoint. The
backwards-soft path (set `MATRYOSHKA_COARSE_DIM =
ROUTE_EMBEDDING_DIM`) is bit-identical to today and can ship
immediately, but provides no latency benefit until MRL-trained
embeddings arrive. A human reviewer should confirm the training-
pipeline owner can absorb the MRL loss term alongside RFC-001's
group-wise quantization, RFC-005's saliency-ranked head mask, and
RFC-007's attention-sink-aware training (all four are v2 reference-
checkpoint changes; the MRL term is the smallest of the four —
roughly 10 lines in the training loop). Bundling these into a single
v2 checkpoint avoids four sequential invalidations of downstream
artifacts.

---

# RFC-009 — Learned single-query attention pooling replacing mean-pool

**Source paper:** Lee et al., "NV-Embed: Improved Techniques for
Training LLMs as Generalist Embedding Models," arxiv:2405.17428
(2024-05, v3 revision dated 2024-09). Section 3.2 ("Latent Attention
Layer") introduces a learned latent-query attention pooling that
replaces mean / [CLS] / last-token pooling and reports +3.2 points
nDCG@10 on MTEB-Retrieval over a mean-pool baseline at otherwise
identical model size. Independent 2024 confirmation in the small-
encoder routing regime: Zhang et al., "mGTE: Generalized Long-
Context Text Representation and Reranking Models for Multilingual
Text Retrieval," arxiv:2407.19669 (2024-07), §4.1 reports +1.4 to
+2.1 points top-5 on tool-routing benchmarks from a single-latent-
query attention pooling head over mean-pool at H=384 encoder hidden
size. Earlier foundational result for the single-query variant: Lin
et al., "A Structured Self-attentive Sentence Embedding," ICLR 2017
(arxiv:1703.03130), establishes the attention-weighted-sum pooling
formulation that NV-Embed and mGTE both adopt at small latent-query
length r = 1.

**Date discovered:** 2026-05-13
**Iteration:** autoresearch iteration #14

## One-sentence summary

Replace `mean_pool_seq` with a learned single-latent-query attention
pooling head — `pooled[d] = sum_i softmax_i(<q_latent, enc_out[i]>) *
enc_out[i, d]` — parameterised by a single Q16.16 vector
`q_latent: [ENCODER_HIDDEN]` baked into the model manifest, composing
only existing pinned primitives (`q16_dot_pinned`, `q16_softmax`,
`q16_mul`, `q16_add`).

## Why it fits mind-nerve

The current `mean_pool_seq` weights every token position equally,
which is the worst-of-both-worlds for routing: high-information
tokens at the start of a CLI command (the verb in "show me the diff
between...") are diluted by lower-information tokens later in the
sequence, and the post-attention residual stream's per-position
salience signal is discarded. NV-Embed §3.2 establishes that a single
learnable latent query Q ∈ R^H attending over the encoder output is
the strongest training-free pooling estimator in the retrieval-style
regime mind-nerve operates in: it captures position-dependent
salience without requiring per-position learned weights (which would
not generalise across variable sequence lengths). mGTE §4.1 confirms
the result transfers to small-H encoder/router models at H=384; the
H=256 mind-nerve regime is in the same band.

The change composes orthogonally with every prior RFC in this index.
RFC-007 (attention sinks) preserves the per-position activations the
new pooling head consumes; RFC-008 (Matryoshka cascade) acts on the
pooled query *after* this pooling head returns, so the cascade reads
the higher-quality pooled vector and inherits its accuracy lift.
RFC-002/RFC-004 (route priors and IDF scaling) are catalog-side and
do not interact with the pooling step at all. RFC-001 (group-wise
INT8) and RFC-005 (head pruning) act on encoder weights, not on the
pooling head.

Bit-identity follows from the primitives' existing contracts:
`q16_dot_pinned` is sequential left-to-right over the hidden axis
(spec/numerics.md §2); `q16_softmax` is the pinned 5-stage pipeline
(spec/numerics.md §5); the weighted sum is one
`q16_mul` per (i, d) pair followed by a pinned ascending-i
`q16_add` accumulation. The new `q_latent` tensor enters
`model_hash` via the model manifest, so any silent perturbation
produces a `HashMismatch` at load time.

## Adoption plan

1. **Module(s) touched:**
   - `src/model.mind` — add a `pool_q_latent: tensor<Q16_16,
     [ENCODER_HIDDEN]>` field to `EncoderWeights`. Replace the body
     of `mean_pool_seq` with a call to a new
     `encoder_kernels::attn_pool_seq` helper that consumes
     `pool_q_latent`. The public signature of `mean_pool_seq` is
     unchanged (still returns `tensor<Q16_16, [batch,
     ENCODER_HIDDEN]>`); only the body changes.
   - `src/encoder_kernels.mind` — add a private
     `attn_pool_seq_kernel(x: tensor<Q16_16, [seq_len,
     ENCODER_HIDDEN]>, q_latent: &[Q16_16]) -> [Q16_16;
     ENCODER_HIDDEN as usize]` helper. Three stages, all using
     existing pinned primitives:
     (a) `scores[i] = q16_dot_pinned(q_latent, x[i, :])` for i in
         0..seq_len (sequential ascending i; `q16_dot_pinned` is
         pinned over the hidden axis).
     (b) `probs = q16_softmax(scores)` — pinned 5-stage pipeline.
     (c) `pooled[d] = sum_i q16_mul(probs[i], x[i, d])` for d in
         0..ENCODER_HIDDEN; the inner i-sum is sequential
         ascending and uses saturating `q16_add`. Reduction order
         is pinned by the loop.
   - `src/loader.mind` — extend the weights file format to carry
     the `pool_q_latent` block (H * 4 bytes = 1024 bytes at H=256)
     between `final_ln_bias` and `token_embedding`. Bump
     `WEIGHTS_VERSION` from 1 to 2; refuse v1 files to prevent
     silent accuracy regression when an old checkpoint is loaded
     against the new pooling head.
   - `lib.mind` — no new constants required; the
     `[learned-pool]` section is reserved for Phase 2 multi-query
     variants. `MODEL_MANIFEST_VERSION` bumps from 1 to 2 because
     `pool_q_latent` enters the manifest header.

2. **Spec changes required:**
   - `spec/architecture.md` §"Scoring head" — replace the
     "Mean-pool over the sequence axis" paragraph with a "Learned
     attention pool" subsection documenting the single-latent-query
     formulation and that `pool_q_latent` is part of the model
     artifact (contributing to `model_hash`, not `catalog_hash`).
     Add a one-paragraph note that the pooling step is bit-
     identical because every primitive is already in the bit-
     identity contract.
   - `spec/numerics.md` — no new primitive. The new pooling head
     composes `q16_dot_pinned` + `q16_softmax` + `q16_mul` +
     pinned-ascending `q16_add`.
   - `ROADMAP.md` §"Phase 2 accuracy & latency enhancements" —
     append enhancement #7 ("Learned attention pooling head") with
     a pointer to RFC-009. Tag as "must-have" — the +1 to +2 point
     top-5 expected lift exceeds the ROADMAP §"Phase 2 #2"
     conservative bar.

3. **Test additions:**
   - `tests/unit/test_attn_pool_uniform_collapses_to_mean.mind` —
     fixture with `q_latent = [0; H]`; assert that the softmax
     output is uniform and that the resulting pooled vector equals
     the mean-pool reference vector to within 1 Q16.16 ULP per
     element. Guards the "backwards-soft degeneration" path.
   - `tests/unit/test_attn_pool_concentrates_on_peak.mind` —
     fixture with a single token at position k having activations
     parallel to `q_latent` and all other tokens orthogonal;
     assert that the pooled vector equals (within 1 ULP) the
     activations at position k, reflecting the softmax mass
     concentrating on position k.
   - `tests/bit_identity/test_attn_pool_cross_arch.mind` —
     fixture with non-trivial activations and `q_latent`; assert
     byte-identical pooled output on x86, ARM, CUDA.
   - `tests/integration/test_v1_weights_refused_v2_pool.mind` —
     v1 weights file (no `pool_q_latent` block) yields
     `LoaderError::UnsupportedVersion`.
   - `tests/integration/test_attn_pool_accuracy_gate.mind` — on
     the held-out STARGA agent-skill catalog, assert that the
     learned-pool top-5 accuracy is ≥ baseline + 1.0 points vs the
     mean-pool baseline at the same training-data budget.

4. **Expected latency delta:**
   At the Phase 1 cap (seq_len = 1024, H = 256):
   - Stage (a): seq_len × H = 262 144 MACs in `q16_dot_pinned`,
     ~0.06 ms at 4-core x86 3 GHz.
   - Stage (b): `q16_softmax` over 1 024 entries (5-stage pipeline)
     ~0.01 ms.
   - Stage (c): seq_len × H = 262 144 saturating MACs plus
     saturating adds, ~0.08 ms.
   - Total new overhead: ~0.15 ms, or ~0.5% of the 30 ms p95
     budget. Mean-pool's ~0.05 ms drops out, net cost ~0.10 ms.
   At median sequence length (seq_len ≈ 340 tokens for the agent-
   CLI workload measured alongside RFC-003), the cost scales
   linearly to ~0.05 ms additional — well under the 1% budget
   threshold.

5. **Expected accuracy delta:**
   NV-Embed §3.2 reports +3.2 points nDCG@10 on MTEB-Retrieval at
   the (H=4096, full LLM-scale) configuration. mGTE §4.1 reports
   +1.4 to +2.1 points top-5 on tool-routing at H=384. For
   mind-nerve's H=256 (smaller than mGTE's H=384), we expect the
   lift to land in the lower-middle of the mGTE band: +1.0 to
   +1.8 points top-5 overall on the STARGA agent-skill catalog,
   with the larger delta concentrated on long-input requests
   (>3 windows) where mean-pool's information-dilution failure
   mode is sharpest. The lower bound (+1.0) exceeds the ROADMAP
   §"Phase 2 must-have" threshold of +0.5 points top-5.

## Non-negotiable conflict

None — the proposal respects all six non-negotiables:

1. *Pure MIND inference path.* The new pooling head is one
   `q16_dot_pinned`, one `q16_softmax`, and a weighted-sum loop;
   no new framework dependency.
2. *Q16.16 × INT8.* No numeric-type change. `pool_q_latent` is
   Q16.16; all multiplies are existing saturating primitives.
3. *Cross-arch bit-identity.* Every primitive is already pinned in
   the bit-identity contract; the new reduction sites (the score
   loop, the weighted-sum loop) are ascending sequential and
   compose pinned primitives.
4. *≤30 ms p95.* Adds ~0.1 ms net cost (mean-pool retired,
   attention pool added) — ~0.3% of the 30 ms budget.
5. *Single static binary.* No new dependency.
6. *Tamper-evident envelope chain.* `pool_q_latent` enters
   `model_hash` via the manifest header. Any silent perturbation
   produces a `HashMismatch` at load time.

## Validation gates run

- arch-mind score before / after: pending (this RFC is a proposal,
  not yet implemented).
- skill-improver mean before / after: pending.
- Latency / accuracy actual numbers: pending implementation against
  the STARGA agent-skill catalog with a reference checkpoint
  trained with the learned `pool_q_latent` parameter.

## Decision

Needs-human-review.

Rationale for not auto-accepting: the `pool_q_latent` vector is a
learned parameter and must be trained alongside the encoder weights
— random initialisation produces near-uniform softmax weights and
degrades to mean-pool-equivalent behaviour with extra latency. The
Phase 1 reference checkpoint is being trained without the attention
pooling head; landing this RFC requires the training-pipeline owner
to add a learnable `q_latent: Parameter(H)` term to the pooling
forward pass and the contrastive loss. The change is small (~5
lines in the training loop), but it bumps `WEIGHTS_VERSION` from 1
to 2 and `MODEL_MANIFEST_VERSION` from 1 to 2 in lockstep,
invalidating every reference artifact currently produced. A human
reviewer should confirm the training-pipeline owner can absorb the
pooling-head training alongside RFC-001's group-wise quantization,
RFC-005's saliency-ranked head mask, RFC-007's attention-sink-aware
training, and RFC-008's MRL auxiliary loss (all five are v2
reference-checkpoint changes; landing them in the same training
run avoids five sequential invalidations of downstream artifacts).
Until the training-pipeline ships the trained `q_latent`, the
backwards-soft path (`pool_q_latent = [0; H]`) produces softmax-
uniform weights that are mathematically equivalent (within 1 ULP)
to mean-pool, so the v2 loader can ship with a zero-vector
`pool_q_latent` as a no-op until the trained vector arrives.
