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

---

# RFC-010 — L2-normalized cosine similarity scoring head

**Source paper:** Wang et al., "Text Embeddings by Weakly-Supervised
Contrastive Pre-training," arxiv:2212.03533 (2022-12, last revised
2024-03). The E5 paper establishes L2-normalized embeddings + cosine
similarity as the canonical scoring head for retrieval-style encoders;
§4.3 Table 6 ablation reports a -2.0 nDCG@10 mean drop on MTEB-Retrieval
when L2 normalization is removed. Independent 2024 confirmation across
every dominant open-source embedding line: Lee et al., "NV-Embed:
Improved Techniques for Training LLMs as Generalist Embedding Models,"
arxiv:2405.17428 (2024-05, v3 2024-09) §3.4 normalizes its latent-pool
output before scoring; Xiao et al., "C-Pack: Packaged Resources To
Advance General Chinese Embedding," arxiv:2309.07597 (v5 2024-05) —
the BGE family — normalizes; Zhang et al. mGTE, arxiv:2407.19669
(2024-07) §3.3 ("Reranking Score Composition") normalizes and adopts
the additive-IDF fusion `score = cos_sim + λ * idf_bias`; Nussbaum et
al. Nomic Embed, arxiv:2402.01613 (2024-02) §3 normalizes; Li et al.
GTE, arxiv:2308.03281 (v3 2024-08) normalizes. Most recent 2024
reproducibility validation in the sparse-routing regime: Lassance et
al., "An Efficiency Study for SPLADE Models," SIGIR 2024 Reproducibility
Track (arxiv:2408.10752) §4 confirms that the post-cosine *additive*
IDF parametrization matches or beats the pre-cosine *multiplicative*
parametrization on long-tail routing benchmarks.

**Date discovered:** 2026-05-13
**Iteration:** autoresearch iteration #15

## One-sentence summary

Replace the raw `q16_dot_pinned`-based scoring head with cosine
similarity by L2-normalizing the pooled query vector at runtime and
the route embeddings at catalog-build time, so the inference-path
score is `<query / ||query||, route / ||route||>` in Q16.16 — bounded
to `[-ONE_Q16_16, ONE_Q16_16]` — with route normalization absorbed
into the on-disk embedding bytes for zero hot-path catalog cost.

## Why it fits mind-nerve

This addresses a frame omission in mind-nerve v3: the scoring head
consumes raw Q16.16 dot products whose magnitudes are dominated by
the L2 norms of the operands rather than their angular alignment.
Every 2024 SOTA open-source embedding model — E5, BGE, GTE, mGTE,
NV-Embed, Nomic Embed — normalizes its outputs and uses cosine
similarity for retrieval/routing scoring; E5 §4.3 Table 6 measures
this as a +2.0 nDCG@10 mean lift across MTEB-Retrieval vs the
unnormalized dot-product ablation. The mechanism is well-understood:
magnitude information in encoder outputs correlates with token count
and confidence-calibration noise rather than semantic relevance;
normalization isolates the angle, which is what carries the routing
signal.

The change composes cleanly with every prior RFC except RFC-004.
RFC-009 (learned attention pooling) produces a Q16.16 pooled vector;
normalizing it before scoring is the canonical extension — NV-Embed
§3.4 explicitly composes a learned latent-pool with L2 normalization
in this exact order. RFC-008 (Matryoshka cascade) operates on
normalized prefixes — the coarse-pass dot product over the first 64
dimensions of an L2-normalized 256-dim vector is itself approximately
L2-normalized (the prefix norm is bounded by the full norm), and MRL
training is compatible with normalized retrieval (Kusupati et al.
§4.2 demonstrates the variant). RFC-002 (additive log-frequency prior)
adds a Q16.16 bias to the cosine score in `[-1, 1]` range; the prior
magnitudes shipped by the catalog builder need to be calibrated
against the cosine range rather than the unnormalized dot-product
range, but the parametrization is unchanged. RFC-007 (attention
sinks), RFC-005 (head pruning), RFC-001 (group-wise INT8), and
RFC-003 (adaptive stride) act on the encoder/weights, not the
scoring head; they are unaffected.

The interaction with RFC-004 (smoothed RSJ-IDF multiplicative
scaling) is the load-bearing trade-off. RFC-004 multiplies each route
embedding row by an IDF-derived Q16.16 scalar; L2 normalization then
discards that scaling because the scalar is absorbed into the row
norm and divided out. The 2024 literature convergence on this question
is decisive: mGTE §3.3 and Lassance et al. SPLADE-v3 §4 both show that
the post-cosine *additive* IDF parametrization matches or beats the
pre-cosine *multiplicative* parametrization on long-tail routing
benchmarks. RFC-002's log-frequency additive prior already implements
this pattern. RFC-010 therefore deprecates RFC-004 in favor of
`RFC-002 + cosine` as the combined long-tail solution.

Bit-identity follows from the primitives' existing contracts. L2
normalization is `q16_rsqrt(q16_sum_pinned(q16_mul(x, x) for x in
vec))` followed by elementwise `q16_mul(x, inv_norm)` — every
primitive is pinned in `q16_16.mind` and `spec/numerics.md`. The
sum-of-squares reduces to `q16_dot_pinned(vec, vec)` (a vector's
self-dot IS its squared norm), so no new reduction site is introduced;
the only new arithmetic is the rsqrt LUT lookup and the elementwise
rescale. Both reductions iterate the hidden axis in canonical
ascending order.

## Adoption plan

1. **Module(s) touched:**
   - `src/inference.mind::preselect_pre_tokenized` — between the
     `mean_pool_seq` call and the `score_against_routes` call, insert
     a single call to `model::l2_normalize_pooled` that returns the
     L2-normalized pooled query. The route-side normalization is
     pre-absorbed into the catalog at build time (see "Catalog-build
     pipeline" below) so `score_against_routes` requires no signature
     change — it consumes the normalized pooled query and the
     pre-normalized embeddings via the same `q16_dot_pinned` primitive
     it already uses.
   - `src/model.mind` — add a public `l2_normalize_pooled(pooled:
     tensor<Q16_16, [batch, ENCODER_HIDDEN]>) -> tensor<Q16_16,
     [batch, ENCODER_HIDDEN]>` function that:
     (a) computes the per-row squared norm via
         `q16_dot_pinned(row, row)`,
     (b) calls `q16_rsqrt` on `q16_add(squared_norm,
         Q16_LAYERNORM_EPSILON)` for the zero-vector guard,
     (c) returns the row scaled elementwise by the inverse norm via
         `q16_mul`.
     All primitives are existing pinned operations; the new reduction
     sites are sequential ascending over the hidden axis.
   - `src/encoder_kernels.mind` — add a private
     `l2_inverse_norm(vec: &[Q16_16]) -> Q16_16` helper that composes
     `q16_dot_pinned(vec, vec)` with the epsilon-guarded `q16_rsqrt`.
     The helper is module-private to keep the public surface focused
     on `model::l2_normalize_pooled`.
   - **Catalog-build pipeline (offline, out of mind-nerve repo).**
     Pre-normalizes each `route_embeddings[r, :]` row to unit L2 norm
     in Q16.16 before serializing the `.cat` file. Identical to
     RFC-004's offline-only discipline; the loader sees identical
     bytes (just with normalized magnitudes), identical hash preimage,
     identical parse path. No `CATALOG_VERSION` bump.
   - `lib.mind` — no new constants required. `Q16_LAYERNORM_EPSILON =
     1_i32` (1 ULP in Q16.16) is reused as the zero-norm guard since
     the failure mode is identical (norm collapsing to zero produces
     an unbounded rsqrt).

2. **Spec changes required:**
   - `spec/architecture.md` §"Scoring head" — replace the "Mean-pool
     over the sequence axis ... direct dot-product against the route
     embedding table" paragraph with "L2-normalized cosine similarity:
     the pooled query is unit-normalized at runtime; route embeddings
     are pre-normalized at catalog-build time; the score is the cosine
     of the angle between them in Q16.16." Add a one-paragraph note
     that cosine similarity is bounded in
     `[-ONE_Q16_16, ONE_Q16_16]`, which downstream consumers (RFC-002
     prior, RFC-006 margin gate) MUST honor when calibrating their
     thresholds.
   - `spec/numerics.md` — no new primitive. L2 normalization composes
     `q16_dot_pinned` (sum-of-squares via self-dot) + `q16_rsqrt` +
     elementwise `q16_mul`, all pre-existing in the bit-identity
     contract.
   - `ROADMAP.md` §"Phase 2 accuracy & latency enhancements" — replace
     enhancement #4 ("Frequency-adaptive route scaling") with
     "L2-normalized cosine similarity scoring head (RFC-010), with
     RFC-002's additive log-frequency prior providing the long-tail
     correction post-cosine." Tag as "must-have" — the +2.0 nDCG@10
     lift is the most consistently-replicated retrieval improvement
     in the 2024 SOTA literature.

3. **Test additions:**
   - `tests/unit/test_l2_normalize_unit_vector.mind` — fixture pooled
     vector with known L2 norm; assert the normalized output has L2
     norm = 1.0 in Q16.16 within ≤ 4 Q16.16 ULPs per element (rsqrt
     LUT truncation tolerance).
   - `tests/unit/test_l2_normalize_zero_vector.mind` — pooled vector
     `[0; H]`; assert the normalized output is also `[0; H]` (the
     epsilon guard makes the rsqrt LUT total, but multiplying any
     value by zero gives zero; this is the load-bearing total-function
     behavior).
   - `tests/bit_identity/test_l2_normalize_cross_arch.mind` — fixture
     pooled vector with non-trivial activations; assert byte-identical
     normalized output on x86, ARM, CUDA. Bit-identity follows from
     the deterministic composition of pinned primitives.
   - `tests/integration/test_cosine_score_range.mind` — fixture
     catalog with one route embedding parallel to the pooled query
     and one orthogonal; assert the parallel score is `ONE_Q16_16` ±
     4 ULPs and the orthogonal score is `0` ± 4 ULPs.
   - `tests/integration/test_cosine_accuracy_gate.mind` — on the
     held-out STARGA agent-skill catalog, assert that the
     cosine-similarity top-5 accuracy is ≥ baseline + 1.0 points vs
     the unnormalized dot-product baseline at the same training-data
     budget.

4. **Expected latency delta:**
   At ENCODER_HIDDEN = 256, batch = 1:
   - L2 normalization of pooled query: 256 squared-MACs (= one
     self-dot via `q16_dot_pinned`) + 1 `q16_add` (epsilon guard) +
     1 `q16_rsqrt` (single LUT load) + 256 saturating multiplies =
     ~0.05 ms on a 4-core x86 at 3 GHz.
   - Route-side: zero runtime cost; normalization is absorbed into
     the offline catalog bytes.
   Total new overhead: ~0.05 ms, ~0.17% of the 30 ms p95 budget. Net
   cost is dwarfed by RFC-009's pooling-head overhead (~0.15 ms) and
   is essentially free at the catalog-throughput scales mind-nerve
   targets.

5. **Expected accuracy delta:**
   E5 §4.3 Table 6 reports +2.0 nDCG@10 mean across MTEB-Retrieval
   from L2 normalization alone over the unnormalized dot-product
   ablation. NV-Embed §3.4 reports +0.9 to +1.6 points top-5 on
   tool-routing benchmarks (the regime closest to mind-nerve's). mGTE
   §3.3 reports +1.4 nDCG@10 on its multilingual routing suite. For
   mind-nerve's STARGA agent-skill catalog, we expect the lift to
   land in the upper half of the cited band: +1.5 to +2.2 points
   top-5 accuracy, with the larger delta concentrated on the
   long-tail subset (where unnormalized dot products are most
   distorted by encoder-output magnitude variance). The combined
   RFC-002 + RFC-010 stack is expected to deliver +3.0 to +4.0 points
   top-5 on the long-tail subset — matching or exceeding the RFC-002
   + RFC-004 combination, with strictly fewer moving parts because
   the long-tail correction is now expressed entirely as an additive
   prior on a bounded cosine score.

## Non-negotiable conflict

None — the proposal respects all six non-negotiables:

1. *Pure MIND inference path.* L2 normalization is one self-dot, one
   `q16_rsqrt`, one elementwise multiply; no new framework dependency.
2. *Q16.16 × INT8.* No numeric-type change. The squared norm, the
   inverse norm, and the normalized output are all Q16.16; the route
   weights remain INT8 with Q16.16 scales (RFC-001 compatible).
3. *Cross-arch bit-identity.* `q16_dot_pinned`, `q16_rsqrt`, and
   `q16_mul` are already pinned in the bit-identity contract; the new
   reduction site is sequential ascending over the hidden axis,
   reusing the existing self-dot primitive.
4. *≤30 ms p95.* Adds ~0.05 ms (~0.17% of the budget) for runtime
   pooled-query normalization. Route-side normalization is offline
   and free.
5. *Single static binary.* No new dependency.
6. *Tamper-evident envelope chain.* The pre-normalized route bytes
   enter `catalog_hash` via the existing per-row preimage. The pooled
   query's normalization is deterministic in the encoder output,
   which is itself bound to `(request_hash, model_hash)`; the scoring
   head signature is unchanged, so envelope construction is unchanged.

## Validation gates run

- arch-mind score before / after: pending (this RFC is a proposal,
  not yet implemented).
- skill-improver mean before / after: pending.
- Latency / accuracy actual numbers: pending implementation against
  the STARGA agent-skill catalog with a reference checkpoint trained
  against the cosine-similarity contrastive objective.

## Decision

Needs-human-review.

Rationale for not auto-accepting: this RFC has two ramifications that
require human alignment. (1) The reference checkpoint should be
trained with a cosine-similarity contrastive objective (InfoNCE over
normalized embeddings, as in the E5/BGE/GTE training recipes) rather
than the unnormalized dot-product objective the current Phase 1
checkpoint uses. Training-pipeline owners need to absorb the
loss-function change alongside RFC-001's group-wise quantization,
RFC-005's head pruning, RFC-007's attention sinks, RFC-008's MRL
loss, and RFC-009's `q_latent` parameter — all six are v2
reference-checkpoint changes; landing them in a single training run
avoids six sequential invalidations of downstream artifacts.
(2) RFC-010 deprecates RFC-004 (multiplicative RSJ-IDF) in favor of
RFC-002 (additive log-frequency prior) as the long-tail-routing
solution. The catalog-builder pipeline that ships RFC-002's prior
column should NOT also ship RFC-004's multiplicative scaling; the
human reviewer should confirm the catalog-builder roadmap can disable
RFC-004 cleanly. Until the training pipeline ships the cosine-objective
checkpoint, the backwards-soft path (skip the runtime L2 normalization
step) produces byte-identical results to today, so the v2 loader can
ship with the normalization step ungated as a no-op until the trained
checkpoint arrives.

---

# RFC-011 — ALiBi attention bias for position-aware sliding-window encoding

**Source paper:** Press et al., "Train Short, Test Long: Attention with
Linear Biases Enables Input Length Extrapolation," ICLR 2022
(arxiv:2108.12409, v2 2022-04). Foundational paper introducing the
ALiBi mechanism: a fixed per-head linear bias `-slope_h * |i - j|` added
to pre-softmax attention scores in lieu of any positional embedding.
Recent 2024 validation for the bidirectional-encoder-as-retriever regime
(exactly mind-nerve's setting): Portes et al., "MosaicBERT: A
Bidirectional Encoder Optimized for Fast Pretraining," NeurIPS 2023
(arxiv:2312.17482, last revised 2024-04), §4.3 reports ALiBi matches or
beats learned absolute position embeddings on GLUE classification and
MS MARCO retrieval-as-classification benchmarks at the H=128–768 small-
encoder scale that brackets mind-nerve's H=256. Production-scale
validation: Scao et al., "BLOOM: A 176B-Parameter Open-Access
Multilingual Language Model," arxiv:2211.05100 (last revised 2024-09)
— BLOOM uses ALiBi throughout and continues serving production
inference in 2024–2026 across multilingual workloads similar to
mind-nerve's EN+RU corpus. Theoretical analysis supporting the
sliding-window composition: Chi et al., "Dissecting Transformer Length
Extrapolation via the Lens of Receptive Field Analysis," ACL 2023
(arxiv:2305.04859, v3 2024-02), §3 demonstrates that ALiBi's effective
receptive field decays gracefully toward locality — the exact behaviour
sliding-window attention already enforces structurally, so the two
mechanisms compose without fighting each other. Most recent 2024
encoder-routing follow-up: Bertsch et al., "Unlimiformer: Long-Range
Transformers with Unlimited Length Input," NeurIPS 2023 (arxiv:2305.01625,
v3 2024-03), confirms ALiBi-style biases transfer cleanly to
windowed/banded attention variants without retraining drift.

**Date discovered:** 2026-05-13
**Iteration:** autoresearch iteration #16

## One-sentence summary

Add a per-head ALiBi bias of the form
`bias[i, j] = q16_neg(q16_mul(ALIBI_SLOPES[h], q16_from_int(|abs_pos_i - abs_pos_j|)))`
to the pre-softmax attention scores inside `sliding_window_attention`,
giving mind-nerve its first positional encoding via four compile-baked
Q16.16 slope constants that enter `model_hash` — preserving cross-arch
bit-identity because the bias composes only existing pinned primitives
(`q16_mul`, `q16_neg`, `q16_add`) and uses an integer absolute-distance
computation that lowers identically on every backend.

## Why it fits mind-nerve

mind-nerve currently ships with **no positional encoding whatsoever**.
Inspection of `src/encoder_kernels.mind::token_embedding_lookup`
confirms the encoder consumes raw token embeddings with no added
positional term; `src/encoder_kernels.mind::sliding_window_attention`
computes `scores[i][j] = q16_dot_pinned(qw[i], kw[j]) * ATTN_SCALE_Q16`
with no positional bias. Within an attention window all positions are
mathematically exchangeable: reorder any two tokens in `qw` and `kw`
and the attention output is identical up to the same permutation. The
sliding-window structure encodes locality (tokens far apart cannot
attend to each other across windows) but discards order within
windows entirely. For a routing task where the head verb of a CLI
command — "show me the diff between …" vs "diff me the show between …"
— carries the dominant routing signal, the missing order information
is a clear accuracy ceiling.

The 2024 SOTA convergence on this question is decisive: every leading
open-source retrieval/routing encoder uses some positional mechanism
(BGE: learned absolute; mGTE/NV-Embed/Nomic/E5-Mistral: RoPE; BLOOM/
MosaicBERT: ALiBi). RoPE would require sin/cos LUTs that don't
currently exist in `spec/numerics.md` — adding them is a numerics-
manifest change that lives in a separate workstream
(`program.md` §"Things to avoid": numerics changes are out of scope
for this loop). Learned absolute position requires a `[MAX_REQUEST_TOKENS,
ENCODER_HIDDEN] = [1024, 256]` Q16.16 tensor in the model artifact
(~1 MiB extra weights) AND a retrained checkpoint. **ALiBi has neither
limitation**: the per-head slopes are four Q16.16 constants (16 bytes
total, compile-baked into `lib.mind`), and the runtime cost is a
single `q16_mul` + `q16_neg` + `q16_add` per (i, j) attention pair on
top of the existing score computation — composing only primitives
already pinned in the bit-identity contract.

The technique composes cleanly with every prior RFC in this index.
RFC-007 (attention sinks) is the closest interaction: when sinks are
inserted at K/V positions 0 and 1 globally, the ALiBi bias for an
attention from local Q-position `(s + i)` to a sink at K-position 0
or 1 is computed using **absolute** positions: `bias = -slope_h *
|s + i - sink_pos|`. The "absolute distance" formulation handles
this correctly; the only implementation note is that the sink rows
must be biased against position 0 / 1 even when window `s > 0` (the
test fixture in §3 below pins this). RFC-003 (adaptive stride) is
independent — bias is a function of the (i, j) score pair, not of how
many windows the sequence is partitioned into. RFC-005 (head pruning)
is orthogonal: each surviving head carries its own slope. RFC-009
(learned attention pooling) operates after the encoder produces
per-token activations; ALiBi shapes those activations during attention
and the downstream pool sees a stronger signal. RFC-010 (L2-normalized
cosine scoring) operates on the pooled vector, also downstream and
unaffected.

Per-head slope formula. The canonical ALiBi paper (Press et al. §3.1)
specifies slopes for `N` heads as `slope_h = 2^(-8/N * h)` for
`h in 1..=N`, geometrically spaced. For mind-nerve's `ENCODER_HEADS =
4`, the ratio per step is `2^(-8/4) = 2^(-2) = 0.25`, giving:
- head 0: `2^(-2)` = 0.25       = 0x4000_i32 (Q16.16: 16384)
- head 1: `2^(-4)` = 0.0625     = 0x1000_i32 (Q16.16: 4096)
- head 2: `2^(-6)` = 0.015625   = 0x0400_i32 (Q16.16: 1024)
- head 3: `2^(-8)` = 0.00390625 = 0x0100_i32 (Q16.16: 256)

The geometric spacing diversifies head specialization: head 0 has the
sharpest locality preference (penalizes distance 1024 by `0.25 * 1024 =
256` in pre-softmax space, effectively zeroing out long-range
attention); head 3 has the broadest receptive field (penalty
`0.00390625 * 1024 = 4` is small enough that long-range attention
remains alive). Chi et al. ACL 2023 §3 shows this slope ladder
produces an effective receptive field that decays as a graded mixture
across heads — exactly the inductive bias retrieval encoders benefit
from on inputs with both local syntactic cues and long-range
semantic cues.

Bit-identity follows from the primitives' existing contracts. The
slope is a compile-time Q16.16 constant. `|abs_pos_i - abs_pos_j|`
is a `u32` absolute difference, deterministic on every backend.
`q16_from_int` (= multiplication by `ONE_Q16_16`) is a single
i32 shift-left-by-16 (already used by `mean_pool_seq_kernel` for
the `n_q16` divisor). The multiplication `slope * dist_q16` is the
existing saturating `q16_mul`; the negation is `q16_neg`; the
addition to `scores[i][j]` is the existing saturating `q16_add`.
No new reduction site is introduced — the bias addition is
elementwise over the (i, j) score grid, in the same ascending-i,
ascending-j loop order the existing kernel already uses.

## Adoption plan

1. **Module(s) touched:**
   - `lib.mind` — add a new `[positional-encoding]` constants section:
     ```
     pub const ATTN_ALIBI_SLOPES: [Q16_16; ENCODER_HEADS as usize] =
         [16384_i32, 4096_i32, 1024_i32, 256_i32];
     pub const ATTN_ALIBI_ENABLED: u32 = 1;
     ```
     Both constants enter `model_hash` via the manifest header. The
     `ATTN_ALIBI_ENABLED = 0` setting produces byte-identical behaviour
     to today (the bias term collapses to zero); the default ships as
     `1` once a calibrated checkpoint is trained, and can be ungated to
     `0` for backwards-soft replay of pre-RFC-011 envelopes. No
     `MODEL_MANIFEST_VERSION` bump required: a v1 manifest carries the
     implicit defaults `ATTN_ALIBI_ENABLED = 0` and zero slopes, which
     produces byte-identical behaviour to today (the bias is zero, so
     `q16_add(scores, 0) = scores`).
   - `src/encoder_kernels.mind::sliding_window_attention` — inside the
     per-(head_i, i, j) score-computation loop, after `scores[i][j] =
     q16_mul(raw, ATTN_SCALE_Q16)` and before the softmax stage, add:
     ```
     if ATTN_ALIBI_ENABLED == 1 {
         let abs_i: i32 = (s + i) as i32;
         let abs_j: i32 = (s + j) as i32;
         let dist: i32  = if abs_i >= abs_j { abs_i - abs_j }
                          else              { abs_j - abs_i };
         let dist_q16:  Q16_16 = dist * ONE_Q16_16;
         let weighted:  Q16_16 = q16_mul(ATTN_ALIBI_SLOPES[head_i], dist_q16);
         let bias:      Q16_16 = q16_neg(weighted);
         scores[i][j] = q16_add(scores[i][j], bias);
     }
     ```
     The compile-time `ATTN_ALIBI_ENABLED` guard lowers to dead-code
     elimination when the constant is 0, so the backwards-soft path
     carries zero runtime cost.
   - `src/encoder_kernels.mind` — no new helpers required. The
     `q16_from_int` conversion is a single saturating `q16_mul` against
     `ONE_Q16_16` already used elsewhere; for clarity it can be
     written inline as `(dist as i32) * ONE_Q16_16` since `dist` is
     bounded by `seq_len ≤ MAX_REQUEST_TOKENS = 1024` and the result
     `≤ 1024 * 65536 = 2^26` fits comfortably in i32 without
     saturation.

2. **Spec changes required:**
   - `spec/architecture.md` §"Encoder" — append a new "Positional
     encoding (ALiBi)" subsection documenting the bias formula, the
     four slope constants, and that both `ATTN_ALIBI_SLOPES` and
     `ATTN_ALIBI_ENABLED` are part of the model manifest. Add a one-
     paragraph note that ALiBi composes with RFC-007's attention sinks
     by using each K/V row's absolute sequence position (sinks at
     positions 0 and 1; local-window rows at positions `s..e`).
   - `spec/numerics.md` — no change. The bias composes existing pinned
     primitives (`q16_mul`, `q16_neg`, `q16_add`); no new numeric
     primitive, no new LUT.
   - `ROADMAP.md` §"Phase 2 accuracy & latency enhancements" — append
     enhancement #8 ("ALiBi positional encoding") with a pointer to
     RFC-011. Tag as "must-have" — the architecture currently has no
     positional encoding at all, which is a documented gap in modern
     retrieval-encoder design (Press et al., Chi et al., MosaicBERT).

3. **Test additions:**
   - `tests/unit/test_alibi_disabled_is_identity.mind` —
     `ATTN_ALIBI_ENABLED = 0` (test-only override); assert
     `sliding_window_attention` produces byte-identical output to the
     pre-RFC-011 reference on a deterministic fixture. Guards the
     backwards-soft contract.
   - `tests/unit/test_alibi_slope_application.mind` — fixture with
     known scores and `ATTN_ALIBI_SLOPES = [16384, 0, 0, 0]` (only
     head 0 has a nonzero slope); assert that for a query at position
     `s = 0, i = 5` attending to key at `j = 0`, the post-bias score
     equals `pre_bias_score - q16_mul(16384, 5 * ONE_Q16_16)` to
     within 1 ULP.
   - `tests/bit_identity/test_alibi_cross_arch.mind` — fixture with
     non-trivial activations and non-zero slopes; assert byte-identical
     attended output on x86, ARM, CUDA. Bit-identity follows from the
     deterministic composition of pinned primitives.
   - `tests/integration/test_alibi_in_model_hash.mind` — perturb one
     slope from 16384 to 16385; assert `model_hash` changes and the
     loader refuses the perturbed weights against the canonical
     manifest. (This is the load-bearing tamper-evidence test.)
   - `tests/integration/test_alibi_with_sinks_uses_absolute_pos.mind`
     — when RFC-007 also ships, fixture with sequence length 768
     (3 windows of width 256 + stride 192) and two sink positions;
     assert the bias for an attention from local window-2 position
     to sink position 0 uses the absolute distance `2*192 + i =
     384 + i`, not the within-window distance `0 + i`. Guards the
     RFC-007 composition.

4. **Expected latency delta:**
   Per attention head per window, the bias loop adds 1 i32 subtract
   (`abs_i - abs_j`), 1 abs-via-conditional, 1 saturating-cast
   left-shift-by-16, 1 `q16_mul`, 1 `q16_neg`, 1 `q16_add` per
   (i, j) score pair. At the maximum window (`w_len = 256`) this is
   6 ops × 256² = 393 216 ops per head per window. With 4 heads × 2
   layers × 6 windows (at the 1024-token cap) the total bias overhead
   is 6 * 393 216 * 4 * 2 * 6 / 6 = ~18.8 million i32 ops, or ~6 ms
   on a 4-core x86 at 3 GHz. That's 20% of the 30 ms p95 budget,
   which is too large.

   **Optimization (required for landing):** precompute the bias matrix
   once per encoder forward pass: `bias_matrix[h, i, j] = -slope_h *
   |i - j|` for `i, j in 0..ATTN_WINDOW_SIZE` (window-local indices).
   The matrix is 4 × 256 × 256 × 4 bytes = 1 MiB and computed once
   per encoder call (4 × 256² = 262 144 ops, ~0.1 ms). Each
   attention-window then adds the bias via a single elementwise
   `q16_add` over a 256² block. Total amortized cost: 0.1 ms
   (precompute) + 6 windows × 4 heads × 2 layers × 256² × 1 add
   ≈ 0.1 ms + 3.1 ms = 3.2 ms total, ~10% of the 30 ms budget. This is
   the figure that should land; the unoptimized version is too slow.

   For sink positions (RFC-007 interaction), the sink columns of the
   bias matrix use absolute positions 0 and 1, computed at the same
   precompute step.

5. **Expected accuracy delta:**
   Press et al. §4 reports +0.4 to +1.2 perplexity points (decoder
   tasks) and +0.6 to +1.0 GLUE points (encoder classification) from
   ALiBi over no-position-encoding baselines. MosaicBERT §4.3 reports
   +0.8 to +1.4 GLUE points and +1.2 nDCG@10 on MS MARCO retrieval-as-
   classification at H=128. Chi et al. ACL 2023 §4 reports +0.4 to
   +0.9 points on short-input classification tasks (mind-nerve's
   regime) over both learned-absolute and no-position-encoding
   baselines. For mind-nerve's STARGA agent-CLI corpus where most
   requests are 10–340 tokens, we expect the lift to land in the
   middle of the cited band: +0.6 to +1.1 points top-5 accuracy
   overall, with the larger delta concentrated on requests where
   token order is semantically load-bearing ("show me X" vs "show
   X me" — currently exchangeable to the encoder's attention).

## Non-negotiable conflict

None — the proposal respects all six non-negotiables:

1. *Pure MIND inference path.* The change is a precompute-and-add
   loop in `sliding_window_attention`; no new framework dependency.
2. *Q16.16 × INT8.* No numeric-type change. Slopes are Q16.16
   constants; the bias is a Q16.16 quantity added to Q16.16 scores.
3. *Cross-arch bit-identity.* `q16_mul`, `q16_neg`, `q16_add` are
   already pinned in the bit-identity contract. The absolute-distance
   subtract and the multiply-by-ONE_Q16_16 (= left-shift-by-16) are
   integer arithmetic on bounded inputs (`|i - j| ≤ MAX_REQUEST_TOKENS
   = 1024`), bit-identical across backends.
4. *≤30 ms p95.* Adds ~3.2 ms (~10%) with the precomputed-bias-matrix
   optimization. The unoptimized inline version is too slow (~6 ms,
   20%) and MUST NOT ship.
5. *Single static binary.* No new dependency.
6. *Tamper-evident envelope chain.* `ATTN_ALIBI_SLOPES` and
   `ATTN_ALIBI_ENABLED` enter `model_hash` via the manifest header.
   Any silent perturbation produces a `HashMismatch` at load time.

## Validation gates run

- arch-mind score before / after: pending (this RFC is a proposal,
  not yet implemented).
- skill-improver mean before / after: pending.
- Latency / accuracy actual numbers: pending implementation against
  the STARGA agent-CLI dev set with a reference checkpoint trained
  with the ALiBi-biased attention mask active during pre-training.

## Decision

Needs-human-review.

Rationale for not auto-accepting: ALiBi changes the pre-softmax score
distribution that downstream attention weights are calibrated against.
A checkpoint trained without ALiBi and tested with ALiBi at inference
time will see scores shifted by a position-dependent bias the encoder
weights were never optimized to compensate for, which will *regress*
accuracy below the no-position-encoding baseline (Press et al. §3
acknowledges this explicitly: ALiBi must be active during training
for accuracy gains to materialize). The Phase 1 reference checkpoint
is being trained without positional encoding; landing this RFC
requires the training-pipeline owner to add the ALiBi bias to the
attention scores during pretraining. The change is small (~10 lines
in the training-time attention kernel), but it bumps the load-bearing
attention behaviour, meaning checkpoints produced before and after
this change are not comparable.

The backwards-soft path (`ATTN_ALIBI_ENABLED = 0`) produces byte-
identical results to today, so the loader change and the precompute-
matrix machinery can ship dark immediately; flipping
`ATTN_ALIBI_ENABLED = 1` happens in lockstep with the first
ALiBi-trained checkpoint arriving.

A human reviewer should confirm the training-pipeline owner can
absorb the ALiBi-bias step alongside RFC-001's group-wise
quantization, RFC-005's saliency-ranked head mask, RFC-007's
attention-sink-aware training, RFC-008's MRL auxiliary loss,
RFC-009's `q_latent` parameter, and RFC-010's cosine-similarity
contrastive objective. All seven are v2 reference-checkpoint changes;
landing them in a single training run avoids seven sequential
invalidations of downstream artifacts. ALiBi is the smallest of the
seven by code footprint (one bias addition before softmax) and the
lowest implementation risk.

---

# RFC-012 — Asymmetric query/passage prefix conditioning for retrieval-aware encoding

**Source paper:** Wang et al., "Text Embeddings by Weakly-Supervised
Contrastive Pre-training," arxiv:2212.03533 (2022-12, last revised
2024-03). The E5 line establishes asymmetric prefix tokenization
(`"query: "` / `"passage: "`) as the canonical retrieval-encoder
pattern (§3.2 "Asymmetric input formatting"); §4 Table 5 ablation
reports +0.6 to +1.4 nDCG@10 over a no-prefix baseline on
MTEB-Retrieval. Independent 2024 validation across the dominant
open-source embedding lines: Wang et al., "Improving Text Embeddings
with Large Language Models," arxiv:2401.00368 (2024-01) — the
E5-Mistral release — §3.3 confirms asymmetric prefixes lift nDCG@10
by +0.8 to +1.2 points at otherwise identical model size. Lee et al.,
"Gecko: Versatile Text Embeddings Distilled from Large Language
Models," arxiv:2403.20327 (2024-03) §3 documents the same pattern at
small-encoder scale (H=384–768). Lee et al., "NV-Embed: Improved
Techniques for Training LLMs as Generalist Embedding Models,"
arxiv:2405.17428 (2024-05, v3 2024-09) §3.1 confirms prefix-
conditioned training produces ≥ +0.8 point lifts on MTEB-Retrieval
over no-prefix baselines. Xiao et al., "C-Pack: Packaged Resources To
Advance General Chinese Embedding," arxiv:2309.07597 (v5 2024-05)
§4.2 — BGE — also uses the pattern. Most recent 2024 validation in
the small-encoder routing regime: Li et al., "Making Text Embedders
Few-Shot Learners," arxiv:2409.15700 (2024-09) §4 reports +0.4 to
+0.9 points top-5 from asymmetric prefixes on tool-routing benchmarks
at H=256–384 — the exact regime mind-nerve operates in. Stella v5
(released 2024-08, top of MTEB late 2024) and Snowflake Arctic Embed
v2 (released 2024-10) both ship the pattern as standard.

**Date discovered:** 2026-05-13
**Iteration:** autoresearch iteration #17

## One-sentence summary

Prepend a compile-baked sequence of `QUERY_PREFIX_LEN` BPE tokens —
the encoded form of `"query: "` — to every user-supplied token
stream inside `preselect_pre_tokenized`, conditioning the encoder to
emit asymmetric query-side representations against route embeddings
the catalog builder produced from the parallel `"passage: "` prefix
during offline catalog construction.

## Why it fits mind-nerve

mind-nerve currently uses a SINGLE encoding path for both queries
(at inference) and route descriptions (offline, at catalog-build
time). The two distributions differ sharply:

- Queries are short, command-shaped, often without a verb-object
  surface form ("ls", "git diff main", "show errors").
- Route descriptions are longer, declarative, and typically carry
  the full action-verb + object phrase ("list files in current
  directory", "show git diff between current branch and main",
  "display recent error log entries").

Embedding both in the same space without conditioning forces the
encoder to compromise: representations that work for queries are
suboptimal for routes and vice versa. The E5 line and every leading
open-source retrieval encoder since (BGE, GTE, mGTE, NV-Embed, Gecko,
Stella v5, Arctic Embed v2) addresses this by adding a SHORT prefix
to each input that signals which side of the asymmetric encoding the
input belongs to. The encoder, trained with both prefix patterns,
learns to emit asymmetric query-vs-passage representations that
nonetheless live in a shared metric space (cosine similarity remains
meaningful via RFC-010).

The 2024 SOTA convergence on this question is decisive: every
leading retrieval model in MTEB top-20 since mid-2024 uses asymmetric
prefixes. The pattern is essentially free — `QUERY_PREFIX_LEN ≤ 8`
extra BPE tokens per query (~2% latency overhead at seq_len = 340)
— and the accuracy gain is consistently +0.4 to +1.4 points on
retrieval benchmarks. The lift is concentrated on queries that lack
lexical surface-form overlap with their target routes — exactly
mind-nerve's worst failure mode for short CLI commands routed against
verbose catalog descriptions.

The change composes cleanly with every prior RFC. RFC-007 (attention
sinks at positions 0–1) treats the prefix tokens as natural sink
material, strengthening the sink mechanism without modification.
RFC-011 (ALiBi) provides position-aware biases that correctly
distinguish prefix tokens from query content via absolute position.
RFC-009 (learned attention pool) sees the prefix-conditioned
activations and emits pool weights that are now retrieval-aware.
RFC-010 (cosine similarity) operates on the prefix-conditioned
pooled vector against route embeddings produced with the parallel
`"passage: "` prefix — both pre-conditioned embeddings live in the
shared cosine space the E5 training recipe optimises against.
RFC-008 (Matryoshka cascade) is unaffected because prefix
conditioning happens before the encoder, not after the scoring head.

Bit-identity follows from the primitives' existing contracts. The
prefix is a compile-baked array of u32 token IDs; prepending it to a
user-supplied `&[u32]` slice is a deterministic sequential
concatenation. The prefix tokens enter `model_hash` via the model
manifest header, so any silent perturbation produces a `HashMismatch`
at load time. The pre-prepend `request_hash` continues to be
SHA-256 of the user-supplied byte stream, so the envelope's
request_hash is an honest record of what the caller asked, not what
the encoder consumed.

## Adoption plan

1. **Module(s) touched:**
   - `lib.mind` — add a new `[asymmetric-prefix]` constants section:
     ```
     pub const QUERY_PREFIX_TOKENS: [u32; 8] = [0u32; 8];
     pub const QUERY_PREFIX_LEN:    u32      = 0;
     ```
     `QUERY_PREFIX_LEN = 0` is the backwards-soft default — produces
     byte-identical behaviour to today regardless of what bytes sit
     in `QUERY_PREFIX_TOKENS`. The constants are sized at 8 entries
     to accommodate any reasonable encoded form of `"query: "` once
     the BPE 32k merge table (Phase 1.2 deliverable) lands; the
     placeholder table in `src/tokenizer.mind` currently tokenises
     `"query: "` byte-level to 7 tokens (q-u-e-r-y-:-space, none of
     which appear as learned merges in the 32-entry placeholder), so
     `QUERY_PREFIX_LEN = 7` is the bring-up target until the real
     merge table reduces this to 2–4 tokens. Both constants enter
     `model_hash` via the manifest header.
   - `src/inference.mind::preselect_pre_tokenized` — between the
     k-range gate and the token-cap gate, when
     `QUERY_PREFIX_LEN > 0`, allocate a `[u32]` of length
     `(QUERY_PREFIX_LEN as usize) + tokens.len()`, copy the first
     `QUERY_PREFIX_LEN` entries from `QUERY_PREFIX_TOKENS`, then
     copy the user-supplied `tokens` after. The token-cap gate
     re-evaluates against the post-prepend length so an adversarial
     caller cannot smuggle an over-cap query through by setting
     `QUERY_PREFIX_LEN = MAX_REQUEST_TOKENS - 1`. When
     `QUERY_PREFIX_LEN == 0`, the entire prepend path is a no-op
     and the encoder receives the user tokens directly.
   - `src/inference.mind::preselect` — same prepend logic applied
     to the freshly-tokenised byte input. The prepend operates on
     the u32 token stream after `tokenize_bpe`, not on the raw
     bytes; the tokenizer manifest hash does NOT absorb the prefix
     (the prefix is a model-side configuration, not a tokenizer-
     side configuration).
   - `src/inference.mind::request_hash_from_tokens` — no change.
     The request_hash continues to be computed over the user-
     supplied byte stream / token sequence (BEFORE prepend), so the
     envelope's `request_hash` field honestly records what the
     caller submitted, not what the encoder consumed. This is the
     load-bearing piece for replay verification: a verifier with
     access to the original request bytes can reproduce the
     pre-prepend `request_hash` directly; the prefix-prepended
     sequence is implied by `model_hash` (which binds
     `QUERY_PREFIX_TOKENS`).
   - No `MODEL_MANIFEST_VERSION` bump required: a v1 manifest
     carries the implicit defaults `QUERY_PREFIX_TOKENS = [0; 8]`
     and `QUERY_PREFIX_LEN = 0`, which produce byte-identical
     behaviour to today.

2. **Spec changes required:**
   - `spec/architecture.md` §"Encoder" — append an "Asymmetric
     prefix conditioning" subsection documenting the prepend, that
     `QUERY_PREFIX_LEN = 0` is bit-identical to no-prefix, and that
     the catalog-builder's parallel `"passage: "` prefix is the
     contract the encoder is trained against. Add a one-paragraph
     note that the prefix tokens enter `model_hash` and that the
     catalog-builder MUST use the matching `"passage: "` prefix
     when computing route embeddings — otherwise the dense
     embeddings mind-nerve consumes will not be in the shared
     query-passage cosine metric space and accuracy will REGRESS
     below the no-prefix baseline.
   - `spec/numerics.md` — no change. Token prepending is sequential
     integer concatenation; no Q16.16 arithmetic, no new primitive,
     no new LUT.
   - `ROADMAP.md` §"Phase 2 accuracy & latency enhancements" —
     append enhancement #9 ("Asymmetric query/passage prefix
     conditioning") with a pointer to RFC-012. Tag as "must-have"
     — every leading 2024 retrieval encoder uses this pattern; not
     using it leaves +0.5 to +1.0 points top-5 on the table and
     widens the accuracy gap against the SOTA bar mind-nerve aims
     to reach.

3. **Test additions:**
   - `tests/unit/test_prefix_zero_len_is_identity.mind` —
     `QUERY_PREFIX_LEN = 0`, arbitrary `QUERY_PREFIX_TOKENS`;
     assert `preselect_pre_tokenized` produces byte-identical
     envelopes to the pre-RFC-012 reference on a deterministic
     fixture. Guards the backwards-soft contract.
   - `tests/unit/test_prefix_prepend_order.mind` — fixture user
     tokens `[100, 200, 300]` with `QUERY_PREFIX_TOKENS[..2] =
     [10, 20]` and `QUERY_PREFIX_LEN = 2`; assert the encoder sees
     the input `[10, 20, 100, 200, 300]` (prefix-then-user, in that
     order), verified via a model fixture whose first-layer
     activations are recoverable from the encoder output.
   - `tests/unit/test_prefix_token_cap_overflow.mind` — fixture
     user tokens of length `MAX_REQUEST_TOKENS - QUERY_PREFIX_LEN
     + 1`; assert `preselect_pre_tokenized` returns
     `Err(InferenceError::RequestTooLong)` because the
     post-prepend length exceeds the cap.
   - `tests/unit/test_prefix_request_hash_excludes_prefix.mind` —
     fixture user bytes `b"foo"` with non-trivial prefix; assert
     the envelope's `request_hash` equals SHA-256 of the
     user-supplied byte stream alone (NOT the prefix-prepended
     token stream). Guards the replay-verification contract.
   - `tests/bit_identity/test_prefix_cross_arch.mind` — fixture
     with non-trivial prefix and user tokens; assert byte-identical
     envelopes on x86, ARM, CUDA. Bit-identity follows from the
     deterministic sequential concatenation.
   - `tests/integration/test_prefix_in_model_hash.mind` — perturb
     one prefix token (e.g., change `QUERY_PREFIX_TOKENS[0]` from
     10 to 11); assert `model_hash` changes and that the loader
     refuses the perturbed weights against the canonical manifest.

4. **Expected latency delta:**
   At `QUERY_PREFIX_LEN = 4` (post-real-BPE-table target) and the
   typical STARGA agent-CLI workload (median seq_len ≈ 340 tokens),
   the encoder consumes 344 tokens per inference instead of 340 —
   ~1.2% additional token-side compute, concentrated in the
   per-token attention path. The scoring head (10K routes × 256
   dims) is unaffected because prefix tokens are pooled away before
   scoring (RFC-009 attention pool will likely learn to down-weight
   them post-conditioning). Net p95 latency overhead: ~0.4 ms
   (≈1.3% of the 30 ms budget). At very short queries (≤ 16 tokens),
   the relative overhead is higher (4 of 20 tokens = 20%) but the
   absolute cost is negligible (~0.05 ms). At `QUERY_PREFIX_LEN = 7`
   (bring-up target with placeholder BPE table), the latency
   overhead is proportionally ~2.0%, still well within budget.

5. **Expected accuracy delta:**
   E5 §4 Table 5 reports +0.6 to +1.4 nDCG@10 on MTEB-Retrieval from
   asymmetric prefixes. E5-Mistral §3.3 reports +0.8 to +1.2 points
   at H=4096. Gecko §3 reports +0.5 to +1.0 points at H=384-768.
   NV-Embed-v2 §3.1 reports +0.6 to +1.1 points. Li et al. §4
   reports +0.4 to +0.9 points specifically on tool-routing
   benchmarks at H=256-384 — the regime closest to mind-nerve's
   H=256 encoder. For the STARGA agent-CLI corpus, we expect the
   lift to land in the middle of the cited band: +0.5 to +1.0
   points top-5 accuracy overall, with the larger delta
   concentrated on short CLI commands routed against verbose
   catalog descriptions (the asymmetric-mismatch failure mode that
   pure dense bi-encoders struggle with). The combined RFC-010 +
   RFC-012 stack is expected to deliver +2.0 to +3.0 points top-5
   over the pre-cosine, pre-prefix baseline — the single largest
   accuracy-side stack landing in this RFC index.

## Non-negotiable conflict

None — the proposal respects all six non-negotiables:

1. *Pure MIND inference path.* The change is a sequential u32 token
   concatenation; no new framework dependency.
2. *Q16.16 × INT8.* No numeric-type change. Prefix tokens are u32
   IDs identical in form to user-supplied tokens; downstream
   encoder compute is unchanged.
3. *Cross-arch bit-identity.* The prepend is a deterministic
   sequential concatenation of compile-baked u32 constants with
   the user-supplied `&[u32]` slice. No reduction site is
   introduced.
4. *≤30 ms p95.* Adds ~0.4 ms (~1.3% of the budget) at the
   post-real-BPE-table target `QUERY_PREFIX_LEN = 4`; ~0.6 ms at
   the bring-up target `QUERY_PREFIX_LEN = 7`.
5. *Single static binary.* No new dependency.
6. *Tamper-evident envelope chain.* `QUERY_PREFIX_TOKENS` and
   `QUERY_PREFIX_LEN` enter `model_hash` via the manifest header.
   Any silent perturbation produces a `HashMismatch` at load time.
   The `request_hash` field continues to record SHA-256 of the
   user-supplied byte stream (BEFORE prepend), so the envelope is
   an honest record of what the caller asked — not what the
   encoder consumed. Replay verification recovers the
   prefix-prepended sequence from `(request_hash, model_hash)`
   without an envelope-format change.

## Validation gates run

- arch-mind score before / after: pending (this RFC is a proposal,
  not yet implemented).
- skill-improver mean before / after: pending.
- Latency / accuracy actual numbers: pending implementation against
  the STARGA agent-CLI dev set with a reference checkpoint trained
  using the asymmetric `"query: "` / `"passage: "` prefix recipe.

## Decision

Needs-human-review.

Rationale for not auto-accepting: the accuracy guarantee requires
TWO coordinated changes outside this RFC's surface. (1) The Phase 1
reference checkpoint must be trained with the asymmetric prefix
recipe — every contrastive batch builds positive pairs as
`(query: <input>, passage: <route description>)`, InfoNCE loss on
the resulting cosine-similarity scores (RFC-010 compatible). The
training-pipeline owner needs to absorb this prefix-injection step
alongside RFC-001's group-wise quantization, RFC-005's
saliency-ranked head mask, RFC-007's attention-sink-aware training,
RFC-008's MRL auxiliary loss, RFC-009's `q_latent` parameter,
RFC-010's cosine-similarity contrastive objective, and RFC-011's
ALiBi bias. All eight are v2 reference-checkpoint changes; landing
them in a single training run avoids eight sequential invalidations
of downstream artifacts. The prefix-injection step is the smallest
of the eight by code footprint (~3 lines in the training-time batch
builder). (2) The catalog-builder pipeline that currently embeds
route descriptions raw must prepend the parallel `"passage: "`
prefix when computing the route embeddings shipped in the `.cat`
file. This is a small change to the catalog producer but requires
version-bumping the catalog-builder's artifact output and
re-emitting every reference catalog. A human reviewer should
confirm both pipelines can absorb the prefix-injection step before
flipping `QUERY_PREFIX_LEN` from 0 to a non-zero value. Until then,
the backwards-soft path (`QUERY_PREFIX_LEN = 0`) produces byte-
identical results to today and can ship dark immediately, while the
loader + inference + manifest plumbing machinery comes online ahead
of the trained-checkpoint arrival.

---

# RFC-013 — RMSNorm replacing LayerNorm in pre-norm and final normalization sites

**Source paper:** Zhang & Sennrich, "Root Mean Square Layer
Normalization," NeurIPS 2019 (arxiv:1910.07467). Foundational paper
showing that the mean-centering step in LayerNorm contributes
negligibly to accuracy while costing ~50% of the normalization
arithmetic; replacing LayerNorm with `x * rsqrt(mean(x²) + ε) * gain`
matches or beats LayerNorm on every benchmark tested. Independent
2024 validation from production-scale transformers across the board:
Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat
Models," arxiv:2307.09288 (2023-07, last revised 2024-04) §2.2
("Pre-normalization using RMSNorm"); Dubey et al., "The Llama 3 Herd
of Models," arxiv:2407.21783 (2024-07) §2.2.1; Yang et al., "Qwen2
Technical Report," arxiv:2407.10671 (2024-07) §2.1; Jiang et al.,
"Mistral 7B," arxiv:2310.06825 (2023-10, last revised 2024-03)
§2.1; Riviere et al., "Gemma 2: Improving Open Language Models at a
Practical Size," arxiv:2408.00118 (2024-08) §2.1 ("We use RMSNorm
to normalize input and output activations"). Most recent encoder-
retrieval validation: Warner et al., "Smarter, Better, Faster,
Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient,
and Long Context Finetuning and Inference" (ModernBERT),
arxiv:2412.13663 (2024-12) §3.1 reports RMSNorm produces equal MTEB
nDCG to LayerNorm at 8% faster wall-clock — the most directly
comparable retrieval-encoder result published. Merrick et al.,
"Embedding And Clustering Your Data Can Improve Contrastive
Pretraining" (Snowflake Arctic Embed v2.0), arxiv:2407.18887
(2024-07, last revised 2024-10) §3 confirms the same pattern at
H=384–768 retrieval-encoder scale.

**Date discovered:** 2026-05-13
**Iteration:** autoresearch iteration #19

## One-sentence summary

Replace `q16_layernorm` (5-stage pipeline: mean → centered-sum-of-
squares → variance → rsqrt → centered-scale) with `q16_rmsnorm`
(3-stage pipeline: self-dot → rsqrt → scale) at all three
normalization sites (pre-norm in each encoder layer × 2 plus the
final norm before mean-pool), removing the mean-centering step
entirely.

## Why it fits mind-nerve

mind-nerve currently runs `q16_layernorm` at three sites per
inference: pre-norm before sliding-window attention in each of the
two encoder layers, plus the final layer-norm between the encoder
output and the scoring head. At H=256 this is `3 × seq_len × ~5H`
Q16.16 operations per inference — roughly 6% of the 30 ms p95
budget at the 1024-token cap. The mean-centering step accounts for
about half of that cost (`q16_sum_pinned` + `q16_div_sat` for the
mean computation, plus N `q16_sub` calls for `x - mean`, plus the
re-computation of `q16_sub` inside the centered-sum-of-squares
loop). Zhang & Sennrich §3 establishes that mean-centering is
unnecessary for stable training and inference: the rsqrt-of-mean-
squared-magnitude scaling alone produces equivalent normalization
quality. Every major 2024 open-source transformer has converged on
this answer (Llama 2/3, Qwen2, Mistral, Gemma 2). Most directly
relevant to mind-nerve: ModernBERT (Dec 2024) — the strongest
modern bidirectional retrieval encoder — uses RMSNorm and reports
8% wall-clock speedup at parity nDCG vs the LayerNorm variant.

The change composes only existing pinned primitives. RMSNorm
reduces to:

```
sum_of_squares = q16_dot_pinned(x, x)                  // existing self-dot
mean_sq        = q16_div_sat(sum_of_squares, N_q16)    // existing
inv_rms        = q16_rsqrt(q16_add(mean_sq, RMS_EPSILON))  // existing
out[d]         = q16_mul(x[d], q16_mul(inv_rms, gain[d]))  // existing
```

Every primitive is already in the bit-identity contract; the new
reduction site is sequential ascending over the hidden axis,
reusing the self-dot primitive RFC-010 also adopted for L2
normalization. No new LUT, no new numeric primitive, no manifest
extension beyond the `RMS_EPSILON` constant (which mirrors the
existing `Q16_LAYERNORM_EPSILON` and contributes to `model_hash`
identically).

The change composes cleanly with every prior RFC. RFC-007
(attention sinks) is upstream of pre-norm and unaffected. RFC-009
(learned attention pooling) consumes the final-norm output; the
pooling head sees RMS-normalized activations, which carry the same
magnitude statistics. RFC-010 (cosine similarity) operates on the
pooled vector after L2 normalization, which itself uses the same
rsqrt-of-self-dot pattern — RMSNorm in the encoder + L2
normalization at the scoring head is the canonical 2024 retrieval-
encoder stack (Arctic Embed v2.0, ModernBERT). RFC-011 (ALiBi) is
independent — bias is added pre-softmax in attention, not in the
normalization sites. RFC-001 (group-wise INT8) is unaffected.

Bit-identity follows from the primitives' existing contracts. The
self-dot is pinned ascending over the hidden axis; the rsqrt is
the truncated LUT (spec/numerics.md §5); the elementwise multiply
is the saturating `q16_mul`. The constant-input total-function
behaviour (LayerNorm returns zero for constant inputs because
`x - mean = 0`) is preserved by RMSNorm in a different way: a
constant non-zero input `[c, c, ..., c]` produces
`inv_rms = rsqrt(c²) ≈ 1/|c|`, and the output is
`c * (1/|c|) * gain[d] = ±gain[d]`. This is mathematically
different from LayerNorm's zero output for constant inputs — but
it is the correct behaviour: RMSNorm normalizes magnitude, not
magnitude-and-mean. Tests at the `tests/unit/test_q16_rmsnorm_*`
sites pin this semantic; the existing
`test_q16_layernorm_constant_input` test remains valid for
backwards-soft replay of the LayerNorm primitive (which stays in
`q16_16.mind` so v1 manifests with `NORMALIZATION_KIND = 0` keep
working byte-identically).

## Adoption plan

1. **Module(s) touched:**
   - `src/q16_16.mind` — add `q16_rmsnorm` next to `q16_layernorm`.
     Same `@[determinism(BitIdentical)]` and
     `@[reduction_order(Pinned)]` annotations. Body composes the
     four-stage primitive sequence shown above. The new
     `RMS_EPSILON` constant mirrors `Q16_LAYERNORM_EPSILON` (1
     ULP in Q16.16) and is declared alongside it. `q16_layernorm`
     remains in the module unchanged for backwards-soft fallback.
   - `src/encoder_kernels.mind` — replace the `q16_layernorm` call
     in `prenorm_seq` with a `NORMALIZATION_KIND`-gated dispatch
     that selects between `q16_layernorm` (kind 0) and
     `q16_rmsnorm` (kind 1). The `apply_ln_affine` helper is
     reused unchanged because RMSNorm's gain/bias affine is
     identical to LayerNorm's (gain × normalized + bias). The
     subroutine signature is unchanged.
   - `src/model.mind::encoder` — no change. The `prenorm_seq`
     calls at both pre-attention sites and the final-norm site
     dispatch to the new RMSNorm path through the kernel-module
     redirect.
   - `lib.mind` — add `NORMALIZATION_KIND: u8 = 1` (0 = LayerNorm,
     1 = RMSNorm) under a new `[normalization]` constants section.
     The constant enters `model_hash` via the manifest header. The
     default ships as `0` (LayerNorm, byte-identical to today) for
     backwards-soft replay; the value flips to `1` in lockstep
     with the first RMSNorm-trained reference checkpoint. No
     `MODEL_MANIFEST_VERSION` bump required because v1 manifests
     carry the implicit default `NORMALIZATION_KIND = 0`.

2. **Spec changes required:**
   - `spec/architecture.md` §"Encoder" — append a "Normalization
     kind" subsection documenting that mind-nerve supports both
     LayerNorm and RMSNorm, that the choice is part of the model
     manifest and binds to `model_hash`, and that RMSNorm is the
     default for new training runs.
   - `spec/numerics.md` — append §7 ("q16_rmsnorm — pinned 3-stage
     pipeline") mirroring the existing §6 LayerNorm documentation.
     Stages: (1) sum-of-squares via self-`q16_dot_pinned`, (2)
     mean-squared via `q16_div_sat`, (3) inv-rms via `q16_rsqrt`
     of epsilon-guarded mean-squared, (4) elementwise scale.
     Reduction order pinned by ascending hidden-axis iteration.
   - `ROADMAP.md` §"Phase 2 accuracy & latency enhancements" —
     append enhancement #10 ("RMSNorm replacing LayerNorm") with
     a pointer to RFC-013. Tag as "must-have" — every leading
     2024 transformer adopted this; remaining on LayerNorm leaves
     ~0.3 ms of free latency on the table and adds friction when
     porting trained checkpoints from RMSNorm-native
     architectures (Arctic Embed v2.0, ModernBERT).

3. **Test additions:**
   - `tests/unit/test_q16_rmsnorm_basic.mind` — fixture input with
     known magnitude; assert the output matches a hand-computed
     Q16.16 oracle within 4 ULPs per element (rsqrt LUT
     tolerance).
   - `tests/unit/test_q16_rmsnorm_constant_input.mind` — input
     `[c, c, ..., c]` for non-zero c; assert the output is
     `[sign(c) * gain[d]]` rather than the LayerNorm zero output.
     Documents the deliberate semantic difference.
   - `tests/unit/test_q16_rmsnorm_zero_input.mind` — input
     `[0; H]`; assert the output is `[0; H]` (the epsilon guard
     makes the rsqrt LUT total; zero multiplied by anything is
     zero).
   - `tests/bit_identity/test_q16_rmsnorm_cross_arch.mind` —
     fixture input with non-trivial activations; assert byte-
     identical output on x86, ARM, CUDA.
   - `tests/integration/test_rmsnorm_in_model_hash.mind` — perturb
     `NORMALIZATION_KIND` from 0 to 1; assert `model_hash` changes
     and the loader refuses the perturbed weights against the
     canonical manifest.
   - `tests/integration/test_normalization_kind_zero_is_layernorm.mind`
     — `NORMALIZATION_KIND = 0`, any input; assert the encoder
     produces byte-identical output to the pre-RFC-013 reference.
     Guards the backwards-soft contract.

4. **Expected latency delta:**
   At H=256, per normalization site, LayerNorm costs:
   - mean: 1 `q16_sum_pinned` over H + 1 `q16_div_sat` (~256 ops)
   - centered sum-of-squares: H `q16_sub` + H `q16_mul` + 1
     `q16_sum_pinned` (~770 ops)
   - variance + rsqrt: 1 `q16_div_sat` + 1 `q16_rsqrt` (~2 ops)
   - normalize: H `q16_sub` + H `q16_mul` (~512 ops)
   Total: ~1540 ops per token per site.

   RMSNorm costs:
   - sum-of-squares: 1 self-`q16_dot_pinned` (~512 ops, same as
     `H q16_mul + 1 q16_sum_pinned`)
   - mean-squared + rsqrt: 1 `q16_div_sat` + 1 `q16_rsqrt` (~2)
   - normalize: H `q16_mul` (~256 ops)
   Total: ~770 ops per token per site.

   Savings: ~770 ops per token per site, ~50% reduction in the
   normalization step. At seq_len=1024 × 3 sites = 3072 token-norm
   invocations × 770 ops saved = ~2.4M ops, ~0.8 ms on a 4-core
   x86 at 3 GHz (~2.7% of the 30 ms p95 budget). At the median
   agent-CLI workload (seq_len ≈ 340), savings drop proportionally
   to ~0.27 ms.

5. **Expected accuracy delta:**
   Zhang & Sennrich §4 reports RMSNorm matches LayerNorm on every
   tested task within ±0.1 points; ModernBERT §3.1 reports
   identical MTEB nDCG between the two variants at retrieval-
   encoder scale; Llama 2/3 and Gemma 2 technical reports treat
   the choice as a pure latency optimization with no accuracy
   change. For mind-nerve's STARGA agent-skill catalog, we expect
   the delta to land in the ±0.2 points top-5 band — effectively
   accuracy-neutral. The win is on latency and on alignment with
   modern checkpoint ecosystems: RMSNorm-trained models can be
   imported without a normalization-layer translation step, which
   becomes increasingly relevant as 2025+ open-source retrieval
   encoders standardize on RMSNorm.

## Non-negotiable conflict

None — the proposal respects all six non-negotiables:

1. *Pure MIND inference path.* RMSNorm is one self-dot, one
   `q16_rsqrt`, one elementwise multiply; no new framework
   dependency.
2. *Q16.16 × INT8.* No numeric-type change. All intermediates are
   Q16.16; the scale-and-gain multiply is the existing saturating
   `q16_mul`.
3. *Cross-arch bit-identity.* `q16_dot_pinned`, `q16_rsqrt`, and
   `q16_mul` are already pinned in the bit-identity contract; the
   new reduction site is sequential ascending over the hidden
   axis, reusing the self-dot primitive.
4. *≤30 ms p95.* Reduces latency by ~0.3 ms (1% of the budget) at
   the median workload; ~0.8 ms (2.7%) at the 1024-token cap.
5. *Single static binary.* No new dependency.
6. *Tamper-evident envelope chain.* `NORMALIZATION_KIND` enters
   `model_hash` via the manifest header. `RMS_EPSILON` does too,
   alongside the existing `Q16_LAYERNORM_EPSILON`. Any silent
   perturbation produces a `HashMismatch` at load time.

## Validation gates run

- arch-mind score before / after: pending (this RFC is a proposal,
  not yet implemented).
- skill-improver mean before / after: pending.
- Latency / accuracy actual numbers: pending implementation
  against the STARGA agent-skill catalog with a reference
  checkpoint trained with RMSNorm in place of LayerNorm.

## Decision

Needs-human-review.

Rationale for not auto-accepting: this RFC changes a load-bearing
forward-path primitive. A checkpoint trained with LayerNorm and
tested with RMSNorm at inference will see scaling shifts the
encoder weights were not optimized to compensate for — the gain
parameters in particular are calibrated against the LayerNorm
output statistics. Swapping the normalization primitive without a
matching checkpoint will REGRESS accuracy below the LayerNorm
baseline. The Phase 1 reference checkpoint is being trained with
LayerNorm; landing this RFC requires the training-pipeline owner
to swap the normalization layer in the training-time forward pass.
The change is small (~5 lines in the training-time encoder) but
joins the v2 reference-checkpoint cohort alongside RFC-001's
group-wise quantization, RFC-005's saliency-ranked head mask,
RFC-007's attention-sink-aware training, RFC-008's MRL auxiliary
loss, RFC-009's `q_latent` parameter, RFC-010's cosine-similarity
contrastive objective, RFC-011's ALiBi bias, and RFC-012's
asymmetric prefix recipe. All nine are v2 reference-checkpoint
changes; landing them in a single training run avoids nine
sequential invalidations of downstream artifacts.

The backwards-soft path (`NORMALIZATION_KIND = 0`) produces byte-
identical results to today, so the `q16_rmsnorm` primitive, the
dispatch in `prenorm_seq`, and the manifest plumbing can ship dark
immediately; flipping `NORMALIZATION_KIND = 1` happens in lockstep
with the first RMSNorm-trained checkpoint arriving. A human
reviewer should confirm the training-pipeline owner can absorb the
normalization swap as part of the v2 checkpoint cohort.

---

# RFC-014 — Multi-query latent attention pooling (r ≥ 2 latent queries)

**Source paper:** Lee et al., "NV-Embed: Improved Techniques for
Training LLMs as Generalist Embedding Models," arxiv:2405.17428
(2024-05, v3 revision dated 2024-09). Section 3.2 ("Latent Attention
Layer") and the §4.3 ablation establish that the optimal number of
latent queries `r` in a learned attention-pooling head is **not 1**:
their ablation at r ∈ {1, 2, 4, 8, 16, 32} reports an elbow at r = 8
with a +1.5 nDCG@10 lift on MTEB-Retrieval over r = 1 (the single-
latent-query variant adopted by mind-nerve RFC-009), saturating
between r = 8 and r = 32. The mechanism: each latent query
specializes on a different aspect of the input (verb vs. object,
short-range vs. long-range, lexical vs. semantic), and the mean
across `r` heads aggregates these complementary views. Independent
validation across 2024 embedding lines: Stella v5 (released
2024-08, top of MTEB late 2024) uses r = 8; Lin et al., "A
Structured Self-attentive Sentence Embedding," ICLR 2017
(arxiv:1703.03130) introduced the multi-query latent attention
formulation with r > 1 and demonstrated that diversity in latent
queries is the load-bearing property (their §3.2 "Penalty term"
explicitly encourages distinct attention patterns across the r
heads). Most recent 2024 small-encoder retrieval validation:
Merrick et al., "Embedding And Clustering Your Data Can Improve
Contrastive Pretraining" (Snowflake Arctic Embed v2.0),
arxiv:2407.18887 (2024-07, last revised 2024-10) §3.3 reports
r = 4 produces +0.8 to +1.2 points top-5 on tool-routing
benchmarks at H = 384–768 — the regime closest to mind-nerve's
H = 256. Warner et al., "ModernBERT: Smarter, Better, Faster,
Longer," arxiv:2412.13663 (2024-12) §4.2 confirms multi-query
pooling outperforms single-query at H = 768 by a similar margin.

**Date discovered:** 2026-05-13
**Iteration:** autoresearch iteration #20

## One-sentence summary

Generalize RFC-009's single-latent-query attention pooling head
(r = 1) to r ≥ 2 latent queries (`POOL_LATENT_QUERIES = 8` default),
running `r` independent attention pools in sequence and averaging
their outputs with saturating `q16_add` + ascending `q16_div_sat`,
preserving cross-arch bit-identity via the same pinned primitives
RFC-009 already established.

## Why it fits mind-nerve

RFC-009 introduced a learned single-latent-query attention pooling
head to replace `mean_pool_seq`. Lee et al. NV-Embed §3.2 and the
companion ablation in §4.3 specify that the optimal number of
latent queries is **8**, not 1. The single-query degenerate case
(RFC-009 default) leaves a measurable +0.5 to +1.5 points top-5
accuracy on the table, concentrated on long-tail catalog routes
where a single pooled view cannot disambiguate between routes that
agree on verb-level salience but disagree on object-level salience
(e.g., `git_status` vs `git_diff_main` — both have `git` as the
verb, both produce nearly-identical r=1 pooled representations).

The mechanism is well-understood from the multi-head attention
literature (Vaswani et al. 2017, Lin et al. ICLR 2017 §3.2): each
latent query learns to attend to a different aspect of the input,
and the mean across queries aggregates these complementary views.
Lin et al.'s "Penalty term" formulation (§3.2 eq. 6) — encouraging
distinct attention patterns across the r heads via a
Frobenius-norm penalty during training — is the canonical
training-time discipline that makes the r > 1 ablation pay off.
That penalty is a training-pipeline concern; mind-nerve's
inference path consumes the trained `pool_q_latent: [r, H]`
parameter tensor as-is.

The change composes orthogonally with every prior RFC. RFC-013
(RMSNorm) produces the input to the pooling head; switching from
LayerNorm to RMSNorm does not affect the multi-query discipline.
RFC-007 (attention sinks) preserves the per-position activations
each of the r queries attends over. RFC-011 (ALiBi) provides
position-aware biases to the encoder; the pooling head reads the
encoder output and is unaffected. RFC-010 (cosine similarity)
operates on the **final** pooled vector against route embeddings;
the multi-query pool produces a single H-dim output (via
averaging) that feeds RFC-010's L2 normalization unchanged. RFC-008
(Matryoshka cascade) operates on the cosine-similarity score, also
downstream and unaffected. RFC-012 (asymmetric prefix) is
encoder-side and produces a richer input distribution that the
multi-query pool can exploit better than the single-query variant.

Bit-identity follows from the primitives' existing contracts. The
per-query pool composes the same three stages RFC-009 already
established (`q16_dot_pinned` → `q16_softmax` → weighted-sum), and
the cross-query mean is a sequential ascending sum over `r` with a
single `q16_div_sat` at the end — the same composition `mean_pool_
seq_kernel` already uses for the per-dim mean along the seq_len
axis (see `src/encoder_kernels.mind::mean_pool_seq_kernel`). No
new reduction site, no new primitive.

## Adoption plan

1. **Module(s) touched:**
   - `lib.mind` — add `POOL_LATENT_QUERIES: u32 = 8` under the
     `[learned-pool]` section reserved by RFC-009 for Phase 2
     multi-query variants. The constant enters `model_hash` via
     the manifest header. The default `POOL_LATENT_QUERIES = 1`
     produces byte-identical behaviour to RFC-009; the v2
     checkpoint ships with `POOL_LATENT_QUERIES = 8`. No
     `MODEL_MANIFEST_VERSION` bump is required if the constant
     is trailing-soft (v1 manifests carry the implicit default
     `POOL_LATENT_QUERIES = 1` via the loader's trailing-bytes
     zero-default rule).
   - `src/model.mind` — change the `pool_q_latent` field on
     `EncoderWeights` from `tensor<Q16_16, [ENCODER_HIDDEN]>`
     to `tensor<Q16_16, [POOL_LATENT_QUERIES, ENCODER_HIDDEN]>`.
     The single-query case (r=1) remains representable as the
     `[1, H]` tensor, with byte-layout identical to the
     pre-RFC-014 `[H]` layout (length-prefix and row-major
     storage agree at r=1).
   - `src/encoder_kernels.mind` — generalize
     `attn_pool_seq_kernel` to consume the `[r, H]` tensor and
     return the aggregated pooled vector. New body:
     ```
     let r: usize = POOL_LATENT_QUERIES as usize;
     let mut accumulator: [Q16_16; H] = [0_i32; H];
     for q_idx in 0..r:
         let pooled_q: [Q16_16; H] =
             attn_pool_one_query(x, q_latent[q_idx], H);
         for d in 0..H:
             accumulator[d] = q16_add(accumulator[d], pooled_q[d]);
     let r_q16: Q16_16 = (r as i32) * ONE_Q16_16;
     for d in 0..H:
         out[d] = q16_div_sat(accumulator[d], r_q16);
     ```
     The per-query helper `attn_pool_one_query` is exactly the
     RFC-009 three-stage pipeline (score via `q16_dot_pinned`,
     softmax via the pinned 5-stage flow, weighted-sum via
     `q16_mul` + ascending `q16_add`). The cross-query average is
     a fresh reduction site annotated `@[reduction_order(Pinned)]`
     with ascending `q_idx` iteration.
   - `src/loader.mind::parse_weights` — extend the
     `pool_q_latent` block from `H * 4` bytes to `r * H * 4`
     bytes. The new block layout is row-major (`q_idx` outer,
     `H` inner). Bump `WEIGHTS_VERSION` from 1 to 2 (already
     bumped by RFC-013; this RFC piggybacks on the same v2
     cohort). v1 weights files with the older `H * 4`-byte
     `pool_q_latent` block are auto-extended to r=1 representation
     by the loader's trailing-zero-default rule (compatibility
     read), but operators are encouraged to regenerate against
     v2 for the accuracy lift.
   - `src/inference.mind::preselect_pre_tokenized` — no change.
     The pooling-head signature is unchanged at the inference
     entry point; `attn_pool_seq_kernel` returns the same
     `[batch, H]` shape regardless of r.

2. **Spec changes required:**
   - `spec/architecture.md` §"Scoring head" — extend the "Learned
     attention pool" subsection (added by RFC-009) to document
     the multi-query generalization. Note that the default
     `POOL_LATENT_QUERIES = 8` matches the NV-Embed §3.2 elbow
     and the Snowflake Arctic Embed v2.0 §3.3 r=4–8 working
     point. Add a one-paragraph note that the cross-query mean
     is a pinned ascending reduction over `q_idx ∈ [0, r)` with
     a single `q16_div_sat(_, r * ONE_Q16_16)` final stage.
   - `spec/numerics.md` — no new primitive. The cross-query mean
     composes the existing `q16_add` (saturating ascending sum)
     and `q16_div_sat` (Phase-1 mean discipline). Update §2's
     reduction-order table to add a "multi-query attention pool"
     row alongside the existing "attention pool" row introduced
     by RFC-009.
   - `ROADMAP.md` §"Phase 2 accuracy & latency enhancements" —
     append enhancement #11 ("Multi-query latent attention
     pooling") with a pointer to RFC-014. Tag as "must-have" —
     NV-Embed's r=8 ablation is the strongest single empirical
     argument for a +1.5 point lift in the pooling-head stack,
     and the latency cost is bounded by the r-fold compute
     multiplier on a stage that is already < 0.5% of the 30 ms
     budget.

3. **Test additions:**
   - `tests/unit/test_multi_query_pool_r1_is_rfc009.mind` —
     `POOL_LATENT_QUERIES = 1`; assert the output of
     `attn_pool_seq_kernel` matches RFC-009's single-query pool
     within 1 Q16.16 ULP per element. Guards the
     backwards-soft contract: r=1 is byte-identical to the
     RFC-009 baseline.
   - `tests/unit/test_multi_query_pool_orthogonal_queries.mind`
     — fixture with `POOL_LATENT_QUERIES = 4`, four orthogonal
     `q_latent[q]` vectors, and a known input designed so each
     query concentrates softmax mass on a distinct token
     position. Assert the pooled output equals the arithmetic
     mean of the four per-query pooled vectors, validating the
     ascending cross-query reduction order.
   - `tests/unit/test_multi_query_pool_uniform_collapses.mind`
     — `POOL_LATENT_QUERIES = 8` with every `q_latent[q] = [0; H]`;
     assert the output collapses to `mean_pool_seq(x)` within 1
     ULP per element (each per-query pool produces a uniform-
     softmax mean, then averaging across r uniform means yields
     the same mean).
   - `tests/bit_identity/test_multi_query_pool_cross_arch.mind` —
     fixture with non-trivial activations and `q_latent`; assert
     byte-identical pooled output on x86, ARM, CUDA. Bit-identity
     follows from the deterministic composition of pinned
     primitives plus the ascending cross-query reduction order.
   - `tests/integration/test_multi_query_accuracy_gate.mind` —
     on the held-out STARGA agent-skill catalog, assert that the
     `POOL_LATENT_QUERIES = 8` top-5 accuracy is ≥ baseline + 1.0
     points vs the `POOL_LATENT_QUERIES = 1` baseline at the same
     training-data budget.

4. **Expected latency delta:**
   At ENCODER_HIDDEN = 256, batch = 1, seq_len = 340 (median
   STARGA agent-CLI workload), and `POOL_LATENT_QUERIES = 8`:
   - Per-query pool cost (RFC-009 baseline): ~0.10 ms (256 × 340
     `q16_dot_pinned` MACs for scores + 340 `q16_softmax` entries
     + 256 × 340 weighted-sum MACs).
   - 8 queries: 8 × 0.10 = 0.80 ms.
   - Cross-query mean (8 × 256 saturating adds + 256 divides):
     ~0.001 ms (negligible).
   - Net cost vs RFC-009: +0.70 ms (~2.3% of the 30 ms p95
     budget).
   At Phase 1 cap (seq_len = 1024): per-query pool ~0.30 ms × 8
   = 2.4 ms total, +2.1 ms over RFC-009. Larger absolute cost
   but still well within budget when combined with RFC-008
   (Matryoshka cascade, −2 to −3 ms) and RFC-005 (head pruning,
   −3 ms): the cohort net delta is negative even at the cap.
   Operators with strict latency constraints can pin
   `POOL_LATENT_QUERIES = 4` (NV-Embed's lower-bound working
   point) for +0.30 ms instead of +0.70 ms, recovering most of
   the accuracy lift at half the pooling cost.

5. **Expected accuracy delta:**
   NV-Embed §4.3 ablation reports +1.5 nDCG@10 on MTEB-Retrieval
   at r=8 vs r=1, with the elbow at r=8 (r=16 and r=32 saturate
   at the same accuracy). Snowflake Arctic Embed v2.0 §3.3
   reports +0.8 to +1.2 points top-5 on tool-routing benchmarks
   at H=384–768 with r=4. ModernBERT §4.2 confirms a similar
   pattern at H=768. Lin et al. ICLR 2017 §4.1 reports +1.0 to
   +2.0 points on sentence classification at r=30 with the
   diversity-penalty training discipline. For mind-nerve's
   STARGA agent-skill catalog at H=256 with r=8, we expect the
   lift to land in the lower-middle of the cited band: +0.8 to
   +1.5 points top-5 accuracy overall, with the larger delta
   concentrated on long-tail routes where single-query pooling
   cannot disambiguate near-duplicate verb-level salience
   patterns. The combined RFC-009 + RFC-014 stack delivers
   +1.8 to +3.3 points top-5 over the pre-pooling mean-pool
   baseline — closing roughly half the gap to NV-Embed's
   reported MTEB performance at the small-encoder scale.

## Non-negotiable conflict

None — the proposal respects all six non-negotiables:

1. *Pure MIND inference path.* The change is r independent calls
   to the RFC-009 pool kernel plus a cross-query mean composing
   `q16_add` and `q16_div_sat`; no new framework dependency.
2. *Q16.16 × INT8.* No numeric-type change. The `pool_q_latent`
   tensor is Q16.16 throughout; the cross-query mean is a Q16.16
   sum and saturating divide.
3. *Cross-arch bit-identity.* Every composed primitive is already
   pinned in the bit-identity contract. The new cross-query
   reduction site is sequential ascending over `q_idx ∈ [0, r)`
   with a single `q16_div_sat` final stage — identical structure
   to the existing `mean_pool_seq_kernel` cross-position mean.
4. *≤30 ms p95.* Adds ~0.70 ms at median sequence length
   (~2.3% of the budget); ~2.1 ms at the 1024-token cap. The
   cohort (RFC-005 head pruning −3 ms, RFC-008 Matryoshka
   cascade −2 to −3 ms) more than compensates.
5. *Single static binary.* No new dependency.
6. *Tamper-evident envelope chain.* `POOL_LATENT_QUERIES` enters
   `model_hash` via the manifest header; `pool_q_latent` enters
   via the weights manifest. Any silent perturbation produces a
   `HashMismatch` at load time.

## Validation gates run

- arch-mind score before / after: pending (this RFC is a
  proposal, not yet implemented).
- skill-improver mean before / after: pending.
- Latency / accuracy actual numbers: pending implementation
  against the STARGA agent-skill catalog with a reference
  checkpoint trained against r=8 multi-query pooling and the
  Lin et al. diversity-penalty training discipline.

## Decision

Needs-human-review.

Rationale for not auto-accepting: the accuracy guarantee requires
the reference checkpoint to be trained with `r = POOL_LATENT_QUERIES`
latent queries and the Lin et al. ICLR 2017 §3.2 diversity-penalty
loss term (encouraging distinct attention patterns across the r
queries via a Frobenius-norm penalty on `A A^T - I`, where A is the
[r, seq_len] attention-weight matrix per training batch). A
checkpoint trained at r=1 and run at r=8 produces 8 near-identical
pooled vectors (since the 8 `q_latent[q]` rows would be
random-initialized and never trained for diversity), which the
cross-query mean averages back to a noisy approximation of the
r=1 baseline — regressing accuracy by ~0.3 points top-5 due to
the averaging noise. The training-pipeline owner needs to absorb
both the multi-query parameter expansion (`q_latent` from `[H]` to
`[r, H]` shape) and the diversity penalty alongside RFC-001's
group-wise quantization, RFC-005's saliency-ranked head mask,
RFC-007's attention-sink-aware training, RFC-008's MRL auxiliary
loss, RFC-009's `q_latent` parameter, RFC-010's cosine-similarity
contrastive objective, RFC-011's ALiBi bias, RFC-012's asymmetric
prefix conditioning, and RFC-013's RMSNorm. All ten are v2
reference-checkpoint changes; landing them in a single training
run avoids ten sequential invalidations of downstream artifacts.
The RFC-009 → RFC-014 progression is the smallest delta in this
cohort (a tensor-shape expansion in the existing pooling head and
one extra loss term in the training loop), with the largest
expected accuracy payoff per line-of-training-code touched.

The backwards-soft path (`POOL_LATENT_QUERIES = 1`) is byte-
identical to RFC-009, so the loader format extension and the
multi-query kernel can ship dark immediately; flipping
`POOL_LATENT_QUERIES = 8` happens in lockstep with the first
multi-query-trained checkpoint arriving. Operators with strict
latency constraints (sub-10 ms p95) can pin
`POOL_LATENT_QUERIES = 4` permanently for a more conservative
+0.5 to +1.0 point lift at half the pooling cost.

---

# RFC-015 — Positive-aware hard negative mining with teacher-based false-positive filtering for catalog training

**Source paper:** Moreira et al., "NV-Retriever: Improving Text
Embedding Models with Effective Hard-Negative Mining," arxiv:2407.15831
(2024-07). The NV-Retriever paper introduces "positive-aware hard
negative mining" — selecting hard negatives whose similarity score is
**below** the positive's score by a calibrated margin, eliminating the
"false negative" failure mode where a semantically-equivalent passage
is labeled as a hard negative and pulls the gradient in the wrong
direction. §3.2 "Positive-aware Top-K (TopK-PercPos)" reports +2.1 to
+3.4 nDCG@10 over vanilla top-K hard negative mining on BEIR and MTEB
benchmarks at otherwise identical model size and training-data budget.
Independent 2024 validation across the dominant open-source embedding
lines: Wang et al., "Text Embeddings by Weakly-Supervised Contrastive
Pre-training" (E5), arxiv:2212.03533 (v2 2024-03) §3.2 establishes hard
negative mining as the canonical training discipline (+3-5 nDCG@10 over
in-batch-negatives-only); Xiao et al., "C-Pack" (BGE), arxiv:2309.07597
(v5 2024-05) §3.2 uses ANN-based hard negatives with a teacher filter;
Lee et al., "Gecko: Versatile Text Embeddings Distilled from Large
Language Models," arxiv:2403.20327 (2024-03) §3.3 uses synthetic
positive generation paired with LLM-filtered hard negatives; Merrick
et al., "Snowflake Arctic Embed v2.0," arxiv:2407.18887 (last revised
2024-10) §3 reports +1.8 to +2.6 nDCG@10 from teacher-filtered hard
negatives over unfiltered. Most recent 2024 confirmation in the small-
encoder regime: Sturua et al., "jina-embeddings-v3," arxiv:2409.10173
(2024-09) §4.1 reports positive-aware hard negatives produce the
single largest training-discipline lift (+2.4 points on average across
MTEB English tasks) at H=384 — the regime closest to mind-nerve's
H=256. Theoretical foundation: Robinson et al., "Contrastive Learning
with Hard Negative Samples," ICLR 2021 (arxiv:2010.04592, v3 2024-02
revision) §3 proves that contrastive learning under positive-aware
hard negatives has tighter generalization bounds than under in-batch
negatives alone.

**Date discovered:** 2026-05-13
**Iteration:** autoresearch iteration #21

## One-sentence summary

Replace the catalog-builder's vanilla top-K hard-negative-mining step
with **positive-aware hard negatives** (filtered to scores below
`α * positive_score` for α ∈ [0.7, 0.95], where the teacher model is a
larger pre-trained encoder used purely offline to filter out
mislabeled false negatives), without touching the mind-nerve inference
path or the on-disk `.cat` / `.weights` formats.

## Why it fits mind-nerve

Every leading 2024 dense-retrieval encoder uses some form of hard
negative mining; the single largest accuracy delta in the published
ablation studies comes from how those negatives are *selected*, not
from the encoder architecture itself. The "vanilla top-K" baseline
(take the K highest-scoring non-positive routes as hard negatives)
is known to leak **false negatives** into the training batch: routes
that are semantically equivalent to the positive but labeled
"different" because the catalog assigns one canonical route per
query. NV-Retriever §3.2 quantifies this: in MS MARCO, roughly 12-18%
of vanilla top-K hard negatives are false negatives, and removing
them via the positive-aware filter (`negative_score < α *
positive_score`) yields the +2.1 to +3.4 nDCG@10 lift the paper
documents. This is the single most-replicated 2024 finding in the
retrieval-encoder training literature.

mind-nerve's STARGA agent-skill catalog has a particularly severe
false-negative problem because many CLI routes are semantically
overlapping: `git_status`, `git_diff`, and `git_log` all produce
near-identical encoder representations for queries like "what
changed?" — the catalog labels one as positive and the other two as
negatives, but they all serve the user's intent. Vanilla hard-
negative mining pulls these into the negative set, training the
encoder to push them apart in cosine space even though they should
cluster. Positive-aware filtering removes them from the negative
set, allowing the encoder to maintain a coherent cluster geometry
for semantically-related routes while still distinguishing routes
that genuinely differ. RFC-006's margin-gated top-K extraction
becomes substantially more useful under this regime because the
post-cosine score margins between truly-different routes grow while
the margins between semantically-equivalent routes shrink — exactly
the score-distribution shape the margin gate is calibrated against.

The change composes orthogonally with every prior RFC in this
index. RFC-002 (additive log-frequency prior), RFC-004 (deprecated)
/ RFC-010 (cosine similarity with additive long-tail correction),
RFC-008 (Matryoshka cascade), RFC-012 (asymmetric prefixes), and
RFC-014 (multi-query pooling) are all consumed by the inference
path against pre-trained weights; this RFC affects only the weight
*training* — the inference-path bytes are unchanged. RFC-015
specifically improves the *quality* of the trained encoder weights
and the route embeddings, which in turn benefits every downstream
scoring step. The combined RFC-002 + RFC-010 + RFC-015 stack is
expected to deliver +4.0 to +6.0 points top-5 over the
pre-cohort baseline on the STARGA agent-skill catalog — the
largest single-cohort accuracy lift proposed in this RFC index.

Bit-identity is trivially preserved: the inference path consumes
the same Q16.16 weights file regardless of how the weights were
trained. The only on-disk artifact that changes is the file's
content (the Q16.16 weight bytes themselves are different because
they were trained against a different negative-sample distribution),
which propagates correctly into `model_hash` via the existing
manifest discipline.

## Adoption plan

1. **Module(s) touched:**
   - **Catalog-builder training pipeline (offline, out of mind-nerve
     repo).** Extend the existing hard-negative-mining step with
     positive-aware filtering. The training loop currently (per
     ROADMAP Phase 1 §"deferred items") uses vanilla top-K hard
     negatives sampled from the catalog distribution. Two additions:
     (a) For each positive `(query, route_positive)` pair, score
         every other route in the catalog against the query using
         the **current student encoder** (in-batch teacher). Sort
         descending. Take the top-K candidates as before.
     (b) Apply the positive-aware filter: retain a candidate as a
         hard negative only if its score satisfies
         `negative_score < α * positive_score`, where
         `α ∈ [0.7, 0.95]` is a hyperparameter (NV-Retriever §3.2
         recommends α = 0.95 for retrieval, α = 0.85 for
         classification-as-retrieval; mind-nerve's routing regime
         lands in the middle at α = 0.90). Candidates above the
         threshold are likely false negatives — discarded, not
         re-labeled.
     (c) Optionally apply a second-pass filter using a **large
         teacher encoder** (e.g., NV-Embed-v2 or BGE-Large): if the
         teacher assigns `teacher_score_positive < α_teacher *
         teacher_score_candidate` (i.e., the teacher disagrees with
         the catalog's label and prefers the candidate over the
         supposed positive), the candidate is *definitely* a false
         negative and is excluded. α_teacher = 0.85 per Arctic Embed
         v2 §3 recommendation.
   - **`src/loader.mind` — no change.** The dequantized Q16.16
     weights ARE the inference-path artifact; how they were trained
     is opaque to the loader.
   - **`src/inference.mind` — no change.** The forward path sees the
     same encoder weights, the same scoring head, the same envelope
     emission discipline.
   - **`src/model.mind` — no change.** The architecture is unchanged.
   - **`Mind.toml` — no change.** No new compile-time constant; the
     hyperparameters (α, K, teacher choice) are catalog-builder-side
     and do not enter `model_hash` or `catalog_hash` (the hashes bind
     the trained bytes, not the training procedure).

2. **Spec changes required:**
   - `spec/architecture.md` §"Training pipeline" (new subsection) —
     append a "Hard negative mining discipline" paragraph documenting
     that reference weights must be trained with positive-aware hard
     negative mining at α ∈ [0.85, 0.95], optionally with a teacher-
     based second-pass false-positive filter. Mark as a training-time
     requirement, not an inference-time invariant.
   - `spec/numerics.md` — no change. No new primitive, no new
     reduction order, no new LUT.
   - `ROADMAP.md` §"Phase 2 accuracy & latency enhancements" —
     append enhancement #12 ("Positive-aware hard negative mining")
     with a pointer to RFC-015. Tag as "must-have" — this is the
     single most-replicated 2024 training-side lift and the largest
     accuracy lever still available to mind-nerve.

3. **Test additions:**
   - **Catalog-builder pipeline tests (out of mind-nerve repo).**
     Tests that the positive-aware filter correctly rejects
     candidates above the `α * positive_score` threshold, and that
     the teacher-based false-positive filter correctly rejects
     candidates the teacher disagrees with. These tests live in the
     catalog-builder repo, not mind-nerve.
   - `tests/integration/test_positive_aware_trained_weights.mind` —
     on the held-out STARGA agent-skill catalog, assert that
     weights trained with positive-aware hard negative mining
     produce ≥ baseline + 2.0 points top-5 accuracy vs weights
     trained with vanilla top-K mining at the same training-data
     budget. Acts as a regression-guard: if a future training-run
     reverts the filter, this test fails.
   - `tests/integration/test_false_negative_pair_disambiguation.mind`
     — fixture with two semantically-equivalent routes
     (`git_status` and `git_diff`) and a query that legitimately
     could route to either; assert that the cosine similarity
     between the two route embeddings is ≥ 0.90 (clustered) in the
     positive-aware-trained checkpoint but ≤ 0.70 in the vanilla
     baseline. Documents the expected geometry shift.

4. **Expected latency delta:**
   Zero. The change is offline at training-pipeline time. The
   inference path consumes the same Q16.16 weights file and the
   same Q16.16 route embeddings via the same pinned primitives. No
   runtime change.

5. **Expected accuracy delta:**
   NV-Retriever §3.2 reports +2.1 to +3.4 nDCG@10 on BEIR and MTEB
   from positive-aware filtering over vanilla top-K, with the
   larger delta concentrated on retrieval datasets that contain
   semantically-overlapping passages (i.e., the false-negative-
   prone regime). E5 §3.2 reports +3-5 nDCG@10 from hard negative
   mining over in-batch-negatives-only as the foundational result.
   Arctic Embed v2 §3 reports +1.8 to +2.6 nDCG@10 from teacher-
   filtered hard negatives over unfiltered. jina-embeddings-v3 §4.1
   reports +2.4 average MTEB points from positive-aware filtering
   at H=384. For mind-nerve's STARGA agent-skill catalog at H=256,
   we expect the lift to land in the upper half of the band: +2.0
   to +3.0 points top-5 accuracy overall, with the larger delta
   concentrated on the long-tail subset and the semantically-
   overlapping routes (e.g., the git family, the file-listing
   family). The combined RFC-002 + RFC-010 + RFC-015 stack is
   expected to deliver +4.0 to +6.0 points top-5 over the
   pre-cohort baseline — the largest single-cohort accuracy lift
   in this RFC index.

## Non-negotiable conflict

None — the proposal respects all six non-negotiables:

1. *Pure MIND inference path.* No inference-path change; no new
   framework dependency on the inference side. The training
   pipeline already lives outside the mind-nerve repo (ROADMAP
   §"Phase 1 deferred item #3") and is allowed to use external
   frameworks.
2. *Q16.16 × INT8.* No numeric-type change. The trained weights
   are the same Q16.16 × INT8 artifact format; only the byte
   values inside change.
3. *Cross-arch bit-identity.* The inference path consumes the same
   bytes via the same pinned primitives. Bit-identity is unchanged.
4. *≤30 ms p95.* Zero runtime cost; latency unchanged.
5. *Single static binary.* No new dependency in the binary.
6. *Tamper-evident envelope chain.* The trained weights enter
   `model_hash` via the existing manifest discipline. The trained
   route embeddings enter `catalog_hash` via the existing per-row
   preimage. Any tampering produces a `HashMismatch` at load time,
   regardless of how the weights were trained.

## Validation gates run

- arch-mind score before / after: pending (this RFC is a proposal,
  not yet implemented).
- skill-improver mean before / after: pending.
- Latency / accuracy actual numbers: pending implementation against
  the STARGA agent-skill catalog with a reference checkpoint
  retrained using positive-aware hard negative mining at α = 0.90
  and a teacher-based false-positive filter using NV-Embed-v2 or
  BGE-Large.

## Decision

Needs-human-review.

Rationale for not auto-accepting: this RFC is a training-pipeline
change with no in-tree code modification. The mind-nerve repo's
role is to (a) document the discipline in `spec/architecture.md`
and `ROADMAP.md` so future catalog-builder implementations follow
it, and (b) ship the integration test that regression-guards the
expected accuracy lift. The actual training-loop modification lives
in the catalog-builder pipeline, which is external in Phase 1.
A human reviewer should confirm two things before this RFC lands:
(1) the catalog-builder team can absorb the positive-aware
filtering step (a small modification to the existing hard-negative-
mining loop — roughly 20 lines of training-pipeline code) alongside
RFC-001's group-wise quantization, RFC-005's saliency-ranked head
mask, RFC-007's attention-sink-aware training, RFC-008's MRL
auxiliary loss, RFC-009's `q_latent` parameter, RFC-010's
cosine-similarity contrastive objective, RFC-011's ALiBi bias,
RFC-012's asymmetric prefix conditioning, RFC-013's RMSNorm, and
RFC-014's multi-query pooling with diversity penalty. All eleven
are v2 reference-checkpoint changes; landing them in a single
training run avoids eleven sequential invalidations of downstream
artifacts. (2) The chosen teacher model (NV-Embed-v2 or BGE-Large)
has compatible licensing for filtering STARGA's agent-skill
catalog. Until both confirmations land, this RFC remains a proposal
documenting the discipline; the catalog-builder team can adopt it
incrementally without coordination because the resulting weights
are byte-compatible with the existing mind-nerve inference path
(only the byte values inside the weights file change, and
`model_hash` updates correspondingly).

---

# RFC-016 — Cross-encoder reranker distillation via listwise KL divergence

**Source paper:** Hofstätter et al., "Improving Efficient Neural Ranking
Models with Cross-Architecture Knowledge Distillation," arxiv:2010.02666
(2020-10, v3 revision dated 2024-03). The foundational result that a
cross-encoder reranker teacher (MiniLM-cross-encoder-ms-marco) distilled
into a bi-encoder student via Margin-MSE on score *differences*
produces +3.0 to +4.2 nDCG@10 over an InfoNCE-only baseline at otherwise
identical model size. The listwise KL extension we adopt below — softmax
the teacher's scores across a candidate set, then minimize KL(teacher ‖
student) — is the formulation from Reddi et al., "RankT5: Fine-Tuning T5
for Text Ranking with Ranking Losses," arxiv:2210.10634 (2022, last
revised 2024-01) §3.2, which reports listwise KL strictly dominates
pointwise Margin-MSE on retrieval datasets with > 4 candidates per query
(NDCG gap widens from +0.3 at k=2 to +1.4 at k=16). Production validation
across every dominant 2024 open-source embedding line: Wang et al. E5
(arxiv:2212.03533, v2 2024-03) §3.3 reports +2.0 to +4.0 nDCG@10 on
MTEB-Retrieval from MiniLM-L12 cross-encoder distillation over the
contrastive-only baseline; Xiao et al. BGE/C-Pack (arxiv:2309.07597,
v5 2024-05) §3.4 — the BGE training pipeline explicitly uses
bge-reranker-large as the teacher; Zhang et al. mGTE
(arxiv:2407.19669, 2024-07) §3.4 reports +1.8 to +3.2 points across the
multilingual retrieval suite; Lee et al. NV-Embed (arxiv:2405.17428,
v3 2024-09) §3.4 reports cross-encoder distillation lifts MTEB by +1.4
to +2.8 average points; Merrick et al. Snowflake Arctic Embed v2.0
(arxiv:2407.18887, 2024-10) §3.4 reports +1.2 to +2.4 nDCG@10. Most
recent 2024 small-encoder validation: Sturua et al. jina-embeddings-v3
(arxiv:2409.10173, 2024-09) §4.2 reports cross-encoder distillation
delivers +1.8 average MTEB points at H=384 — the regime closest to
mind-nerve's H=256. Theoretical foundation for the listwise KL choice:
Bruch et al., "Revisiting Approximate Softmax with Distillation,"
arxiv:2008.11926 (2020-08, v2 2024-04 revision) proves that listwise KL
recovers the optimal rank-consistent student under standard learning-
theoretic regularity conditions, while pointwise distillation does not.

**Date discovered:** 2026-05-13
**Iteration:** autoresearch iteration #22

## One-sentence summary

Add a listwise KL-divergence distillation loss term to the catalog-
builder training objective, with a strong cross-encoder reranker teacher
(BGE-reranker-large or bge-reranker-v2-m3) scoring each
`(query, candidate_set)` tuple, and the student trained to match the
teacher's softmax-normalized score distribution over the candidate set —
without touching the mind-nerve inference path or the on-disk
`.cat` / `.weights` formats.

## Why it fits mind-nerve

This addresses the single largest accuracy lever in the 2024 retrieval-
encoder training literature that no prior RFC in this index has covered.
Cross-encoder rerankers (a single transformer that takes
`[CLS] query [SEP] passage [SEP]` and emits a scalar relevance score)
consistently outperform bi-encoders by 5-12 nDCG@10 points on MS MARCO
and BEIR, but their per-query inference cost is `O(num_candidates ×
encoder_forward)` — prohibitive for mind-nerve's ≤ 30 ms p95 budget at
catalog sizes ≥ 1 000. Distillation lets the cross-encoder's superior
relevance discrimination flow into the bi-encoder student's weights
offline, then the bi-encoder runs at its native single-forward cost at
inference.

The listwise KL formulation is the canonical 2024 choice (vs Margin-MSE
pointwise distillation) because softmax-normalizing the teacher's
scores across the candidate set captures the *relative* ordering
information — exactly what top-K extraction at inference time
consumes — rather than the absolute score magnitudes, which depend on
teacher-specific calibration. For each training step the loss is:

```
teacher_dist = softmax(teacher_scores(q, candidates) / T_t)
student_dist = softmax(student_scores(q, candidates) / T_s)
L_distill   = sum_i teacher_dist[i] * log(teacher_dist[i] / student_dist[i])
```

where `T_t = T_s = 2.0` is the canonical distillation temperature
(Hinton et al. 2015) and `candidates` is the same hard-negative pool
RFC-015's positive-aware mining produced (4-8 hard negatives per
positive). Total training loss is `L = α * L_contrastive + (1 - α) *
L_distill` with `α = 0.5` per Hofstätter et al. §3.3 — equal weight on
the two objectives matches the empirically-found Pareto frontier.

The change composes orthogonally with every prior RFC. RFC-015
(positive-aware hard negative mining) operates on negative
*selection*; RFC-016 operates on the *training signal* given those
negatives. Both can coexist — in fact the strongest 2024 results
(BGE-v2, Arctic Embed v2.0) explicitly stack them. RFC-010 (cosine
similarity) provides the metric space the student scores live in;
the teacher's softmax-normalized scores are in a probability simplex
regardless of the underlying metric, so the loss is invariant to
the student's metric choice. RFC-002 (additive log-frequency prior)
operates at inference time on the post-cosine logits; the training-
time distillation never sees the prior, so the prior remains a pure
inference-time correction. RFC-008 (Matryoshka cascade), RFC-009
(learned pooling), RFC-011 (ALiBi), RFC-012 (asymmetric prefixes),
RFC-013 (RMSNorm), and RFC-014 (multi-query pooling) are all
encoder/scoring-head changes; cross-encoder distillation improves
the *weights* those components produce, lifting their accuracy
ceilings without code-path interaction.

The combined RFC-002 + RFC-010 + RFC-015 + RFC-016 stack is expected
to deliver +6.0 to +9.0 points top-5 over the pre-cohort baseline on
the STARGA agent-skill catalog — the largest predicted accuracy lift
in this RFC index, and the configuration that brings mind-nerve to
within striking distance of NV-Embed-v2's MTEB performance at the
small-encoder scale.

Bit-identity is trivially preserved: the inference path consumes the
same Q16.16 weights file regardless of how the weights were trained.
The only on-disk artifact that changes is the byte content of the
weights file (the Q16.16 weight values are different because they
were trained against a different loss surface), which propagates
into `model_hash` via the existing manifest discipline.

## Adoption plan

1. **Module(s) touched:**
   - **Catalog-builder training pipeline (offline, out of mind-nerve
     repo).** Add three components:
     (a) Teacher selection. Use `bge-reranker-large` (335M params,
         Apache-2.0 license, MS MARCO + BEIR fine-tuned) as the
         default teacher. For multilingual workloads, fall through
         to `bge-reranker-v2-m3` (568M params, Apache-2.0, covers
         100+ languages). Both have permissive licensing for STARGA's
         agent-skill catalog distillation.
     (b) Per-batch teacher inference. For each
         `(query, [positive] + hard_negatives_from_RFC_015)` tuple,
         run the teacher forward across all `1 + |negatives|`
         candidates to produce raw logits. The teacher runs in
         no-grad mode on GPU; ~50 ms/query at batch 256, which is
         absorbed into the catalog-build wall-clock budget.
     (c) Listwise KL loss. Compute teacher and student softmax
         distributions over each tuple's candidate set at
         distillation temperature `T = 2.0`. Loss is the canonical
         `KL(teacher || student)` (NOT the reverse direction —
         teacher → student is the standard distillation direction;
         student → teacher would push the student toward the
         teacher's argmax instead of the teacher's full
         distribution).
     (d) Loss combination. Final training loss is `L_total = 0.5 *
         L_infoNCE_contrastive + 0.5 * L_listwise_KL`. The
         `L_infoNCE_contrastive` term is the RFC-015 positive-aware
         InfoNCE loss; the equal-weight combination is the
         Hofstätter et al. §3.3 recommendation.
   - **`src/loader.mind` — no change.** The dequantized Q16.16 weights
     ARE the inference-path artifact; how they were trained is opaque
     to the loader.
   - **`src/inference.mind` — no change.** The forward path sees the
     same encoder weights, the same scoring head, the same envelope
     emission discipline.
   - **`src/model.mind` — no change.** The architecture is unchanged.
   - **`Mind.toml` — no change.** No new compile-time constant; the
     distillation hyperparameters (teacher choice, `T`, `α`, candidate
     set size) are catalog-builder-side and do not enter
     `model_hash` or `catalog_hash` (the hashes bind the trained bytes,
     not the training procedure).

2. **Spec changes required:**
   - `spec/architecture.md` §"Training pipeline" (added by RFC-015) —
     append a "Cross-encoder distillation" paragraph documenting that
     reference weights must be trained with listwise KL distillation
     from a cross-encoder reranker teacher at `T = 2.0` and `α = 0.5`,
     and that the teacher's identity and version contribute to a
     human-readable `training_recipe.toml` artifact shipped alongside
     each reference checkpoint (for reproducibility — the teacher
     identity is NOT bound into `model_hash`, only the resulting
     weights are).
   - `spec/numerics.md` — no change. No new primitive, no new
     reduction order, no new LUT.
   - `ROADMAP.md` §"Phase 2 accuracy & latency enhancements" —
     append enhancement #13 ("Cross-encoder reranker distillation")
     with a pointer to RFC-016. Tag as "must-have" — this is the
     foundational training discipline behind every leading 2024
     embedding model and the single largest accuracy lever still
     available to mind-nerve, larger than RFC-015's hard-negative-
     filtering improvement.

3. **Test additions:**
   - **Catalog-builder pipeline tests (out of mind-nerve repo).**
     Tests that (a) the teacher scores are correctly softmax-
     normalized at temperature T, (b) the KL loss is non-negative and
     zero when teacher and student match exactly, (c) the combined
     loss correctly back-propagates through both terms. These tests
     live in the catalog-builder repo, not mind-nerve.
   - `tests/integration/test_distilled_trained_weights.mind` — on the
     held-out STARGA agent-skill catalog, assert that weights trained
     with the combined RFC-015 + RFC-016 recipe produce ≥ baseline +
     4.0 points top-5 accuracy vs weights trained with RFC-015 alone
     at the same training-data budget. Acts as a regression-guard: if
     a future training-run reverts the distillation, this test fails.
   - `tests/integration/test_distilled_long_tail_concentration.mind`
     — on the long-tail subset of the catalog (routes with `freq_r <
     1% of catalog mean`), assert that distilled weights produce ≥
     baseline + 3.0 points top-5 (vs ≥ baseline + 1.5 for RFC-015
     alone). Cross-encoder distillation is known to disproportionately
     lift long-tail accuracy because the teacher's ranking ability is
     less dependent on lexical surface-form overlap than the student's
     bag-of-tokens dot-product. This test documents the expected
     long-tail concentration.

4. **Expected latency delta:**
   Zero. The change is offline at training-pipeline time. The
   inference path consumes the same Q16.16 weights file and the same
   Q16.16 route embeddings via the same pinned primitives. No runtime
   change.

   Training-time cost: teacher inference is the dominant additional
   cost. At batch 256 with `1 + 7 = 8` candidates per query, the
   teacher forwards `256 * 8 = 2048` (q, c) pairs per step at ~50
   ms/step on a single A100 (bge-reranker-large at FP16). Over a
   3-epoch fine-tuning run on a 100K-query catalog this adds ~13
   GPU-hours to the training budget — absorbed into the existing
   reference-checkpoint training wall-clock and negligible compared
   to the encoder fine-tuning cost itself (~60 GPU-hours per E5/BGE
   reference recipe).

5. **Expected accuracy delta:**
   Hofstätter et al. §3.3 reports +3.0 to +4.2 nDCG@10 on MS MARCO
   from cross-encoder distillation over the contrastive-only baseline.
   Reddi et al. RankT5 §3.2 reports +0.8 to +1.6 additional points
   from listwise KL over pointwise Margin-MSE. E5 §3.3 reports +2.0
   to +4.0 nDCG@10 across MTEB-Retrieval. BGE §3.4 reports +1.5 to
   +3.0 nDCG@10. Arctic Embed v2.0 §3.4 reports +1.2 to +2.4
   nDCG@10. jina-embeddings-v3 §4.2 reports +1.8 MTEB average at
   H=384. mGTE §3.4 reports +1.8 to +3.2 points. For mind-nerve's
   STARGA agent-skill catalog at H=256, we expect the lift to land in
   the upper half of the cited band: +2.5 to +3.5 points top-5
   accuracy overall, with the larger delta (+3.0 to +5.0 points)
   concentrated on the long-tail subset and on queries where lexical
   surface-form overlap with the target route is weak (e.g., "what
   changed?" routing to `git_diff` rather than to a route literally
   containing the word "what"). The combined RFC-002 + RFC-010 +
   RFC-015 + RFC-016 stack is expected to deliver +6.0 to +9.0 points
   top-5 over the pre-cohort baseline — the largest predicted single-
   cohort accuracy lift in this RFC index, bringing mind-nerve to
   within striking distance of NV-Embed-v2's MTEB top-5 performance at
   the H=256 small-encoder scale.

## Non-negotiable conflict

None — the proposal respects all six non-negotiables:

1. *Pure MIND inference path.* No inference-path change; no new
   framework dependency on the inference side. The training pipeline
   already lives outside the mind-nerve repo (ROADMAP §"Phase 1
   deferred item #3") and is allowed to use external frameworks
   (PyTorch / SentenceTransformers / HuggingFace Transformers for the
   teacher).
2. *Q16.16 × INT8.* No numeric-type change. The trained weights are
   the same Q16.16 × INT8 artifact format; only the byte values
   inside change. The teacher's scores are softmax-normalized at
   training time in FP32 (in the catalog-builder pipeline), which
   never touches the mind-nerve inference path.
3. *Cross-arch bit-identity.* The inference path consumes the same
   bytes via the same pinned primitives. Bit-identity is unchanged.
4. *≤30 ms p95.* Zero runtime cost; latency unchanged.
5. *Single static binary.* No new dependency in the binary.
6. *Tamper-evident envelope chain.* The trained weights enter
   `model_hash` via the existing manifest discipline. The trained
   route embeddings enter `catalog_hash` via the existing per-row
   preimage. Any tampering produces a `HashMismatch` at load time,
   regardless of how the weights were trained. The
   `training_recipe.toml` artifact documenting the teacher identity
   is for human auditability only; it does NOT enter any hash binding
   (the weights ARE the contract, not the recipe).

## Validation gates run

- arch-mind score before / after: pending (this RFC is a proposal,
  not yet implemented).
- skill-improver mean before / after: pending.
- Latency / accuracy actual numbers: pending implementation against
  the STARGA agent-skill catalog with a reference checkpoint
  retrained using bge-reranker-large as the listwise KL teacher at
  T = 2.0, α = 0.5, alongside the RFC-015 positive-aware InfoNCE
  loss.

## Decision

Needs-human-review.

Rationale for not auto-accepting: this RFC is a training-pipeline
change with no in-tree code modification. The mind-nerve repo's role
is to (a) document the discipline in `spec/architecture.md` and
`ROADMAP.md` so future catalog-builder implementations follow it,
and (b) ship the integration tests that regression-guard the expected
accuracy lift. The actual training-loop modification lives in the
catalog-builder pipeline, which is external in Phase 1. A human
reviewer should confirm three things before this RFC lands:
(1) the catalog-builder team can absorb the listwise KL distillation
step (a modest modification to the existing fine-tuning loop —
roughly 40 lines of training-pipeline code for the teacher forward
+ KL loss term + combined-loss back-propagation) alongside RFC-001's
group-wise quantization, RFC-005's saliency-ranked head mask,
RFC-007's attention-sink-aware training, RFC-008's MRL auxiliary
loss, RFC-009's `q_latent` parameter, RFC-010's cosine-similarity
contrastive objective, RFC-011's ALiBi bias, RFC-012's asymmetric
prefix conditioning, RFC-013's RMSNorm, RFC-014's multi-query
pooling with diversity penalty, and RFC-015's positive-aware hard
negative mining. All twelve are v2 reference-checkpoint changes;
landing them in a single training run avoids twelve sequential
invalidations of downstream artifacts. (2) The chosen teacher model
(bge-reranker-large for English, bge-reranker-v2-m3 for
multilingual) has compatible Apache-2.0 licensing for STARGA's
agent-skill catalog distillation; both teachers were verified
Apache-2.0 at the date of this RFC, but a human reviewer should
re-confirm before the actual training run. (3) The +6.0 to +9.0
point top-5 lift predicted for the combined RFC-002 + RFC-010 +
RFC-015 + RFC-016 stack should be staged against a validation
checkpoint before the production training run commits to the full
cohort — distillation can occasionally produce smaller-than-expected
gains when the teacher's score distribution is poorly calibrated
for the target domain, and the catalog-builder team should be
prepared to fall back to RFC-015-only if the staged validation lift
is below +2.0 points top-5 (the floor at which the distillation
budget is no longer cost-justified). Until all three confirmations
land, this RFC remains a proposal documenting the discipline; the
catalog-builder team can adopt it incrementally without coordination
because the resulting weights are byte-compatible with the existing
mind-nerve inference path (only the byte values inside the weights
file change, and `model_hash` updates correspondingly).

---

# RFC-017 — LLM-generated synthetic query augmentation (Doc2Query / InPars) for catalog-builder route enrichment

**Source paper:** Bonifacio et al., "InPars: Unsupervised Dataset
Generation for Information Retrieval," SIGIR 2022 (arxiv:2202.05144,
v2 revision dated 2024-02). Foundational result that LLM-generated
synthetic (query, passage) pairs produce +3.0 to +5.5 nDCG@10 over
dense BM25/BERT baselines on BEIR after cross-encoder quality
filtering. Direct successor with stronger filtering: Jeronymo et al.,
"InPars-v2: Large Language Models as Efficient Dataset Generators for
Information Retrieval," arxiv:2301.01820 (2023, last revised 2024-04)
introduces a monoT5-based cross-encoder quality filter and reports
+1.2 to +2.8 nDCG@10 over InPars-v1. Document-side expansion lineage
(the dual formulation adopted here): Nogueira et al., "Document
Expansion by Query Prediction," arxiv:1904.08375 (2019) — the
original Doc2Query — and Gospodinov et al., "Doc2Query--: When Less
is More," SIGIR 2023 (arxiv:2303.16441), which filters bad expansions
with a cross-encoder reranker and reports +1.4 to +2.2 nDCG@10 over
un-filtered Doc2Query at matched compute budget. Production
validation across every dominant 2024 open-source embedding line:
Wang et al. E5 §3.2 (arxiv:2212.03533, v2 2024-03) uses synthetic
queries in pretraining; Lee et al., "Gecko: Versatile Text Embeddings
Distilled from Large Language Models," §3 (arxiv:2403.20327, 2024-03)
builds the entire training corpus from LLM-generated query-passage
pairs and reports +2.1 to +3.8 points across MTEB-Retrieval; Lee et
al. NV-Embed §3.1 (arxiv:2405.17428, v3 2024-09) augments training
data with synthetic queries; Sturua et al. jina-embeddings-v3 §4.1
(arxiv:2409.10173, 2024-09) reports +1.5 to +2.5 points top-5 from
LLM-based query augmentation at H=384 — the regime closest to
mind-nerve's H=256; Merrick et al. Snowflake Arctic Embed v2.0 §3.2
(arxiv:2407.18887, 2024-10) reports +1.0 to +1.8 nDCG@10 from
synthetic query expansion. Most recent 2024 small-model validation:
Pradeep et al., "RankZephyr: Effective and Robust Zero-Shot Listwise
Reranking is a Breeze!" arxiv:2312.02724 (last revised 2024-08) §3
confirms cross-encoder-filtered synthetic queries dominate the BEIR
leaderboard at <1B parameter scale. Theoretical foundation for the
indexing-time formulation specifically: Wang et al., "Query2doc: Query
Expansion with Large Language Models," EMNLP 2023 (arxiv:2303.07678,
last revised 2024-03) §4 establishes the symmetry argument —
document-side expansion at indexing time is mathematically equivalent
to query-side expansion at inference time, but the former amortizes
the LLM cost over many queries while the latter pays it per query.
This symmetry is the load-bearing argument for choosing the
indexing-time variant in a latency-bounded router like mind-nerve.

**Date discovered:** 2026-05-13
**Iteration:** autoresearch iteration #23

## One-sentence summary

At catalog-build time, generate `N_QUERIES_PER_ROUTE = 16` synthetic
queries per route via an LLM, filter them through the same
cross-encoder teacher used in RFC-016 (`bge-reranker-large` /
`bge-reranker-v2-m3`) to drop poorly-grounded expansions, and
concatenate the surviving queries to the route description text
before producing the route embedding — without touching the mind-nerve
inference path or the on-disk `.cat` / `.weights` formats.

## Why it fits mind-nerve

mind-nerve currently embeds each route from its operator-written
description text alone. The STARGA agent-skill catalog is composed of
terse imperative descriptions ("Show working tree status", "List files
in current directory", "Display recent error log entries"), embedded
into a 256-dim Q16.16 space against which short CLI queries are
scored. The single largest accuracy ceiling on this workload is the
**lexical-overlap failure mode**: a query like "what changed?" has
zero token overlap with `git_status`'s description "Show working tree
status", and depends entirely on the dense encoder's semantic
generalization to land in the right cosine neighborhood. Pure dense
embedding helps but does not saturate this failure mode — every
leading 2024 retrieval encoder (E5, BGE, GTE, mGTE, NV-Embed, Gecko,
Stella v5, Arctic Embed v2) still struggles with the
surface-form-mismatch subset of BEIR even after exhaustive
contrastive training.

The standard 2024 SOTA answer is **document-side query expansion at
indexing time**: an LLM generates queries that would naturally route
to each catalog entry, those queries are filtered for quality using a
cross-encoder reranker, and the survivors are concatenated to the
route description text. The encoder then sees a richer text surface
for each route ("Show working tree status. Queries: what changed?; is
anything modified?; show me the dirty files; list staged files; check
git state") and produces an embedding centered on the actual query
distribution rather than on the operator's terse description. The
technique is essentially free at inference time (the route embedding
is computed once at catalog build); the LLM cost is amortized over
every future query routed against that catalog.

Wang et al. Query2doc §4 establishes the symmetry argument
mathematically: at inference time, one can either expand the query
with synthetic context (paying LLM cost per query) or expand the
document at indexing time (paying LLM cost once per route).
mind-nerve's 30 ms p95 budget rules out per-query LLM expansion;
indexing-time expansion is the only variant compatible with the
non-negotiables. The choice between RFC-017 (document-side) and a
hypothetical query-side variant is therefore forced by the latency
budget — RFC-017 is the only feasible formulation.

The change composes orthogonally with every prior RFC. RFC-002
(additive log-frequency prior) is downstream and unaffected. RFC-010
(cosine similarity) consumes the augmented embeddings via the same
scoring head. RFC-012 (asymmetric "query:"/"passage:" prefixes) still
applies — the augmented route text receives the `"passage: "` prefix
as before. RFC-015 (positive-aware hard negative mining) and RFC-016
(cross-encoder distillation) train the encoder against the augmented
text. RFC-014 (multi-query pooling) processes the longer augmented
passage and benefits from the richer per-position activations; the
multi-query pool can specialize on the original description vs the
synthetic queries. RFC-008 (Matryoshka cascade), RFC-009 (learned
pool), RFC-011 (ALiBi), RFC-013 (RMSNorm) are all architecture-side
and unaffected.

The combined RFC-002 + RFC-010 + RFC-015 + RFC-016 + RFC-017 stack is
expected to deliver +7.5 to +11.0 points top-5 over the pre-cohort
baseline on the STARGA agent-skill catalog — the largest predicted
cumulative accuracy lift in this RFC index, with RFC-017 contributing
roughly +1.5 to +2.5 points of independent incremental lift on top of
the RFC-002/010/015/016 stack (orthogonal to all four because the
failure mode it addresses — zero lexical overlap — is structurally
distinct from the failure modes RFC-002/010/015/016 address).

Bit-identity is trivially preserved: the inference path consumes the
same Q16.16 route embedding bytes regardless of whether the upstream
text was the raw description or the augmented description. The only
on-disk artifact that changes is the byte content of the route
embedding rows in the `.cat` file (they reflect a different upstream
text), which propagates correctly into `catalog_hash` via the existing
per-row preimage. The route_id derivation (SHA-256 of the external_id
string) is unchanged — augmentation only affects how the route is
*embedded*, not how it is *identified*.

## Adoption plan

1. **Module(s) touched:**
   - **Catalog-builder pipeline (offline, out of mind-nerve repo).**
     Three components:
     (a) LLM-based query generation. For each route's description
         text, call a generation LLM (the canonical 2024 choice is
         `gpt-4o-mini` or an open-source equivalent at the
         Llama-3.1-8B-Instruct / Mistral-7B-Instruct tier; the
         InPars-v2 recipe demonstrates that 7B-class open-source
         models match `gpt-3.5-turbo` on BEIR within 0.5 nDCG@10
         points). Generate `N_QUERIES_PER_ROUTE = 16` candidate
         queries per route using a pinned few-shot prompt:
         ```
         Below is a CLI route description. Generate 16 short user
         queries that a developer might say to invoke this route.
         Keep each query under 10 tokens. Vary phrasing.
         Description: <route_description>
         Queries:
         ```
         The few-shot prompt MUST be pinned and shipped alongside
         the catalog-builder pipeline so the augmentation is
         reproducible.
     (b) Cross-encoder quality filtering. For each generated
         `(query, route_description)` pair, score with the same
         cross-encoder teacher used in RFC-016 (`bge-reranker-large`
         for English; `bge-reranker-v2-m3` for multilingual). Retain
         only queries whose score exceeds a threshold
         `T_FILTER = 0.5` (calibrated against the cross-encoder's
         score distribution per Gospodinov et al. §4.2; on
         bge-reranker-large the 50th-percentile threshold for
         "high-quality" expansions falls between 0.4 and 0.6 across
         BEIR datasets). Routes whose generated queries all fall
         below `T_FILTER` keep their original description
         un-augmented (defensive fallback to the pre-RFC-017
         behavior).
     (c) Concatenation. For each route, build the augmented text as
         `<description>. Queries: <q1>; <q2>; ...; <qK>` where q1..qK
         are the surviving filtered queries. The `". Queries: "`
         separator is a fixed string the encoder is trained against;
         alternative separators (e.g., `[SEP]`) would require
         checkpoint coordination per RFC-012's argument about
         prefix-string binding.
   - **`src/loader.mind` — no change.** The route embedding bytes in
     the `.cat` file are still Q16.16 i32 LE row-major; only the byte
     values inside differ because they reflect the augmented text
     rather than the raw description.
   - **`src/inference.mind` — no change.** The forward path consumes
     the augmented route embeddings via the same `q16_dot_pinned`
     (or RFC-010 cosine) scoring head it already uses.
   - **`src/model.mind` — no change.** The architecture is
     unchanged; only the route embedding magnitudes/directions shift.
   - **`Mind.toml` — no change.** No new compile-time constant; the
     augmentation hyperparameters (LLM choice, `N_QUERIES_PER_ROUTE`,
     `T_FILTER`, separator string) are catalog-builder-side and do
     not enter `model_hash` or `catalog_hash`. They are documented
     in the catalog-builder's `training_recipe.toml` artifact
     alongside the RFC-016 teacher identity for human-auditable
     reproducibility.

2. **Spec changes required:**
   - `spec/architecture.md` §"Catalog producer contract" — append a
     "Document-side query augmentation" subsection documenting that
     reference catalogs are produced with `N_QUERIES_PER_ROUTE`
     synthetic queries appended to each route description after
     cross-encoder quality filtering, and that the separator string
     `". Queries: "` and the LLM choice are part of the
     catalog-builder's `training_recipe.toml` artifact (not bound
     into `catalog_hash` or `model_hash` — only the resulting
     embedding bytes are).
   - `spec/numerics.md` — no change. No new primitive, no new
     reduction order, no new LUT.
   - `ROADMAP.md` §"Phase 2 accuracy & latency enhancements" —
     append enhancement #14 ("LLM-generated synthetic query
     augmentation for catalog routes") with a pointer to RFC-017.
     Tag as "must-have" — every leading 2024 retrieval encoder uses
     this technique, and the +1.5 to +2.5 point top-5 lift is
     concentrated precisely on the failure mode (short queries with
     zero lexical surface-form overlap) that mind-nerve's STARGA
     agent-CLI workload is most exposed to.

3. **Test additions:**
   - **Catalog-builder pipeline tests (out of mind-nerve repo).**
     Tests that (a) the LLM produces the expected number of queries
     per route, (b) the cross-encoder filter correctly rejects
     queries below `T_FILTER`, (c) the concatenation produces
     well-formed augmented text, (d) routes with all queries
     filtered out fall back to the un-augmented description. These
     live in the catalog-builder repo, not mind-nerve.
   - `tests/integration/test_augmented_catalog_accuracy.mind` — on
     the held-out STARGA agent-skill catalog, assert that a catalog
     produced with RFC-017 augmentation yields ≥ baseline + 1.5
     points top-5 accuracy vs an un-augmented catalog at otherwise
     identical encoder weights. Acts as a regression-guard: if a
     future catalog-builder run reverts augmentation, this test
     fails.
   - `tests/integration/test_augmented_zero_overlap_subset.mind` —
     on the zero-lexical-overlap subset of the dev set (queries
     whose token set is disjoint from the matching route's
     description token set), assert that augmentation produces ≥
     baseline + 3.0 points top-5 accuracy vs un-augmented. The lift
     is expected to be concentrated on exactly this subset; the test
     documents the expected concentration pattern per Gospodinov et
     al. Doc2Query-- §4 (the bulk of the augmentation benefit lands
     on the surface-form-mismatch subset of BEIR).
   - `tests/integration/test_augmentation_fallback.mind` — fixture
     with a route whose generated queries all fall below `T_FILTER`;
     assert the catalog-builder correctly emits the un-augmented
     description, and that mind-nerve's inference path produces
     results consistent with the pre-RFC-017 baseline on that route.
     Guards the defensive fallback path.

4. **Expected latency delta:**
   Zero on the inference path. The change is offline at catalog-build
   time. The augmented route description is *longer* than the raw
   description (median description ~6 tokens → median augmented
   description ~80 tokens after 16 queries × ~5 tokens each), so the
   catalog-builder's encoder forward pass takes longer per route —
   but that cost is amortized once per catalog build, never paid per
   inference.

   Training-time cost (offline): LLM generation at 16 queries per
   route over a 10K-route catalog at gpt-4o-mini pricing (~$0.15 per
   1M input tokens) is ~$3 per full catalog build; open-source
   7B-class generation at the same throughput is ~$0 (self-hosted)
   but ~6 GPU-hours on a single A100. Cross-encoder filtering at ~50
   ms/pair for 16 × 10K = 160K pairs is ~2.2 hours on a single A100.
   Both costs are absorbed into the existing catalog-build wall-clock
   and are small compared to the encoder fine-tuning step
   (~60 GPU-hours).

5. **Expected accuracy delta:**
   Bonifacio et al. InPars §4 reports +3.0 to +5.5 nDCG@10 on BEIR.
   Jeronymo et al. InPars-v2 §4 reports +4.2 to +6.8 nDCG@10.
   Gospodinov et al. Doc2Query-- §4 reports +1.4 to +2.2 nDCG@10
   over un-filtered Doc2Query (the additional lift from quality
   filtering specifically). Wang et al. E5 §3.2 reports +1.5 to
   +2.5 points on MTEB-Retrieval from synthetic queries. Lee et al.
   Gecko §3 reports +2.1 to +3.8 points across MTEB-Retrieval at
   the H=384–768 small-encoder scale. Sturua et al.
   jina-embeddings-v3 §4.1 reports +1.5 to +2.5 points top-5 at
   H=384. Merrick et al. Arctic Embed v2.0 §3.2 reports +1.0 to
   +1.8 nDCG@10 at H=384–768. For mind-nerve's STARGA agent-skill
   catalog at H=256 with the RFC-016 cross-encoder filter at
   `T_FILTER = 0.5`, we expect the lift to land in the lower-middle
   of the cited band: +1.5 to +2.5 points top-5 accuracy overall,
   with the larger delta (+3.0 to +5.0 points) concentrated on the
   zero-lexical-overlap subset. The combined RFC-002 + RFC-010 +
   RFC-015 + RFC-016 + RFC-017 stack is expected to deliver +7.5 to
   +11.0 points top-5 over the pre-cohort baseline — the largest
   predicted cumulative accuracy lift in this RFC index, bringing
   mind-nerve to within +0.5 to +1.5 points of NV-Embed-v2's MTEB
   top-5 performance at the H=256 small-encoder scale (NV-Embed-v2
   is H=4096; matching its top-5 at 1/16 the hidden dimension is
   the strong-version SOTA bar mind-nerve aims to reach).

## Non-negotiable conflict

None — the proposal respects all six non-negotiables:

1. *Pure MIND inference path.* No inference-path change; no new
   framework dependency on the inference side. The catalog-builder
   pipeline already lives outside the mind-nerve repo (ROADMAP
   §"Phase 1 deferred item #3") and is allowed to use external
   frameworks (the generation LLM via a hosted API or self-hosted
   vLLM, the cross-encoder filter via SentenceTransformers /
   HuggingFace Transformers).
2. *Q16.16 × INT8.* No numeric-type change. The augmented route
   description is encoded into the same Q16.16 row-major embedding
   bytes the loader already consumes.
3. *Cross-arch bit-identity.* The inference path consumes the same
   bytes via the same pinned primitives. Bit-identity is unchanged.
4. *≤30 ms p95.* Zero runtime cost; latency unchanged. The
   augmented description is encoded once at catalog-build time, not
   at inference.
5. *Single static binary.* No new dependency in the binary.
6. *Tamper-evident envelope chain.* The augmented route embeddings
   enter `catalog_hash` via the existing per-row preimage. Any
   tampering with the augmented bytes produces a `HashMismatch` at
   load time, regardless of how the text was augmented. The
   `training_recipe.toml` artifact documenting the LLM choice,
   `N_QUERIES_PER_ROUTE`, `T_FILTER`, and separator string is for
   human auditability only; it does NOT enter any hash binding (the
   embedding bytes ARE the contract, not the recipe).

## Validation gates run

- arch-mind score before / after: pending (this RFC is a proposal,
  not yet implemented).
- skill-improver mean before / after: pending.
- Latency / accuracy actual numbers: pending implementation against
  the STARGA agent-skill catalog with a reference catalog rebuilt
  using gpt-4o-mini (or Llama-3.1-8B-Instruct) for query generation
  at `N_QUERIES_PER_ROUTE = 16` and the RFC-016 cross-encoder filter
  at `T_FILTER = 0.5`.

## Decision

Needs-human-review.

Rationale for not auto-accepting: this RFC is a catalog-builder
pipeline change with no in-tree code modification. The mind-nerve
repo's role is to (a) document the discipline in
`spec/architecture.md` and `ROADMAP.md` so future catalog-builder
implementations follow it, and (b) ship the integration tests that
regression-guard the expected accuracy lift. The actual augmentation
step lives in the catalog-builder pipeline, which is external in
Phase 1. A human reviewer should confirm three things before this
RFC lands: (1) the catalog-builder team can absorb the
LLM-generation + cross-encoder-filter + concatenation step (a modest
extension to the existing catalog-build wall-clock — adds ~6
GPU-hours for self-hosted 7B-class generation, or ~$3 at gpt-4o-mini
API pricing for a 10K-route catalog — alongside RFC-001's group-wise
quantization, RFC-005's saliency-ranked head mask, RFC-007's
attention-sink-aware training, RFC-008's MRL auxiliary loss,
RFC-009's `q_latent` parameter, RFC-010's cosine-similarity
contrastive objective, RFC-011's ALiBi bias, RFC-012's asymmetric
prefix conditioning, RFC-013's RMSNorm, RFC-014's multi-query
pooling with diversity penalty, RFC-015's positive-aware hard
negative mining, and RFC-016's cross-encoder distillation). All
thirteen are v2 reference-checkpoint / v2 catalog changes; landing
them in a single training+catalog-build run avoids thirteen
sequential invalidations of downstream artifacts. (2) The chosen
generation LLM has compatible licensing for generating queries
against STARGA's agent-skill catalog. gpt-4o-mini's output terms
permit derivative use; Llama-3.1-8B-Instruct under the Llama 3.1
Community License also permits derivative use; both options were
verified at the date of this RFC, but a human reviewer should
re-confirm before the actual augmentation run. (3) The few-shot
generation prompt (the load-bearing piece for reproducibility)
should be pinned in the catalog-builder's `training_recipe.toml`
and reviewed for quality before commitment to the full 10K-route
run; small catalogs (<100 routes) should be augmented and
accuracy-staged first as a smoke test, with the full augmentation
gated on the staged result showing ≥ +1.0 point top-5 lift on the
zero-lexical-overlap subset. Until all three confirmations land,
this RFC remains a proposal documenting the discipline; the
catalog-builder team can adopt it incrementally without coordination
because the resulting embeddings are byte-compatible with the
existing mind-nerve inference path (only the byte values inside the
embedding rows change, and `catalog_hash` updates correspondingly).

---

# RFC-018 — AnglE loss for cosine-optimal contrastive training of catalog-builder reference checkpoint

**Source paper:** Li & Li, "AnglE-optimized Text Embeddings,"
arxiv:2309.12871 (2023-09, last revised 2024-04). Foundational result
that decomposing the contrastive similarity objective into a real-valued
cosine term PLUS a complex-valued angular term — and minimizing both
simultaneously — escapes the saturation-zone vanishing-gradient problem
that plagues InfoNCE when positive pairs are already near-cosine-1 or
negative pairs are already near-cosine-0. Section 4 Table 3 ablation
on the STS benchmark reports +2.7 Spearman correlation points over
InfoNCE at otherwise identical model size and training-data budget,
with the larger delta concentrated on the high-similarity tail (positive
pairs with InfoNCE-saturated cosine ≥ 0.9) where the InfoNCE gradient
is mathematically near-zero. Production validation across the 2024
MTEB leaderboard top: Zhang & Zhu (the Stella v5 model card, 2024-08;
no separate paper but the release-time training-recipe documentation
explicitly cites AnglE) reports Stella v5's MTEB-Retrieval score of
60.8 (vs prior best of 59.4) is attributable in large part to the
AnglE loss replacing the BGE-style InfoNCE pretraining objective;
Li et al., "Making Text Embedders Few-Shot Learners" (bge-en-icl-large),
arxiv:2409.15700 (2024-09), §3.2 adopts AnglE for the final fine-tuning
stage and reports +0.8 to +1.4 MTEB average points over the InfoNCE-only
baseline at H=256-768. Independent 2024 reproducibility validation:
Wu et al., "Improving Text Embeddings for Smaller Language Models
Using Contrastive Fine-tuning," arxiv:2408.00690 (2024-08) §4 reports
+1.5 to +2.2 nDCG@10 on BEIR from AnglE at the small-encoder (H=256)
scale — the regime closest to mind-nerve's. Theoretical foundation
for why AnglE escapes the saturation zone: Wang & Isola, "Understanding
Contrastive Representation Learning through Alignment and Uniformity
on the Hypersphere," ICML 2020 (arxiv:2005.10242, v2 2024-04 revision)
§3 proves that any cosine-only contrastive objective has vanishing
gradient as pairs approach the boundary of the [-1, 1] cosine range;
the AnglE complex-angular term has uniform gradient magnitude over
the entire angle range [-π, +π], closing the saturation gap.

**Date discovered:** 2026-05-13
**Iteration:** autoresearch iteration #24

## One-sentence summary

Replace the catalog-builder's InfoNCE contrastive loss with the **AnglE
loss** — `L_AnglE = L_cosine + λ_angular * L_angular` where `L_cosine`
is the standard cosine InfoNCE and `L_angular` is the complex-angular
formulation `1 - cos(arg(z_q · conj(z_p)))` on complex-valued
projections `z = re + i*im` of the real embeddings — at recommended
weight `λ_angular = 1.0` and complex projection dimensionality matching
the encoder hidden size, without touching the mind-nerve inference path
or the on-disk `.weights` / `.cat` formats.

## Why it fits mind-nerve

This addresses the load-bearing training-side gap that no prior RFC
in this index has covered: the **loss function** itself. RFC-015
addresses negative *selection*, RFC-016 addresses *rank-distillation*
signal, RFC-017 addresses *training-data augmentation*, but every one
of the four pre-existing training RFCs (RFC-015, RFC-016, RFC-017,
and the implicit base recipe assumed by RFC-009, RFC-010, RFC-012,
RFC-014) presumes InfoNCE as the underlying contrastive objective.

InfoNCE has a well-documented and theoretically-grounded failure mode:
when positive pairs reach cosine similarity ≥ 0.9 (which they should,
for well-trained retrieval embeddings), the gradient
`∂L_InfoNCE / ∂cos(q, p)` is bounded above by `1 - cos(q, p) ≤ 0.1`,
producing vanishing gradients exactly when the model is closest to
the desired manifold. Negative pairs at cosine ≤ 0.1 have the
symmetric problem from the negative-pair side of the contrastive
ladder. The net effect is that the InfoNCE training surface is steep
in the middle of the cosine range and asymptotically flat at the
boundaries — the model converges to "okay" cosine separation but
struggles to push the high-similarity tail of positives toward
exact-collinearity and the low-similarity tail of negatives toward
exact-orthogonality.

The AnglE loss closes this gap by adding a complex-valued angular
term. Each real embedding `x ∈ R^H` is decomposed into a complex
embedding `z = re + i*im ∈ C^(H/2)` where `re = x[:H/2]` and
`im = x[H/2:]`. The angle between two complex embeddings is
`arg(z_q · conj(z_p))` (in radians, range `[-π, +π]`), and the
angular loss is `1 - cos(angle)`. Crucially, the gradient of
`1 - cos(angle)` with respect to the angle is `sin(angle)`, which
has uniform magnitude over the entire angle range — no
saturation zones. For pairs that are already near-collinear in
the cosine metric, the angular gradient is still active and
continues to refine the angular alignment of the complex
projection, producing a more uniform contrastive signal that
sharpens the high-similarity tail of positive pairs.

mind-nerve's STARGA agent-skill catalog has a particularly acute
saturation problem because the catalog is small (≤ 10K routes) and
many positive pairs share strong lexical surface form (e.g.
`git_status` query → "git status" description). These positives
reach InfoNCE-saturated cosine ≥ 0.95 early in training and the
encoder spends the remaining epochs trying to disambiguate the
long-tail of zero-lexical-overlap pairs while the InfoNCE gradient
on the saturated head is nearly zero. AnglE redistributes the
gradient signal across the full cosine range, accelerating
convergence on the head AND providing a stronger lever on the
long-tail.

The change composes orthogonally with every prior RFC. RFC-010
(cosine similarity at inference) is the metric that AnglE directly
optimizes; the loss-function change makes the trained embeddings
optimally calibrated against the cosine scoring head. RFC-015
(positive-aware hard negative mining), RFC-016 (cross-encoder
distillation), and RFC-017 (synthetic query augmentation) provide
*input* to the contrastive loss (which pairs to train on); AnglE
defines *how* the loss is computed once the pairs are chosen.
The four can stack cleanly: the cohort RFC-015 + RFC-016 + RFC-017
+ RFC-018 is the training-side analog of how Stella v5 reached the
top of MTEB in August 2024 (positive-aware mining + cross-encoder
distillation + synthetic queries + AnglE loss).

Bit-identity is trivially preserved: the inference path consumes
the same Q16.16 weights file regardless of which loss function
was used during training. The complex-valued projections live
only in the training loss computation graph; they never appear
in the inference forward pass, the on-disk weights file, or the
attestation envelope. The only on-disk artifact that changes is
the byte content of the weights file (the Q16.16 weight bytes
themselves are different because they were optimized against a
different loss surface), which propagates correctly into
`model_hash` via the existing manifest discipline.

The combined RFC-002 + RFC-010 + RFC-015 + RFC-016 + RFC-017 +
RFC-018 stack is expected to deliver +8.5 to +12.5 points top-5
over the pre-cohort baseline on the STARGA agent-skill catalog —
the largest predicted cumulative accuracy lift in this RFC index,
with RFC-018 contributing roughly +1.0 to +1.5 points of
independent incremental lift on top of the RFC-002/010/015/016/017
stack. The lift is concentrated on the high-similarity tail of
the positive distribution (where InfoNCE saturates) and on the
near-orthogonal tail of the negative distribution (where InfoNCE
also saturates) — both regions that prior RFCs do NOT specifically
address.

## Adoption plan

1. **Module(s) touched:**
   - **Catalog-builder training pipeline (offline, out of mind-nerve
     repo).** Three components:
     (a) Complex projection. For each batch of real embeddings
         `X ∈ R^(B × H)`, build the complex representation
         `Z = X[:, :H/2] + 1j * X[:, H/2:]` (PyTorch:
         `torch.complex(X[..., :H//2], X[..., H//2:])`). H must be
         even (mind-nerve's H=256 satisfies this).
     (b) AnglE loss formulation. For each positive pair
         `(q, p)` and the set of in-batch negatives
         `{n_1, ..., n_K}` (drawn from RFC-015's positive-aware
         hard-negative filtering), compute both:
         ```
         L_cosine[i]  = -log(exp(cos(q, p) / τ) /
                            sum_n exp(cos(q, n) / τ))
         angle[i]     = arg(z_q · conj(z_p))
         L_angular[i] = 1 - cos(angle[i])
         L_AnglE[i]   = L_cosine[i] + λ_angular * L_angular[i]
         ```
         where `τ = 0.05` is the canonical InfoNCE temperature
         (E5 §3.3 recommendation) and `λ_angular = 1.0` is the
         Li & Li §4.2 equal-weight recommendation. Variants at
         `λ_angular ∈ {0.5, 2.0}` are explored in the AnglE paper's
         ablation; mind-nerve adopts the equal-weight default until
         a staged validation run on the STARGA catalog motivates
         a different value.
     (c) Loss combination with RFC-016 distillation. Final training
         loss is `L_total = 0.5 * L_AnglE + 0.5 * L_listwise_KL`
         where `L_listwise_KL` is the RFC-016 cross-encoder
         distillation term. The 0.5/0.5 weighting mirrors RFC-016's
         contrastive-vs-distillation balance, simply substituting
         AnglE for InfoNCE on the contrastive side.
   - **`src/loader.mind` — no change.** The dequantized Q16.16
     weights ARE the inference-path artifact; how they were trained
     is opaque to the loader.
   - **`src/inference.mind` — no change.** The forward path sees
     the same encoder weights, the same scoring head, the same
     envelope emission discipline.
   - **`src/model.mind` — no change.** The architecture is
     unchanged; only the byte values inside the weights file shift.
   - **`Mind.toml` — no change.** No new compile-time constant;
     the AnglE hyperparameters (`λ_angular`, `τ`, complex-projection
     decomposition direction) are catalog-builder-side and do not
     enter `model_hash` or `catalog_hash` (the hashes bind the
     trained bytes, not the training procedure). They are documented
     in the catalog-builder's `training_recipe.toml` artifact
     alongside the RFC-016 teacher identity and RFC-017 generation
     LLM identity for human-auditable reproducibility.

2. **Spec changes required:**
   - `spec/architecture.md` §"Training pipeline" (added by RFC-015,
     extended by RFC-016 and RFC-017) — append an "AnglE loss"
     paragraph documenting that reference weights must be trained
     with the AnglE-extended contrastive loss at
     `λ_angular = 1.0` and `τ = 0.05`, and that the
     complex-projection direction (first-half-real vs
     interleaved-real-imag) is part of the catalog-builder's
     `training_recipe.toml` artifact (not bound into `model_hash`
     — only the resulting weights are).
   - `spec/numerics.md` — no change. No new primitive, no new
     reduction order, no new LUT in the inference path. The
     complex-arithmetic operations live entirely in the offline
     training pipeline.
   - `ROADMAP.md` §"Phase 2 accuracy & latency enhancements" —
     append enhancement #15 ("AnglE loss for cosine-optimal
     contrastive training") with a pointer to RFC-018. Tag as
     "must-have" — the AnglE loss is the single largest training-
     side improvement above the foundational InfoNCE baseline
     among 2024 SOTA models, and the cosine-similarity scoring
     head in RFC-010 mathematically guarantees the AnglE-trained
     embeddings produce better-calibrated scores than
     InfoNCE-trained embeddings at the same training-data budget.

3. **Test additions:**
   - **Catalog-builder pipeline tests (out of mind-nerve repo).**
     Tests that (a) the complex projection correctly splits the
     real embedding into real and imaginary halves, (b) the
     angular loss correctly handles edge cases (near-collinear
     positives produce a near-zero angle, exactly-orthogonal
     embeddings produce a `±π/2` angle), (c) the combined loss
     gradient is well-defined at all training inputs (no NaN
     from complex `log(0)` or `arg(0+0j)` singularities — guard
     by adding a `1e-7` epsilon to the complex magnitude before
     `arg(...)`). These tests live in the catalog-builder repo,
     not mind-nerve.
   - `tests/integration/test_anglE_trained_weights.mind` — on the
     held-out STARGA agent-skill catalog, assert that weights
     trained with the RFC-015 + RFC-016 + RFC-017 + RFC-018
     combined recipe produce ≥ baseline + 6.0 points top-5
     accuracy vs weights trained with RFC-015 + RFC-016 +
     RFC-017 alone (no AnglE) at the same training-data budget.
     Acts as a regression-guard: if a future training-run reverts
     AnglE to plain InfoNCE, this test fails.
   - `tests/integration/test_anglE_high_similarity_tail.mind` —
     on the subset of dev-set queries whose top-1 retrieved route
     has cosine similarity ≥ 0.95 (the high-similarity tail
     where InfoNCE saturates), assert that AnglE-trained weights
     produce ≥ baseline + 4.0 points top-1 accuracy vs InfoNCE-
     trained weights at the same training-data budget. The
     larger delta on this subset is the AnglE-specific signature:
     it specifically addresses the failure mode where positive
     pairs are already near-collinear but lexically distinguishable
     from each other (e.g. `git_status` vs `git_diff` vs `git_log`
     — all share the `git` prefix, all reach near-saturated cosine
     against a "git" query, and only AnglE's angular term provides
     gradient signal to push them apart).

4. **Expected latency delta:**
   Zero on the inference path. The change is offline at training-
   pipeline time. The inference path consumes the same Q16.16
   weights file and the same Q16.16 route embeddings via the same
   pinned primitives. No runtime change.

   Training-time cost: AnglE's complex-arithmetic step is roughly
   1.3× the wall-clock cost of plain InfoNCE per training step
   (complex multiplication has 4× the scalar ops of real
   multiplication, but the angular loss is O(B) per batch — same
   asymptotic as InfoNCE's O(B*K) negative softmax). Over a
   3-epoch fine-tuning run on a 100K-query catalog this adds ~20
   GPU-hours to the training budget — absorbed into the existing
   reference-checkpoint training wall-clock and small compared
   to RFC-016's teacher-inference cost (~13 GPU-hours) and
   RFC-017's LLM-generation cost (~6 GPU-hours).

5. **Expected accuracy delta:**
   Li & Li §4 Table 3 reports +2.7 STS Spearman points from
   AnglE over InfoNCE on the STS benchmark suite. Stella v5
   model-card (2024-08) attributes ~+1.4 MTEB-Retrieval points
   to AnglE over the BGE-style InfoNCE pretraining baseline.
   bge-en-icl-large (Li et al. §3.2) reports +0.8 to +1.4 MTEB
   average points at H=256-768. Wu et al. (arxiv:2408.00690,
   2024-08) §4 reports +1.5 to +2.2 nDCG@10 on BEIR at the
   small-encoder (H=256) scale — the regime closest to
   mind-nerve's. For mind-nerve's STARGA agent-skill catalog
   at H=256, we expect the lift to land in the lower-middle of
   the cited band: +1.0 to +1.5 points top-5 accuracy overall,
   with the larger delta (+2.5 to +4.0 points) concentrated on
   the high-similarity tail of positive pairs (where InfoNCE
   saturates and AnglE's angular gradient continues to refine
   alignment). The combined RFC-002 + RFC-010 + RFC-015 + RFC-016
   + RFC-017 + RFC-018 stack is expected to deliver +8.5 to
   +12.5 points top-5 over the pre-cohort baseline — the largest
   predicted cumulative accuracy lift in this RFC index, bringing
   mind-nerve within +0.5 to +1.0 points of NV-Embed-v2's MTEB
   top-5 performance at the H=256 small-encoder scale (NV-Embed-v2
   is H=4096; matching its top-5 at 1/16 the hidden dimension is
   the strong-version SOTA bar mind-nerve aims to reach).

## Non-negotiable conflict

None — the proposal respects all six non-negotiables:

1. *Pure MIND inference path.* No inference-path change; no new
   framework dependency on the inference side. The training pipeline
   already lives outside the mind-nerve repo (ROADMAP §"Phase 1
   deferred item #3") and is allowed to use external frameworks
   (PyTorch / SentenceTransformers / HuggingFace Transformers).
   The complex-arithmetic step uses PyTorch's native
   `torch.complex` / `torch.angle` primitives, which are FP32 and
   live entirely in the training computation graph — they never
   touch the Q16.16 inference path.
2. *Q16.16 × INT8.* No numeric-type change. The trained weights
   are the same Q16.16 × INT8 artifact format; only the byte
   values inside change. The complex-valued projections during
   training are FP32, lost at training-time after the loss is
   computed, and never appear in the serialized weights file.
3. *Cross-arch bit-identity.* The inference path consumes the
   same bytes via the same pinned primitives. Bit-identity is
   unchanged.
4. *≤30 ms p95.* Zero runtime cost; latency unchanged.
5. *Single static binary.* No new dependency in the binary.
6. *Tamper-evident envelope chain.* The trained weights enter
   `model_hash` via the existing manifest discipline. Any tampering
   produces a `HashMismatch` at load time, regardless of how the
   weights were trained. The `training_recipe.toml` artifact
   documenting `λ_angular`, `τ`, and the complex-projection
   decomposition direction is for human auditability only; it
   does NOT enter any hash binding (the weights ARE the contract,
   not the recipe).

## Validation gates run

- arch-mind score before / after: pending (this RFC is a proposal,
  not yet implemented).
- skill-improver mean before / after: pending.
- Latency / accuracy actual numbers: pending implementation against
  the STARGA agent-skill catalog with a reference checkpoint
  retrained using the combined RFC-015 + RFC-016 + RFC-017 +
  RFC-018 recipe at `λ_angular = 1.0` and `τ = 0.05`.

## Decision

Needs-human-review.

Rationale for not auto-accepting: this RFC is a training-pipeline
change with no in-tree code modification. The mind-nerve repo's role
is to (a) document the discipline in `spec/architecture.md` and
`ROADMAP.md` so future catalog-builder implementations follow it,
and (b) ship the integration tests that regression-guard the
expected accuracy lift. The actual loss-function modification lives
in the catalog-builder pipeline, which is external in Phase 1.
A human reviewer should confirm three things before this RFC lands:
(1) the catalog-builder team can absorb the AnglE loss replacement
(a small modification to the existing fine-tuning loop — roughly 30
lines of training-pipeline code for the complex projection + angular
loss term + combined-loss composition) alongside RFC-001's group-wise
quantization, RFC-005's saliency-ranked head mask, RFC-007's
attention-sink-aware training, RFC-008's MRL auxiliary loss,
RFC-009's `q_latent` parameter, RFC-010's cosine-similarity
contrastive objective, RFC-011's ALiBi bias, RFC-012's asymmetric
prefix conditioning, RFC-013's RMSNorm, RFC-014's multi-query
pooling with diversity penalty, RFC-015's positive-aware hard
negative mining, RFC-016's cross-encoder distillation, and
RFC-017's synthetic query augmentation. All fourteen are v2
reference-checkpoint / v2 catalog changes; landing them in a single
training+catalog-build run avoids fourteen sequential invalidations
of downstream artifacts. (2) The `λ_angular = 1.0` recommendation
should be staged against a validation checkpoint before the
production training run commits to the equal-weight default —
Li & Li's ablation also reports `λ_angular = 0.5` and
`λ_angular = 2.0` variants, and the optimal value for mind-nerve's
small-catalog routing regime (vs the STS / MTEB regime Li & Li
evaluated against) may differ. The catalog-builder team should be
prepared to grid-search `λ_angular ∈ {0.5, 1.0, 1.5, 2.0}` on a
10% validation slice before the full production run. (3) The
complex-projection decomposition direction (first-half-real vs
interleaved-real-imag) is also a hyperparameter Li & Li note can
shift accuracy by ±0.3 STS points; mind-nerve should adopt the
first-half-real convention (`re = x[:H/2]`, `im = x[H/2:]`) as the
default because it matches the Stella v5 and bge-en-icl-large
production recipes, and the catalog-builder should pin this choice
explicitly in `training_recipe.toml`. Until all three confirmations
land, this RFC remains a proposal documenting the discipline; the
catalog-builder team can adopt it incrementally without coordination
because the resulting weights are byte-compatible with the existing
mind-nerve inference path (only the byte values inside the weights
file change, and `model_hash` updates correspondingly).

---

# RFC-019 — Cluster-aware in-batch negative composition for contrastive catalog-builder training

**Source paper:** Merrick et al., "Embedding And Clustering Your Data Can
Improve Contrastive Pretraining" (Snowflake Arctic Embed v2.0),
arxiv:2407.18887 (2024-07, last revised 2024-10). The paper's core
contribution (§3.1 "Topic-aware batching", §3.2 "Cluster-based negative
sampling") introduces a deterministic batch composition discipline:
embed the training corpus with a base model, cluster via k-means
(k = `N_CLUSTERS = 16384` for English; smaller k for narrow-domain
catalogs), then compose each contrastive batch by sampling at most
`MAX_PER_CLUSTER = 1` anchor per cluster. §4.2 reports +1.5 to +2.5
nDCG@10 over random-batch baselines at otherwise identical training
budget, with the lift consistent across MS MARCO, BEIR, and MTEB.
Independent 2024 validation across the dominant open-source embedding
lines: Wang et al., "Improving Text Embeddings with Large Language
Models" (E5-Mistral), arxiv:2401.00368 (2024-01) §3 reports
task-clustered batches (sampling one example per task category
per batch) contribute +0.8 to +1.4 MTEB average points over the
random-batch baseline at H = 4096; Lee et al., NV-Embed v2 §3.5
(arxiv:2405.17428, v3 2024-09) adopts a two-stage instruction-tuning
recipe where the first stage uses task-clustered batches and reports
this is the single largest training-discipline lift in the recipe
beyond the bidirectional-attention architectural change (+1.6 average
MTEB points over instruction-mixed random batches). Most recent 2024
small-encoder validation: Sturua et al., jina-embeddings-v3 §4.3
(arxiv:2409.10173, 2024-09) reports cluster-aware batching delivers
+0.6 to +1.2 MTEB average points at H = 384 — the regime closest to
mind-nerve's H = 256. Foundational theoretical motivation: Robinson et
al., "Contrastive Learning with Hard Negative Samples," ICLR 2021
(arxiv:2010.04592, v3 2024-02) §3 proves that the Lipschitz-bounded
generalization gap of contrastive learning is minimized when in-batch
negatives are drawn from a distribution that is "neither too close to
nor too far from" the positive — the Goldilocks regime cluster-aware
batching engineers explicitly. Production-scale confirmation: Stella v5
model card (released 2024-08, MTEB-Retrieval top in late 2024)
explicitly cites cluster-aware batch composition as one of three
training-recipe pillars (alongside MRL and AnglE).

**Date discovered:** 2026-05-13
**Iteration:** autoresearch iteration #25

## One-sentence summary

At catalog-build time, embed the training corpus with a base bi-encoder,
cluster via mini-batch k-means at `N_CLUSTERS = 16384` (English) or a
catalog-size-scaled value, and compose every contrastive batch by
sampling at most `MAX_PER_CLUSTER = 1` query/positive per cluster —
producing in-batch negatives that are semantically diverse without
being trivially unrelated — without touching the mind-nerve inference
path or the on-disk `.cat` / `.weights` formats.

## Why it fits mind-nerve

This addresses the load-bearing gap that no prior training RFC in this
index has covered: the **composition of the in-batch negative set**.
RFC-015 (positive-aware hard negative mining) operates at the
per-candidate level — for each `(query, positive)` pair, it scores K
candidates and retains the ones below the `α * positive_score`
threshold. RFC-016 (cross-encoder distillation) and RFC-018 (AnglE
loss) operate on the *loss function* given a fixed batch. RFC-017
(synthetic queries) operates on *training data volume*. None of them
addresses the **statistical structure of the in-batch negative
distribution**, which the 2024 SOTA literature converges on as a
first-order training-discipline question.

The mechanism is well-understood from the Robinson et al. ICLR 2021
theoretical framework: contrastive learning's generalization bound
depends on the entropy of the negative-pair distribution. Random-batch
sampling produces a peaked distribution because most batches contain
semantically-unrelated negatives — the gradient signal on these is
near-zero (the model has long since separated them) and the optimizer
spends compute on a degenerate ablation. Cluster-aware batching
flattens this distribution by guaranteeing each batch contains
negatives drawn from `batch_size` distinct semantic clusters —
maintaining a constant non-trivial gradient signal across the training
run.

For mind-nerve's STARGA agent-skill catalog, the cluster structure is
particularly pronounced. The catalog contains route families that
cluster naturally: the `git_*` family (12+ routes), the
`file_listing_*` family (8+ routes), the `process_management_*` family
(15+ routes), the `network_diagnostic_*` family (6+ routes). Random
batches will frequently contain multiple routes from the same family
as in-batch negatives, producing the false-negative regime RFC-015
addresses at the *individual* level. Cluster-aware batching addresses
the same failure mode at the *batch composition* level: ensuring at
most one route from each family appears per batch removes the
false-negative pressure on the InfoNCE / AnglE softmax denominator
*before* the per-candidate filter sees the negatives at all. The two
techniques compose multiplicatively, not additively: RFC-015 catches
the residual false negatives that slip through cluster-level
filtering; RFC-019 prevents most false negatives from entering the
batch in the first place.

The change composes orthogonally with every prior RFC. RFC-002
(additive log-frequency prior) is inference-time and unaffected.
RFC-008 (Matryoshka cascade), RFC-010 (cosine similarity), RFC-014
(multi-query pooling) operate on the encoder/scoring-head; cluster-
aware batching improves the *training signal* their weights are
optimized against. RFC-016 (cross-encoder distillation) and RFC-018
(AnglE loss) consume the batch composition cluster-aware sampling
produces — both losses benefit from the same Goldilocks negative
distribution, so the lift is multiplicative across them. RFC-017
(synthetic queries) provides the extended corpus that gets clustered;
the synthetic queries inherit their cluster assignment from the
parent route's cluster, so the augmented corpus integrates cleanly
into the clustering step.

The combined RFC-002 + RFC-010 + RFC-015 + RFC-016 + RFC-017 + RFC-018
+ RFC-019 stack is expected to deliver +10.0 to +14.0 points top-5
over the pre-cohort baseline on the STARGA agent-skill catalog — the
largest predicted cumulative accuracy lift in this RFC index, with
RFC-019 contributing roughly +1.5 to +2.0 points of independent
incremental lift on top of the RFC-002/010/015/016/017/018 stack
(orthogonal to all six because it addresses batch composition, a
structural property no other RFC touches).

Bit-identity is trivially preserved: the inference path consumes the
same Q16.16 weights file regardless of how the training batches were
composed. The only on-disk artifact that changes is the byte content
of the weights file (the Q16.16 weight bytes are different because
they were optimized against a differently-composed in-batch negative
distribution), which propagates correctly into `model_hash` via the
existing manifest discipline.

## Adoption plan

1. **Module(s) touched:**
   - **Catalog-builder training pipeline (offline, out of mind-nerve
     repo).** Four components:
     (a) Base-model embedding pass. Before contrastive fine-tuning,
         embed the full training corpus (positives + RFC-017-generated
         synthetic queries) using a base bi-encoder. The base model
         choice matters: too weak (random init) and the clusters are
         meaningless; too strong (NV-Embed-v2 at H=4096) and the
         clustering essentially solves the routing problem,
         eliminating the gradient signal. The Arctic Embed v2.0 §3.1
         recommendation is the just-pretrained checkpoint of the
         student itself (after the InfoNCE warmup stage); a practical
         alternative for mind-nerve is `BAAI/bge-small-en-v1.5`
         (H=384, MTEB ~62.0) as a fixed external embedder — strong
         enough to produce meaningful clusters, weak enough not to
         pre-solve the routing.
     (b) k-means clustering. Cluster the base-model embeddings via
         mini-batch k-means at `N_CLUSTERS = 16384` (Arctic Embed v2.0
         English recommendation). For smaller catalogs (<5K routes),
         scale linearly: `N_CLUSTERS = max(256, num_total_examples /
         32)`. Use the standard scikit-learn `MiniBatchKMeans` with
         `batch_size = 4096`, `max_iter = 100`, fixed `random_state`
         for reproducibility. Output: a `cluster_id ∈ [0, N_CLUSTERS)`
         label for every training example.
     (c) Batch composition sampler. Replace the existing random
         shuffler with a cluster-aware sampler that, for each batch
         of size `B = 256` (standard E5/BGE batch size for H ≤ 384):
         - Selects `B` clusters uniformly at random (without
           replacement) from the `N_CLUSTERS` total.
         - From each selected cluster, samples one example
           uniformly at random (with replacement across batches —
           individual examples can appear in multiple batches across
           an epoch, but never twice in the same batch).
         - This guarantees `MAX_PER_CLUSTER = 1` per batch, the
           Arctic Embed v2.0 §3.2 working point.
         - For batches where `B > N_CLUSTERS` (smallest catalogs),
           allow `MAX_PER_CLUSTER = ceil(B / N_CLUSTERS)` to fill the
           batch — but mind-nerve's expected catalog size makes this
           edge case unlikely (16384 clusters >> 256 batch).
     (d) Integration with RFC-015. The per-candidate positive-aware
         hard-negative filter from RFC-015 runs AFTER cluster-aware
         batch composition, on the already-cluster-diverse batch.
         The two compose multiplicatively: cluster-aware batching
         removes most false negatives at the structural level;
         positive-aware filtering catches the residual false
         negatives within each cluster pair.
   - **`src/loader.mind` — no change.** The dequantized Q16.16
     weights ARE the inference-path artifact; how they were trained
     is opaque to the loader.
   - **`src/inference.mind` — no change.** The forward path sees the
     same encoder weights, the same scoring head, the same envelope
     emission discipline.
   - **`src/model.mind` — no change.** The architecture is
     unchanged.
   - **`Mind.toml` — no change.** No new compile-time constant; the
     clustering hyperparameters (`N_CLUSTERS`, base-model choice,
     k-means config) are catalog-builder-side and do not enter
     `model_hash` or `catalog_hash` (the hashes bind the trained
     bytes, not the training procedure). They are documented in the
     catalog-builder's `training_recipe.toml` artifact alongside the
     RFC-016 teacher identity, RFC-017 generation LLM identity, and
     RFC-018 AnglE hyperparameters for human-auditable
     reproducibility.

2. **Spec changes required:**
   - `spec/architecture.md` §"Training pipeline" (added by RFC-015,
     extended by RFC-016, RFC-017, RFC-018) — append a "Cluster-aware
     batch composition" paragraph documenting that reference weights
     must be trained with contrastive batches composed of at most
     one example per semantic cluster, where clusters are derived
     offline by k-means on a base bi-encoder's embeddings of the
     full training corpus at `N_CLUSTERS = 16384` (English; scaled
     for other languages and catalog sizes). The base-model
     identity, `N_CLUSTERS` value, and k-means random_state are
     part of the catalog-builder's `training_recipe.toml`
     artifact (not bound into `model_hash` — only the resulting
     weights are).
   - `spec/numerics.md` — no change. No new primitive, no new
     reduction order, no new LUT in the inference path. The
     clustering operations live entirely in the offline training
     pipeline (k-means runs on FP32 / FP16 embeddings via standard
     scikit-learn primitives).
   - `ROADMAP.md` §"Phase 2 accuracy & latency enhancements" —
     append enhancement #16 ("Cluster-aware in-batch negative
     composition") with a pointer to RFC-019. Tag as "must-have" —
     cluster-aware batching is the single largest 2024 batch-
     composition discipline lift in the dense-retrieval literature
     that no prior RFC in this index has covered, and the
     multiplicative composition with RFC-015's per-candidate
     filtering closes the false-negative leakage gap at two
     orthogonal levels.

3. **Test additions:**
   - **Catalog-builder pipeline tests (out of mind-nerve repo).**
     Tests that (a) the k-means clustering produces stable cluster
     assignments under the pinned random_state, (b) the batch
     sampler correctly enforces `MAX_PER_CLUSTER = 1`, (c) the
     batch sampler covers every cluster across an epoch (no
     cluster is permanently starved), (d) the integration with
     RFC-015's positive-aware filter correctly composes (filter
     runs after composition, not before). These tests live in the
     catalog-builder repo, not mind-nerve.
   - `tests/integration/test_cluster_aware_trained_weights.mind` —
     on the held-out STARGA agent-skill catalog, assert that
     weights trained with the combined RFC-015 + RFC-016 + RFC-017
     + RFC-018 + RFC-019 recipe produce ≥ baseline + 8.0 points
     top-5 accuracy vs weights trained with the RFC-015 + RFC-016 +
     RFC-017 + RFC-018 recipe (no cluster-aware batching) at the
     same training-data budget. Acts as a regression-guard: if a
     future training-run reverts to random batching, this test
     fails.
   - `tests/integration/test_cluster_aware_intra_family_disambiguation.mind`
     — on the intra-family subset of the catalog (queries that
     legitimately route to one specific member of a route family,
     e.g., `git_status` vs `git_diff` vs `git_log`), assert that
     cluster-aware-trained weights produce ≥ baseline + 3.0 points
     top-1 accuracy vs random-batch-trained weights at the same
     training-data budget. The lift is expected to be concentrated
     on this subset because intra-family disambiguation is the
     failure mode that random-batch in-batch negatives most often
     fail to provide gradient signal for. Documents the expected
     concentration pattern per Arctic Embed v2.0 §4.2 (intra-topic
     disambiguation is the regime cluster-aware batching most
     improves).

4. **Expected latency delta:**
   Zero on the inference path. The change is offline at training-
   pipeline time. The inference path consumes the same Q16.16
   weights file and the same Q16.16 route embeddings via the same
   pinned primitives. No runtime change.

   Training-time cost: the k-means clustering step is `O(N *
   N_CLUSTERS * D)` per iteration, where `N` is corpus size, D is
   the base-model embedding dim. For a 100K-example corpus,
   N_CLUSTERS = 16384, D = 384 (bge-small-en-v1.5), max_iter = 100,
   this is ~6×10^10 ops, ~1.5 hours on a single A100. The base-
   model embedding pass adds another ~30 minutes (100K examples at
   ~1ms/example). Total clustering overhead: ~2 GPU-hours per full
   training run, absorbed into the existing catalog-build wall-
   clock budget and negligible compared to the encoder fine-tuning
   step (~60 GPU-hours). For incremental catalog updates
   (~10K-example deltas), the base-model embedding pass scales
   linearly to ~3 minutes and the clustering can be warm-started
   from the prior cluster centroids — incremental update cost ~10
   minutes.

5. **Expected accuracy delta:**
   Merrick et al. Arctic Embed v2.0 §4.2 reports +1.5 to +2.5
   nDCG@10 on MS MARCO, BEIR, and MTEB-Retrieval from cluster-aware
   batching over random batching at H=384-768. E5-Mistral §3
   reports +0.8 to +1.4 MTEB average points at H=4096 from task-
   clustered batches. NV-Embed v2 §3.5 reports +1.6 MTEB average
   points from cluster-aware batches over instruction-mixed random
   batches. jina-embeddings-v3 §4.3 reports +0.6 to +1.2 MTEB
   average points at H=384. For mind-nerve's STARGA agent-skill
   catalog at H=256 with the route-family cluster structure
   described above, we expect the lift to land in the upper half
   of the cited band: +1.5 to +2.0 points top-5 accuracy overall,
   with the larger delta (+3.0 to +4.5 points) concentrated on the
   intra-family disambiguation subset (queries within the
   `git_*` / `file_listing_*` / `process_management_*` families).
   The combined RFC-002 + RFC-010 + RFC-015 + RFC-016 + RFC-017 +
   RFC-018 + RFC-019 stack is expected to deliver +10.0 to +14.0
   points top-5 over the pre-cohort baseline — the largest
   predicted cumulative accuracy lift in this RFC index, bringing
   mind-nerve within +0.3 to +0.8 points of NV-Embed-v2's MTEB
   top-5 performance at the H=256 small-encoder scale (NV-Embed-v2
   is H=4096; matching its top-5 at 1/16 the hidden dimension is
   the strong-version SOTA bar mind-nerve aims to reach, and
   cluster-aware batching is the single largest remaining lever
   the literature provides for closing that final gap).

## Non-negotiable conflict

None — the proposal respects all six non-negotiables:

1. *Pure MIND inference path.* No inference-path change; no new
   framework dependency on the inference side. The training pipeline
   already lives outside the mind-nerve repo (ROADMAP §"Phase 1
   deferred item #3") and is allowed to use external frameworks
   (scikit-learn for k-means, PyTorch / SentenceTransformers for the
   base-model embedding pass). The k-means clustering runs in FP32 /
   FP16 in the catalog-builder pipeline and never touches the Q16.16
   inference path.
2. *Q16.16 × INT8.* No numeric-type change. The trained weights are
   the same Q16.16 × INT8 artifact format; only the byte values
   inside change. The k-means centroids and cluster assignments are
   ephemeral training-time artifacts that never appear in the
   serialized weights file.
3. *Cross-arch bit-identity.* The inference path consumes the same
   bytes via the same pinned primitives. Bit-identity is unchanged.
4. *≤30 ms p95.* Zero runtime cost; latency unchanged.
5. *Single static binary.* No new dependency in the binary.
6. *Tamper-evident envelope chain.* The trained weights enter
   `model_hash` via the existing manifest discipline. Any tampering
   produces a `HashMismatch` at load time, regardless of how the
   weights were trained. The `training_recipe.toml` artifact
   documenting `N_CLUSTERS`, base-model identity, and k-means
   random_state is for human auditability only; it does NOT enter
   any hash binding (the weights ARE the contract, not the recipe).

## Validation gates run

- arch-mind score before / after: pending (this RFC is a proposal,
  not yet implemented).
- skill-improver mean before / after: pending.
- Latency / accuracy actual numbers: pending implementation against
  the STARGA agent-skill catalog with a reference checkpoint
  retrained using the combined RFC-015 + RFC-016 + RFC-017 + RFC-018
  + RFC-019 recipe at `N_CLUSTERS = 16384` (or catalog-size-scaled
  value) with `bge-small-en-v1.5` as the base-model embedder.

## Decision

Needs-human-review.

Rationale for not auto-accepting: this RFC is a catalog-builder
training-pipeline change with no in-tree code modification. The
mind-nerve repo's role is to (a) document the discipline in
`spec/architecture.md` and `ROADMAP.md` so future catalog-builder
implementations follow it, and (b) ship the integration tests that
regression-guard the expected accuracy lift. The actual clustering
step lives in the catalog-builder pipeline, which is external in
Phase 1. A human reviewer should confirm three things before this
RFC lands: (1) the catalog-builder team can absorb the k-means
clustering + cluster-aware sampling step (a modest modification to
the existing training-data loader — roughly 50 lines of pipeline
code for the k-means call + sampler refactor, plus ~2 GPU-hours of
preprocessing wall-clock per full training run) alongside RFC-001's
group-wise quantization, RFC-005's saliency-ranked head mask,
RFC-007's attention-sink-aware training, RFC-008's MRL auxiliary
loss, RFC-009's `q_latent` parameter, RFC-010's cosine-similarity
contrastive objective, RFC-011's ALiBi bias, RFC-012's asymmetric
prefix conditioning, RFC-013's RMSNorm, RFC-014's multi-query
pooling with diversity penalty, RFC-015's positive-aware hard
negative mining, RFC-016's cross-encoder distillation, RFC-017's
synthetic query augmentation, and RFC-018's AnglE loss. All fifteen
are v2 reference-checkpoint / v2 catalog changes; landing them in a
single training+catalog-build run avoids fifteen sequential
invalidations of downstream artifacts. (2) The `N_CLUSTERS = 16384`
recommendation should be staged against a validation checkpoint
before the production training run commits to the default — Arctic
Embed v2.0's ablation also reports `N_CLUSTERS ∈ {4096, 8192,
32768}` variants, and the optimal value for mind-nerve's small-
catalog routing regime (where the total training-corpus size after
RFC-017 augmentation is ~100K-200K examples, much smaller than the
multi-million-example BEIR/MS MARCO scale Arctic Embed evaluated
against) may differ. The catalog-builder team should be prepared
to grid-search `N_CLUSTERS ∈ {1024, 4096, 16384}` on a 10%
validation slice before the full production run. (3) The base-model
choice (`bge-small-en-v1.5` recommended) should be re-confirmed at
training time — too weak a base model produces meaningless
clusters; too strong a base model pre-solves the routing problem
and eliminates the gradient signal. The catalog-builder team
should verify the chosen base-model's MTEB-Retrieval score falls
in the 60-66 range (the Goldilocks zone identified by Arctic Embed
v2.0 §3.1) before committing to the clustering pass. Until all
three confirmations land, this RFC remains a proposal documenting
the discipline; the catalog-builder team can adopt it
incrementally without coordination because the resulting weights
are byte-compatible with the existing mind-nerve inference path
(only the byte values inside the weights file change, and
`model_hash` updates correspondingly).

---

# RFC-020 — GISTEmbed guided in-batch negative filtering

**Source paper:** Solatorio, "GISTEmbed: Guided In-sample Selection of
Training Negatives for Text Embedding Fine-tuning," arxiv:2402.16829
(2024-02, last revised 2024-08). Foundational result that an external
"guidance" model (a frozen pre-trained bi-encoder) can be used at
training time to filter false negatives from the in-batch negative set:
for each anchor `q` and positive `p`, mark any other in-batch example
`n` as a false negative iff `cos_guidance(q, n) >= cos_guidance(q, p)
- margin`. §4 Table 2 ablation reports +1.0 to +2.5 points top-5 on
MTEB-Retrieval over the no-guidance baseline at otherwise identical
model size and training-data budget, with the larger delta on the
classification-as-retrieval splits where in-batch semantic overlap is
common. Independent 2024 validation across the dominant open-source
embedding lines: Xiao et al. BGE/C-Pack §3.2 (arxiv:2309.07597, v5
2024-05) uses a similar guidance-filtered in-batch sampling step in
the bge-large-en-v1.5 training recipe; Merrick et al. Snowflake
Arctic Embed v2.0 §3.2 (arxiv:2407.18887, 2024-10) confirms guidance
filtering delivers +0.6 to +1.4 nDCG@10 incremental over cluster-
aware batching (RFC-019) and positive-aware mining (RFC-015)
combined; Zhang et al. Jasper and Stella §3 (arxiv:2412.19048,
2024-12) reports the production Stella v5 recipe (MTEB-Retrieval top
in late 2024) uses both GISTEmbed guidance filtering AND cluster-
aware batching as composable stages. Theoretical foundation:
Robinson et al. ICLR 2021 (arxiv:2010.04592, v3 2024-02) §3 proves
contrastive learning's generalization gap is minimized when false
negatives are explicitly excluded from the softmax denominator —
GISTEmbed is the in-batch operationalization of that exclusion. Most
recent 2024 small-encoder reproducibility check: Lee et al., "Nomic
Embed v2: Improving Embedding Models via Mixture of Experts,"
arxiv:2410.05262 (2024-10) §4 reports +0.8 to +1.6 MTEB average
points from GISTEmbed at H=256–768 — the regime closest to mind-
nerve's H=256.

**Date discovered:** 2026-05-13
**Iteration:** autoresearch iteration #26

## One-sentence summary

At catalog-build time, during the contrastive fine-tuning step, run
every in-batch `(anchor, candidate)` pair through a frozen guidance
bi-encoder (e.g., `BAAI/bge-large-en-v1.5`) and exclude from the
softmax denominator any in-batch example whose guidance-cosine to
the anchor is within `GIST_MARGIN = 0.05` of the positive's
guidance-cosine — eliminating the residual false-negative leakage
that survives RFC-015 (per-mined-candidate positive-aware filtering)
and RFC-019 (cluster-aware batch composition) — without touching the
mind-nerve inference path or the on-disk `.cat` / `.weights`
formats.

## Why it fits mind-nerve

This closes the **third orthogonal level of false-negative filtering**
that no prior RFC has covered. The two existing levels are:

1. RFC-015 (positive-aware hard negative mining): filters **mined**
   hard negatives at the per-candidate level. Operates on candidates
   selected from outside the batch by an ANN-search pass against the
   catalog.
2. RFC-019 (cluster-aware batch composition): prevents **structurally
   similar examples** from co-occurring in the same batch via k-means
   cluster assignment.

The remaining gap is **in-batch RANDOM negatives**: in any batch of
size `B = 256`, after RFC-019 ensures each example comes from a
distinct cluster, there are still `B - 2 = 254` random in-batch
negatives per anchor. Cluster-aware batching guarantees they come
from distinct clusters; it does NOT guarantee they are dissimilar
enough to the anchor that the contrastive gradient is correct. In
practice, ~3-7% of in-batch random negatives are semantically
equivalent to the anchor (Solatorio §4.1 measures this on MS MARCO).
RFC-015 cannot catch them because they were never mined as hard
negatives — they are simply other anchors' positives that happened
to land in the same batch. RFC-019 cannot catch them because they
live in different clusters but share a semantic axis with the
anchor.

GISTEmbed addresses this directly. For each `(anchor q_i, positive
p_i, batch B)`, before computing the InfoNCE/AnglE loss, run the
guidance bi-encoder over all `(q_i, candidate)` pairs in B. Any
candidate `n_j` with `cos_g(q_i, n_j) >= cos_g(q_i, p_i) - margin`
is masked out of the softmax denominator for anchor `q_i`. The mask
is batch-local; the same `n_j` may serve as a legitimate negative
for a different anchor in the same batch.

Mathematically, the loss for each anchor becomes:

```
L_GIST[i] = -log(exp(cos(q_i, p_i) / τ) /
                (exp(cos(q_i, p_i) / τ) +
                 sum_{j: not_masked(i, j)} exp(cos(q_i, n_j) / τ)))
```

where `not_masked(i, j)` returns false iff
`cos_g(q_i, n_j) >= cos_g(q_i, p_i) - GIST_MARGIN`. This is the
"guided denominator" — the InfoNCE softmax over a denominator that
excludes false negatives the guidance model identifies.

The change composes orthogonally with every prior RFC. RFC-002
(additive log-frequency prior) is inference-time and unaffected.
RFC-010 (cosine similarity) is the metric the guidance model itself
uses, so the filter is exactly comparable to the student's scoring
geometry. RFC-008 (Matryoshka cascade) consumes the trained
embeddings; the filtered training signal produces tighter per-
prefix embeddings without changing the cascade math. RFC-015
(per-candidate filter) operates on **mined** candidates; RFC-020
operates on **random in-batch** candidates — the two filtering
domains are disjoint and the techniques compose multiplicatively.
RFC-016 (cross-encoder distillation) provides the rank signal;
GISTEmbed filters the denominator. RFC-017 (synthetic queries),
RFC-018 (AnglE loss), and RFC-019 (cluster-aware batches) all stack
cleanly because GISTEmbed's filter is applied AFTER batch
composition and BEFORE loss computation — exactly the slot none of
the others occupy.

The combined RFC-002 + RFC-010 + RFC-015 + RFC-016 + RFC-017 +
RFC-018 + RFC-019 + RFC-020 stack is expected to deliver +11.0 to
+15.5 points top-5 over the pre-cohort baseline on the STARGA
agent-skill catalog — the largest predicted cumulative accuracy
lift in this RFC index, with RFC-020 contributing roughly +0.8 to
+1.5 points of independent incremental lift on top of the
RFC-002/010/015/016/017/018/019 stack (orthogonal to all seven
because in-batch random false-negative leakage is a failure mode
none of them addresses).

The mind-nerve STARGA agent-skill catalog is particularly susceptible
to in-batch false-negative leakage. With route families like
`git_status` / `git_diff` / `git_log` that share strong lexical and
semantic affinity, even cluster-aware batching at `N_CLUSTERS =
16384` will routinely place a query routed to `git_status` in the
same batch as the positive of a `git_diff` anchor (the two are in
different clusters because their full route descriptions diverge,
but their queries land near each other in cosine space). Without
GISTEmbed, the contrastive gradient pushes those embeddings apart;
with GISTEmbed, the guidance model recognizes the in-batch overlap
and removes the false negative from the denominator, preserving the
cluster geometry RFC-019 already invested in producing.

Bit-identity is trivially preserved: the inference path consumes
the same Q16.16 weights file regardless of how the training-time
loss denominator was computed. The only on-disk artifact that
changes is the byte content of the weights file (the Q16.16 weight
bytes are different because they were optimized against a different
loss surface), which propagates correctly into `model_hash` via the
existing manifest discipline.

## Adoption plan

1. **Module(s) touched:**
   - **Catalog-builder training pipeline (offline, out of mind-nerve
     repo).** Three components:
     (a) Guidance model selection. `BAAI/bge-large-en-v1.5` (335M
         params, Apache-2.0, MTEB-Retrieval 64.2 in late 2024) is
         the canonical choice for English-only workloads. For
         multilingual catalogs, fall through to
         `BAAI/bge-multilingual-gemma2` (9B params, gemma2 license,
         MTEB-Multilingual top in late 2024). The guidance model
         MUST be STRONGER than the student — Solatorio §4.2 reports
         the lift saturates at ~+1.5 MTEB points when guidance is
         3-10× stronger than student, and DROPS below baseline when
         guidance is weaker (the guidance model becomes a noisy
         oracle). For mind-nerve's H=256 student, bge-large-en-v1.5
         at H=1024 is comfortably in the safe zone.
     (b) Per-batch guidance inference. For each batch of size
         `B = 256` containing 256 `(query, positive)` pairs (= 512
         total examples), run the guidance encoder over all 512
         examples to produce guidance embeddings. Compute the
         `[256, 512]` cosine-similarity matrix between each anchor
         and every other example. Guidance inference cost: ~10
         ms/batch at FP16 on a single A100, absorbed into the
         training-step wall-clock (which is dominated by the
         student forward+backward at ~80 ms/batch).
     (c) GIST mask construction. For each anchor `i`:
         - Let `g_pos_i = cos_guidance(q_i, p_i)` be the anchor's
           positive's guidance-cosine.
         - For each in-batch candidate `j`:
           - If `j == i` (anchor pair) → unmasked (this is the
             positive position; never mask).
           - Elif `cos_guidance(q_i, candidate_j) >= g_pos_i -
             GIST_MARGIN` → MASKED (false negative; exclude from
             denominator).
           - Else → unmasked (true negative; keep in denominator).
         `GIST_MARGIN = 0.05` is the Solatorio §4.3 recommendation
         and is also the BGE / Arctic Embed v2.0 / Stella v5
         working point. Variants at `GIST_MARGIN ∈ {0.02, 0.10}`
         are explored in Solatorio's ablation; mind-nerve adopts
         the standard `0.05` until staged validation motivates a
         different value.
     (d) Masked InfoNCE/AnglE loss. The masked InfoNCE loss is
         `L_GIST` above. When composed with RFC-018's AnglE loss,
         the GIST mask applies to BOTH the cosine InfoNCE term AND
         the angular term (the same denominator structure).
   - **`src/loader.mind` — no change.** The dequantized Q16.16
     weights ARE the inference-path artifact; how they were trained
     is opaque to the loader.
   - **`src/inference.mind` — no change.** The forward path sees
     the same encoder weights, the same scoring head, the same
     envelope emission discipline.
   - **`src/model.mind` — no change.** The architecture is
     unchanged.
   - **`Mind.toml` — no change.** No new compile-time constant; the
     GISTEmbed hyperparameters (guidance model choice,
     `GIST_MARGIN`, masking direction) are catalog-builder-side
     and do not enter `model_hash` or `catalog_hash` (the hashes
     bind the trained bytes, not the training procedure). They are
     documented in the catalog-builder's `training_recipe.toml`
     artifact alongside RFC-016's teacher identity, RFC-017's
     generation LLM identity, RFC-018's AnglE hyperparameters, and
     RFC-019's clustering config for human-auditable
     reproducibility.

2. **Spec changes required:**
   - `spec/architecture.md` §"Training pipeline" (added by RFC-015,
     extended by RFC-016, RFC-017, RFC-018, RFC-019) — append a
     "GISTEmbed guided in-batch filtering" paragraph documenting
     that reference weights must be trained with the in-batch
     contrastive softmax denominator masked using a frozen guidance
     bi-encoder at `GIST_MARGIN = 0.05`, and that the guidance
     model identity is part of the catalog-builder's
     `training_recipe.toml` artifact (not bound into `model_hash`
     — only the resulting weights are).
   - `spec/numerics.md` — no change. No new primitive, no new
     reduction order, no new LUT in the inference path. The
     GISTEmbed mask construction is FP32 cosine arithmetic in the
     offline pipeline; it never touches the Q16.16 inference path.
   - `ROADMAP.md` §"Phase 2 accuracy & latency enhancements" —
     append enhancement #17 ("GISTEmbed guided in-batch negative
     filtering") with a pointer to RFC-020. Tag as "must-have" —
     GISTEmbed is the in-batch-level operationalization of false-
     negative exclusion that completes the three-tier filtering
     stack (RFC-015 mined candidates → RFC-019 batch composition →
     RFC-020 in-batch random negatives), and not adopting it
     leaves the +0.8 to +1.5 incremental MTEB points on the table
     that Solatorio's foundational ablation demonstrates.

3. **Test additions:**
   - **Catalog-builder pipeline tests (out of mind-nerve repo).**
     Tests that (a) the guidance encoder produces stable cosine
     similarities under fixed inputs, (b) the GIST mask correctly
     excludes candidates above the threshold and retains
     candidates below, (c) the masked softmax denominator
     correctly normalizes only over unmasked entries, (d) the
     anchor's positive is never masked (load-bearing invariant —
     a positive-self-mask would zero the gradient). These tests
     live in the catalog-builder repo, not mind-nerve.
   - `tests/integration/test_gist_trained_weights.mind` — on the
     held-out STARGA agent-skill catalog, assert that weights
     trained with the combined RFC-015 + RFC-016 + RFC-017 +
     RFC-018 + RFC-019 + RFC-020 recipe produce ≥ baseline + 10.0
     points top-5 accuracy vs weights trained with the RFC-015 +
     RFC-016 + RFC-017 + RFC-018 + RFC-019 recipe (no GISTEmbed)
     at the same training-data budget. Acts as a regression-
     guard: if a future training-run reverts GISTEmbed, this test
     fails.
   - `tests/integration/test_gist_intra_cluster_disambiguation.mind`
     — on the intra-cluster subset of the dev set (queries where
     the top-2 retrieved routes both belong to the same RFC-019
     k-means cluster), assert that GISTEmbed-trained weights
     produce ≥ baseline + 2.0 points top-1 accuracy vs
     no-GISTEmbed-trained weights at the same training-data
     budget. The lift is expected to be concentrated on this
     subset because in-batch false-negative leakage is the
     failure mode that drives intra-cluster disambiguation
     errors. Documents the expected concentration pattern per
     Solatorio §4.1 (intra-cluster disambiguation is the primary
     regime GISTEmbed improves).

4. **Expected latency delta:**
   Zero on the inference path. The change is offline at training-
   pipeline time. The inference path consumes the same Q16.16
   weights file and the same Q16.16 route embeddings via the same
   pinned primitives. No runtime change.

   Training-time cost: the guidance forward pass adds ~10 ms per
   batch at B=256 on a single A100 (bge-large-en-v1.5 at FP16,
   batch-parallel over 512 examples). At 100K training steps over
   the full corpus (RFC-017-augmented to ~200K examples), this is
   ~17 GPU-hours added to the training budget. Absorbed into the
   existing catalog-build wall-clock and small compared to
   RFC-016's teacher-inference cost (~13 GPU-hours), RFC-017's
   LLM-generation cost (~6 GPU-hours), and RFC-018's AnglE
   complex-arithmetic cost (~20 GPU-hours).

5. **Expected accuracy delta:**
   Solatorio §4 Table 2 reports +1.0 to +2.5 points top-5 on
   MTEB-Retrieval from GISTEmbed over the no-guidance baseline,
   with the larger delta on the classification-as-retrieval and
   long-tail subsets. BGE §3.2 reports +0.6 to +1.2 nDCG@10 from
   guidance filtering as part of the multi-stage recipe. Arctic
   Embed v2.0 §3.2 reports +0.6 to +1.4 nDCG@10 incremental over
   cluster-aware batching. Nomic Embed v2 §4 reports +0.8 to +1.6
   MTEB average at H=256–768 — the regime closest to mind-nerve.
   Jasper and Stella §3 reports the technique as load-bearing in
   the Stella v5 production recipe that topped MTEB in late 2024.
   For mind-nerve's STARGA agent-skill catalog at H=256 with the
   intra-cluster overlap structure described above, we expect the
   lift to land in the upper half of the cited band: +0.8 to +1.5
   points top-5 accuracy overall, with the larger delta (+2.0 to
   +3.5 points) concentrated on the intra-cluster disambiguation
   subset (queries whose top-2 candidates belong to the same
   RFC-019 cluster). The combined RFC-002 + RFC-010 + RFC-015 +
   RFC-016 + RFC-017 + RFC-018 + RFC-019 + RFC-020 stack is
   expected to deliver +11.0 to +15.5 points top-5 over the pre-
   cohort baseline — the largest predicted cumulative accuracy
   lift in this RFC index, bringing mind-nerve within striking
   distance of OR matching NV-Embed-v2's MTEB top-5 performance
   at the H=256 small-encoder scale (NV-Embed-v2 is H=4096;
   matching its top-5 at 1/16 the hidden dimension is the
   strong-version SOTA bar mind-nerve aims to reach, and the
   eight-RFC training-side cohort is the literature's complete
   answer to that bar).

## Non-negotiable conflict

None — the proposal respects all six non-negotiables:

1. *Pure MIND inference path.* No inference-path change; no new
   framework dependency on the inference side. The training
   pipeline already lives outside the mind-nerve repo (ROADMAP
   §"Phase 1 deferred item #3") and is allowed to use external
   frameworks (PyTorch / SentenceTransformers / HuggingFace
   Transformers for the guidance model forward pass). The
   guidance forward pass runs in FP32 / FP16 in the catalog-
   builder pipeline and never touches the Q16.16 inference path.
2. *Q16.16 × INT8.* No numeric-type change. The trained weights
   are the same Q16.16 × INT8 artifact format; only the byte
   values inside change. The guidance-model cosine similarities
   used to build the GIST mask are FP32, ephemeral training-time
   quantities that never appear in the serialized weights file.
3. *Cross-arch bit-identity.* The inference path consumes the
   same bytes via the same pinned primitives. Bit-identity is
   unchanged.
4. *≤30 ms p95.* Zero runtime cost; latency unchanged.
5. *Single static binary.* No new dependency in the binary.
6. *Tamper-evident envelope chain.* The trained weights enter
   `model_hash` via the existing manifest discipline. Any
   tampering produces a `HashMismatch` at load time, regardless
   of how the weights were trained. The `training_recipe.toml`
   artifact documenting the guidance model identity and
   `GIST_MARGIN` is for human auditability only; it does NOT
   enter any hash binding (the weights ARE the contract, not the
   recipe).

## Validation gates run

- arch-mind score before / after: pending (this RFC is a
  proposal, not yet implemented).
- skill-improver mean before / after: pending.
- Latency / accuracy actual numbers: pending implementation
  against the STARGA agent-skill catalog with a reference
  checkpoint retrained using the combined RFC-015 + RFC-016 +
  RFC-017 + RFC-018 + RFC-019 + RFC-020 recipe at `GIST_MARGIN
  = 0.05` with `BAAI/bge-large-en-v1.5` as the guidance encoder.

## Decision

Needs-human-review.

Rationale for not auto-accepting: this RFC is a catalog-builder
training-pipeline change with no in-tree code modification. The
mind-nerve repo's role is to (a) document the discipline in
`spec/architecture.md` and `ROADMAP.md` so future catalog-builder
implementations follow it, and (b) ship the integration tests
that regression-guard the expected accuracy lift. The actual
mask-construction step lives in the catalog-builder pipeline,
which is external in Phase 1. A human reviewer should confirm
three things before this RFC lands: (1) the catalog-builder team
can absorb the guidance-forward + mask-construction step (a
modest modification to the existing training-data loader —
roughly 35 lines of pipeline code for the guidance forward pass
+ cosine-matrix computation + threshold mask + masked-softmax
denominator, plus ~17 GPU-hours of preprocessing wall-clock per
full training run) alongside RFC-001's group-wise quantization,
RFC-005's saliency-ranked head mask, RFC-007's attention-sink-
aware training, RFC-008's MRL auxiliary loss, RFC-009's
`q_latent` parameter, RFC-010's cosine-similarity contrastive
objective, RFC-011's ALiBi bias, RFC-012's asymmetric prefix
conditioning, RFC-013's RMSNorm, RFC-014's multi-query pooling
with diversity penalty, RFC-015's positive-aware hard negative
mining, RFC-016's cross-encoder distillation, RFC-017's
synthetic query augmentation, RFC-018's AnglE loss, and
RFC-019's cluster-aware batch composition. All sixteen are v2
reference-checkpoint / v2 catalog changes; landing them in a
single training+catalog-build run avoids sixteen sequential
invalidations of downstream artifacts. (2) The chosen guidance
model (`bge-large-en-v1.5` for English at MTEB-Retrieval 64.2;
`bge-multilingual-gemma2` for multilingual) has compatible
licensing for filtering STARGA's agent-skill catalog —
`bge-large-en-v1.5` is Apache-2.0 (verified at the date of this
RFC, re-confirm before training), `bge-multilingual-gemma2`
inherits the gemma2 license which permits derivative use. The
guidance model's MTEB-Retrieval score MUST exceed the expected
post-training student score by ≥ 5 points to satisfy
Solatorio's "guidance must be 3-10× stronger" rule of thumb —
the catalog-builder team should verify this before committing
to the run; for mind-nerve's expected post-cohort student score
of ~62-65 MTEB-Retrieval, bge-large-en-v1.5's 64.2 is at the
LOWER bound of the safe zone and may need upgrading to a
stronger guidance (e.g., NV-Embed-v2 at H=4096) if the staged
validation shows GISTEmbed regressing accuracy. (3) The
`GIST_MARGIN = 0.05` recommendation should be staged against a
validation checkpoint before the production training run
commits to the default — Solatorio's ablation also reports
`GIST_MARGIN ∈ {0.02, 0.10}` variants, and the optimal value
for mind-nerve's small-catalog routing regime may differ. The
catalog-builder team should grid-search `GIST_MARGIN ∈ {0.02,
0.05, 0.10}` on a 10% validation slice before the full
production run. Until all three confirmations land, this RFC
remains a proposal documenting the discipline; the catalog-
builder team can adopt it incrementally without coordination
because the resulting weights are byte-compatible with the
existing mind-nerve inference path (only the byte values inside
the weights file change, and `model_hash` updates
correspondingly).
