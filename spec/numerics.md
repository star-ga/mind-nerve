# Numerics

Authoritative per-operator reduction-order specification for the
mind-nerve inference path. Implementations in `src/` MUST match this
document. `mindc` MUST lint against it. Discrepancies are bugs against
this spec, not against the code.

This spec is `numerics.md@v1`.

## Section 1 — Overview

Every reduction performed inside the mind-nerve inference path uses
**exactly one of two strategies**, declared per-operator at the type
level. There is no third strategy. There is no per-backend variation
of which strategy applies. Backend-specific lowering may parallelise
inside the chosen shape, but the shape itself is fixed.

The two strategies are:

1. **`sequential`** — left-to-right scalar accumulation. Element `0`
   first, element `N-1` last. No reordering. No partial accumulation
   into thread-local buffers that get combined out of order.
2. **`tree_associative_fixed`** — pre-built binary reduction tree of
   depth `ceil(log2(N))`, leaves padded with the identity element,
   pair-wise reduction from leaves toward the root. The tree shape is
   identical on every backend; lane assignment is backend-specific but
   may not change the pair-wise composition order.

Both strategies are deterministic and produce a single canonical
output for any given input. Bit-identity across architectures is a
property of pinning the shape, not of pinning the hardware.

All inference-path arithmetic is fixed-point Q16.16, represented as a
signed 32-bit two's-complement integer. The semantics of Q16.16
multiply are bit-exact and defined in §4. The overflow policy is
**Saturating**, never Wrapping (see §4 for rationale).

No IEEE-754 operation is permitted anywhere inside the `mind_nerve`
module. `mindc` enforces this via the module-level invariant
`@[invariant no_float_ops]` (see §6).

### Audit trigger

This document closes finding **A2** of the Phase 0 architectural
audit, which observed:

> "Q16.16 cross-arch bit-identity through reduction-order pinned by
> compile-time topology is convention, not constraint. Softmax,
> layer-norm, and Q16.16-overflow rules are undefined. `mindc` has to
> make the call, which means the lowering pass — not the spec — is
> the bit-identity contract."

After this document lands, the spec **is** the contract. The lowering
pass executes the contract; it does not author it.

## Section 2 — Per-operator reduction strategy

Every reduction site reachable from `preselect` MUST appear in this
table. Adding a new reduction without adding a row here is a `mindc`
lint error (§6 rule 2).

| Operator | Reduction | Strategy | Tree depth | Notes |
|---|---|---|---|---|
| `q16_sum(x: [i32; N]) -> i32` | scalar accumulate over `N` elements | `sequential` | n/a | Iterates `acc = q16_add_sat(acc, x[i])` for `i = 0..N`. Initial `acc = 0`. Used by `q16_dot`, `q16_softmax`, `q16_layernorm`. |
| `q16_max(x: [i32; N]) -> i32` | scalar max over `N` elements | `sequential` | n/a | Iterates `acc = max(acc, x[i])` for `i = 0..N`. Initial `acc = MIN_Q16 = -2_147_483_648`. Used by `q16_softmax`. |
| `q16_min(x: [i32; N]) -> i32` | scalar min over `N` elements | `sequential` | n/a | Initial `acc = MAX_Q16 = 2_147_483_647`. Diagnostics only, not in forward pass. |
| `q16_dot(a, b: [i32; N]) -> i32` | sum of products | `sequential` | n/a | Elementwise `q16_mul` followed by `q16_sum`. Both stages sequential. |
| `q16_softmax(x: [i32; N]) -> [i32; N]` | 5-stage: max → shift → exp → sum → divide | `sequential` (each stage) | n/a | Stage order pinned (see §5). `q16_exp` is a fixed lookup table; never a polynomial. |
| `q16_layernorm(x, gamma, beta: [i32; H]) -> [i32; H]` | mean → sum_of_squares → variance → rsqrt → elementwise affine | `sequential` (each stage) | n/a | Stage order pinned: `mean → sum_of_squares → variance → rsqrt → normalize → affine`. `q16_rsqrt` is a fixed lookup table. |
| `q16_attention_scores(Q, K) -> S` | per-element dot product `Q[i] · K[j]` over `hidden_dim` | `sequential` per element | n/a | Output is `tensor<Q16_16, [seq_len, seq_len]>`. Reduction is per output element over `hidden_dim`; no cross-element reduction. |
| `q16_attention_apply(S, V) -> Y` | per-element dot product `S[i] · V[:, d]` over `seq_len` | `sequential` per element | n/a | Output is `tensor<Q16_16, [seq_len, hidden_dim]>`. Reduction is over `seq_len` per `(i, d)`. |
| `q16_gated_residual(x, residual, gate: [i32; H]) -> [i32; H]` | elementwise blend, no reduction | n/a | n/a | No reduction means no strategy tag. Still requires `@[determinism(BitIdentical)]`. |
| `q16_top_k(logits: [i32; N], k: u32) -> ([RouteId; k], [i32; k])` | `k` passes of `q16_argmax` over `N`-wide mask | `sequential` (`k` passes) | n/a | Mask-and-rerun, not heap-based. Tie-break by `SHA-256(route_id) ascending` is part of the operator contract (see `spec/architecture.md §Classifier head`). |
| `q16_argmax(x: [i32; N], mask: [bool; N]) -> u32` | scalar argmax with masking | `sequential` | n/a | On exact tie, lower index wins inside this primitive; outer `q16_top_k` overrides with `SHA-256(route_id)` ordering. |
| `q16_canonical_hash(top_k: TopK) -> [u8; 32]` | SHA-256 over length-prefixed `(route_id, score)` pairs | `sequential` | n/a | Length-prefix order: `id_len:u32 || route_id || score:i32_le`, pairs emitted in result order. |

### Coverage requirements

- Signature, strategy tag, and determinism annotation MUST appear
  together at the function definition site. None of the three is
  optional.
- The function MUST be inside the `mind_nerve` module so the
  module-level `@[invariant no_float_ops]` applies.
- Helpers called from these operators inherit the same three
  requirements transitively.
- Any future operator missing from this table is a `mindc` lint error
  referencing this document.

## Section 3 — Tree-associative-fixed strategy

No operator in §2 currently uses this strategy. The strategy is
specified here so future operators that adopt it have a single
authoritative shape and so auditors can verify the lint catches
deviations.

The shape, for a reduction over `N` elements with associative identity
element `e`:

1. **Padding.** Let `N' = next_power_of_two(N)`. Pad input to length
   `N'` by appending `(N' - N)` copies of `e`. Identity is
   operator-specific: `0` for sum, `MIN_Q16` for max, `MAX_Q16` for
   min.
2. **Tree depth.** `D = log2(N')`.
3. **Pair-wise step.** At depth `d` from leaves (`d = 0` at leaves,
   `d = D` at root), pair element `i` with element `i + stride` where
   `stride = 2^d`. Combine pairs in increasing `i` order. The output
   of depth `d` becomes the input of depth `d + 1`. No skipping, no
   chunk-and-recombine, no log-then-linear fallback.
4. **Output.** After `D` rounds, exactly one element remains.

### Cross-backend invariant

Backend lowering MAY:

- Assign different threads / warps / lanes to different `i` indices
  at a given depth.
- Use SIMD intrinsics to compute several pair-wise reductions in
  parallel within a single depth level.
- Fuse depth levels into a single kernel.

Backend lowering MUST NOT:

- Reorder pair-wise composition. The pair `(x[i], x[i + stride])` at
  depth `d` is the only legal pair.
- Use a chunk-based recursive fallback when `N` is large. There is
  exactly one tree shape per `N`.
- Substitute Kahan summation, pairwise summation with non-binary
  fan-in, or any other "better" reduction.

Backend-specific parallelism is the lowering pass's problem. The spec
fixes the order of combination, not the hardware that performs it.

## Section 4 — Q16.16 multiply contract

The only legal Q16.16 multiply in the inference path is:

```
fn q16_mul(a: i32, b: i32) -> i32 {
    let wide:    i64 = (a as i64) * (b as i64);
    let shifted: i64 = wide >> 16;   // arithmetic shift; sign-extended
    if shifted > MAX_Q16 as i64 { return MAX_Q16; }
    if shifted < MIN_Q16 as i64 { return MIN_Q16; }
    return shifted as i32;
}
```

Where `MAX_Q16 = 2_147_483_647` (`i32::MAX`) and
`MIN_Q16 = -2_147_483_648` (`i32::MIN`).

### Properties

1. **Width.** Intermediate product is `i64`. The product of two `i32`
   values fits in `i64` without overflow; the widening multiply never
   wraps.
2. **Rounding.** None. Arithmetic right shift by 16 truncates toward
   negative infinity for negative magnitudes. No round-half-to-even,
   no round-toward-zero correction. This is the cheapest deterministic
   choice every backend can reproduce identically.
3. **Overflow policy.** **Saturating.** If the shifted result exceeds
   the representable range, clamp to `MAX_Q16` or `MIN_Q16`. Wrapping
   is forbidden.
4. **Sign.** Preserved by arithmetic shift on two's-complement; no
   special case for negative inputs.

### Why Saturating, not Wrapping

A wrapping Q16.16 multiply at the attention score layer can flip a
near-saturation positive score to a large negative one in a single
operation, producing `exp(near_zero)` in softmax and silently zeroing
the corresponding attention weight. The model "forgets" tokens whose
pre-softmax scores happened to overflow.

Saturating multiply caps the value at the representable maximum. The
attention weight is large but finite, softmax still produces a
probability distribution, and behaviour degrades gracefully.
Saturation also matches the auditable failure mode: "this multiply hit
the ceiling" is visible in intermediate tensors; a wrap is invisible
until the chain hash diverges.

### Companion saturating operators

`q16_add_sat`, `q16_sub_sat`, and `q16_div_sat` use the same widen-
clamp pattern. Divide-by-zero saturates to `MAX_Q16` (positive
dividend) or `MIN_Q16` (negative dividend) rather than panicking; this
keeps the inference path total and reproduces the failure value
identically across backends. A softmax denominator that reaches zero
indicates the max-subtract stage produced shifted inputs at the table
floor — itself a documented degenerate case (§5).

### What `mindc` rejects

A Q16.16 multiply written as `(a as i64 * b as i64) >> 16` cast back
to `i32` without the saturation clamp is a lint error, even if the
output range is statically provable to fit. The single legal multiply
is `q16_mul`; no exceptions.

## Section 5 — Softmax detailed flow

Softmax is the highest-risk reduction site because it composes two
reductions (max + sum) around a numerically stable `max-subtract`
step. The full pinned flow:

```
fn q16_softmax(x: [i32; N]) -> [i32; N] {
    // Stage 1: max (sequential left-to-right)
    let m: i32 = q16_max(x);

    // Stage 2: shift (elementwise, no reduction)
    let shifted: [i32; N] = [q16_sub_sat(x[i], m) for i in 0..N];

    // Stage 3: exp lookup (elementwise, no reduction)
    let exped: [i32; N] = [q16_exp(shifted[i]) for i in 0..N];

    // Stage 4: sum (sequential left-to-right)
    let s: i32 = q16_sum(exped);

    // Stage 5: divide (elementwise, no reduction)
    return [q16_div_sat(exped[i], s) for i in 0..N];
}
```

Stages may not be fused, reordered, or replaced with a single-pass
approximation. A backend that performs "online softmax" in a single
fused kernel is acceptable **only if** it reproduces the bits this
five-stage form produces; that is the backend's burden, not the
spec's.

### `q16_exp` contract

`q16_exp(x: i32) -> i32` MUST be a fixed lookup table:

- At least **256 entries**.
- Covers input range `[-16.0, 0.0]` in Q16.16. Inputs above zero are
  impossible after max-subtract; inputs below `-16.0` clamp to entry
  `0`.
- Shipped with the model weights blob (the table contributes to
  `model_hash`).
- **Truncated lookup**, no interpolation. Interpolation adds a
  multiply per call whose bit-result depends on multiply ordering;
  truncation removes that failure surface.
- Inputs at or above zero clamp to entry `N-1` (output is
  Q16.16(1.0) = 65536).

**Polynomials are forbidden.** Polynomial evaluation requires several
`q16_mul` calls and the result depends on associativity choices
(left-to-right vs Horner vs Estrin). Lookup is the only form that
survives backend lowering without per-step ordering rules.

### `q16_rsqrt` contract

Used by `q16_layernorm`. Same shape as `q16_exp`: fixed truncated
lookup table, ≥ 256 entries, covers the positive input range of
variance values, table bytes contribute to `model_hash`. No
Newton-Raphson iteration (the iteration order would diverge under
fused-multiply-add reassociation).

### Degenerate input handling

Documented outputs that `tests/unit/test_q16_16.mind` MUST verify:

- **All-equal input** `x = [v, ..., v]`: after max-subtract, shifted
  is all zero; `q16_exp(0) = 65536`; sum is `N × 65536`; each output
  is `Q16.16(1/N)` rounded down per `q16_div_sat`. Bit-identical
  across backends.
- **All-zero input**: identical to all-equal.
- **Single non-zero** `x = [0, ..., 0, v, 0, ..., 0]` with `v > 0`:
  shifted positions are `-v` (clamped to table floor if below
  `-16.0`) and `0` at peak; exp is 65536 at peak, ≈ 0 elsewhere; sum
  ≈ 65536; output is Q16.16(~1.0) at the peak position.

## Section 6 — `mindc` lint contract

`mindc` MUST reject the following at compile time. These are compile
errors, not warnings.

### Rule 1 — Missing determinism annotation

Every function reachable from `preselect` MUST carry
`@[determinism(BitIdentical)]`:

```
error[E_NERVE_001]: function reachable from `preselect` lacks
  `@[determinism(BitIdentical)]` annotation
  --> src/inference.mind:line:col
     | fn helper(...)
     | ^^^^^^^^^^^^^ add @[determinism(BitIdentical)] above this signature
  = note: see spec/numerics.md §1 and §6 rule 1
```

### Rule 2 — Reduction without strategy tag

Any function performing a fold, accumulate, sum, max, min, or
argmax-style operation over a slice MUST carry
`@[reduction_strategy("sequential" | "tree_associative_fixed")]`:

```
error[E_NERVE_002]: reduction operator lacks reduction-strategy tag
  --> src/model.mind:line:col
     | fn unsafe_sum(x: [i32]) -> i32
     | ^^^^^^^^^^^^^ declare @[reduction_strategy("sequential")] or
     |               @[reduction_strategy("tree_associative_fixed")]
  = note: see spec/numerics.md §2 operator table
```

### Rule 3 — Illegal Q16.16 multiply

The only legal Q16.16 multiply is `q16_mul`. Any other multiply
pattern on Q16.16-typed values is rejected:

```
error[E_NERVE_003]: forbidden Q16.16 multiply pattern
  --> src/model.mind:line:col
     | let c: Q16_16 = (a as i64 * b as i64 >> 16) as i32;
     |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ use q16_mul(a, b)
  = note: see spec/numerics.md §4
```

### Rule 4 — IEEE-754 operation inside the module

The `mind_nerve` module carries `@[invariant no_float_ops]`. Any
`f32`, `f64`, `math::sqrt`, `math::exp`, or other floating-point
operation inside the module is rejected:

```
error[E_NERVE_004]: floating-point operation inside `mind_nerve`
  violates `@[invariant no_float_ops]`
  --> src/inference.mind:line:col
     | let score: f32 = (raw as f32) / 65536.0;
     |                   ^^^^^^^^^^^^^^^^^^^^^^ no IEEE-754 in inference path
  = note: scores convert to decimal only at the wire boundary;
          inference internals remain in Q16.16
  = note: see spec/numerics.md §1 and spec/architecture.md
          §Bit-identity contract
```

### Rule 5 — Implementation-defined reduction

Calling a backend-provided `reduce_sum`, `reduce_max`, or any kernel
documented as "implementation-defined" is rejected even with a
strategy tag:

```
error[E_NERVE_005]: implementation-defined reduction is not bit-identity safe
  --> src/model.mind:line:col
     | let s = backend::reduce_sum(x);
     |          ^^^^^^^^^^^^^^^^^^^^^ has implementation-defined associativity;
     |          use q16_sum or a tree_associative_fixed reduction
  = note: see spec/numerics.md §2 allow-list
```

### Lint scope

Rules 1–5 apply transitively. A function called from `preselect` that
calls a function that violates any rule is itself a compile error
(the error points at the leaf violation).

## Section 7 — Test obligations

### Unit-level proofs (`tests/unit/test_q16_16.mind`)

1. **`q16_mul` saturation.** For inputs near the representable
   boundary (`MAX_Q16`, `MIN_Q16`, `MAX_Q16 - 1`, …), the function
   returns the saturated value bit-identically across backends.
2. **`q16_mul` sign correctness.** All 16 sign-pair combinations of
   `(small_pos, small_neg, large_pos, large_neg)` match the spec.
3. **`q16_sum` order sensitivity.** Reordering the input array
   produces a *different* sum (saturation is not associative). If a
   backend reorders for performance, this test fails. This is the
   regression guard.
4. **`q16_max` and `q16_min`.** Edge cases of all-equal, monotone
   increasing, monotone decreasing, single peak/valley.
5. **`q16_softmax` degenerate inputs.** The three cases in §5 produce
   the documented outputs bit-for-bit.
6. **`q16_exp` / `q16_rsqrt` table determinism.** Table bytes,
   hashed, match the hashes committed to the model weights blob.
7. **`q16_layernorm` stage order.** Sum-of-squares computed in the
   wrong stage order produces a different variance; the test fails
   if the pipeline is reordered.

### Cross-architecture proofs (`tests/bit_identity/`)

1. **At least 100 randomised inputs per §2 operator.** Same input
   vector, lowered through `mindc` to two distinct architectures,
   produces byte-identical output. Phase 1: `{x86-cpu, cuda}`. Phase
   2: add `{arm-cpu, webgpu}`. Phase 3: add `{npu}`.
2. **End-to-end `preselect` parity.** For at least 100 randomised
   `(weights, catalog, request)` triples, every backend in scope for
   the phase produces a byte-identical `result_hash`.
3. **No `std::sort` or implementation-defined ordering.** Grep'ing
   the lowered binary for those symbols returns zero hits; this is a
   lint-equivalent CI gate.
4. **Saturation cross-arch.** Inputs designed to trigger saturation
   in `q16_mul`, `q16_add_sat`, `q16_sub_sat`, and `q16_div_sat`
   produce the same saturated values on every backend. This is the
   most fragile cross-arch property; it gets its own test suite.

### Failure semantics

One byte of divergence fails the CI gate. There is no "approximately
bit-identical" and no warning threshold. The runner emits the first
divergent byte's offset, a hex dump of the surrounding region, and
exits non-zero.

## Section 8 — Versioning

This spec is `numerics.md@v1`. Stable contract.

### Major-version-bump changes (`v2` and incompatible with `v1`)

- Adding, removing, or reclassifying any row in §2.
- Changing the `q16_mul` overflow policy from Saturating to anything
  else.
- Changing the softmax stage order or the `q16_exp` lookup table
  shape.
- Changing the layer-norm pipeline order.
- Removing or weakening any §6 lint rule.

### Non-breaking changes (still `v1`)

- Adding a new operator that uses an existing strategy, provided the
  row is appended to §2 with the same annotation discipline.
- Adding a new §6 lint rule strictly more restrictive than existing
  rules.
- Clarifications, examples, additional test obligations in §7.

### Migration on major bump

A bump to `numerics.md@v2` invalidates the `model_hash` for every
v1-trained mind-nerve weights blob, because the numerical semantics
of the inference path change and the same weights now produce
different bits.

- All v1 attestation envelopes remain **cryptographically valid** —
  chain hashes are still correct over their original inputs.
- All v1 attestation envelopes remain **semantically valid for v1
  consumers** — anyone replaying against a v1 mind-runtime gets the
  same bits.
- A v2 mind-runtime MUST refuse to extend a v1 chain. The chain
  resets with a documented `chain_reset_reason: numerics_v2_migration`
  envelope.
- Downstream consumers (mind-mem v4 audit composition, any host that
  builds composite decisions on top of mind-nerve results) MUST NOT mix
  v1 and v2 results within a single decision without explicit
  acknowledgement at the consumer layer.

### Withdrawn versions

None. v1 is the first version. If v1 is ever withdrawn, this section
gets a row recording the withdrawal date, the replacing version, and
the migration path. Versions are never silently removed.
