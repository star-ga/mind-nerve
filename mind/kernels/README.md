# mind-nerve/mind/kernels — Phase A1.2 Q16.16 Encoder Kernels

STARGA internal — Phase A1.2 of the BGE-small-en-v1.5 native MIND encoder port.

## Overview

This directory contains the Q16.16 MIND kernel family for the mind-nerve
BGE-small-en-v1.5 fine-tune encoder. All arithmetic is integer-only (i64
encoding Q16.16 or Q32.32 fixed-point). No f32 anywhere in this surface.

## Files

| File | LOC (approx) | Purpose |
|---|---|---|
| `matmul_q16.mind` | ~155 | Dense Q16.16 matrix multiply, 3 shape regimes + scoring |
| `batched_matmul_q16.mind` | ~160 | Per-head Q·Kᵀ and attn·V |
| `layernorm_q16.mind` | ~135 | LayerNorm with Q32.32 variance accumulator |
| `gelu_q16.mind` | ~70 | GELU via tanh approximation (Hendrycks/Gimpel) |
| `l2_norm_q16.mind` | ~60 | Single-vector L2 normalisation |
| `embedding_q16.mind` | ~100 | Word+position+token_type gather + 3-way sum |
| `sliding_window.mind` | ~115 | Later-window-wins framing helpers |
| `topk_q16.mind` | ~165 | K-min-heap top-K selection (K ≤ 32) |
| `encode.mind` | ~340 | Top-level 12-layer BERT encoder driver |

Total: ~1,300 LOC (dense, no generated boilerplate).

## Canonical Reduction Orders

Reduction order is fixed and documented per-file. Summary:

**matmul_q16**: outer `i` (output rows), middle `k` (contraction, ascending),
inner `j` (output cols). Q32.32 accumulator inside `k` loop, narrowed to
Q16.16 on store.

**batched_matmul_q16 (Q·Kᵀ)**: `h=0..11`, `i=0..T-1`, `k=0..31`, `j=0..T-1`.
Scale by Q16.16(11585) = 1/sqrt(32) applied at store.

**batched_matmul_q16 (attn·V)**: `h=0..11`, `i=0..T-1`, `k=0..T-1`, `j=0..31`.

**layernorm**: per token `i`, mean accumulation `j=0..H-1`, variance
accumulation `j=0..H-1`, normalise write `j=0..H-1`.

**l2_norm**: sum-of-squares `j=0..n-1`, then normalise `j=0..n-1`.

**topk**: scan `i=0..N-1` ascending. Tie-break: equal scores → smaller
index wins.

**sliding_window (later-window-wins)**: windows dispatched ascending
`n=0,1,2,...`. Each window overwrites output slots `[start, end)`. For any
token in an overlap region the last window to write wins — equivalent to
assigning each token to the highest-indexed window that covers it.

## Q16.16 ABI

- All activations and weights are `i64` values encoding Q16.16 fixed-point.
  Integer part: `raw >> 16`. Fractional part: `raw & 0xFFFF`.
- Q32.32 accumulators are `i64` values with 32 fractional bits; narrowed
  to Q16.16 by `>> 16` after accumulation.
- `eps=1e-12` from BERT's LayerNorm is below the Q16.16 ULP (~1.5e-5) and
  is treated as 0. Zero-variance rows are clamped to 1 (defensive).
- The GELU approximation swaps erf-exact for the Hendrycks/Gimpel tanh form;
  max-abs error ≤ 0.001 pre-quantisation. Post-Q16.16 the tanh LUT provides
  bit-identical cross-substrate results.
- Softmax uses Q32.32 denominator accumulation (spec §3.1 decision).

## LUT Dependencies (A1.1)

These files import from `luts.*` (resolved at link time when A1.1 lands):

| File | Import |
|---|---|
| `layernorm_q16.mind` | `luts.sqrt_q16` (for `rsqrt_q16`) |
| `gelu_q16.mind` | `luts.tanh_q16` |
| `l2_norm_q16.mind` | `luts.sqrt_q16` (for `rsqrt_q16`) |
| `encode.mind` | `luts.softmax_q16` |

Until A1.1 commits, `mindc --emit-ir` produces a `const.i64 0` stub for
each cross-module import call site (one WARN per `use` statement). This is
expected behavior — the parse is clean, linkage resolves at `--emit-shared`.

## Sliding-Window Rule (spec §3.3)

Window: size=256, stride=192, overlap=64.

For T ≤ 256: single window, no framing (dense attention over all T tokens).

For T > 256: windows dispatched ascending (n=0, 1, 2, ...). Each window
covers `[n*192, min(n*192+256, T))`. Overlap tokens are overwritten by the
later window. `winning_window_index(t) = t / 192` gives the canonical
winning window for any token t (used by the A1.4 bit-identity harness).

## Build

Requires mindc v0.4.4 with `std-surface` and `cross-module-imports` features:

```
cd <mind-checkout>
cargo run --features "std-surface cross-module-imports" \
    --bin mindc -- <mind-nerve>/mind/kernels/<file>.mind --emit-ir
```

Full shared-library build (after A1.1 lands):

```
cd <mind-checkout>
cargo run --features "std-surface cross-module-imports mlir-build" \
    --bin mindc -- <mind-nerve>/mind/kernels/encode.mind \
    --emit-shared <mind-nerve>/python/mind_nerve/_native/libmind_nerve_encoder.so \
    --target=x86_64-unknown-linux-gnu
```

## Phase A1.2 Status

- [x] `matmul_q16.mind` — parse-clean
- [x] `batched_matmul_q16.mind` — parse-clean
- [x] `layernorm_q16.mind` — parse-clean (1 expected WARN: luts import stub)
- [x] `gelu_q16.mind` — parse-clean (1 expected WARN: luts import stub)
- [x] `l2_norm_q16.mind` — parse-clean (1 expected WARN: luts import stub)
- [x] `embedding_q16.mind` — parse-clean
- [x] `sliding_window.mind` — parse-clean
- [x] `topk_q16.mind` — parse-clean
- [x] `encode.mind` — parse-clean (2 expected WARNs: std.vec + luts import stubs)
- [ ] Integration test vs Q16.16 numpy reference — blocked on A1.1 LUT linkage
- [ ] p95 latency gate (≤ 50 ms, T=256, i7-5930K) — blocked on A1.3 wheel
- [ ] CUDA bit-identity — deferred to A2 (v0.4.1) per spec §3.2
