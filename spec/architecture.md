# Architecture

Authoritative design document. Implementation in `src/` and `cli/` MUST match
this spec. Discrepancies are bugs against this document, not against the code.

## Functional contract

```
preselect : (Request, RouteCatalog, K) -> [RouteId; K]
```

- `Request` is a UTF-8 byte sequence, length ≤ 1024 tokens after BPE encoding.
  Requests longer than 1024 tokens are rejected at the wire handshake with
  `RequestTooLong`. 1024 covers ≥99% of realistic agent-CLI request lengths;
  longer requests should be split or summarised by the caller.
- `RouteCatalog` is a static set of `RouteId`s with associated route
  descriptors (skill descriptions, tool docstrings, agent capability prose).
  The catalog is hashed at load time; classification is conditioned on the
  catalog hash.
- `K` is the number of routes to return; default 5, max 64.
- Output is a deterministically ordered list of route IDs. Ordering is by
  relevance descending, tie-broken by `SHA-256(route_id) ascending`.

## Model shape

Encoder + direct scoring head. No decoder. Total parameter budget ≤ 4M
(encoder ~0.8M INT8 weights + route embedding table sized by
`|RouteCatalog|` × hidden — at 4400 routes that is ~1.1M Q16.16 values).

The architecture has been through two reductions to fit a strict CPU latency
budget on 4-core x86:

1. **Drop the decoder.** A decoder cross-attending to a fixed route table
   has the same expressive power as a single matmul between a pooled query
   vector and the route embedding table, at a fraction of the FLOPs.
2. **Shrink the encoder.** Fleet-consensus latency analysis showed 12-layer
   sliding-window encoders at hidden=384 cannot fit 30 ms p95 on commodity
   4-core x86 even with all SIMD optimizations applied. The encoder is
   reduced to 2 layers at hidden=256, paired with INT8 weight quantization
   and a window K/V overlap cache. This brings analytical worst-case
   latency at the 1024-token cap to ~16 ms, leaving half the budget for
   tokenization, scoring, top-K, and envelope SHA-256.

### Encoder

- 2 layers
- Hidden dimension 256
- 4 attention heads (head dim 64)
- **Sliding-window self-attention.** Window 256 tokens, stride 192 (overlap
  64). At the worst-case 1024-token request this is 5 windows × 256² scores
  instead of 1024² scores — a 4× FLOP reduction with no measurable loss on
  intent classification, because route relevance is dominated by local
  phrase semantics.
- **K/V overlap cache.** The 64-token overlap between adjacent windows
  re-uses K and V projections rather than recomputing them. Reduces
  per-layer KV-projection MACs by ~25%.
- **Fused QKV+O projection kernel.** Q, K, V projections are emitted as a
  single (3 × hidden² × seq) matmul rather than three separate matmuls,
  amortising load/store overhead. Output projection is fused into the same
  kernel sequence to keep activations in cache. mindc lowering enforces the
  fusion; reduction order is pinned.
- **No feed-forward sublayer** — attention + gated residual only.
- Pre-norm; gated residual replaces standard residual to compensate for the
  missing FFN's representational role.
- Window overlap-add: token positions inside the overlap region receive
  contributions from both adjacent windows. Reduction is sum (not mean) —
  pinning the reduction order is the compiler's responsibility, not a
  runtime decision (see [`numerics.md`](numerics.md)).
- Vocabulary: 32k BPE in Phase 1 (English-only). Phase 2 expands to a
  multilingual vocab (likely 48-64k) sized to cover the 12 Tier-1
  languages and the Tier-2 / Tier-3 fallbacks under
  [`quality_targets.md`](quality_targets.md) §"Multilingual language
  policy". Per-language eval gates are tracked there.

### Weight storage discipline

The inference path is **Q16.16 activations × INT8 weights**:

- Weights are stored as `i8` per-output-channel-quantized integers.
- A per-output-channel `i32` scale tensor (Q16.16) multiplies the integer
  product to recover Q16.16 activations.
- MAC primitive: `((act_i32 as i64) * (weight_i8 as i64)) * scale_i32`
  accumulated in `i64`, shifted `>> 16`, saturated to `i32`. The rounding
  mode is round-to-nearest-even, pinned by mindc into a single scalar
  primitive that emits identical bytes on every backend.
- This relaxes the "pure Q16.16 everywhere" framing to "Q16.16 in flight,
  INT8 on disk and on the multiplier weight side." Cross-arch bit-identity
  is preserved because the saturated-MAC primitive is the same bytes on
  every architecture.
- Per-channel scales contribute to the weight manifest hash, which
  contributes to `model_hash`. Tampering with a scale tensor breaks the
  attestation chain on first inference.

### Query pooling

After the encoder, the per-token hidden states are reduced to a single
`(hidden,)` query vector by mean pooling. Mean pool is chosen over [CLS]
because (a) sliding-window attention does not propagate a global token
cleanly across windows, and (b) mean pool is a single reduction with a
trivially pinned order.

### Direct scoring head

- Route embedding table: `(|RouteCatalog|, hidden)`, Q16.16. Loaded at
  catalog-hash time; bit-identical across runs of the same catalog.
- Score vector: `query @ route_table.T` — one matmul, `|RouteCatalog|`
  inner products.
- Top-K extraction: argpartition then exact sort on the K-slice, deterministic
  tie-break by `SHA-256(route_id) ascending` (see
  [`tests/unit/test_top_k.mind`](../tests/unit/test_top_k.mind)).
- The route embedding table is the only `|RouteCatalog|`-dependent parameter
  block. Adding routes does not require retraining the encoder; it requires
  populating new rows in the table from the same distillation procedure that
  produced the rest of the table.

## Numerical strategy

Activations in the inference path are Q16.16 throughout. Weights are INT8
(per-output-channel) with Q16.16 scale tensors; see "Weight storage
discipline" above. Softmax, layer norm, attention, mean pool, and the scoring
matmul are reduction-order pinned by the compile-time topology — no runtime
non-determinism. See [`numerics.md`](numerics.md) for the per-op reduction
table and the five mindc lint codes (`E_NERVE_001..005`) that enforce it.

Quantization-aware training is required; post-training quantization is
explicitly out of scope. Reference weights ship with quantization noise ≤
0.02 in top-5 accuracy vs the FP32 reference checkpoint used during training,
measured at INT8-weights / Q16.16-activations.

The Phase 1 training pipeline may run in FP32, but the INT8/Q16.16 inference
checkpoint is the canonical artifact. The FP32 reference checkpoint is
diagnostic only and is never distributed.

## Latency budget

Target: p95 ≤ 30 ms end-to-end on 4-core x86 CPU at single-batch, 1024-token
worst-case request. MIND-compiled SIMD codegen — no interpreter, no Python
loop overhead, no BLAS dependency. INT8×i32 → i64 multiply-accumulate maps
to AVX2 VPDPBUSD (x86) and NEON SDOT (ARM) intrinsics generated by
mind-runtime's lowering passes.

Decomposition (worst-case 1024-token request):

| Stage | Budget |
|---|---|
| Tokenize (32k BPE, byte fallback) | 2 ms |
| Encoder forward (2L sliding-window, INT8 weights, K/V cache) | 12 ms |
| Mean pool + direct scoring (4400-route catalog) | 5 ms |
| Top-K extraction | 1 ms |
| Attestation envelope (SHA-256 over 212-byte preimage) | 5 ms |
| Slack | 5 ms |
| **Total** | **30 ms** |

Encoder budget reflects 5 windows × 2 layers × (fused QKV+O projection +
per-window attention) at hidden=256, head_dim=64, with K/V overlap cache
amortising the 64-token overlap region. The matmul-dominated scoring step
is the next biggest stage because the 4400-route table is 4× larger than
the per-layer projection matrices.

Phase 1 may exceed budget on architectures other than x86-CPU; Phase 2 must
meet budget on ARM as well. CUDA and WebGPU are not budgeted on latency in
Phase 1 (throughput-mode, batch ≥ 32).

Typical-case is much cheaper. A 256-token request runs 1 window × 2 layers,
roughly 1/5 the encoder cost. The 30 ms budget is the worst case at the
1024-token cap; p50 on typical agent requests lands in the 5–8 ms range.

Requests longer than 1024 tokens are rejected at the wire handshake. The cap
is not negotiable at runtime; clients that need longer-context routing
should split or summarise upstream.

## Catalog hashing

`RouteCatalog` produces a `CatalogHash = SHA-256(canonical-serialization)`
where canonical serialization is:

```
for each route in routes sorted by SHA-256(route_id):
    emit length-prefixed route_id
    emit length-prefixed description SHA-256
```

The hash is computed at load and pinned into the attestation envelope for
every inference performed against this catalog. Two hosts with the same
catalog and the same model produce identical attestation chains.

## Attestation envelope

Every inference emits an envelope. The v2 layout is 212 bytes packed,
little-endian for all multi-byte integers. Authoritative byte-level contract
is `tests/unit/test_evidence.mind §1`; `src/evidence.mind` carries the
matching offset constants.

```
offset  size  field
  0     1     version             (= 2)
  1     1     entry_kind          (1=Inference, 2=ModelLoad, 3=CatalogLoad)
  2     2     wire_version        (u16 LE; numerics-pin major version)
  4     4     k                   (u32 LE; 1..=MAX_TOP_K)
  8     8     timestamp_ms        (i64 LE; monotonic, NOT wall-clock)
 16     1     architecture        (1=x86_64, 2=aarch64, 3=cuda)
 17     1     reserved            (MUST be 0)
 18     2     chain_reset_reason  (u16 LE: 0=Continuation, 1=ModelSwap,
                                   2=CatalogChanged, 3=ClockReset)
 20    32     model_hash          (SHA-256 of weights manifest)
 52    32     tokenizer_hash      (SHA-256 of tokenizer vocab + merges)
 84    32     catalog_hash        (SHA-256 of canonical RouteCatalog)
116    32     request_hash        (SHA-256 of request bytes)
148    32     result_hash         (SHA-256 of canonical(top_k))
180    32     chain_prev          (SHA-256 of previous envelope, zero if first)
====   ===
TOTAL  212
```

`chain_curr` is NOT stored in the envelope. It is computed at verification
time as `SHA-256(212-byte serialization)`, which lets the verifier detect
tampering with any field without an additional integrity slot. A v2-aware
verifier walks the chain by recomputing `chain_curr` for envelope *i* and
comparing to `chain_prev` of envelope *i+1*. The first envelope in a chain
has `chain_prev = 0`.

`architecture` and `entry_kind` enum values start at 1 so an uninitialised
zero byte is unambiguously invalid. `reserved` MUST be zero on construction
AND on deserialize — a verifier that sees non-zero reserved bytes refuses
the envelope and emits `ReservedByteNonZero`. This catches future-version
envelopes being forced through a v2 verifier.

`request_hash` MUST be non-zero in every valid envelope (SOTA-track #2:
input-fingerprinted attestation). An all-zero `request_hash` is rejected by
both `emit()` (construction gate) and `deserialize()` / `verify()` (inbound
gate) with `ZeroRequestHash`. This invariant ensures that every attested
envelope is bound to a concrete, non-sentinel request fingerprint — a hand-
crafted or replayed envelope with an unbound `request_hash` cannot pass
verification. `chain_prev = zeros32()` remains valid for the first envelope
in a chain (chain-start sentinel); the non-zero invariant applies only to
`request_hash`.

Phase 1 architecture enum covers `x86_64`, `aarch64`, `cuda`. Phase 2 extends
to `webgpu` (= 4) and `npu` (= 5) without bumping `version` — the u8 field
reserves all 256 slots for future backends.

## Bit-identity contract

For every `(model_hash, tokenizer_hash, catalog_hash, request_hash)` tuple,
the produced `result_hash` MUST be identical across:

- x86 CPU
- ARM CPU
- CUDA (any compute capability ≥ 7.0)
- WebGPU (any conformant implementation)
- NPU (any backend mind-runtime supports at NPU lowering)

`tests/bit_identity/` is the load-bearing test suite. CI gate fails if any
backend diverges by even one byte.

This contract holds against quantization-aware-trained Q16.16 weights only.
FP32 reference checkpoints are explicitly not bit-identical across
architectures and are diagnostic-only.

## Model-hash discipline

A single `model_hash` covers the full inference artefact: encoder weights,
route embedding table, BPE vocabulary, BPE merge rules, and the per-op
reduction-order pin table emitted by mindc. Any one of those changing
produces a new model_hash. A run that loads weights whose manifest hash
does not match `model_hash` refuses to start — there is no recovery path
that keeps the chain valid.

Per-tensor hashes are recorded in a manifest, and the manifest's SHA-256 is
the value compared at load time. This means a tampered weight tensor breaks
the manifest hash, which breaks the model_hash, which breaks the chain — a
single tampered byte at any layer of the artefact is detected before the
first inference runs.

### Neuron-hash aggregation rule

Each tensor entry in the manifest carries a `neuron_hash: [u8; 32]` field —
the SHA-256 of the tensor's Q16.16 byte layout (i32 little-endian per element,
row-major for matrices). The hashes are committed into the manifest aggregate
via the `build_manifest_preimage` function in `src/loader.mind`:

1. Entries are sorted ascending by tensor name (ASCII lexicographic order).
2. A fixed-layout preimage is built:
   - 4-byte magic `"MNPM"` (0x4D 0x4E 0x50 0x4D).
   - u32 LE entry count.
   - Per entry: u32 LE name length, name bytes, u32 LE rows, u32 LE cols,
     32-byte `neuron_hash`.
3. `manifest_aggregate = SHA-256(preimage)`.

The aggregate is the value embedded in the weights file `model_hash` field
and compared at load time. Because the sort is canonical and the preimage is
self-describing, any reordering of tensors in the catalog (e.g. by a
different build tool) that produces the same sorted order yields the same
aggregate.

## Per-neuron manifest

The public function `manifest_export(manifest: &TensorManifest) -> [u8]` in
`src/loader.mind` emits a deterministic UTF-8 JSON document:

```json
{
  "version": 1,
  "tensors": [
    {
      "name": "<tensor_name>",
      "shape": [<rows>, <cols>],
      "neuron_hash": "<64-hex-chars>"
    }
  ],
  "aggregate": "<64-hex-chars>"
}
```

Rules:
- Keys within each tensor object are emitted in the fixed order: `name`,
  `shape`, `neuron_hash`.
- The `tensors` array is sorted by tensor name (ascending, alphabetical).
- All hex strings are lowercase.
- No trailing comma after the last element in any array or object.
- The output is byte-identical on every backend for the same input manifest.

The `aggregate` field is the hex encoding of `TensorManifest::aggregate` —
the SHA-256 of the canonical `build_manifest_preimage` bytes. This is the
value that crosses into the MindLLM handshake as `mind_nerve_hash`.

Tensor naming convention (alphabetical = canonical order):
- `encoder.final_ln_bias`
- `encoder.final_ln_gain`
- `encoder.layer{NN}.ln_bias`   (NN = zero-padded two-digit layer index)
- `encoder.layer{NN}.ln_gain`
- `encoder.layer{NN}.residual_gate`
- `encoder.layer{NN}.wk`
- `encoder.layer{NN}.wq`
- `encoder.layer{NN}.wv`
- `encoder.layer{NN}.wo`
- `token_embedding`

## MindLLM cross-binding handshake

The handshake protocol is defined in `integrations/mindllm_attestation.mind`.
It produces a `BindingRecord` that allows an independent verifier to confirm
that a given mind-nerve model and a given MindLLM model were attested together
at a specific point in time.

### Protocol summary

```
binding_msg = SHA-256(mind_nerve_model_hash ++ mindllm_model_hash ++ nonce)
signature   = Ed25519_sign(private_key, binding_msg)
```

where `++` is byte concatenation and `mind_nerve_model_hash` is the
`aggregate` field from `manifest_export()`.

### Verifier algorithm (no private key required)

1. Assert that none of `mind_nerve_hash`, `mindllm_hash`, `nonce`,
   `signature`, or `signer_pubkey` is all-zero bytes.
2. Recompute `binding_msg = SHA-256(mind_nerve_hash ++ mindllm_hash ++ nonce)`.
3. Call `Ed25519_verify(signer_pubkey, binding_msg, signature)`.
4. Accept if and only if the signature is valid.

The verifier needs only:
- The published `BindingRecord` (200 bytes, format in
  `integrations/mindllm_attestation.mind §SERIALIZATION`).
- The signer's public key (32 bytes, from a trust anchor).
- A SHA-256 implementation and an Ed25519 verifier per RFC 8032.

### BindingRecord wire format (200 bytes)

```
offset  size  field
  0     4     magic "MNBA" (0x4D 0x4E 0x42 0x41)
  4     2     version u16 LE = 1
  6     2     reserved = 0
  8    32     mind_nerve_hash  (SHA-256 manifest aggregate)
 40    32     mindllm_hash     (SHA-256 MindLLM manifest aggregate)
 72    32     nonce            (caller-supplied, 32 bytes)
104    64     signature        (Ed25519, 64 bytes)
168    32     signer_pubkey    (Ed25519 public key, 32 bytes)
---
200 bytes total
```

`chain_curr` for this record is `SHA-256(200 bytes)`, analogous to the
attestation envelope chain discipline, and allows binding records to be
chained across model upgrades.

## What this architecture is not

- Not a small generative model. The scoring head emits discrete route IDs;
  there is no token-by-token output path.
- Not a vector database. The classifier emits discrete route IDs, not
  embeddings. Vector retrieval is a different problem; mind-nerve composes
  with vector retrieval upstream if the host wants it.
- Not a replacement for the calling LLM. mind-nerve decides which routes the
  calling LLM sees; the LLM still does the actual task.
- Not a tool execution engine. mind-nerve emits route IDs; the host calls the
  tool.

## Backwards-soft architecture switches (Phase 1, autoresearch IMPLEMENT)

12 architecture refinements have landed as compile-time switches in
`src/lib.mind` and the kernel/scoring modules. Every switch defaults to
a value that keeps the binary byte-identical to the pre-RFC build until
a calibrated reference checkpoint flips it. Flipping a switch is a
versioned architecture change (new `model_hash`), never a runtime knob.

| RFC | Switch (default) | Implemented in | Effect when flipped |
|---|---|---|---|
| RFC-001 | `WEIGHTS_VERSION_V2 = 2`; `GROUP_SIZE = 32` (v1 default) | `loader.mind` | Group-wise INT8 weight quantization with shared Q16.16 scales (AWQ-style). |
| RFC-002 | Catalog format v2 trailing prior block (v1 default) | `loader.mind`, `inference.mind` | Q16.16 per-route log-frequency / PMI prior added to logits before top-K. |
| RFC-003 | `STRIDE_LOW = 96`, `STRIDE_MID = 192`, `STRIDE_HIGH = 256` (thresholds disabled) | `encoder_kernels.mind`, `model.mind` | Content-fingerprinted three-way adaptive window stride. |
| RFC-005 | `HEAD_MASK_DEFAULT_ALL_ALIVE` (all heads alive) | `encoder_kernels.mind` | Compile-time bitmask zeros dead attention heads' contribution. |
| RFC-007 | `NUM_SINK_TOKENS = 0` | `encoder_kernels.mind` | StreamingLLM-style attention sinks at fixed absolute positions. |
| RFC-008 | `MATRYOSHKA_COARSE_DIM = ROUTE_EMBEDDING_DIM`, `K_COARSE_MULTIPLIER = 1` (cascade disabled) | `lib.mind`, `model.mind`, `inference.mind`, `top_k.mind` | Two-pass scoring: coarse dot over first `D'` dims → top-`αk` shortlist → full rerank. |
| RFC-009 | `POOLING_KIND = POOLING_KIND_MEAN` | `encoder_kernels.mind` | Learned single-latent-query attention pooling replaces mean-pool. |
| RFC-010 | `COSINE_SCORING_ENABLED = 0` | `lib.mind`, `model.mind`, `inference.mind` | Pooled query is L2-normalized before the scoring head (cosine sim). |
| RFC-011 | `ATTN_ALIBI_ENABLED = 0`; slopes `[1/4, 1/16, 1/64, 1/256]` in Q16.16 | `lib.mind`, `encoder_kernels.mind` | Per-head linear distance bias on attention scores (ALiBi). |
| RFC-012 | `QUERY_PREFIX_LEN = 0` (8 reserved slots) | `lib.mind`, `inference.mind` | Asymmetric query / passage prefix conditioning (INSTRUCTOR-style). |
| RFC-013 | `NORMALIZATION_KIND = NORMALIZATION_KIND_LAYERNORM` | `q16_16.mind`, `encoder_kernels.mind` | RMSNorm replaces LayerNorm at pre-norm + final-norm sites. |
| RFC-014 | `POOL_LATENT_QUERIES = 1` (single query, byte-identical to RFC-009 default-on) | `encoder_kernels.mind` | Multi-query latent attention pooling (NV-Embed-style, `r ≥ 2`). |

23 further RFCs were drafted by the loop and SKIPPED — they describe
training-time disciplines (loss formulations, hard-negative mining,
curriculum, distillation, EMA, SAM, R-Drop, ANCE refresh, GradCache,
RetroMAE pretraining, etc.) and are absorbed into the offline catalog-
builder's INT8 weight + Q16.16 scale bytes via `model_hash` /
`catalog_hash` without touching the inference surface.

The full draft index, including SKIP rationale per RFC and source-paper
citations, lives at `RFCs/INDEX.md`.

## Open questions, Phase 1

These are explicitly unresolved and will be answered by Phase 1 implementation:

1. Whether 2 encoder layers with no FFN sublayer are sufficient at the
   sliding-window receptive field at calibrated accuracy targets, or
   whether 3-4 layers become necessary on long requests where intent
   depends on cross-window context
2. Whether the route embedding table benefits from frozen vs trainable
   embeddings at scale (1000+ routes vs 100 routes)
4. Whether attestation envelope SHA-256 dominates p99 latency at large
   catalog sizes (10,000+ routes) and needs a hash cache
5. Whether window 256 / stride 192 is the right local-receptive-field
   tradeoff or whether 512/384 is needed for legal/medical request
   distributions where intent words cluster further apart than agent-CLI
   request distributions
