# mind-nerve/mind/exports — C-ABI Export Surface

Phase A1.3 — C-ABI entry points for libmind_nerve_encoder.so.

## ABI Version

**ABI version: 1** (encoded in the `mn_encoder_version()` string).

Breaking changes increment the ABI version. Consumers must verify the
version string at startup before calling any other entry point.

## Six Entry Points

```c
/* Allocate an encoder handle backed by a pre-quantised Q16.16 weight blob.
 * weights_blob: i64 address of a flat Q16.16 weight buffer (~142 MB).
 *               Produced offline by quantize_phase1_to_q16.py.
 * len:          byte length of the weight buffer.
 * Returns:      opaque i64 handle, or 0 on allocation failure.
 */
int64_t mn_encoder_init(int64_t weights_blob, int64_t len);

/* Run a full 12-layer BERT encoder forward pass (BGE-small-en-v1.5 fine-tune).
 * handle:    opaque handle from mn_encoder_init.
 * token_ids: flat i64 buffer of T token IDs (widened from int32).
 * n_tokens:  sequence length T, 1 ≤ T ≤ 512.
 * out_vec:   caller-provided i64 buffer of 384 elements; on return holds the
 *             Q16.16 L2-normalised embedding.
 * Returns 0 on success, -1 on null handle.
 */
int64_t mn_encoder_encode(int64_t handle, int64_t token_ids,
                          int64_t n_tokens, int64_t out_vec);

/* Compute dot-product scores: catalog × query_vector.
 * handle:     opaque handle (null-checked; not otherwise used).
 * qv:         i64 buffer of 384 Q16.16 elements (query embedding).
 * catalog:    i64 buffer of N×384 Q16.16 elements (pre-L2-normalised rows).
 * n_rows:     number of catalog entries N.
 * out_scores: caller-provided i64 buffer of N Q16.16 dot products.
 * Returns 0 on success, -1 on null handle.
 */
int64_t mn_encoder_score(int64_t handle, int64_t qv,
                         int64_t catalog, int64_t n_rows,
                         int64_t out_scores);

/* Select the top-K highest-scoring candidates from a score array.
 * scores:     i64 buffer of N Q16.16 scores.
 * n:          number of scores.
 * k:          number of results to return; K ≤ 32.
 * out_idx:    caller-provided i64 buffer of K; filled with catalog indices,
 *             sorted descending by score.
 * out_scores: caller-provided i64 buffer of K Q16.16 scores (sorted desc).
 * Returns 0 on success.
 */
int64_t mn_encoder_topk(int64_t scores, int64_t n, int64_t k,
                        int64_t out_idx, int64_t out_scores);

/* Release the handle and all scratch buffers.
 * handle: opaque handle from mn_encoder_init.
 * Returns 0.
 */
int64_t mn_encoder_free(int64_t handle);

/* Return the i64 address of a null-terminated build-id string.
 * Format: "mind-nerve-encoder/<api_ver>;mindc/<ver>;abi/<abi_ver>;q16.16"
 * Example: "mind-nerve-encoder/1;mindc/0.4.4;abi/1;q16.16"
 * The MIND heap stores one ASCII byte per i64 slot (stride 8 bytes).
 * Python ctypes consumers: use the _NativeRuntime.version() helper which
 * handles the stride-8 decode automatically.
 */
int64_t mn_encoder_version(void);
```

## Wire Protocol

### Q16.16 encoding

All activations, weights, and scores cross the FFI boundary as Q16.16
fixed-point integers packed inside i64 (int64_t) values:

```
float_value → q16 = int(round(float_value * 65536))
q16 → float  = q16 / 65536.0
```

No f32 is passed to or from the .so at any point.

### Buffer layout

All buffers are flat row-major `i64[]` arrays. Each `i64` element occupies
8 bytes (little-endian). Shapes:

| Buffer          | Shape       | Bytes          |
|----------------|-------------|----------------|
| token_ids      | (T,)        | T × 8          |
| out_vec / qv   | (384,)      | 3 072          |
| catalog        | (N, 384)    | N × 3 072      |
| out_scores     | (N,)        | N × 8          |
| out_idx        | (K,)        | K × 8          |
| out_scores (k) | (K,)        | K × 8          |

### Handle memory ownership

`mn_encoder_init` allocates all scratch buffers for the maximum sequence
length (256 tokens). The caller owns only:

- The weight blob (must remain valid until `mn_encoder_free`).
- All caller-provided output buffers.

Scratch buffers (activations, intermediate tensors) are owned by the handle.

### Thread safety

A single handle is **not thread-safe** (scratch buffers are not protected).
Multiple simultaneous requests require multiple handles sharing one weight blob.

## Sliding-Window Rule (T > 256)

For sequences longer than 256 tokens, the encoder uses the
"later-window-wins" rule (spec §3.3):

- Window size: 256 tokens, stride: 192 tokens, overlap: 64 tokens.
- Windows dispatched ascending (n = 0, 1, 2, ...).
- Each window's output overwrites the global buffer for tokens [start, end).
- The final window provides the CLS-token embedding for the whole sequence.

For T ≤ 256 this degenerates to a single dense attention pass, identical
to the Phase-1 PyTorch reference.

## Build

Requires mindc v0.10.2 (`std-surface` is default; project mode via
`mind/Mind.toml` — see `tests/mindc_gate.sh`):

```bash
./tools/build_native_encoder.sh
# or with an existing checkout:
MIND_CHECKOUT=<mind-checkout> ./tools/build_native_encoder.sh
```

The output `.so` is staged to `python/mind_nerve/_native/libmind_nerve_encoder.so`
alongside the native runtime `libmindnerve.so`. Both are bundled into the
wheel via `pyproject.toml` package-data.

## Status (re-assessed 2026-08-10)

- [x] `c_abi.mind` — all 6 pub fn symbols defined, compiles under mindc 0.10.2 project mode.
- [x] `_native.py` — ctypes binding for all 6 entry points.
- [x] `inference.py` — MIND_NERVE_BACKEND selector (native / pytorch); native is the default.
- [x] `tools/build_native_encoder.sh` — build pipeline scaffolding.
- [x] `libmind_nerve_encoder.so` — **SHIPPED 2026-05-19**: bundled in the
      published wheel with a real `encoder_weights.q16.bin` blob (closes
      roadmap task #59).
- [ ] Native link of the full kernel tree as one `.so` — blocked on mindc
      symbol-visibility/mangling (private helpers emit as global symbols in
      every object → multiple-definition at link). Tracked as a mind-repo
      follow-up from the 2026-08-10 audit.
- [ ] CUDA variant (`--target=cuda`) — the bit-identity harness's CUDA leg is
      an unimplemented sentinel (`CUDA_DEFERRED_TO_V0_4_1`); hardware is no
      longer a blocker (U1 RTX 3080 live). Task #57 stays open until the CUDA
      emit path exists and the SHA run is recorded.

## A1.4 harness status (re-assessed 2026-08-10)

The bit-identity harness (`tests/bit_identity/`) runs green today on the
pytorch and native backends (CI `bit-identity-cpu` job). Historical
blockers 1–2 below are cleared (the encoder `.so` ships; the runtime loads
`encoder_weights.q16.bin`); the CUDA leg remains a sentinel until the CUDA
emit path exists (task #57).

1. ~~`tools/build_native_encoder.sh` must succeed~~ — done; the wheel bundles
   the resulting `.so`.
2. `route_table.q16.bin` — the offline-quantized route table remains
   deferred; the native path scores against `route_table.npy` today.
3. The A1.1 LUT precision gate (≤ 1 ULP-eq error on 1M samples) must pass.
4. The A1.2 single-forward-pass bit-identity vs Q16.16 numpy reference must pass.
