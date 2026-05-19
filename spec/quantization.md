# Quantization

Authoritative specification for the offline FP32 → Q16.16 quantizer that
produces `route_table.q16.bin` (and, when applicable,
`encoder_weights.q16.bin`). Implementations in `tools/` and consumers in
`python/mind_nerve/` MUST match this document. Discrepancies are bugs
against this spec, not against the code.

This spec is `quantization.md@v1`. Quantizer version `1.0`.

## Section 1 — Scope

The Phase 1 PyTorch trainer (`python/mind_nerve/mind_train.py`) emits an
FP32 SentenceTransformer checkpoint plus a precomputed catalog at
`route_table.npy` (shape `(N_routes, hidden_dim)`, dtype `float32`). The
native MIND encoder (`mind/kernels/encode.mind`) consumes Q16.16 fixed-
point weight tables via the heap ABI defined in
[`spec/numerics.md`](numerics.md) §4.

The offline quantizer bridges these two artifacts. It is **offline**
(not on the inference hot path), runs once per checkpoint, and emits a
runtime artifact that ships out-of-band (the binary is large and never
committed to git).

In scope:
- `route_table.q16.bin` — concatenated Q16.16 per-row catalog
  embeddings, row-major, i64 LE per element.
- `route_table.q16.meta.json` — reproducibility metadata.
- (Optional) `encoder_weights.q16.bin` — when a checkpoint exposes a
  separable Q16.16-quantizable encoder head. Phase 1 does NOT emit
  this; reserved for Phase 6.x once `mn_encoder_init` consumes a real
  weights blob.

Out of scope:
- Online quantization on the inference path. The inference path
  consumes already-quantized bytes; it never quantizes f32.
- Quantization-aware training. The Phase 1 trainer runs FP32; the
  quantizer is a post-training step. Quantization-aware training is
  tracked separately under `spec/architecture.md §"Numerical strategy"`.

## Section 2 — Quantization scheme

### 2.1 Fixed-point format

- **Format**: Q16.16 signed two's-complement.
- **Bit width (value)**: 32 bits. Range `[INT32_MIN, INT32_MAX]` =
  `[-2_147_483_648, 2_147_483_647]`.
- **Float interpretation**: `f = q / 2^16`. Q16.16 `1.0` = integer
  `65536`. Q16.16 `MAX` ≈ `32767.999984741`.
- **Bit width (on-disk encoding)**: each Q16.16 value is widened to
  `int64` little-endian on disk. The high 32 bits are the
  sign-extension of the low 32 bits. This matches the MIND heap ABI's
  i64-only `__mind_load_i64` (see `mind/kernels/encode.mind`) without
  requiring a byte-swap or repack at load time.
- **Endianness**: little-endian for all multi-byte integers. Big-
  endian hosts are out of scope for Phase 1.
- **Alignment**: 8-byte natural alignment per element. The file size
  is exactly `N_rows * hidden_dim * 8` bytes.

### 2.2 Scale

- `SCALE = 2^16 = 65536`.
- A float `f` becomes integer `q = round(f * SCALE)`.

The scale is fixed. Per-tensor scales, per-channel scales, and learned
scales are explicitly out of scope for the route table (the table is a
single tensor whose dynamic range is bounded by L2-normalized
embeddings, `|f| ≤ 1.0`). A future per-output-channel INT8 weight
encoder ([`spec/architecture.md §"Weight storage discipline"`](architecture.md))
will own its own scale tensor; that path is not this quantizer's
responsibility.

### 2.3 Rounding rule

**Round-half-to-even (banker's rounding).** Implemented as
`numpy.round(x, 0)` on the `float64` product `x = f * SCALE` with an
explicit destination buffer; the result is cast to `int64` via
`astype(np.int64, casting='unsafe')` *after* the round, *after* the
saturation clamp in §2.4.

Round-half-to-even is the IEEE-754 default rounding mode and the only
rule NumPy's `np.round` implements deterministically across platforms.
Truncation (`int()`), round-half-away-from-zero, and Python's built-in
`round()` (which is banker's-rounding only for floats but truncates for
integers and exhibits floating-point representation artifacts) are all
forbidden in this quantizer.

### 2.4 Saturation

If the rounded integer exceeds the `[INT32_MIN, INT32_MAX]` range,
clamp to the nearest boundary:

```
q = clip(round(f * SCALE), INT32_MIN, INT32_MAX)
```

Saturation matches the inference-path `q16_mul` saturation policy (see
`spec/numerics.md` §4). A quantized value at the boundary will, on the
hot path, produce a saturated multiply rather than wrapping silently.

### 2.5 Determinism

The quantizer MUST be deterministic in the strong sense:

1. **Same machine, two runs, same input** → byte-identical output.
2. **Same input, two machines (x86 / ARM)** → byte-identical output.

Determinism is achieved by:
- Performing the multiply in `float64`, never in `float32` (the
  intermediate product has 52 bits of mantissa; SCALE is exactly
  representable; the round-half-to-even rule has a single answer for
  every finite `float64` input).
- Using `numpy.round` (which calls libm `rint` under FE_TONEAREST,
  pinned by NumPy's `_MM_SET_ROUNDING_MODE` initialization) rather
  than Python's built-in `round` (which depends on the CPython
  build's rounding helpers).
- Casting to `int64` only after the round and clamp, never before.
- Iterating tensors in canonical (sorted-by-name) order in the meta
  JSON, so the `aggregate` SHA-256 is reproducible across runs.

A bit-identity test (`tests/python/test_quantize_phase1.py`) quantizes
the same input twice and asserts byte-identical output. This is the
regression gate.

## Section 3 — File layout

### 3.1 `route_table.q16.bin`

Flat binary blob. No header.

```
+--------------------------------------------------------------+
|  row[0]: hidden_dim × int64 LE  (Q16.16 widened to i64)      |
|  row[1]: hidden_dim × int64 LE                                |
|  ...                                                          |
|  row[N-1]: hidden_dim × int64 LE                              |
+--------------------------------------------------------------+
```

- Row-major (C-order): row `i`, column `j` lives at byte offset
  `(i * hidden_dim + j) * 8`.
- File size: `N_rows * hidden_dim * 8` bytes exactly.
- No padding rows, no trailing magic, no checksum embedded in the
  file. The SHA-256 of the file bytes lives in the meta JSON; a
  consumer that wants integrity validation reads the meta first.

This layout is consumed verbatim by `mn_encoder_init` when it receives
the blob's heap address (see `python/mind_nerve/_native.py`
`_NativeRuntime.init`) and by the catalog scoring path in
`python/mind_nerve/inference.py`.

### 3.2 `route_table.q16.meta.json`

Deterministic UTF-8 JSON document. Keys are emitted in fixed order; the
output is byte-identical for any given input.

```json
{
  "schema_version": 1,
  "kind": "mind_nerve.quantize.route_table",
  "quantizer_version": "1.0",
  "n_rows": 4400,
  "hidden_dim": 384,
  "scale": 65536,
  "rounding": "half_to_even",
  "saturation": "int32",
  "dtype_disk": "int64_le",
  "dtype_value": "int32_q16_16",
  "byte_size": 13516800,
  "sha256": "<64-hex-chars>",
  "source": {
    "catalog_npy_path": "<path-as-string>",
    "catalog_npy_sha256": "<64-hex-chars>",
    "catalog_npy_dtype": "float32",
    "checkpoint_path": "<path-as-string or null>",
    "checkpoint_hash": "<64-hex-chars or null>"
  },
  "stats": {
    "min_q16": -65536,
    "max_q16": 65536,
    "saturated_count": 0
  },
  "produced_at_iso": "<RFC-3339 UTC>",
  "produced_by": "mind-nerve quantize"
}
```

Rules:
- Top-level keys in fixed order: `schema_version`, `kind`,
  `quantizer_version`, `n_rows`, `hidden_dim`, `scale`, `rounding`,
  `saturation`, `dtype_disk`, `dtype_value`, `byte_size`, `sha256`,
  `source`, `stats`, `produced_at_iso`, `produced_by`.
- Hex strings lowercase, 64 characters, no `0x` prefix.
- `produced_at_iso` is the only non-deterministic field. To compare
  two meta JSONs for content-equality the consumer SHOULD ignore
  `produced_at_iso`. The `sha256` of the binary covers all the
  semantically meaningful bytes.
- No trailing comma after the last element of any array or object.
- Pretty-printed with `indent=2` and a trailing newline so it is
  human-readable in a git diff (the JSON itself is not committed; the
  pretty form is for downstream inspection).

### 3.3 Hash gate

The meta JSON's `sha256` field is the SHA-256 of the `.bin` file
bytes, lowercase hex. A consumer that loads `route_table.q16.bin`
SHOULD recompute the SHA-256 and reject the blob if it differs. The
hash is the integrity gate; the quantizer is the producer.

This hash does **not** participate in the inference-path `model_hash`
chain (see `spec/architecture.md §"Model-hash discipline"`). The
inference path independently rehashes the catalog at load time; the
meta SHA-256 is for human/CI verification of the quantizer output.

## Section 4 — Round-trip semantics

The pure-Python helpers `f32_to_q16(f)` and `q16_to_f32(q)` in
`tools/quantize_phase1_to_q16.py` MUST satisfy:

1. **Quantization error.** For any `f` in the L2-normalized embedding
   range `[-1.0, 1.0]`:
   ```
   abs(q16_to_f32(f32_to_q16(f)) - f) < 2 * 2^-16  ≈ 3.0518e-5
   ```
   This is the standard fixed-point quantization error (`2^-(fractional_bits+1)`
   maximum half-LSB). The factor of two covers the round-half-to-even
   resolution.

2. **Idempotence on the integer side.** For any integer `q` in
   `[INT32_MIN, INT32_MAX]`:
   ```
   f32_to_q16(q16_to_f32(q)) == q
   ```
   This is exact: integer → float → integer with no rounding loss in
   the round trip, because every Q16.16 integer is exactly
   representable in `float64`.

3. **Saturation symmetry.** `f32_to_q16(2.0e6)` saturates to
   `INT32_MAX`; `f32_to_q16(-2.0e6)` saturates to `INT32_MIN`.

These invariants are verified at the unit-test level by
`tests/python/test_quantize_phase1.py`.

## Section 5 — Bit-identity gate

The quantizer is the producer of an inference-path artifact. Two
quantizations of the same input on the same machine MUST produce
byte-identical `.bin` outputs. This is the **Phase 6.2
reproducibility claim**.

The test (`tests/python/test_quantize_phase1.py`) runs:

1. Generate a synthetic FP32 catalog `(100, 384)` from a seeded NumPy
   PRNG.
2. Save to `route_table.npy`.
3. Invoke the quantizer twice with different output directories.
4. Assert `sha256(bin_a) == sha256(bin_b)`.

A future cross-arch test (Phase 7) will run the same fixture on x86
and ARM and assert the same SHA-256. Phase 1 ships the same-machine
gate only; the cross-arch claim is a forward commitment.

## Section 6 — CLI surface

`mind-nerve quantize` is the single user-facing entry point.

```
mind-nerve quantize \
    --input    <pytorch-checkpoint-dir or :none:>  \
    --catalog  <route_table.npy>                    \
    --output   <output-dir>                         \
    [--hidden-dim 384]                              \
    [--dry-run]
```

Behaviour:

- `--catalog` is required. The catalog `.npy` file is the
  authoritative input. The Phase 1 trainer always emits one.
- `--input` is OPTIONAL. When supplied, the path is hashed into the
  meta JSON's `source.checkpoint_hash` field (SHA-256 over file bytes
  in sorted-path order, matching `mind_train._compute_checkpoint_hash`).
  When omitted or the literal token `:none:`, `source.checkpoint_*`
  fields are `null`. The Phase 1 quantizer does NOT read the
  PyTorch weights — the catalog `.npy` already carries the encoder
  output. Future encoder-weight quantization will read the checkpoint.
- `--output` defaults to `$MIND_NERVE_RUNTIME_DIR` if set, else
  `~/.cache/mind-nerve/q16/`. The directory is created with mode
  `0700` if absent.
- `--hidden-dim` defaults to `384` (BGE-small-en-v1.5). If the
  catalog's column count differs, the quantizer rejects with a clear
  error.
- `--dry-run` prints the meta JSON to stdout and exits without
  writing.

Exit codes: `0` on success, `1` on runtime error (missing input, bad
shape), `2` on argument error.

## Section 7 — Versioning

This spec is `quantization.md@v1`. The `quantizer_version` field in the
meta JSON is `1.0`.

### Major-version-bump changes (incompatible with `v1`)

- Changing `SCALE` away from `2^16`.
- Changing the rounding rule away from round-half-to-even.
- Changing the saturation range away from `int32`.
- Changing the on-disk encoding (e.g. switching from i64 LE to i32 LE
  packed, or adding a header).

A bump invalidates every `route_table.q16.bin` produced by v1. The
runtime loader (`python/mind_nerve/inference.py`) MUST refuse to load
a v2 blob with a v1 meta JSON.

### Non-breaking changes (still `v1`)

- Adding fields to the meta JSON (consumers ignore unknown keys).
- Adding new optional CLI flags with safe defaults.
- Adding new optional output files (e.g. `encoder_weights.q16.bin`).
