# Bit-identity test suite

The load-bearing correctness gate. Runs fixed-input fixtures against every
in-scope backend and verifies that all backends produce the same masked output
bytes — byte for byte.

Phase 1 exit criterion: "One byte of divergence fails the build."

## Scope by phase

| Phase | Backends in scope |
|---|---|
| Phase 1 | x86 CPU, CUDA |
| Phase 2 | x86 CPU, ARM CPU, CUDA, WebGPU, NPU |

A backend not in scope for the current phase is allowed to diverge. A
backend in scope that diverges by even one byte fails the build.

## Directory layout

```
tests/bit_identity/
├── README.md              this file
├── run.sh                 CI entry point
├── gen_fixtures.py        deterministic fixture generator (stdlib only)
├── verify.py              divergence diagnostic tool (stdlib only)
└── fixtures/
    ├── MANIFEST           machine-readable (request, catalog, golden) list
    ├── request_001.mic2   16 tokens, k=5
    ├── request_002.mic2   256 tokens, k=10
    ├── request_003.mic2   1024 tokens, k=64
    ├── catalog_44.bin     44-route catalog, deterministic Q16.16 embeddings
    ├── catalog_440.bin    440-route catalog
    ├── catalog_4400.bin   4400-route catalog
    └── expected/
        ├── request_001_catalog_44.sha256
        ├── request_001_catalog_440.sha256
        ├── request_001_catalog_4400.sha256
        ├── request_002_catalog_44.sha256
        ├── request_002_catalog_440.sha256
        ├── request_002_catalog_4400.sha256
        ├── request_003_catalog_44.sha256
        ├── request_003_catalog_440.sha256
        └── request_003_catalog_4400.sha256
```

## What gets hashed

The harness does NOT hash the raw stdout verbatim. Before hashing, it zeros
two envelope fields that are intentionally backend-specific:

- `timestamp_ms` (envelope bytes 8-15, 8 bytes): wall-clock value, not
  bit-identical across runs. The harness pins this via
  `MIND_NERVE_TEST_INJECT_MS` (default: 1000000).
- `architecture` (envelope byte 16, 1 byte): encodes `x86_64=1 / aarch64=2 /
  cuda=3`. Naturally differs per backend.

The masked frame SHA-256 must be identical across every in-scope backend.

Result fields that MUST be identical:
- `result_hash` (which route IDs were returned, in which order)
- All score bytes for every returned route
- `model_hash`, `tokenizer_hash`, `catalog_hash`, `request_hash`

## Current status (Phase 1.2)

The binary has not been built yet. `mindc` toolchain bringup is Phase 1.3.

`run.sh` exits 2 with the message:

    [bit-id] BINARY NOT BUILT: cpu binary not found at: ...
    [bit-id] Bit-identity check SKIPPED — binary not built.
    [bit-id] This is expected at Phase 1.2 (toolchain bringup).
    [bit-id] CI FAILS FAST here by design — fix: build the binary.

This is the correct and expected behaviour. CI fails non-zero, which is
right — the gate must not silently pass when the binary is absent.

## Running the gate

### Normal gate mode (CI)

```sh
# From repo root:
bash tests/bit_identity/run.sh
```

Requires:
- `./mind-nerve-cpu` executable (compiled for x86/ARM CPU)
- `./mind-nerve-cuda` executable (compiled for CUDA) — only if `nvidia-smi` succeeds
- `./fixtures/model.weights` — or set `MIND_NERVE_MODEL` env var
- All golden hashes populated in `fixtures/expected/`

Exits 0 iff all backends agree with golden and pairwise.
Exits 1 on any divergence.
Exits 2 if the binary is not built.
Exits 4 if fixture files are missing.

### Single-backend mode (development)

```sh
# Run only the CPU backend, skip pairwise comparison:
bash tests/bit_identity/run.sh --backend cpu

# Run only CUDA:
bash tests/bit_identity/run.sh --backend cuda
```

In single-backend mode, the harness still compares against the golden hash
but skips pairwise cross-backend comparison.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MIND_NERVE_CPU` | `./mind-nerve-cpu` | Path to x86/ARM cpu binary |
| `MIND_NERVE_CUDA` | `./mind-nerve-cuda` | Path to CUDA binary |
| `MIND_NERVE_ARM` | `./mind-nerve-arm` | Path to ARM binary |
| `MIND_NERVE_MODEL` | `fixtures/model.weights` | Path to weights file |
| `MIND_NERVE_TEST_INJECT_MS` | `1000000` | Pinned timestamp (ms) |

## Populating golden hashes (after the binary builds)

Run the gate once in generate mode against the reference CPU build:

```sh
bash tests/bit_identity/run.sh --generate-golden
```

This runs every (request, catalog) pair against the CPU backend and writes
the masked SHA-256 into `fixtures/expected/*.sha256`. Then commit:

```sh
git add tests/bit_identity/fixtures/expected/
git commit -m "test(bit-id): populate golden hashes from reference cpu build"
```

After the golden hashes are committed, subsequent runs of `run.sh` (without
`--generate-golden`) compare every backend against the locked hashes.

Golden hashes must ONLY be updated when the model weights, tokenizer, or
inference logic changes — and only after an explicit review that the change
is intentional, not a regression. The commit message must call out which
artifact changed and why.

## Divergence diagnostics

When the gate fails, use `verify.py` to understand WHERE in the frame the
bytes diverged:

```sh
# Inspect a single frame:
python3 tests/bit_identity/verify.py /tmp/frame_cpu.bin

# Compare two frames (shows byte-level diff after masking):
python3 tests/bit_identity/verify.py /tmp/frame_cpu.bin /tmp/frame_cuda.bin
```

Output includes:
- All k route IDs (first 16 bytes hex) and scores (Q16.16 as int and float)
- All attestation envelope fields (decoded)
- `chain_curr` computed as SHA-256(212-byte envelope)
- First diverging byte with field name, context hex, and total divergence count

The most common root causes for divergence:
1. Reduction-order not pinned: floating-point accumulation order differs by
   backend. Fix: enforce the mindc lint rules `E_NERVE_001..005`.
2. Tie-break non-determinism: SHA-256(route_id) sort order not enforced.
   Fix: `tests/unit/test_top_k.mind`.
3. Softmax polynomial instead of lookup table: polynomial approximation
   diverges on different FPUs. Fix: use the lookup table implementation.
4. Scale tensor loaded in wrong byte order: per-channel INT8 scales applied
   in LE vs BE order. Fix: explicit LE decode at load time.

## Regenerating fixtures

Fixture content is deterministic from fixed seeds in `gen_fixtures.py`.
Changing a seed invalidates all committed golden hashes and requires a
full golden refresh cycle. Do not change seeds without a planned refresh.

```sh
python3 tests/bit_identity/gen_fixtures.py
```

The `.mic2` request files and catalog `.bin` files are committed to the
repository. They must NOT be generated at CI time.

## Fixture format reference

### .mic2 request files

mic@2 text frame, wire protocol from `cli/main.mind`:

```
mic@2/mind-nerve/preselect
model: __MIND_NERVE_MODEL__
catalog: __MIND_NERVE_CATALOG__
k: <k>
tokens: <csv of u32 BPE token IDs>
.
```

The `model:` and `catalog:` path sentinels are replaced at runtime by
`run.sh` using `MIND_NERVE_MODEL` and `MIND_NERVE_CATALOG` env vars
(or the fixture catalog path directly).

### catalog .bin files

```
[0:4]    magic "MNC1"
[4:8]    route_count (u32 LE)
per route:
  [0:32]   route_id = SHA-256("route_{seq:06d}")
  [32:36]  embedding_dim (u32 LE, always 256)
  [36:36+256*4]  256 x i32 LE Q16.16 embedding values
                 drawn from N(0, 0.02), seed FEED_FACE_CA7A
total per route: 1060 bytes
total file:      8 + n_routes * 1060 bytes
```

### golden hash files (fixtures/expected/*.sha256)

Single line: 64-character lowercase hex SHA-256 digest, newline-terminated.
Contains "PENDING ..." until populated by `--generate-golden`.
