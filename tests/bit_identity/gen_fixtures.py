#!/usr/bin/env python3
"""
tests/bit_identity/gen_fixtures.py

Generates deterministic fixture files for the cross-arch bit-identity harness.
Produces:
  fixtures/request_001.mic2   — 16 tokens, k=5
  fixtures/request_002.mic2   — 256 tokens, k=10
  fixtures/request_003.mic2   — 1024 tokens, k=64
  fixtures/catalog_44.bin     — 44-route catalog, deterministic Q16.16 embeddings
  fixtures/catalog_440.bin    — 440-route catalog
  fixtures/catalog_4400.bin   — 4400-route catalog
  fixtures/model.weights      — deterministic INT8/Q16.16 encoder weights bundle

All outputs are bit-for-bit reproducible from the fixed seeds below. Every
value is drawn from a pure-integer xorshift64 PRNG — NO floating point is used
anywhere in a fixture byte, because the whole point of this harness is
byte-identity and IEEE-754 rounding is exactly the thing MIND removes. No
numpy. No third-party libraries. Pure stdlib.

────────────────────────────────────────────────────────────────────────────
PORTED WIRE FORMATS (2026-08-17)

These fixtures match the PORTED loader in src/loader.mind + src/inference.mind
(the raw-i64 kernel-ABI rewrite that now builds to a real native ELF). The
pre-port fixture layouts (magic + bare u32 route_count, no catalog hash, no
external-id trailer) are dead — the ported parser reads offset 4 as a u16
version, requires a canonical SHA-256 catalog hash at offset 16 and a UTF-8
external-id trailer, and validates a full weights bundle. A stale fixture trips
LD_ERR_VERSION / LD_ERR_HASH / LD_ERR_TRUNCATED and the CLI exits 7.

Catalog (.bin) — consumed by loader_parse_catalog (src/loader.mind). v1
(has_prior = 0), little-endian throughout:
  [0:4]    magic "MNC1"                (0x4D 0x4E 0x43 0x31)
  [4:6]    u16 version = 1
  [6:8]    u16 reserved = 0
  [8:12]   u32 num_routes
  [12:16]  u32 embedding_dim = 256     (== ROUTE_EMBEDDING_DIM)
  [16:48]  32-byte canonical catalog hash (see catalog_preimage below)
  [48:..]  route_ids           num_routes * 32 bytes (unique; A3 gate)
  [..]     embeddings          num_routes * 256 * i32-LE  (Q16.16)
  (v2 only) route_prior        num_routes * i32-LE        — NOT emitted (v1)
  [..]     external-id trailer per route: u16-LE len + len UTF-8 bytes
  [len-4:] u32-LE trailer_off  (== start of the trailer == prior_end)

Canonical catalog hash preimage (loader_parse_catalog, rebuilt from FILE
bytes, route order 0..num_routes, NOT sorted):
  for each route s:
    u32-LE 32 || id[s] (32) || u32-LE (edim*4) || embedding row bytes (edim*4)
  SHA-256 of the concatenation → written at offset 16.

Weights (.weights) — consumed by loader_parse_weights (src/loader.mind). v1
(groupwise = 0), little-endian throughout:
  [0:4]    magic "MNW1"                (0x4D 0x4E 0x57 0x31)
  [4:6]    u16 version = 1
  [6:8]    u16 reserved = 0
  [8:40]   32-byte model_hash          (opaque; MODEL_HASH_BIND_ENABLED = 0)
  [40:72]  32-byte tokenizer_hash      (opaque)
  [72:76]  u32 encoder_layers = 2      (== ENCODER_LAYERS)
  [76:80]  u32 encoder_hidden = 256    (== ENCODER_HIDDEN)
  [80:..]  ENCODER_LAYERS x layer block, each:
             wq,wk,wv,wo   4 * h*h  INT8  (row-major, per output channel)
             scales        4 * h    i32-LE Q16.16 (v1: one scale / out channel)
             ln_gain       h        i32-LE Q16.16
             ln_bias       h        i32-LE Q16.16
             residual_gate h        i32-LE Q16.16
  [..]     final_ln_gain   h        i32-LE Q16.16
           final_ln_bias   h        i32-LE Q16.16
  [..]     token_embedding VOCAB_SIZE * h  i32-LE Q16.16

model_hash binding is OFF (MODEL_HASH_BIND_ENABLED = 0), so model_hash /
tokenizer_hash are opaque deterministic bytes — the loader copies them out and
does not recompute them.

The .mic2 request format mirrors cli/main.mind's mic@2 wire protocol exactly:
  mic@2/mind-nerve/preselect
  model: <placeholder>
  catalog: <placeholder>
  k: <k>
  tokens: <csv of u32 token ids>
  .
model: and catalog: paths are placeholder sentinels the harness substitutes at
runtime via run.sh's sed (MIND_NERVE_MODEL / the catalog path per manifest pair).
"""

import hashlib
import os
import struct

# ---------------------------------------------------------------------------
# Fixed seeds — NEVER change these. Changing a seed invalidates all golden
# hashes committed in fixtures/expected/ and requires a full golden refresh.
# ---------------------------------------------------------------------------

SEED_TOKENS_001 = 0xDEAD_BEEF_0001
SEED_TOKENS_002 = 0xDEAD_BEEF_0002
SEED_TOKENS_003 = 0xDEAD_BEEF_0003
SEED_CATALOG = 0xFEED_FACE_CA7A
SEED_WEIGHTS = 0x5EED_10AD_0001

# Architecture constants — MUST match src/lib.mind. A mismatch trips
# LD_ERR_SHAPE (embedding_dim / encoder shape) in the loader.
VOCAB_SIZE = 32768          # src/lib.mind VOCAB_SIZE (embedding table rows)
TOKEN_VOCAB = 32000         # token-id draw bound (< VOCAB_SIZE, safe index)
HIDDEN_DIM = 256            # ENCODER_HIDDEN == ROUTE_EMBEDDING_DIM
ENCODER_LAYERS = 2          # ENCODER_LAYERS
GROUP_SIZE = 32             # GROUP_SIZE (v2 groupwise scales — unused here)

# Wire versions (v1 catalog, v1 weights — the non-groupwise / no-prior form).
CAT_VERSION = 1
W_VERSION = 1

# Q16.16 scale: 2^16.
Q16_ONE = 65536

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
EXPECTED_DIR = os.path.join(FIXTURE_DIR, "expected")

CATALOG_SIZES = [44, 440, 4400]
REQUESTS = [
    ("request_001", 16, 5, SEED_TOKENS_001),
    ("request_002", 256, 10, SEED_TOKENS_002),
    ("request_003", 1024, 64, SEED_TOKENS_003),
]


# ---------------------------------------------------------------------------
# Minimal deterministic PRNG (xorshift64, stdlib only, pure integer)
# ---------------------------------------------------------------------------


class Xorshift64:
    """
    Xorshift64 PRNG. Deterministic, fast, no stdlib dependency, pure integer.
    Period: 2^64 - 1. Not cryptographic.
    """

    def __init__(self, seed: int):
        # Seed must be non-zero; fold to 64-bit unsigned.
        self._state = (seed & 0xFFFF_FFFF_FFFF_FFFF) or 1

    def next_u64(self) -> int:
        x = self._state
        x ^= (x << 13) & 0xFFFF_FFFF_FFFF_FFFF
        x ^= (x >> 7) & 0xFFFF_FFFF_FFFF_FFFF
        x ^= (x << 17) & 0xFFFF_FFFF_FFFF_FFFF
        self._state = x & 0xFFFF_FFFF_FFFF_FFFF
        return self._state

    def next_u32_bounded(self, upper: int) -> int:
        """Return a uniform u32 in [0, upper). Upper must be > 0."""
        assert upper > 0
        # Rejection sampling to avoid modulo bias.
        threshold = (1 << 64) - ((1 << 64) % upper)
        while True:
            v = self.next_u64()
            if v < threshold:
                return v % upper

    def next_i32_in(self, lo: int, hi: int) -> int:
        """Return a uniform integer in [lo, hi] (inclusive), pure integer."""
        span = hi - lo + 1
        return lo + self.next_u32_bounded(span)


# ---------------------------------------------------------------------------
# Deterministic integer byte streams (no floats, no PRNG state shared across
# sections — every section is reproducible from its own fixed seed).
# ---------------------------------------------------------------------------


def i32_stream_bytes(seed: int, count: int, lo: int, hi: int) -> bytes:
    """`count` little-endian i32 values uniformly in [lo, hi]."""
    rng = Xorshift64(seed)
    out = bytearray(count * 4)
    for i in range(count):
        struct.pack_into("<i", out, i * 4, rng.next_i32_in(lo, hi))
    return bytes(out)


def u8_stream_bytes(seed: int, count: int) -> bytes:
    """`count` raw bytes (INT8 weights interpreted two's-complement)."""
    rng = Xorshift64(seed)
    out = bytearray(count)
    for i in range(count):
        out[i] = rng.next_u64() & 0xFF
    return bytes(out)


def tiled_i32_bytes(seed: int, count: int, lo: int, hi: int,
                    pattern_count: int = 65536) -> bytes:
    """
    `count` i32-LE values, produced by tiling a deterministic pattern block.
    Used for the 32 MiB token-embedding table so generation stays fast while
    remaining bit-for-bit reproducible. Tiling only affects which row content
    repeats; every byte is still fixed by `seed`.
    """
    pc = min(pattern_count, count)
    pat = i32_stream_bytes(seed, pc, lo, hi)
    total_bytes = count * 4
    reps = (total_bytes + len(pat) - 1) // len(pat)
    return (pat * reps)[:total_bytes]


def opaque_hash(label: str) -> bytes:
    """A fixed 32-byte value (model_hash / tokenizer_hash are opaque here)."""
    return hashlib.sha256(label.encode("utf-8")).digest()


# ---------------------------------------------------------------------------
# Route ID derivation
# ---------------------------------------------------------------------------


def route_id_bytes(seq: int) -> bytes:
    """
    Derive the 32-byte RouteId for route number `seq` (0-based).
    RouteId = SHA-256("route_{seq:06d}") — unique per seq, so the loader's
    A3 uniqueness gate (first-duplicate-wins) always passes.
    """
    label = f"route_{seq:06d}".encode("utf-8")
    return hashlib.sha256(label).digest()


def route_external_id(seq: int) -> bytes:
    """UTF-8 external id for the trailer (valid UTF-8, ASCII subset)."""
    return f"route_{seq:06d}".encode("utf-8")


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def build_mic2_request(name: str, n_tokens: int, k: int, seed: int) -> bytes:
    """
    Build a mic@2 preselect request frame. token IDs are drawn uniformly from
    [0, TOKEN_VOCAB) — safely below VOCAB_SIZE so every id is a valid
    embedding-row index. model: / catalog: are placeholder sentinels that
    run.sh replaces at runtime.
    """
    rng = Xorshift64(seed)
    tokens = [rng.next_u32_bounded(TOKEN_VOCAB) for _ in range(n_tokens)]
    token_csv = ",".join(str(t) for t in tokens)

    lines = [
        "mic@2/mind-nerve/preselect",
        "model: __MIND_NERVE_MODEL__",
        "catalog: __MIND_NERVE_CATALOG__",
        f"k: {k}",
        f"tokens: {token_csv}",
        ".",
        "",  # trailing newline
    ]
    return "\n".join(lines).encode("utf-8")


def catalog_preimage_and_hash(ids: list, emb_rows: list) -> bytes:
    """
    Recompute the canonical catalog hash EXACTLY as loader_parse_catalog does
    (v1: no route_prior). Preimage per route s (file order, not sorted):
      u32-LE 32 || id[s] || u32-LE (edim*4) || embedding row bytes
    """
    h = hashlib.sha256()
    edim_bytes = HIDDEN_DIM * 4
    for rid, row in zip(ids, emb_rows):
        h.update(struct.pack("<I", 32))
        h.update(rid)
        h.update(struct.pack("<I", edim_bytes))
        h.update(row)
    return h.digest()


def build_catalog_bin(n_routes: int) -> bytes:
    """
    Build a deterministic v1 catalog binary in the PORTED loader format.
    See the module docstring for the byte-level layout. Every embedding value
    is a pure-integer Q16.16 draw in [-1310, 1310] (~±0.02).
    """
    ids = [route_id_bytes(s) for s in range(n_routes)]

    # Per-route embedding rows: HIDDEN_DIM i32-LE Q16.16 values. Each route
    # gets an independent seed (mixed with the index) so rows differ, but the
    # whole thing is fixed by SEED_CATALOG.
    emb_rows = []
    for s in range(n_routes):
        row_seed = (SEED_CATALOG ^ ((s + 1) * 0x9E37_79B9_7F4A_7C15)) & \
            0xFFFF_FFFF_FFFF_FFFF
        emb_rows.append(i32_stream_bytes(row_seed, HIDDEN_DIM, -1310, 1310))

    ids_section = b"".join(ids)
    emb_section = b"".join(emb_rows)
    prior_section = b""  # v1: no route-prior block

    # External-id trailer: u16-LE len + UTF-8 bytes per route, file order.
    trailer = bytearray()
    for s in range(n_routes):
        ext = route_external_id(s)
        trailer.extend(struct.pack("<H", len(ext)))
        trailer.extend(ext)

    # trailer_off == prior_end == start of the trailer.
    trailer_off = 48 + len(ids_section) + len(emb_section) + len(prior_section)

    chash = catalog_preimage_and_hash(ids, emb_rows)

    buf = bytearray()
    buf.extend(b"MNC1")                         # [0:4]  magic
    buf.extend(struct.pack("<H", CAT_VERSION))  # [4:6]  version
    buf.extend(struct.pack("<H", 0))            # [6:8]  reserved
    buf.extend(struct.pack("<I", n_routes))     # [8:12] num_routes
    buf.extend(struct.pack("<I", HIDDEN_DIM))   # [12:16] embedding_dim
    buf.extend(chash)                           # [16:48] catalog hash
    buf.extend(ids_section)
    buf.extend(emb_section)
    buf.extend(prior_section)
    buf.extend(trailer)
    buf.extend(struct.pack("<I", trailer_off))  # [len-4:] trailer_off pointer
    return bytes(buf)


def build_weights_bin() -> bytes:
    """
    Build a deterministic v1 weights bundle in the PORTED loader format. See
    the module docstring for the byte-level layout. All values are pure-integer
    Q16.16 / INT8 draws; the model_hash binding gate is OFF so the two hash
    fields are opaque. Total size = 80 + LAYERS*per_layer + 2*h*4 + VOCAB*h*4.
    """
    h = HIDDEN_DIM

    buf = bytearray()
    buf.extend(b"MNW1")                           # [0:4]  magic
    buf.extend(struct.pack("<H", W_VERSION))      # [4:6]  version
    buf.extend(struct.pack("<H", 0))              # [6:8]  reserved
    buf.extend(opaque_hash("mind-nerve/fixture/model_hash"))      # [8:40]
    buf.extend(opaque_hash("mind-nerve/fixture/tokenizer_hash"))  # [40:72]
    buf.extend(struct.pack("<I", ENCODER_LAYERS)) # [72:76] encoder_layers
    buf.extend(struct.pack("<I", h))              # [76:80] encoder_hidden

    for layer in range(ENCODER_LAYERS):
        lseed = (SEED_WEIGHTS ^ ((layer + 1) * 0xA24B_AED4_963E_E407)) & \
            0xFFFF_FFFF_FFFF_FFFF
        # wq, wk, wv, wo — 4 * h*h INT8 (row-major).
        buf.extend(u8_stream_bytes(lseed ^ 0x01, 4 * h * h))
        # scales — 4 * h i32-LE Q16.16 (v1: one per output channel), positive
        # around 1.0 so dequantized weights stay small and non-zero.
        buf.extend(i32_stream_bytes(lseed ^ 0x02, 4 * h, Q16_ONE // 2,
                                    3 * Q16_ONE // 2))
        # ln_gain (~1.0), ln_bias (small), residual_gate (0..1.0).
        buf.extend(i32_stream_bytes(lseed ^ 0x03, h, Q16_ONE // 2,
                                    3 * Q16_ONE // 2))
        buf.extend(i32_stream_bytes(lseed ^ 0x04, h, -6554, 6554))
        buf.extend(i32_stream_bytes(lseed ^ 0x05, h, 0, Q16_ONE))

    # final_ln_gain (~1.0), final_ln_bias (small).
    buf.extend(i32_stream_bytes(SEED_WEIGHTS ^ 0x10, h, Q16_ONE // 2,
                                3 * Q16_ONE // 2))
    buf.extend(i32_stream_bytes(SEED_WEIGHTS ^ 0x11, h, -6554, 6554))

    # token_embedding — VOCAB_SIZE * h i32-LE Q16.16 (~32 MiB; tiled).
    buf.extend(tiled_i32_bytes(SEED_WEIGHTS ^ 0x20, VOCAB_SIZE * h,
                               -1310, 1310))
    return bytes(buf)


# ---------------------------------------------------------------------------
# Golden hash placeholders — real hashes land via run.sh --generate-golden.
# ---------------------------------------------------------------------------

GOLDEN_PLACEHOLDER = (
    "PENDING — run 'bash tests/bit_identity/run.sh --generate-golden' "
    "against a reference cpu build to populate this file.\n"
)


def write_golden_placeholder(name: str) -> None:
    path = os.path.join(EXPECTED_DIR, name)
    if os.path.exists(path):
        return  # Never overwrite committed golden hashes.
    with open(path, "w") as f:
        f.write(GOLDEN_PLACEHOLDER)


# ---------------------------------------------------------------------------
# Manifest file — machine-readable list of all (request, catalog) pairs
# ---------------------------------------------------------------------------


def write_manifest(pairs: list) -> None:
    """Write fixtures/MANIFEST listing all (request, catalog, golden) triples."""
    path = os.path.join(FIXTURE_DIR, "MANIFEST")
    with open(path, "w") as f:
        f.write("# Fixture manifest — auto-generated by gen_fixtures.py\n")
        f.write("# Format: request_file|catalog_file|golden_file\n")
        for req_name, cat_size in pairs:
            cat_name = f"catalog_{cat_size}.bin"
            golden_name = f"{req_name}_catalog_{cat_size}.sha256"
            f.write(f"{req_name}.mic2|{cat_name}|{golden_name}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    os.makedirs(EXPECTED_DIR, exist_ok=True)

    print("Generating request fixtures...")
    for req_name, n_tokens, k, seed in REQUESTS:
        data = build_mic2_request(req_name, n_tokens, k, seed)
        path = os.path.join(FIXTURE_DIR, f"{req_name}.mic2")
        with open(path, "wb") as f:
            f.write(data)
        print(f"  {req_name}.mic2  {n_tokens} tokens  k={k}  {len(data)} bytes")

    print("\nGenerating catalog fixtures (ported v1 format)...")
    for n_routes in CATALOG_SIZES:
        data = build_catalog_bin(n_routes)
        name = f"catalog_{n_routes}.bin"
        path = os.path.join(FIXTURE_DIR, name)
        with open(path, "wb") as f:
            f.write(data)
        chash = data[16:48].hex()
        print(f"  {name}  {n_routes} routes  {len(data)} bytes  "
              f"catalog_hash={chash[:16]}...")

    print("\nGenerating weights fixture (ported v1 format)...")
    wdata = build_weights_bin()
    wpath = os.path.join(FIXTURE_DIR, "model.weights")
    with open(wpath, "wb") as f:
        f.write(wdata)
    print(f"  model.weights  {ENCODER_LAYERS} layers  h={HIDDEN_DIM}  "
          f"vocab={VOCAB_SIZE}  {len(wdata)} bytes")

    print("\nInitialising golden hash placeholders (skips existing files)...")
    pairs = []
    for req_name, _, _, _ in REQUESTS:
        for n_routes in CATALOG_SIZES:
            golden_name = f"{req_name}_catalog_{n_routes}.sha256"
            write_golden_placeholder(golden_name)
            pairs.append((req_name, n_routes))
            print(f"  {golden_name}")

    print("\nWriting fixture manifest...")
    write_manifest([(r, c) for r, c in pairs])
    print("  fixtures/MANIFEST")

    print("\nDone. Next step: build the mind-nerve binary, then run:")
    print("  bash tests/bit_identity/run.sh --generate-golden")
    print("to populate fixtures/expected/*.sha256, then run without the flag.")


if __name__ == "__main__":
    main()
