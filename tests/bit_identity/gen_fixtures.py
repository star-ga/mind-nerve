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

All outputs are bit-for-bit reproducible from the fixed seeds below.
No numpy. No third-party libraries. Pure stdlib.

The catalog .bin format is a simple length-prefixed binary layout:
  [0:4]   magic "MNC1" (mind-nerve catalog v1)
  [4:8]   route_count (u32 LE)
  per route:
    [0:32]  route_id (SHA-256 of "route_{seq:06d}")
    [32:36] embedding_dim (u32 LE, always 256)
    [36:36+256*4] embedding (256 x i32 LE, Q16.16 values)

This format is consumed by run.sh when it stubs catalog loading.
The actual mindc-compiled binary will have its own catalog manifest
format; this file documents what the test harness uses directly.

The .mic2 request format mirrors cli/main.mind wire protocol exactly:
  mic@2/mind-nerve/preselect
  model: <placeholder>
  catalog: <placeholder>
  k: <k>
  tokens: <csv of u32 token ids>
  .

model: and catalog: paths are placeholders (the harness substitutes real
paths at runtime via environment variables MIND_NERVE_MODEL and
MIND_NERVE_CATALOG).
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

VOCAB_SIZE = 32000  # 32k BPE vocabulary
HIDDEN_DIM = 256  # route embedding dimension (matches architecture.md)

# Q16.16 range: i32, fractional bits = 16.
# Embeddings drawn from N(0, 0.02) in float, scaled to Q16.16.
# scale = 2^16 = 65536; 0.02 * 65536 = 1310.72 -> we draw integers from
# the discrete approximation using a Box-Muller transform over the PRNG.
EMBED_SCALE = 65536  # 2^16
EMBED_STDDEV_FLOAT = 0.02

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
EXPECTED_DIR = os.path.join(FIXTURE_DIR, "expected")

CATALOG_SIZES = [44, 440, 4400]
REQUESTS = [
    ("request_001", 16, 5, SEED_TOKENS_001),
    ("request_002", 256, 10, SEED_TOKENS_002),
    ("request_003", 1024, 64, SEED_TOKENS_003),
]


# ---------------------------------------------------------------------------
# Minimal deterministic PRNG (xorshift64, stdlib only)
# ---------------------------------------------------------------------------


class Xorshift64:
    """
    Xorshift64 PRNG. Deterministic, fast, no stdlib dependency.
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

    def next_float_01(self) -> float:
        """Return a float in [0, 1) via 53-bit precision."""
        return (self.next_u64() >> 11) / float(1 << 53)

    def next_normal(self) -> float:
        """
        Box-Muller transform. Returns one normal sample N(0,1).
        Consumes two u64 draws.
        """
        import math

        while True:
            u1 = self.next_float_01()
            u2 = self.next_float_01()
            if u1 > 0.0:
                break
        mag = math.sqrt(-2.0 * math.log(u1))
        # Return just one of the two samples for simplicity.
        return mag * math.cos(2.0 * math.pi * u2)


# ---------------------------------------------------------------------------
# Route ID derivation
# ---------------------------------------------------------------------------


def route_id_bytes(seq: int) -> bytes:
    """
    Derive the 32-byte RouteId for route number `seq` (0-based).
    RouteId = SHA-256("route_{seq:06d}").
    This mirrors the tie-break contract: SHA-256(route_id) ascending.
    """
    label = f"route_{seq:06d}".encode("utf-8")
    return hashlib.sha256(label).digest()


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def build_mic2_request(
    name: str,
    n_tokens: int,
    k: int,
    seed: int,
) -> bytes:
    """
    Build a mic@2 preselect request frame.
    token IDs are drawn uniformly from [0, VOCAB_SIZE).
    model: and catalog: paths are placeholder sentinels that the harness
    replaces at runtime via env vars MIND_NERVE_MODEL and MIND_NERVE_CATALOG.
    """
    rng = Xorshift64(seed)
    tokens = [rng.next_u32_bounded(VOCAB_SIZE) for _ in range(n_tokens)]
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


def build_catalog_bin(n_routes: int) -> bytes:
    """
    Build a deterministic catalog binary with `n_routes` routes.
    All routes use the shared SEED_CATALOG; the route index is mixed into
    the per-route seed by XOR to ensure each route has independent draws.
    Format:
      [0:4]   magic "MNC1"
      [4:8]   route_count u32 LE
      per route:
        [0:32]  route_id (32 bytes, SHA-256 of label)
        [32:36] embedding_dim u32 LE (= 256)
        [36:36+256*4] 256 x i32 LE Q16.16 embedding values
    Total size: 8 + n_routes * (32 + 4 + 256*4) = 8 + n_routes * 1060 bytes.
    """
    buf = bytearray()
    # Magic.
    buf.extend(b"MNC1")
    # Route count.
    buf.extend(struct.pack("<I", n_routes))

    rng = Xorshift64(SEED_CATALOG)

    for seq in range(n_routes):
        # route_id = SHA-256 of label.
        rid = route_id_bytes(seq)
        buf.extend(rid)

        # embedding_dim.
        buf.extend(struct.pack("<I", HIDDEN_DIM))

        # 256 Q16.16 embedding values drawn from N(0, 0.02).
        for _ in range(HIDDEN_DIM):
            sample_f = rng.next_normal() * EMBED_STDDEV_FLOAT
            # Clamp to i32 range before rounding.
            scaled = sample_f * EMBED_SCALE
            clamped = max(-2147483648, min(2147483647, int(round(scaled))))
            buf.extend(struct.pack("<i", clamped))

    return bytes(buf)


# ---------------------------------------------------------------------------
# Catalog hash (mirrors architecture.md §Catalog hashing)
# ---------------------------------------------------------------------------


def catalog_hash(n_routes: int) -> bytes:
    """
    Compute the CatalogHash per the spec:
      SHA-256 over (for each route sorted by SHA-256(route_id)):
        length-prefixed route_id
        length-prefixed description SHA-256

    For fixtures, the route description is the fixed label string
    "route_{seq:06d} description" — the same seed used in route_id_bytes.
    Sorting is by SHA-256(route_id) which for our deterministic IDs is
    simply SHA-256(SHA-256("route_{seq:06d}")).
    """
    # Build (sha256_of_route_id, seq) pairs for sorting.
    pairs = []
    for seq in range(n_routes):
        rid = route_id_bytes(seq)
        sort_key = hashlib.sha256(rid).digest()
        desc = f"route_{seq:06d} description".encode("utf-8")
        desc_sha = hashlib.sha256(desc).digest()
        pairs.append((sort_key, rid, desc_sha))

    pairs.sort(key=lambda p: p[0])

    h = hashlib.sha256()
    for _sort_key, rid, desc_sha in pairs:
        # length-prefixed route_id (u32 LE length + bytes)
        h.update(struct.pack("<I", len(rid)))
        h.update(rid)
        # length-prefixed description SHA-256 (u32 LE length + bytes)
        h.update(struct.pack("<I", len(desc_sha)))
        h.update(desc_sha)

    return h.digest()


# ---------------------------------------------------------------------------
# Placeholder golden hash files
# These are written as 64-char hex strings + newline. They contain the
# literal string "PENDING" until run.sh is executed against a reference
# build and the real hashes are committed.
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
    """
    Write fixtures/MANIFEST listing all (request_file, catalog_file, golden_file).
    run.sh reads this to know which pairs to exercise.
    """
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
        size = len(data)
        print(f"  {req_name}.mic2  {n_tokens} tokens  k={k}  {size} bytes")

    print("\nGenerating catalog fixtures...")
    for n_routes in CATALOG_SIZES:
        data = build_catalog_bin(n_routes)
        name = f"catalog_{n_routes}.bin"
        path = os.path.join(FIXTURE_DIR, name)
        with open(path, "wb") as f:
            f.write(data)
        chash = catalog_hash(n_routes).hex()
        print(f"  {name}  {n_routes} routes  {len(data)} bytes  catalog_hash={chash[:16]}...")

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
    print("to populate fixtures/expected/*.sha256 and commit them.")


if __name__ == "__main__":
    main()
