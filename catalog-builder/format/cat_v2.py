"""Catalog v2 binary format encoder/decoder.

v2 layout (backward-compatible extension of the v1 MNC1 format):

  [0:4]    magic "MNC2"                       (version discriminant)
  [4:8]    route_count  u32 LE
  per route  (same byte layout as v1 per-route block):
    [0:32]  route_id       32-byte SHA-256
    [32:36] embedding_dim  u32 LE (always ROUTE_EMBEDDING_DIM = 256)
    [36:36+256*4]  embedding  256 × i32 LE (INT8-quantised, pre-scaled)

  [tail]  prior block  (new in v2)
    magic_tail  "PRIR"  4 bytes
    count       u32 LE  (must equal route_count)
    route_count × i32 LE  Q16.16 log-prior values

  The prior block is always present in v2. Its size is route_count * 4 bytes.
  v1 readers that probe the total length and find the extra tail will
  either ignore it or fail gracefully — the v1 magic "MNC1" is never written
  by this module, so true v1 files are unaffected.

Backward-compatibility contract:
  - v1 files (magic "MNC1") load via ``decode_v1()`` in this module or via
    the existing fixture loader; they are never modified.
  - ``decode_any()`` dispatches on magic and returns a uniform dict.
  - The v2 prior block starts at offset 8 + route_count * ROUTE_BLOCK_BYTES
    and is identified by the "PRIR" sentinel before accessing any prior data.

Q16.16 encoding for the prior column:
  value = log(1 + freq_r)  as float
  encoded = round(value * 65536)  clamped to i32 range
"""

from __future__ import annotations

import struct
from typing import Any

MAGIC_V1: bytes = b"MNC1"
MAGIC_V2: bytes = b"MNC2"
MAGIC_PRIOR: bytes = b"PRIR"

ROUTE_ID_BYTES: int = 32
EMBEDDING_DIM: int = 256
EMBED_FIELD_BYTES: int = 4  # u32 LE dim prefix
ROUTE_BLOCK_BYTES: int = ROUTE_ID_BYTES + EMBED_FIELD_BYTES + EMBEDDING_DIM * 4  # 1060
PRIOR_SENTINEL_BYTES: int = 4
PRIOR_COUNT_BYTES: int = 4
PRIOR_VALUE_BYTES: int = 4  # i32 LE per route

Q16_FRAC_BITS: int = 16
Q16_SCALE: int = 1 << Q16_FRAC_BITS  # 65536
I32_MAX: int = 2_147_483_647
I32_MIN: int = -2_147_483_648


def float_to_q16(value: float) -> int:
    """Convert a float to Q16.16 fixed-point i32, clamped to i32 range."""
    raw = round(value * Q16_SCALE)
    return max(I32_MIN, min(I32_MAX, raw))


def q16_to_float(encoded: int) -> float:
    """Decode a Q16.16 i32 back to float."""
    return encoded / Q16_SCALE


def freq_adaptive_scale(freq_r: float) -> float:
    """Pre-scale factor for a route embedding row before INT8 quantisation (RFC-004).

    Returns max(0.5, 1 / sqrt(freq_r)).
    High-frequency routes are scaled down toward 0.5; rare routes keep 1.0.
    Zero or negative freq_r is treated as 1.0 (no scaling).
    """
    import math

    if freq_r <= 0.0:
        return 1.0
    return max(0.5, 1.0 / math.sqrt(freq_r))


def encode_prior_block(log_priors: list[float]) -> bytes:
    """Encode the trailing prior block: PRIR + count + route_count Q16.16 values."""
    n = len(log_priors)
    parts = [MAGIC_PRIOR, struct.pack("<I", n)]
    for lp in log_priors:
        parts.append(struct.pack("<i", float_to_q16(lp)))
    return b"".join(parts)


def decode_prior_block(data: bytes, route_count: int) -> list[float]:
    """Decode the trailing prior block from v2 catalog bytes.

    Raises ``ValueError`` if the sentinel or count is invalid.
    """
    expected_tail = PRIOR_SENTINEL_BYTES + PRIOR_COUNT_BYTES + route_count * PRIOR_VALUE_BYTES
    if len(data) < expected_tail:
        raise ValueError(f"prior block too short: need {expected_tail} bytes, got {len(data)}")
    tail = data[-expected_tail:]
    sentinel = tail[:PRIOR_SENTINEL_BYTES]
    if sentinel != MAGIC_PRIOR:
        raise ValueError(
            f"prior block sentinel mismatch: expected {MAGIC_PRIOR!r}, got {sentinel!r}"
        )
    (count,) = struct.unpack_from("<I", tail, PRIOR_SENTINEL_BYTES)
    if count != route_count:
        raise ValueError(f"prior block route count {count} != catalog route count {route_count}")
    offset = PRIOR_SENTINEL_BYTES + PRIOR_COUNT_BYTES
    return [
        q16_to_float(struct.unpack_from("<i", tail, offset + i * PRIOR_VALUE_BYTES)[0])
        for i in range(count)
    ]


def encode_v2(
    routes: list[dict[str, Any]],
    log_priors: list[float],
) -> bytes:
    """Encode a v2 catalog blob.

    ``routes`` is a list of dicts with keys ``route_id`` (bytes, 32) and
    ``embedding`` (list[int], 256 pre-scaled INT8-quantised Q16.16 values).
    ``log_priors`` must have the same length as ``routes``.

    Raises ``ValueError`` on shape mismatches.
    """
    if len(routes) != len(log_priors):
        raise ValueError(
            f"routes ({len(routes)}) and log_priors ({len(log_priors)}) must have equal length"
        )
    parts: list[bytes] = [MAGIC_V2, struct.pack("<I", len(routes))]
    for r in routes:
        rid: bytes = r["route_id"]
        if len(rid) != ROUTE_ID_BYTES:
            raise ValueError(f"route_id must be {ROUTE_ID_BYTES} bytes, got {len(rid)}")
        emb: list[int] = r["embedding"]
        if len(emb) != EMBEDDING_DIM:
            raise ValueError(f"embedding must have {EMBEDDING_DIM} values, got {len(emb)}")
        parts.append(rid)
        parts.append(struct.pack("<I", EMBEDDING_DIM))
        parts.extend(struct.pack("<i", v) for v in emb)
    # Trailing prior block
    parts.append(encode_prior_block(log_priors))
    return b"".join(parts)


def decode_v1(data: bytes) -> dict[str, Any]:
    """Decode a v1 catalog binary (magic MNC1).

    Returns ``{"version": 1, "route_count": N, "routes": [...]}``.
    Each route dict has keys ``route_id`` (bytes) and ``embedding`` (list[int]).
    """
    if len(data) < 8:
        raise ValueError("catalog too short for v1 header")
    magic = data[:4]
    if magic != MAGIC_V1:
        raise ValueError(f"expected MNC1 magic, got {magic!r}")
    (route_count,) = struct.unpack_from("<I", data, 4)
    routes = _decode_route_blocks(data, 8, route_count)
    return {"version": 1, "route_count": route_count, "routes": routes}


def decode_v2(data: bytes) -> dict[str, Any]:
    """Decode a v2 catalog binary (magic MNC2).

    Returns ``{"version": 2, "route_count": N, "routes": [...], "log_priors": [...]}``.
    """
    if len(data) < 8:
        raise ValueError("catalog too short for v2 header")
    magic = data[:4]
    if magic != MAGIC_V2:
        raise ValueError(f"expected MNC2 magic, got {magic!r}")
    (route_count,) = struct.unpack_from("<I", data, 4)
    routes = _decode_route_blocks(data, 8, route_count)
    log_priors = decode_prior_block(data, route_count)
    return {"version": 2, "route_count": route_count, "routes": routes, "log_priors": log_priors}


def decode_any(data: bytes) -> dict[str, Any]:
    """Dispatch on magic byte and decode v1 or v2 catalog."""
    if len(data) < 4:
        raise ValueError("catalog blob too short to probe magic")
    magic = data[:4]
    if magic == MAGIC_V1:
        return decode_v1(data)
    if magic == MAGIC_V2:
        return decode_v2(data)
    raise ValueError(f"unrecognised catalog magic {magic!r}")


def _decode_route_blocks(data: bytes, offset: int, route_count: int) -> list[dict[str, Any]]:
    """Parse route_count sequential route blocks starting at ``offset``."""
    routes: list[dict[str, Any]] = []
    for _ in range(route_count):
        if offset + ROUTE_BLOCK_BYTES > len(data):
            raise ValueError("catalog data truncated in route block")
        rid = data[offset : offset + ROUTE_ID_BYTES]
        offset += ROUTE_ID_BYTES
        (dim,) = struct.unpack_from("<I", data, offset)
        offset += EMBED_FIELD_BYTES
        if dim != EMBEDDING_DIM:
            raise ValueError(f"unexpected embedding dim {dim}")
        emb = list(struct.unpack_from(f"<{EMBEDDING_DIM}i", data, offset))
        offset += EMBEDDING_DIM * 4
        routes.append({"route_id": rid, "embedding": emb})
    return routes
