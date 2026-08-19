"""Catalog binary format encoder/decoder — MNC1, the PORTED wire format.

U4b (2026-08-19): this module used to implement a "v2" format (magic
``MNC2`` + a trailing ``PRIR`` prior block) that was never accepted by any
shipping loader — ``src/loader.mind``'s ``loader_parse_catalog`` (the only
real reader) requires a u16 version + u16 reserved header, a canonical
SHA-256 catalog hash at offset 16, and a UTF-8 external-id trailer, none of
which the old ``MNC2`` branch produced. That branch is retired; this module
now implements ``MNC1``, the format the ``.mind`` loader actually reads,
matching ``tests/bit_identity/gen_fixtures.py``'s ``build_catalog_bin``
byte-for-byte. The Q16.16 codec and the RFC-004 frequency-adaptive scale
helper are unchanged (kept verbatim) — only the container format changed.

MNC1 layout (little-endian throughout; see ``src/loader.mind``
``loader_parse_catalog`` and its module docstring for the authoritative
byte-level spec):

  [0:4]    magic "MNC1"
  [4:6]    u16 version           (1 = no route-prior; 2 = inline route-prior)
  [6:8]    u16 reserved          (MUST be zero)
  [8:12]   u32 num_routes
  [12:16]  u32 embedding_dim     (== ROUTE_EMBEDDING_DIM == 256)
  [16:48]  32-byte canonical catalog hash (see ``catalog_preimage_and_hash``)
  [48:..]  route_ids             num_routes * 32 bytes (unique; A3 gate)
  [..]     embeddings            num_routes * 256 * i32-LE (Q16.16)
  (v2 only) route_prior          num_routes * i32-LE Q16.16 log-prior
  [..]     external-id trailer per route: u16-LE len + len UTF-8 bytes
  [len-4:] u32-LE trailer_off    (== start of the trailer)

Canonical catalog hash preimage (rebuilt from the FILE bytes, route order
0..num_routes, NOT sorted), per route ``s``:
  u32-LE 32 || id[s] (32) || u32-LE (edim*4) || embedding row bytes (edim*4)
  || (u32-LE 4 || route_prior[s] (4))     -- only when version == 2
SHA-256 of the concatenation is the 32-byte value written at offset 16.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any

MAGIC_MNC1: bytes = b"MNC1"

ROUTE_ID_BYTES: int = 32
# Embedding dimensionality — must match ROUTE_EMBEDDING_DIM in src/lib.mind.
EMBEDDING_DIM: int = 256
HEADER_BYTES: int = 48  # magic(4) + version(2) + reserved(2) + n(4) + dim(4) + hash(32)

# Wire versions accepted by src/loader.mind's loader_parse_catalog.
CAT_VERSION_1: int = 1  # no route-prior block
CAT_VERSION_2: int = 2  # inline route-prior block

Q16_FRAC_BITS: int = 16
Q16_SCALE: int = 1 << Q16_FRAC_BITS  # 65536
I32_MAX: int = 2_147_483_647
I32_MIN: int = -2_147_483_648


# ---------------------------------------------------------------------------
# Q16.16 codec (kept verbatim from the retired module — unrelated to the
# container format change).
# ---------------------------------------------------------------------------


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


def _clamp_i32(value: int) -> int:
    return max(I32_MIN, min(I32_MAX, int(value)))


# ---------------------------------------------------------------------------
# Canonical catalog hash — mirrors loader_parse_catalog's preimage exactly.
# ---------------------------------------------------------------------------


def catalog_preimage_and_hash(
    route_ids: list[bytes],
    emb_rows: list[bytes],
    prior_rows: list[bytes] | None = None,
) -> bytes:
    """Recompute the canonical MNC1 catalog hash exactly as ``loader_parse_catalog``
    does (``src/loader.mind``). ``emb_rows`` and ``prior_rows`` are already-packed
    little-endian byte strings (256*4 bytes and 4 bytes respectively) — the
    preimage is built from the same bytes the caller is about to write, so the
    hash can never drift from the emitted payload.
    """
    if len(emb_rows) != len(route_ids):
        raise ValueError("route_ids and emb_rows must have equal length")
    if prior_rows is not None and len(prior_rows) != len(route_ids):
        raise ValueError("prior_rows must have the same length as route_ids")
    edim_bytes = EMBEDDING_DIM * 4
    h = hashlib.sha256()
    for i, rid in enumerate(route_ids):
        if len(rid) != ROUTE_ID_BYTES:
            raise ValueError(f"route_id must be {ROUTE_ID_BYTES} bytes, got {len(rid)}")
        h.update(struct.pack("<I", ROUTE_ID_BYTES))
        h.update(rid)
        h.update(struct.pack("<I", edim_bytes))
        h.update(emb_rows[i])
        if prior_rows is not None:
            h.update(struct.pack("<I", 4))
            h.update(prior_rows[i])
    return h.digest()


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


def encode_mnc1(
    route_ids: list[bytes],
    embeddings: list[list[int]],
    external_ids: list[bytes],
    *,
    log_priors: list[float] | None = None,
) -> bytes:
    """Encode a catalog to the MNC1 wire format ``src/loader.mind`` accepts.

    ``route_ids`` — 32-byte SHA-256 ids, one per route, file order (the order
    the loader's A3 uniqueness scan and canonical hash both use).
    ``embeddings`` — one length-``EMBEDDING_DIM`` list of Q16.16 i32 values
    per route.
    ``external_ids`` — one UTF-8-encoded external id per route, written to
    the trailer (validated by the loader; never re-hashed).
    ``log_priors`` — optional, one float per route. When given, the wire
    version is 2 (inline route-prior) and each value is written as a Q16.16
    i32 via :func:`float_to_q16`; when omitted, version 1 is written and the
    loader defaults every route's prior to zero.
    """
    n = len(route_ids)
    if len(embeddings) != n or len(external_ids) != n:
        raise ValueError("route_ids, embeddings, external_ids must have equal length")
    if log_priors is not None and len(log_priors) != n:
        raise ValueError("log_priors must have the same length as route_ids")

    version = CAT_VERSION_2 if log_priors is not None else CAT_VERSION_1
    ids_section = b"".join(route_ids)
    for rid in route_ids:
        if len(rid) != ROUTE_ID_BYTES:
            raise ValueError(f"route_id must be {ROUTE_ID_BYTES} bytes, got {len(rid)}")

    emb_rows: list[bytes] = []
    for row in embeddings:
        if len(row) != EMBEDDING_DIM:
            raise ValueError(f"embedding must have {EMBEDDING_DIM} values, got {len(row)}")
        emb_rows.append(b"".join(struct.pack("<i", _clamp_i32(v)) for v in row))
    emb_section = b"".join(emb_rows)

    prior_rows: list[bytes] | None = None
    prior_section = b""
    if log_priors is not None:
        prior_rows = [struct.pack("<i", float_to_q16(lp)) for lp in log_priors]
        prior_section = b"".join(prior_rows)

    chash = catalog_preimage_and_hash(route_ids, emb_rows, prior_rows)

    trailer = bytearray()
    for ext in external_ids:
        if len(ext) > 0xFFFF:
            raise ValueError("external id exceeds the u16 trailer length field")
        trailer.extend(struct.pack("<H", len(ext)))
        trailer.extend(ext)

    trailer_off = HEADER_BYTES + len(ids_section) + len(emb_section) + len(prior_section)

    buf = bytearray()
    buf.extend(MAGIC_MNC1)
    buf.extend(struct.pack("<H", version))
    buf.extend(struct.pack("<H", 0))
    buf.extend(struct.pack("<I", n))
    buf.extend(struct.pack("<I", EMBEDDING_DIM))
    buf.extend(chash)
    buf.extend(ids_section)
    buf.extend(emb_section)
    buf.extend(prior_section)
    buf.extend(trailer)
    buf.extend(struct.pack("<I", trailer_off))
    return bytes(buf)


# ---------------------------------------------------------------------------
# Decoder — pure-Python reference implementation, mirroring
# loader_parse_catalog's gates exactly. Used by tests and by producers that
# want to self-verify a bundle before writing it.
# ---------------------------------------------------------------------------


def decode_mnc1(data: bytes) -> dict[str, Any]:
    """Decode + fully validate an MNC1 catalog blob (mirrors every
    ``loader_parse_catalog`` gate: magic, version, reserved, hash, trailer).

    Returns ``{"version", "route_count", "routes": [{"route_id",
    "embedding", "external_id"}], "log_priors" (v2 only)}``. Raises
    ``ValueError`` with a message naming the failed gate.
    """
    if len(data) < HEADER_BYTES:
        raise ValueError("catalog too short for MNC1 header")
    if data[:4] != MAGIC_MNC1:
        raise ValueError(f"expected MNC1 magic, got {data[:4]!r}")
    (version,) = struct.unpack_from("<H", data, 4)
    if version not in (CAT_VERSION_1, CAT_VERSION_2):
        raise ValueError(f"unsupported version {version} (loader accepts 1 or 2)")
    has_prior = version == CAT_VERSION_2
    (reserved,) = struct.unpack_from("<H", data, 6)
    if reserved != 0:
        raise ValueError(f"reserved field must be zero, got {reserved}")
    (num_routes,) = struct.unpack_from("<I", data, 8)
    (embedding_dim,) = struct.unpack_from("<I", data, 12)
    if embedding_dim != EMBEDDING_DIM:
        raise ValueError(f"embedding_dim {embedding_dim} != {EMBEDDING_DIM}")
    file_hash = data[16:48]

    ids_off = HEADER_BYTES
    ids_end = ids_off + num_routes * ROUTE_ID_BYTES
    if len(data) < ids_end:
        raise ValueError("catalog truncated in route_ids section")
    route_ids = [data[ids_off + i * 32 : ids_off + i * 32 + 32] for i in range(num_routes)]

    emb_off = ids_end
    emb_bytes = num_routes * EMBEDDING_DIM * 4
    emb_end = emb_off + emb_bytes
    if len(data) < emb_end:
        raise ValueError("catalog truncated in embeddings section")
    emb_rows = [
        data[emb_off + i * EMBEDDING_DIM * 4 : emb_off + (i + 1) * EMBEDDING_DIM * 4]
        for i in range(num_routes)
    ]
    embeddings = [list(struct.unpack(f"<{EMBEDDING_DIM}i", row)) for row in emb_rows]

    prior_off = emb_end
    prior_rows: list[bytes] | None = None
    log_priors: list[float] | None = None
    prior_end = prior_off
    if has_prior:
        prior_bytes = num_routes * 4
        prior_end = prior_off + prior_bytes
        if len(data) < prior_end:
            raise ValueError("catalog truncated in route_prior section")
        prior_rows = [data[prior_off + i * 4 : prior_off + i * 4 + 4] for i in range(num_routes)]
        log_priors = [q16_to_float(struct.unpack("<i", pr)[0]) for pr in prior_rows]

    recomputed = catalog_preimage_and_hash(route_ids, emb_rows, prior_rows)
    if recomputed != file_hash:
        raise ValueError("canonical catalog hash mismatch")

    if len(data) < prior_end + 4:
        raise ValueError("catalog truncated before trailer pointer")
    (trailer_off,) = struct.unpack_from("<I", data, len(data) - 4)
    if trailer_off != prior_end:
        raise ValueError(f"trailer_off {trailer_off} != expected {prior_end}")

    cursor = trailer_off
    trailer_field = len(data) - 4
    external_ids: list[bytes] = []
    for _ in range(num_routes):
        if cursor + 2 > trailer_field:
            raise ValueError("catalog truncated in external-id trailer")
        (elen,) = struct.unpack_from("<H", data, cursor)
        cursor += 2
        if cursor + elen > trailer_field:
            raise ValueError("catalog truncated in external-id trailer")
        ext = data[cursor : cursor + elen]
        ext.decode("utf-8")  # raises UnicodeDecodeError on invalid UTF-8
        external_ids.append(ext)
        cursor += elen
    if cursor != trailer_field:
        raise ValueError("external-id trailer does not end exactly at trailer_off pointer")

    routes = [
        {"route_id": route_ids[i], "embedding": embeddings[i], "external_id": external_ids[i]}
        for i in range(num_routes)
    ]
    result: dict[str, Any] = {"version": version, "route_count": num_routes, "routes": routes}
    if log_priors is not None:
        result["log_priors"] = log_priors
    return result
