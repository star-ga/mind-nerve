#!/usr/bin/env python3
"""
tests/bit_identity/verify.py

Divergence diagnostic tool for the cross-arch bit-identity gate.

Usage:
    python3 tests/bit_identity/verify.py <frame.bin> [<frame2.bin> ...]

Reads one or more raw mic-b binary frames and emits a human-readable
diagnostic for each:
  - magic, k
  - first 16 bytes of each RouteId (hex)
  - score for each route (Q16.16 as integer and float)
  - envelope fields: version, entry_kind, wire_version, k, timestamp_ms,
    architecture, chain_reset_reason
  - first 16 bytes of model_hash, tokenizer_hash, catalog_hash,
    request_hash, result_hash, chain_prev
  - computed chain_curr = SHA-256(212-byte envelope)

When two frame files are given, also emits a byte-level diff showing the
FIRST offset where the frames diverge (after masking timestamp_ms and
architecture, which are intentionally different per backend).

Stdlib only. No numpy.
"""

import hashlib
import os
import struct
import sys

# ---------------------------------------------------------------------------
# mic-b frame layout (from cli/main.mind)
#
#  offset    size          field
#  0         4             magic "MNB1"
#  4         2             k (u16 LE)
#  6         32*k          k x RouteId (32 bytes each)
#  6+32*k    4*k           k x score (i32 LE Q16.16)
#  6+36*k    212           attestation envelope v2
#  TOTAL = 218 + 36*k
# ---------------------------------------------------------------------------

MAGIC = b"MNB1"
ENVELOPE_SIZE = 212

# Attestation envelope v2 layout (from architecture.md):
#  offset  size  field
#    0      1    version
#    1      1    entry_kind   (1=Inference, 2=ModelLoad, 3=CatalogLoad)
#    2      2    wire_version (u16 LE)
#    4      4    k            (u32 LE)
#    8      8    timestamp_ms (i64 LE)
#   16      1    architecture (1=x86_64, 2=aarch64, 3=cuda)
#   17      1    reserved
#   18      2    chain_reset_reason (u16 LE)
#   20     32    model_hash
#   52     32    tokenizer_hash
#   84     32    catalog_hash
#  116     32    request_hash
#  148     32    result_hash
#  180     32    chain_prev
#  ===    ===
#  TOTAL  212

ARCH_NAMES = {1: "x86_64", 2: "aarch64", 3: "cuda", 4: "webgpu", 5: "npu"}
ENTRY_KIND_NAMES = {1: "Inference", 2: "ModelLoad", 3: "CatalogLoad"}
CHAIN_RESET_NAMES = {
    0: "Continuation",
    1: "ModelSwap",
    2: "CatalogChanged",
    3: "ClockReset",
}

Q16_SCALE = 65536.0  # 2^16


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class ParseError(Exception):
    pass


def parse_frame(data: bytes) -> dict:
    """
    Parse a mic-b binary frame into a structured dict.
    Raises ParseError on malformed input.
    """
    if len(data) < 6:
        raise ParseError(f"Frame too short: {len(data)} bytes (minimum 6)")

    if data[0:4] != MAGIC:
        raise ParseError(f"Bad magic: {data[0:4].hex()} (expected 4d4e4231)")

    k = struct.unpack_from("<H", data, 4)[0]
    expected_size = 218 + 36 * k
    if len(data) != expected_size:
        raise ParseError(f"Wrong frame size: got {len(data)}, expected {expected_size} (k={k})")

    routes = []
    for i in range(k):
        offset = 6 + 32 * i
        route_id = data[offset : offset + 32]
        routes.append(route_id)

    scores = []
    score_base = 6 + 32 * k
    for i in range(k):
        offset = score_base + 4 * i
        score_i32 = struct.unpack_from("<i", data, offset)[0]
        scores.append(score_i32)

    env_start = 6 + 36 * k
    env = data[env_start : env_start + ENVELOPE_SIZE]
    envelope = parse_envelope(env)

    # chain_curr is NOT stored; computed as SHA-256(212-byte envelope).
    chain_curr = hashlib.sha256(env).digest()

    return {
        "k": k,
        "routes": routes,
        "scores": scores,
        "envelope": envelope,
        "envelope_raw": env,
        "chain_curr": chain_curr,
        "total_size": len(data),
    }


def parse_envelope(env: bytes) -> dict:
    if len(env) != ENVELOPE_SIZE:
        raise ParseError(f"Envelope wrong size: {len(env)} (expected {ENVELOPE_SIZE})")

    version = env[0]
    entry_kind = env[1]
    wire_version = struct.unpack_from("<H", env, 2)[0]
    k_env = struct.unpack_from("<I", env, 4)[0]
    timestamp_ms = struct.unpack_from("<q", env, 8)[0]
    architecture = env[16]
    reserved = env[17]
    chain_reset_reason = struct.unpack_from("<H", env, 18)[0]
    model_hash = env[20:52]
    tokenizer_hash = env[52:84]
    catalog_hash = env[84:116]
    request_hash = env[116:148]
    result_hash = env[148:180]
    chain_prev = env[180:212]

    return {
        "version": version,
        "entry_kind": entry_kind,
        "wire_version": wire_version,
        "k": k_env,
        "timestamp_ms": timestamp_ms,
        "architecture": architecture,
        "reserved": reserved,
        "chain_reset_reason": chain_reset_reason,
        "model_hash": model_hash,
        "tokenizer_hash": tokenizer_hash,
        "catalog_hash": catalog_hash,
        "request_hash": request_hash,
        "result_hash": result_hash,
        "chain_prev": chain_prev,
    }


# ---------------------------------------------------------------------------
# Masked frame (zeros timestamp_ms and architecture for comparison)
# ---------------------------------------------------------------------------


def mask_frame_bytes(data: bytes) -> bytes:
    """
    Return a copy of the frame with timestamp_ms and architecture zeroed.
    This is the canonical preimage for cross-arch bit-identity comparison.
    """
    k = struct.unpack_from("<H", data, 4)[0]
    env_start = 6 + 36 * k
    out = bytearray(data)

    # Zero timestamp_ms: envelope offset 8, length 8.
    for i in range(8):
        out[env_start + 8 + i] = 0

    # Zero architecture: envelope offset 16, length 1.
    out[env_start + 16] = 0

    return bytes(out)


# ---------------------------------------------------------------------------
# Printer
# ---------------------------------------------------------------------------


def hex16(b: bytes) -> str:
    """Return first 16 bytes as hex, with '...' if longer."""
    if len(b) <= 16:
        return b.hex()
    return b[:16].hex() + "..."


def q16_to_float(i32: int) -> float:
    return i32 / Q16_SCALE


def print_frame(path: str, frame: dict) -> None:
    env = frame["envelope"]
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"FILE:          {path}")
    print(f"{sep}")
    print(f"total_size:    {frame['total_size']} bytes")
    print(f"k:             {frame['k']}")
    print()

    print("Routes (top-K results):")
    for i, (rid, score) in enumerate(zip(frame["routes"], frame["scores"], strict=False)):
        score_f = q16_to_float(score)
        print(f"  [{i:2d}] route_id={hex16(rid)}  score_q16={score:+011d}  ({score_f:+.6f})")
    print()

    print("Attestation envelope:")
    print(f"  version:            {env['version']}")
    print(
        f"  entry_kind:         {env['entry_kind']} "
        f"({ENTRY_KIND_NAMES.get(env['entry_kind'], 'unknown')})"
    )
    print(f"  wire_version:       {env['wire_version']}")
    print(f"  k:                  {env['k']}")
    print(f"  timestamp_ms:       {env['timestamp_ms']}")
    print(
        f"  architecture:       {env['architecture']} "
        f"({ARCH_NAMES.get(env['architecture'], 'unknown')})"
    )
    print(f"  reserved:           {env['reserved']}")
    print(
        f"  chain_reset_reason: {env['chain_reset_reason']} "
        f"({CHAIN_RESET_NAMES.get(env['chain_reset_reason'], 'unknown')})"
    )
    print()

    print(f"  model_hash:         {hex16(env['model_hash'])}")
    print(f"  tokenizer_hash:     {hex16(env['tokenizer_hash'])}")
    print(f"  catalog_hash:       {hex16(env['catalog_hash'])}")
    print(f"  request_hash:       {hex16(env['request_hash'])}")
    print(f"  result_hash:        {hex16(env['result_hash'])}")
    print(f"  chain_prev:         {hex16(env['chain_prev'])}")
    print()
    print(f"  chain_curr (computed): {frame['chain_curr'].hex()}")

    masked = mask_frame_bytes(frame["envelope_raw"])
    # Compute the masked-envelope digest directly (same masking run.sh does for
    # the full frame, but here we expose the envelope portion for diagnostic
    # clarity).
    print(f"  masked_envelope_sha256: {hashlib.sha256(masked).hexdigest()}")


# ---------------------------------------------------------------------------
# Byte-level diff between two masked frames
# ---------------------------------------------------------------------------


def diff_frames(path_a: str, data_a: bytes, path_b: str, data_b: bytes) -> int:
    """
    Emit a byte-level diff of the two masked frames.
    Returns the number of differing bytes.
    """
    masked_a = mask_frame_bytes(data_a)
    masked_b = mask_frame_bytes(data_b)

    hash_a = hashlib.sha256(masked_a).hexdigest()
    hash_b = hashlib.sha256(masked_b).hexdigest()

    print(f"\n{'=' * 60}")
    print("PAIRWISE DIFF (timestamp_ms and architecture masked)")
    print(f"  A: {path_a}")
    print(f"  B: {path_b}")
    print(f"  A masked sha256: {hash_a}")
    print(f"  B masked sha256: {hash_b}")

    if hash_a == hash_b:
        print("  RESULT: IDENTICAL — frames are bit-identical after masking")
        return 0

    print("  RESULT: DIVERGE — locating first differing byte...")

    # Find first divergence.
    min_len = min(len(masked_a), len(masked_b))
    first_diff = None
    diff_count = 0

    for i in range(min_len):
        if masked_a[i] != masked_b[i]:
            if first_diff is None:
                first_diff = i
            diff_count += 1

    if len(masked_a) != len(masked_b):
        diff_count += abs(len(masked_a) - len(masked_b))

    if first_diff is not None:
        # Decode which field this offset falls in.
        k = struct.unpack_from("<H", data_a, 4)[0]
        env_start = 6 + 36 * k
        field = _field_name_at(first_diff, k, env_start)
        print(f"  First divergence at byte {first_diff} ({field})")
        print(f"    A[{first_diff}] = 0x{masked_a[first_diff]:02x}")
        print(f"    B[{first_diff}] = 0x{masked_b[first_diff]:02x}")
        print()
        # Print context (16 bytes around the divergence).
        lo = max(0, first_diff - 8)
        hi = min(min_len, first_diff + 8)
        a_ctx = masked_a[lo:hi].hex()
        b_ctx = masked_b[lo:hi].hex()
        print(f"  Context A bytes [{lo}:{hi}]: {a_ctx}")
        print(f"  Context B bytes [{lo}:{hi}]: {b_ctx}")

    print(f"  Total differing bytes (in common prefix): {diff_count}")

    return diff_count


def _field_name_at(offset: int, k: int, env_start: int) -> str:
    """
    Return a human-readable name for the field at byte `offset` in a
    masked mic-b frame with the given k.
    """
    if offset < 4:
        return "magic"
    if offset < 6:
        return "k_u16"
    if offset < 6 + 32 * k:
        route_idx = (offset - 6) // 32
        return f"route_id[{route_idx}]"
    score_start = 6 + 32 * k
    if offset < score_start + 4 * k:
        score_idx = (offset - score_start) // 4
        return f"score[{score_idx}]"
    if offset < env_start:
        return "padding(?)"
    env_off = offset - env_start
    ENV_FIELDS = [
        (0, 1, "version"),
        (1, 1, "entry_kind"),
        (2, 2, "wire_version"),
        (4, 4, "envelope.k"),
        (8, 8, "timestamp_ms (should be zeroed)"),
        (16, 1, "architecture (should be zeroed)"),
        (17, 1, "reserved"),
        (18, 2, "chain_reset_reason"),
        (20, 32, "model_hash"),
        (52, 32, "tokenizer_hash"),
        (84, 32, "catalog_hash"),
        (116, 32, "request_hash"),
        (148, 32, "result_hash"),
        (180, 32, "chain_prev"),
    ]
    for start, length, name in ENV_FIELDS:
        if start <= env_off < start + length:
            return f"envelope.{name}[+{env_off - start}]"
    return f"envelope+{env_off}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python3 verify.py <frame.bin> [<frame2.bin> ...]")
        print("       Diagnoses mic-b binary frames from the bit-identity gate.")
        sys.exit(1)

    frames = []
    raw_datas = []

    for path in args:
        if not os.path.isfile(path):
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        with open(path, "rb") as f:
            data = f.read()
        try:
            frame = parse_frame(data)
        except ParseError as e:
            print(f"ERROR parsing {path}: {e}", file=sys.stderr)
            sys.exit(1)
        frames.append((path, frame))
        raw_datas.append((path, data))
        print_frame(path, frame)

    if len(raw_datas) == 2:
        diff_frames(
            raw_datas[0][0],
            raw_datas[0][1],
            raw_datas[1][0],
            raw_datas[1][1],
        )
    elif len(raw_datas) > 2:
        # Pairwise diffs for more than 2 frames.
        any_diff = False
        for i in range(len(raw_datas)):
            for j in range(i + 1, len(raw_datas)):
                count = diff_frames(
                    raw_datas[i][0],
                    raw_datas[i][1],
                    raw_datas[j][0],
                    raw_datas[j][1],
                )
                if count > 0:
                    any_diff = True
        if any_diff:
            sys.exit(1)


if __name__ == "__main__":
    main()
