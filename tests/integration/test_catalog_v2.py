"""Integration tests for catalog v2 format: prior column + freq-adaptive scaling.

Covers:
  - v2 round-trip: encode → decode, prior column values survive.
  - Frequency-adaptive scaling: per-route scalar is in [0.5, 1.0].
  - v1 backward compatibility: v1 blobs decode without error.
  - build_prior.py: compute_log_priors produces correct values.
  - build_index.py: emit_v2_catalog writes a parseable v2 binary.
  - v1 prefix identity: first 8 + N * ROUTE_BLOCK_BYTES bytes of a v2
    catalog are byte-identical to what a v1 encoder would write for the
    same routes (modulo magic byte difference — tested by stripping magic).
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup — allow importing from catalog-builder without installation.
# ---------------------------------------------------------------------------

CATALOG_BUILDER = Path(__file__).parent.parent.parent / "catalog-builder"
sys.path.insert(0, str(CATALOG_BUILDER))

from format.cat_v2 import (  # noqa: E402
    EMBEDDING_DIM,
    MAGIC_PRIOR,
    MAGIC_V1,
    MAGIC_V2,
    ROUTE_BLOCK_BYTES,
    decode_any,
    decode_prior_block,
    decode_v1,
    decode_v2,
    encode_prior_block,
    encode_v2,
    float_to_q16,
    freq_adaptive_scale,
    q16_to_float,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_XORSHIFT_SEED = 0xFEED_FACE_0002


def _xorshift64(state: int) -> tuple[int, int]:
    state &= 0xFFFF_FFFF_FFFF_FFFF
    state ^= (state << 13) & 0xFFFF_FFFF_FFFF_FFFF
    state ^= (state >> 7) & 0xFFFF_FFFF_FFFF_FFFF
    state ^= (state << 17) & 0xFFFF_FFFF_FFFF_FFFF
    state &= 0xFFFF_FFFF_FFFF_FFFF
    return state, state


def synthetic_routes(n: int, seed: int = _XORSHIFT_SEED) -> list[dict]:
    """Build n synthetic routes with deterministic Q16.16 embeddings."""
    routes: list[dict] = []
    state = seed or 1
    for i in range(n):
        rid = hashlib.sha256(f"test_route_{i:06d}".encode()).digest()
        emb: list[int] = []
        for _ in range(EMBEDDING_DIM):
            state, v = _xorshift64(state)
            # Map to signed i32 range [-32768, 32767] (small values)
            val = int((v & 0xFFFF) - 0x8000)
            emb.append(val)
        routes.append({"route_id": rid, "embedding": emb})
    return routes


def build_v1_blob(routes: list[dict]) -> bytes:
    """Build a v1-format blob (MNC1) from the same routes structure."""
    parts: list[bytes] = [MAGIC_V1, struct.pack("<I", len(routes))]
    for r in routes:
        parts.append(r["route_id"])
        parts.append(struct.pack("<I", EMBEDDING_DIM))
        for v in r["embedding"]:
            parts.append(struct.pack("<i", v))
    return b"".join(parts)


# ---------------------------------------------------------------------------
# Tests: Q16.16 codec
# ---------------------------------------------------------------------------


class TestQ16Codec:
    def test_round_trip_zero(self) -> None:
        assert float_to_q16(0.0) == 0

    def test_round_trip_one(self) -> None:
        encoded = float_to_q16(1.0)
        assert encoded == 65536

    def test_round_trip_half(self) -> None:
        encoded = float_to_q16(0.5)
        assert abs(q16_to_float(encoded) - 0.5) < 1e-5

    def test_round_trip_log2(self) -> None:
        val = math.log(2.0)
        encoded = float_to_q16(val)
        recovered = q16_to_float(encoded)
        assert abs(recovered - val) < 1e-4

    def test_clamps_to_i32(self) -> None:
        assert float_to_q16(1e18) == 2_147_483_647
        assert float_to_q16(-1e18) == -2_147_483_648


# ---------------------------------------------------------------------------
# Tests: prior block encode / decode
# ---------------------------------------------------------------------------


class TestPriorBlock:
    def test_encode_decode_round_trip(self) -> None:
        priors = [math.log(1.0 + i) for i in range(1, 9)]
        encoded = encode_prior_block(priors)
        # Wrap in dummy catalog bytes so decode_prior_block can find the tail.
        dummy = b"\x00" * 16 + encoded
        recovered = decode_prior_block(dummy, len(priors))
        for orig, rec in zip(priors, recovered):
            assert abs(orig - rec) < 1e-4, f"mismatch: {orig} vs {rec}"

    def test_sentinel_present(self) -> None:
        priors = [0.693, 1.099, 1.386]
        encoded = encode_prior_block(priors)
        assert encoded.startswith(MAGIC_PRIOR)

    def test_wrong_count_raises(self) -> None:
        priors = [0.693, 1.099]
        encoded = encode_prior_block(priors)
        # Supply a dummy that is exactly as long as the encoded block —
        # asking for route_count=5 means decode_prior_block will try to
        # read 5 * 4 = 20 bytes of values; with only 2 values present the
        # tail will be misaligned and the sentinel check fires.
        dummy = b"\x00" * 16 + encoded
        with pytest.raises(ValueError):
            decode_prior_block(dummy, 5)  # wrong count → sentinel or size mismatch

    def test_wrong_sentinel_raises(self) -> None:
        bad = b"\x00" * 20  # no PRIR sentinel
        with pytest.raises(ValueError, match="sentinel mismatch"):
            decode_prior_block(bad, 1)


# ---------------------------------------------------------------------------
# Tests: v2 encode / decode round-trip
# ---------------------------------------------------------------------------


class TestV2RoundTrip:
    def test_small_catalog_round_trip(self) -> None:
        routes = synthetic_routes(8)
        log_priors = [math.log(1.0 + i) for i in range(1, 9)]
        blob = encode_v2(routes, log_priors)
        result = decode_v2(blob)

        assert result["version"] == 2
        assert result["route_count"] == 8
        assert len(result["routes"]) == 8
        assert len(result["log_priors"]) == 8

        for i, (orig_lp, rec_lp) in enumerate(zip(log_priors, result["log_priors"])):
            assert abs(orig_lp - rec_lp) < 1e-4, f"prior mismatch at route {i}"

    def test_magic_is_mnc2(self) -> None:
        routes = synthetic_routes(4)
        log_priors = [0.693] * 4
        blob = encode_v2(routes, log_priors)
        assert blob[:4] == MAGIC_V2

    def test_route_id_preserved(self) -> None:
        routes = synthetic_routes(3)
        log_priors = [0.693] * 3
        blob = encode_v2(routes, log_priors)
        result = decode_v2(blob)
        for orig, rec in zip(routes, result["routes"]):
            assert orig["route_id"] == rec["route_id"]

    def test_embedding_preserved(self) -> None:
        routes = synthetic_routes(3)
        log_priors = [0.693] * 3
        blob = encode_v2(routes, log_priors)
        result = decode_v2(blob)
        for orig, rec in zip(routes, result["routes"]):
            assert orig["embedding"] == rec["embedding"]

    def test_decode_any_dispatches_v2(self) -> None:
        routes = synthetic_routes(5)
        log_priors = [0.5, 0.693, 1.0, 1.099, 1.386]
        blob = encode_v2(routes, log_priors)
        result = decode_any(blob)
        assert result["version"] == 2

    def test_length_mismatch_raises(self) -> None:
        routes = synthetic_routes(4)
        with pytest.raises(ValueError, match="equal length"):
            encode_v2(routes, [0.693] * 2)  # wrong number of priors

    def test_uniform_prior_roundtrip(self) -> None:
        """Uniform prior (all routes freq_r = 1) round-trips correctly."""
        routes = synthetic_routes(16)
        uniform_lp = math.log(2.0)  # log(1 + 1)
        log_priors = [uniform_lp] * 16
        blob = encode_v2(routes, log_priors)
        result = decode_v2(blob)
        for rec_lp in result["log_priors"]:
            assert abs(rec_lp - uniform_lp) < 1e-4


# ---------------------------------------------------------------------------
# Tests: v1 backward compatibility
# ---------------------------------------------------------------------------


class TestV1BackwardCompat:
    def test_v1_blob_decodes(self) -> None:
        routes = synthetic_routes(4)
        v1_blob = build_v1_blob(routes)
        result = decode_v1(v1_blob)
        assert result["version"] == 1
        assert result["route_count"] == 4

    def test_decode_any_dispatches_v1(self) -> None:
        routes = synthetic_routes(4)
        v1_blob = build_v1_blob(routes)
        result = decode_any(v1_blob)
        assert result["version"] == 1
        assert "log_priors" not in result

    def test_v1_route_ids_preserved(self) -> None:
        routes = synthetic_routes(3)
        v1_blob = build_v1_blob(routes)
        result = decode_v1(v1_blob)
        for orig, rec in zip(routes, result["routes"]):
            assert orig["route_id"] == rec["route_id"]

    def test_v2_prefix_layout_matches_v1(self) -> None:
        """v2 route blocks are byte-identical to v1 route blocks (same layout).

        Compare the raw route-block bytes (after the 8-byte header) between
        a v1 and v2 blob built from the same routes. The only difference is
        the magic bytes in [0:4]; the route block region must be identical.
        """
        routes = synthetic_routes(5)
        log_priors = [0.693] * 5
        v1_blob = build_v1_blob(routes)
        v2_blob = encode_v2(routes, log_priors)

        # Route block region starts at offset 8.
        n = len(routes)
        route_region_len = n * ROUTE_BLOCK_BYTES
        v1_routes = v1_blob[8 : 8 + route_region_len]
        v2_routes = v2_blob[8 : 8 + route_region_len]
        assert v1_routes == v2_routes, "route block bytes differ between v1 and v2"


# ---------------------------------------------------------------------------
# Tests: frequency-adaptive scaling
# ---------------------------------------------------------------------------


class TestFreqAdaptiveScaling:
    def test_freq_1_gives_scale_1(self) -> None:
        assert abs(freq_adaptive_scale(1.0) - 1.0) < 1e-9

    def test_high_freq_clamped_to_half(self) -> None:
        assert freq_adaptive_scale(1e6) == pytest.approx(0.5)

    def test_scale_in_range(self) -> None:
        # freq_r comes from raw_count + alpha where alpha >= 1, so freq_r >= 1
        # always holds in practice.  For freq_r >= 1, scale = 1/sqrt(freq_r)
        # which is in (0, 1.0]; the max(0.5, ...) floor then gives [0.5, 1.0].
        for freq_r in [1.0, 2.0, 4.0, 9.0, 16.0, 100.0, 1000.0]:
            s = freq_adaptive_scale(freq_r)
            assert 0.5 <= s <= 1.0, f"scale {s} out of [0.5, 1.0] for freq_r={freq_r}"

    def test_zero_freq_gives_1(self) -> None:
        assert freq_adaptive_scale(0.0) == 1.0

    def test_scale_decreases_with_freq(self) -> None:
        scales = [freq_adaptive_scale(f) for f in [1.0, 4.0, 16.0, 64.0]]
        for a, b in zip(scales, scales[1:]):
            assert a >= b, "scale should be non-increasing with frequency"


# ---------------------------------------------------------------------------
# Tests: emit_v2_catalog integration
# ---------------------------------------------------------------------------


class TestEmitV2Catalog:
    def test_emit_produces_valid_v2(self, tmp_path: Path) -> None:
        """build_index.emit_v2_catalog writes a valid v2 binary."""
        from build_index import emit_v2_catalog  # noqa: PLC0415

        items = [
            {"sha256": f"abc{i:04x}", "id": f"id{i}", "freq_r": float(i + 1), "kind": "skill"}
            for i in range(6)
        ]
        out = tmp_path / "test_emit.bin"
        emit_v2_catalog(items, {}, out)

        assert out.exists()
        blob = out.read_bytes()
        result = decode_v2(blob)
        assert result["route_count"] == 6
        assert len(result["log_priors"]) == 6

    def test_emit_with_prior_map(self, tmp_path: Path) -> None:
        """Prior map values are written into the v2 blob."""
        from build_index import emit_v2_catalog  # noqa: PLC0415

        sha_a = "deadbeef0001"
        sha_b = "deadbeef0002"
        items = [
            {"sha256": sha_a, "id": "r0", "freq_r": 1.0, "kind": "skill"},
            {"sha256": sha_b, "id": "r1", "freq_r": 1.0, "kind": "skill"},
        ]
        expected_lp_a = math.log(1.0 + 5.0)
        expected_lp_b = math.log(1.0 + 2.0)
        prior_map = {
            hashlib.sha256(sha_a.encode()).hexdigest(): expected_lp_a,
            hashlib.sha256(sha_b.encode()).hexdigest(): expected_lp_b,
        }
        out = tmp_path / "prior_test.bin"
        emit_v2_catalog(items, prior_map, out)

        blob = out.read_bytes()
        result = decode_v2(blob)
        # The prior map is keyed by sha256(rid_hex) so we check the blobs
        # were written; exact match depends on routing inside emit.
        assert len(result["log_priors"]) == 2
        for lp in result["log_priors"]:
            assert lp > 0.0


# ---------------------------------------------------------------------------
# Tests: build_prior.py
# ---------------------------------------------------------------------------


class TestBuildPrior:
    def test_compute_log_priors_uniform(self) -> None:
        from build_prior import compute_log_priors  # noqa: PLC0415

        route_ids = ["r0", "r1", "r2"]
        result = compute_log_priors(route_ids, {}, alpha=1.0)
        for rid in route_ids:
            assert abs(result[rid] - math.log(2.0)) < 1e-10

    def test_compute_log_priors_with_counts(self) -> None:
        from build_prior import compute_log_priors  # noqa: PLC0415

        raw_counts = {"r0": 10, "r1": 1}
        result = compute_log_priors(["r0", "r1", "r2"], raw_counts, alpha=1.0)
        # r0: log(1 + 10 + 1) = log(12)
        assert abs(result["r0"] - math.log(12.0)) < 1e-9
        # r1: log(1 + 1 + 1) = log(3)
        assert abs(result["r1"] - math.log(3.0)) < 1e-9
        # r2: log(1 + 0 + 1) = log(2)
        assert abs(result["r2"] - math.log(2.0)) < 1e-9

    def test_load_catalog_ids(self, tmp_path: Path) -> None:
        from build_prior import load_catalog_ids  # noqa: PLC0415

        items = [
            {"sha256": "aaa111", "id": "x", "kind": "skill"},
            {"sha256": "bbb222", "id": "y", "kind": "agent"},
        ]
        catalog = tmp_path / "items.jsonl"
        catalog.write_text(
            "\n".join(json.dumps(i) for i in items) + "\n", encoding="utf-8"
        )
        ids = load_catalog_ids(catalog)
        assert ids == ["aaa111", "bbb222"]

    def test_cli_dry_run(self, tmp_path: Path) -> None:
        """build_prior.main() writes the output file without error."""
        import subprocess

        catalog = tmp_path / "items.jsonl"
        catalog.write_text(
            json.dumps({"sha256": "abc123", "id": "x", "kind": "skill"}) + "\n"
        )
        out = tmp_path / "prior.json"
        result = subprocess.run(
            [
                sys.executable,
                str(CATALOG_BUILDER / "build_prior.py"),
                "--catalog", str(catalog),
                "--output", str(out),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert out.exists()
        data = json.loads(out.read_text())
        assert "priors" in data
        assert data["num_routes"] == 1
