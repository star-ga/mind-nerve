"""Integration tests for the MNC1 catalog format (U4b).

U4b (2026-08-19): the old "v2" format tested here (magic ``MNC2`` + a
trailing ``PRIR`` prior block) was never accepted by any shipping loader —
``src/loader.mind``'s ``loader_parse_catalog`` requires a u16
version + u16 reserved header, a canonical SHA-256 catalog hash, and a
UTF-8 external-id trailer. That branch is retired (see
``catalog-builder/format/cat_v2.py``'s module docstring); this file now pins
the MNC1 format the loader actually reads.

Covers:
  - Q16.16 codec + frequency-adaptive scaling (unchanged, kept verbatim).
  - MNC1 encode -> decode round-trip (v1 no-prior, v2 inline-prior).
  - Every ``loader_parse_catalog`` gate the pure-Python decoder mirrors:
    magic, version, reserved-zero, canonical hash, trailer/UTF-8.
  - build_index.py: ``emit_mnc1_catalog`` writes a parseable MNC1 binary.
  - Cross-language oracle: ``encode_mnc1`` reproduces
    ``tests/bit_identity/gen_fixtures.py``'s ``build_catalog_bin`` bytes
    EXACTLY for the same routes — the two authoritative producers must never
    drift from each other.
  - AE8 (architecture-plan): no live encoder/decoder emits or accepts the
    retired ``MNC2`` magic outside history/docs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import struct
import sys
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Path setup — allow importing from catalog-builder without installation.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_BUILDER = REPO_ROOT / "catalog-builder"
sys.path.insert(0, str(CATALOG_BUILDER))

from format.cat_v2 import (  # noqa: E402
    CAT_VERSION_1,
    CAT_VERSION_2,
    EMBEDDING_DIM,
    HEADER_BYTES,
    MAGIC_MNC1,
    catalog_preimage_and_hash,
    decode_mnc1,
    encode_mnc1,
    float_to_q16,
    freq_adaptive_scale,
    q16_to_float,
)


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def synthetic_routes(n: int, seed: int = 0xFEED_FACE_0002) -> tuple[list[bytes], list[list[int]]]:
    """Build n synthetic (route_id, embedding) pairs, deterministic."""
    route_ids: list[bytes] = []
    embeddings: list[list[int]] = []
    state = seed or 1

    def xorshift64() -> int:
        nonlocal state
        state ^= (state << 13) & 0xFFFF_FFFF_FFFF_FFFF
        state ^= (state >> 7) & 0xFFFF_FFFF_FFFF_FFFF
        state ^= (state << 17) & 0xFFFF_FFFF_FFFF_FFFF
        state &= 0xFFFF_FFFF_FFFF_FFFF
        return state

    for i in range(n):
        route_ids.append(hashlib.sha256(f"test_route_{i:06d}".encode()).digest())
        emb = [int((xorshift64() & 0xFFFF) - 0x8000) for _ in range(EMBEDDING_DIM)]
        embeddings.append(emb)
    return route_ids, embeddings


def synthetic_external_ids(n: int) -> list[bytes]:
    return [f"route_{i:06d}".encode("utf-8") for i in range(n)]


# ---------------------------------------------------------------------------
# Tests: Q16.16 codec (unchanged behavior)
# ---------------------------------------------------------------------------


class TestQ16Codec:
    def test_round_trip_zero(self) -> None:
        assert float_to_q16(0.0) == 0

    def test_round_trip_one(self) -> None:
        assert float_to_q16(1.0) == 65536

    def test_round_trip_half(self) -> None:
        assert abs(q16_to_float(float_to_q16(0.5)) - 0.5) < 1e-5

    def test_round_trip_log2(self) -> None:
        val = math.log(2.0)
        assert abs(q16_to_float(float_to_q16(val)) - val) < 1e-4

    def test_clamps_to_i32(self) -> None:
        assert float_to_q16(1e18) == 2_147_483_647
        assert float_to_q16(-1e18) == -2_147_483_648


class TestFreqAdaptiveScaling:
    def test_freq_1_gives_scale_1(self) -> None:
        assert abs(freq_adaptive_scale(1.0) - 1.0) < 1e-9

    def test_high_freq_clamped_to_half(self) -> None:
        assert freq_adaptive_scale(1e6) == pytest.approx(0.5)

    def test_scale_in_range(self) -> None:
        for freq_r in [1.0, 2.0, 4.0, 9.0, 16.0, 100.0, 1000.0]:
            s = freq_adaptive_scale(freq_r)
            assert 0.5 <= s <= 1.0, f"scale {s} out of [0.5, 1.0] for freq_r={freq_r}"

    def test_zero_freq_gives_1(self) -> None:
        assert freq_adaptive_scale(0.0) == 1.0

    def test_scale_decreases_with_freq(self) -> None:
        scales = [freq_adaptive_scale(f) for f in [1.0, 4.0, 16.0, 64.0]]
        for a, b in zip(scales, scales[1:], strict=False):
            assert a >= b, "scale should be non-increasing with frequency"


# ---------------------------------------------------------------------------
# Tests: MNC1 encode / decode round-trip
# ---------------------------------------------------------------------------


class TestMnc1RoundTrip:
    def test_v1_no_prior_round_trip(self) -> None:
        route_ids, embeddings = synthetic_routes(8)
        external_ids = synthetic_external_ids(8)
        blob = encode_mnc1(route_ids, embeddings, external_ids)
        assert blob[:4] == MAGIC_MNC1
        result = decode_mnc1(blob)
        assert result["version"] == CAT_VERSION_1
        assert result["route_count"] == 8
        assert "log_priors" not in result
        for i, route in enumerate(result["routes"]):
            assert route["route_id"] == route_ids[i]
            assert route["embedding"] == embeddings[i]
            assert route["external_id"] == external_ids[i]

    def test_v2_inline_prior_round_trip(self) -> None:
        route_ids, embeddings = synthetic_routes(6)
        external_ids = synthetic_external_ids(6)
        log_priors = [math.log(1.0 + i) for i in range(1, 7)]
        blob = encode_mnc1(route_ids, embeddings, external_ids, log_priors=log_priors)
        result = decode_mnc1(blob)
        assert result["version"] == CAT_VERSION_2
        assert result["route_count"] == 6
        for orig, rec in zip(log_priors, result["log_priors"], strict=True):
            assert abs(orig - rec) < 1e-4

    def test_empty_catalog_round_trips(self) -> None:
        blob = encode_mnc1([], [], [])
        result = decode_mnc1(blob)
        assert result["route_count"] == 0
        assert result["routes"] == []

    def test_header_bytes_constant(self) -> None:
        route_ids, embeddings = synthetic_routes(1)
        external_ids = synthetic_external_ids(1)
        blob = encode_mnc1(route_ids, embeddings, external_ids)
        # [0:4] magic, [4:6] version, [6:8] reserved, [8:12] n, [12:16] dim,
        # [16:48] hash == 48 bytes total before the first route id.
        assert HEADER_BYTES == 48
        assert blob[HEADER_BYTES : HEADER_BYTES + 32] == route_ids[0]

    def test_length_mismatch_raises(self) -> None:
        route_ids, embeddings = synthetic_routes(4)
        with pytest.raises(ValueError, match="equal length"):
            encode_mnc1(route_ids, embeddings, synthetic_external_ids(2))

    def test_prior_length_mismatch_raises(self) -> None:
        route_ids, embeddings = synthetic_routes(4)
        with pytest.raises(ValueError, match="log_priors"):
            encode_mnc1(route_ids, embeddings, synthetic_external_ids(4), log_priors=[0.1, 0.2])

    def test_bad_route_id_length_raises(self) -> None:
        with pytest.raises(ValueError, match="32 bytes"):
            encode_mnc1([b"short"], [[0] * EMBEDDING_DIM], [b"x"])

    def test_bad_embedding_length_raises(self) -> None:
        route_ids, _ = synthetic_routes(1)
        with pytest.raises(ValueError, match=f"{EMBEDDING_DIM}"):
            encode_mnc1(route_ids, [[0] * (EMBEDDING_DIM - 1)], [b"x"])


# ---------------------------------------------------------------------------
# Tests: every loader_parse_catalog gate the decoder mirrors
# ---------------------------------------------------------------------------


class TestLoaderGates:
    def _valid_blob(self, n: int = 4) -> bytes:
        route_ids, embeddings = synthetic_routes(n)
        return encode_mnc1(route_ids, embeddings, synthetic_external_ids(n))

    def test_rejects_bad_magic(self) -> None:
        blob = bytearray(self._valid_blob())
        blob[0:4] = b"XXXX"
        with pytest.raises(ValueError, match="magic"):
            decode_mnc1(bytes(blob))

    def test_rejects_retired_mnc2_magic(self) -> None:
        """AE8 regression: the retired MNC2 magic must never decode."""
        blob = bytearray(self._valid_blob())
        blob[0:4] = b"MNC2"
        with pytest.raises(ValueError, match="magic"):
            decode_mnc1(bytes(blob))

    def test_rejects_unsupported_version(self) -> None:
        blob = bytearray(self._valid_blob())
        struct.pack_into("<H", blob, 4, 3)
        with pytest.raises(ValueError, match="version"):
            decode_mnc1(bytes(blob))

    def test_rejects_nonzero_reserved(self) -> None:
        blob = bytearray(self._valid_blob())
        struct.pack_into("<H", blob, 6, 1)
        with pytest.raises(ValueError, match="reserved"):
            decode_mnc1(bytes(blob))

    def test_rejects_wrong_embedding_dim(self) -> None:
        blob = bytearray(self._valid_blob())
        struct.pack_into("<I", blob, 12, 128)
        with pytest.raises(ValueError, match="embedding_dim"):
            decode_mnc1(bytes(blob))

    def test_rejects_tampered_embedding_via_hash(self) -> None:
        blob = bytearray(self._valid_blob())
        # Flip one byte inside the first route's embedding — the canonical
        # hash must catch it even though every other gate still passes.
        blob[HEADER_BYTES + 32] ^= 0xFF
        with pytest.raises(ValueError, match="hash"):
            decode_mnc1(bytes(blob))

    def test_rejects_truncated_blob(self) -> None:
        blob = self._valid_blob()[:-10]
        with pytest.raises(ValueError):
            decode_mnc1(blob)

    def test_too_short_for_header_rejected(self) -> None:
        with pytest.raises(ValueError, match="header"):
            decode_mnc1(b"\x00" * 10)


# ---------------------------------------------------------------------------
# Tests: canonical hash helper
# ---------------------------------------------------------------------------


class TestCanonicalHash:
    def test_hash_changes_with_embedding(self) -> None:
        route_ids, embeddings = synthetic_routes(3)
        emb_rows_a = [struct.pack(f"<{EMBEDDING_DIM}i", *row) for row in embeddings]
        emb_rows_b = [struct.pack(f"<{EMBEDDING_DIM}i", *([1] + row[1:])) for row in embeddings]
        hash_a = catalog_preimage_and_hash(route_ids, emb_rows_a)
        hash_b = catalog_preimage_and_hash(route_ids, emb_rows_b)
        assert hash_a != hash_b

    def test_hash_length_is_32(self) -> None:
        route_ids, embeddings = synthetic_routes(2)
        emb_rows = [struct.pack(f"<{EMBEDDING_DIM}i", *row) for row in embeddings]
        assert len(catalog_preimage_and_hash(route_ids, emb_rows)) == 32


# ---------------------------------------------------------------------------
# Tests: emit_mnc1_catalog (build_index.py) integration
# ---------------------------------------------------------------------------


class TestEmitMnc1Catalog:
    def test_emit_produces_valid_mnc1_v1(self, tmp_path: Path) -> None:
        """No prior file -> version 1 (no prior block)."""
        from build_index import emit_mnc1_catalog  # noqa: PLC0415

        items = [
            {"sha256": f"abc{i:04x}", "id": f"id{i}", "freq_r": float(i + 1), "kind": "skill"}
            for i in range(6)
        ]
        out = tmp_path / "test_emit.bin"
        emit_mnc1_catalog(items, {}, out)

        assert out.exists()
        result = decode_mnc1(out.read_bytes())
        assert result["version"] == CAT_VERSION_1
        assert result["route_count"] == 6
        assert "log_priors" not in result

    def test_emit_with_prior_map_produces_v2(self, tmp_path: Path) -> None:
        """A prior file -> version 2 (inline route-prior)."""
        from build_index import emit_mnc1_catalog  # noqa: PLC0415

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
        emit_mnc1_catalog(items, prior_map, out)

        result = decode_mnc1(out.read_bytes())
        assert result["version"] == CAT_VERSION_2
        assert len(result["log_priors"]) == 2
        for lp in result["log_priors"]:
            assert lp > 0.0

    def test_emit_external_ids_are_the_sha256(self, tmp_path: Path) -> None:
        from build_index import emit_mnc1_catalog  # noqa: PLC0415

        items = [{"sha256": "cafef00d", "id": "r0", "freq_r": 1.0, "kind": "skill"}]
        out = tmp_path / "ext_id_test.bin"
        emit_mnc1_catalog(items, {}, out)
        result = decode_mnc1(out.read_bytes())
        assert result["routes"][0]["external_id"] == b"cafef00d"


# ---------------------------------------------------------------------------
# Tests: cross-language oracle vs tests/bit_identity/gen_fixtures.py
# ---------------------------------------------------------------------------


class TestGenFixturesOracle:
    """encode_mnc1 must reproduce gen_fixtures.build_catalog_bin byte-for-byte
    — the fixture generator and the production encoder are two independent
    implementations of the same spec and must never silently diverge."""

    @pytest.fixture(scope="class")
    def gen_fixtures(self) -> Any:
        return _load_module(
            REPO_ROOT / "tests" / "bit_identity" / "gen_fixtures.py", "gen_fixtures_oracle"
        )

    @pytest.mark.parametrize("n_routes", [0, 1, 4, 44])
    def test_matches_gen_fixtures_byte_for_byte(self, gen_fixtures: Any, n_routes: int) -> None:
        gf = gen_fixtures
        expected = gf.build_catalog_bin(n_routes)

        route_ids = [gf.route_id_bytes(s) for s in range(n_routes)]
        embeddings: list[list[int]] = []
        for s in range(n_routes):
            row_seed = (gf.SEED_CATALOG ^ ((s + 1) * 0x9E37_79B9_7F4A_7C15)) & 0xFFFF_FFFF_FFFF_FFFF
            row_bytes = gf.i32_stream_bytes(row_seed, gf.HIDDEN_DIM, -1310, 1310)
            embeddings.append(list(struct.unpack(f"<{gf.HIDDEN_DIM}i", row_bytes)))
        external_ids = [gf.route_external_id(s) for s in range(n_routes)]

        actual = encode_mnc1(route_ids, embeddings, external_ids)
        assert actual == expected


# ---------------------------------------------------------------------------
# Tests: build_prior.py (unchanged by U4b, kept for regression coverage)
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
        assert abs(result["r0"] - math.log(12.0)) < 1e-9
        assert abs(result["r1"] - math.log(3.0)) < 1e-9
        assert abs(result["r2"] - math.log(2.0)) < 1e-9

    def test_load_catalog_ids(self, tmp_path: Path) -> None:
        import json as _json

        from build_prior import load_catalog_ids  # noqa: PLC0415

        items = [
            {"sha256": "aaa111", "id": "x", "kind": "skill"},
            {"sha256": "bbb222", "id": "y", "kind": "agent"},
        ]
        catalog = tmp_path / "items.jsonl"
        catalog.write_text("\n".join(_json.dumps(i) for i in items) + "\n", encoding="utf-8")
        ids = load_catalog_ids(catalog)
        assert ids == ["aaa111", "bbb222"]

    def test_cli_dry_run(self, tmp_path: Path) -> None:
        """build_prior.main() writes the output file without error."""
        import json as _json
        import subprocess

        catalog = tmp_path / "items.jsonl"
        catalog.write_text(_json.dumps({"sha256": "abc123", "id": "x", "kind": "skill"}) + "\n")
        out = tmp_path / "prior.json"
        result = subprocess.run(
            [
                sys.executable,
                str(CATALOG_BUILDER / "build_prior.py"),
                "--catalog",
                str(catalog),
                "--output",
                str(out),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert out.exists()
