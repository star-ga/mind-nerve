"""Integration tests for adaptive window stride threshold calibration.

Covers:
  - percentile computation correctness.
  - T_LOW / T_HIGH clamped to [96, 256] range.
  - Thresholds appear in manifest JSON with correct structure.
  - Runtime selector logic: given T_LOW and T_HIGH, correct stride is chosen.
  - JSONL input format (token_count field) parses correctly.
  - Edge cases: single-sample, all-same-length, extreme values.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import pytest

CATALOG_BUILDER = Path(__file__).parent.parent.parent / "catalog-builder"
sys.path.insert(0, str(CATALOG_BUILDER))

from calibrate_stride import (  # noqa: E402
    P_HIGH,
    P_LOW,
    STRIDE_COMPACT,
    STRIDE_DEFAULT,
    STRIDE_LONG,
    T_MAX,
    T_MIN,
    build_manifest,
    clamp_threshold,
    compute_thresholds,
    load_lengths,
    percentile,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_lengths_file(tmp_path: Path, lengths: list[int], fmt: str = "plain") -> Path:
    p = tmp_path / "lengths.txt"
    if fmt == "plain":
        p.write_text("\n".join(str(n) for n in lengths) + "\n")
    elif fmt == "jsonl":
        p.write_text(
            "\n".join(json.dumps({"token_count": n}) for n in lengths) + "\n"
        )
    return p


# ---------------------------------------------------------------------------
# Tests: percentile computation
# ---------------------------------------------------------------------------


class TestPercentile:
    def test_median_odd(self) -> None:
        vals = [1, 2, 3, 4, 5]
        assert percentile(vals, 50.0) == pytest.approx(3.0)

    def test_median_even(self) -> None:
        vals = [1, 2, 3, 4]
        assert percentile(vals, 50.0) == pytest.approx(2.5)

    def test_p25_simple(self) -> None:
        vals = [10, 20, 30, 40]
        assert percentile(vals, 25.0) == pytest.approx(17.5)

    def test_p75_simple(self) -> None:
        vals = [10, 20, 30, 40]
        assert percentile(vals, 75.0) == pytest.approx(32.5)

    def test_single_element(self) -> None:
        assert percentile([42], 50.0) == pytest.approx(42.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            percentile([], 50.0)


# ---------------------------------------------------------------------------
# Tests: clamp_threshold
# ---------------------------------------------------------------------------


class TestClampThreshold:
    def test_below_min_clamped(self) -> None:
        assert clamp_threshold(10.0) == T_MIN

    def test_above_max_clamped(self) -> None:
        assert clamp_threshold(9999.0) == T_MAX

    def test_mid_value_unchanged(self) -> None:
        assert clamp_threshold(150.0) == 150

    def test_rounding(self) -> None:
        assert clamp_threshold(149.6) == 150
        assert clamp_threshold(149.4) == 149


# ---------------------------------------------------------------------------
# Tests: compute_thresholds
# ---------------------------------------------------------------------------


class TestComputeThresholds:
    def test_normal_distribution_in_range(self) -> None:
        lengths = list(range(50, 300, 5))  # 50 values from 50 to 295
        t_low, t_high = compute_thresholds(lengths)
        assert T_MIN <= t_low <= T_MAX
        assert T_MIN <= t_high <= T_MAX
        assert t_low < t_high

    def test_extreme_low_values_clamp_to_t_min(self) -> None:
        lengths = [5] * 100  # all very short
        t_low, t_high = compute_thresholds(lengths)
        assert t_low == T_MIN
        assert t_high == T_MIN + 1 or t_high == T_MIN  # ordering preserved

    def test_extreme_high_values_clamp_to_t_max(self) -> None:
        lengths = [10000] * 100  # all very long
        t_low, t_high = compute_thresholds(lengths)
        assert t_high == T_MAX

    def test_typical_request_lengths(self) -> None:
        """Realistic distribution: most requests 64-512 tokens."""
        lengths = [64, 96, 128, 128, 192, 192, 256, 256, 384, 512]
        t_low, t_high = compute_thresholds(lengths)
        # Both must be in valid range.
        assert T_MIN <= t_low <= T_MAX
        assert T_MIN <= t_high <= T_MAX
        assert t_low <= t_high

    def test_single_sample(self) -> None:
        t_low, t_high = compute_thresholds([200])
        assert T_MIN <= t_low <= T_MAX
        assert T_MIN <= t_high <= T_MAX


# ---------------------------------------------------------------------------
# Tests: manifest structure
# ---------------------------------------------------------------------------


class TestBuildManifest:
    def test_manifest_keys(self) -> None:
        m = build_manifest(128, 192, 100, "test.txt")
        assert "stride" in m
        assert m["stride"]["STRIDE_FP_T_LOW"] == 128
        assert m["stride"]["STRIDE_FP_T_HIGH"] == 192
        assert m["stride"]["STRIDE_COMPACT"] == STRIDE_COMPACT
        assert m["stride"]["STRIDE_DEFAULT"] == STRIDE_DEFAULT
        assert m["stride"]["STRIDE_LONG"] == STRIDE_LONG

    def test_thresholds_in_range(self) -> None:
        lengths = list(range(100, 300, 10))
        t_low, t_high = compute_thresholds(lengths)
        m = build_manifest(t_low, t_high, len(lengths), "test")
        assert T_MIN <= m["stride"]["STRIDE_FP_T_LOW"] <= T_MAX
        assert T_MIN <= m["stride"]["STRIDE_FP_T_HIGH"] <= T_MAX


# ---------------------------------------------------------------------------
# Tests: file I/O
# ---------------------------------------------------------------------------


class TestLoadLengths:
    def test_plain_format(self, tmp_path: Path) -> None:
        f = write_lengths_file(tmp_path, [128, 256, 512], fmt="plain")
        result = load_lengths(f)
        assert sorted(result) == [128, 256, 512]

    def test_jsonl_format(self, tmp_path: Path) -> None:
        f = write_lengths_file(tmp_path, [64, 192, 384], fmt="jsonl")
        result = load_lengths(f)
        assert sorted(result) == [64, 192, 384]

    def test_skips_zero_and_negative(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.txt"
        f.write_text("0\n-1\n100\n200\n")
        result = load_lengths(f)
        assert result == [100, 200]

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        f = tmp_path / "blanks.txt"
        f.write_text("100\n\n200\n\n300\n")
        result = load_lengths(f)
        assert sorted(result) == [100, 200, 300]


# ---------------------------------------------------------------------------
# Tests: calibrate_stride CLI + manifest file emit
# ---------------------------------------------------------------------------


class TestCalibrateStrideCLI:
    def test_cli_writes_manifest_json(self, tmp_path: Path) -> None:
        import subprocess

        lengths_file = write_lengths_file(
            tmp_path, list(range(96, 512, 8)), fmt="plain"
        )
        out_dir = tmp_path / "stride_out"
        result = subprocess.run(
            [
                sys.executable,
                str(CATALOG_BUILDER / "calibrate_stride.py"),
                "--lengths", str(lengths_file),
                "--output", str(out_dir),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        manifest_path = out_dir / "manifest.json"
        assert manifest_path.exists()
        m = json.loads(manifest_path.read_text())
        assert "stride" in m
        t_low = m["stride"]["STRIDE_FP_T_LOW"]
        t_high = m["stride"]["STRIDE_FP_T_HIGH"]
        assert T_MIN <= t_low <= T_MAX
        assert T_MIN <= t_high <= T_MAX

    def test_cli_writes_toml_fragment(self, tmp_path: Path) -> None:
        import subprocess

        lengths_file = write_lengths_file(tmp_path, [128, 192, 256], fmt="plain")
        out_dir = tmp_path / "toml_out"
        subprocess.run(
            [
                sys.executable,
                str(CATALOG_BUILDER / "calibrate_stride.py"),
                "--lengths", str(lengths_file),
                "--output", str(out_dir),
            ],
            capture_output=True,
            check=True,
        )
        toml_path = out_dir / "manifest.toml"
        assert toml_path.exists()
        toml_text = toml_path.read_text()
        assert "[stride]" in toml_text
        assert "STRIDE_FP_T_LOW" in toml_text
        assert "STRIDE_FP_T_HIGH" in toml_text

    def test_dry_run_no_files(self, tmp_path: Path) -> None:
        import subprocess

        lengths_file = write_lengths_file(tmp_path, [100, 200, 300], fmt="plain")
        result = subprocess.run(
            [
                sys.executable,
                str(CATALOG_BUILDER / "calibrate_stride.py"),
                "--lengths", str(lengths_file),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        # No files written in dry-run.
        assert not (tmp_path / "manifest.json").exists()


# ---------------------------------------------------------------------------
# Tests: runtime stride selector logic
# ---------------------------------------------------------------------------


class TestRuntimeStrideSelector:
    """Verify that the selector described in calibrate_stride.py is correct.

    This test exercises the MIND-side invariant in Python to ensure the
    threshold semantics are unambiguous before the MIND constant flip.
    """

    @staticmethod
    def select_stride(
        request_len: int,
        t_low: int,
        t_high: int,
    ) -> int:
        """Python re-implementation of the RFC-003 runtime selector."""
        if request_len < t_low:
            return STRIDE_COMPACT
        if request_len > t_high:
            return STRIDE_LONG
        return STRIDE_DEFAULT

    def test_below_t_low_selects_compact(self) -> None:
        assert self.select_stride(50, 128, 192) == STRIDE_COMPACT

    def test_above_t_high_selects_long(self) -> None:
        assert self.select_stride(300, 128, 192) == STRIDE_LONG

    def test_in_range_selects_default(self) -> None:
        assert self.select_stride(150, 128, 192) == STRIDE_DEFAULT

    def test_at_t_low_boundary_selects_default(self) -> None:
        assert self.select_stride(128, 128, 192) == STRIDE_DEFAULT

    def test_at_t_high_boundary_selects_default(self) -> None:
        assert self.select_stride(192, 128, 192) == STRIDE_DEFAULT

    def test_manifest_driven_thresholds_in_range(self, tmp_path: Path) -> None:
        """Thresholds from a real manifest file stay in [96, 256]."""
        lengths = list(range(80, 600, 12))  # varied distribution
        t_low, t_high = compute_thresholds(lengths)
        m = build_manifest(t_low, t_high, len(lengths), "synthetic")
        s = m["stride"]
        assert T_MIN <= s["STRIDE_FP_T_LOW"] <= T_MAX
        assert T_MIN <= s["STRIDE_FP_T_HIGH"] <= T_MAX
