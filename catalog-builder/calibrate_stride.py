#!/usr/bin/env python3
"""Calibrate adaptive window-stride thresholds from request-length distribution.

Reads a request-length sample (one integer per line, or JSONL with a
"token_count" field) captured during catalog-build or profiling, then computes
two percentile thresholds:

    STRIDE_FP_T_LOW   — 25th percentile of token lengths
    STRIDE_FP_T_HIGH  — 75th percentile of token lengths

These thresholds are written into the catalog manifest under a ``[stride]``
TOML table (and the companion manifest.json) so the runtime can select a
stride value appropriate for the request length distribution seen during
catalog construction.

Runtime selection logic (described here; implemented in src/lib.mind):
  - request_len < T_LOW  → use ATTN_WINDOW_STRIDE_COMPACT  (192 — tight overlap,
                            short context, faster)
  - request_len > T_HIGH → use ATTN_WINDOW_STRIDE_LONG     (128 — more overlap,
                            long context, better recall)
  - otherwise             → use ATTN_WINDOW_STRIDE          (192 — default)

Thresholds are clamped to [96, 256] matching the constraint in the spec.
They are emitted as integers (token counts, not Q16.16 — strides are counts).

The constant-stride compile-time path in src/lib.mind is NOT removed.
It is gated behind the ``ADAPTIVE_STRIDE_ENABLED = 0`` constant (RFC-003);
when 0 the compile-time ATTN_WINDOW_STRIDE constant dominates and this
manifest section is ignored.  Setting ADAPTIVE_STRIDE_ENABLED = 1 activates
the manifest-driven path and requires a new model_hash (architecture change).

Output files
------------
  <out_dir>/manifest.toml   — TOML fragment; safe to include into main catalog
                              manifest with ``include``
  <out_dir>/manifest.json   — JSON equivalent for build pipeline consumption

Usage
-----
    # From a plain integer-per-line lengths file:
    python3 calibrate_stride.py \\
        --lengths /tmp/request_lengths.txt \\
        --output  catalog-data/stride/

    # From JSONL with "token_count" field:
    python3 calibrate_stride.py \\
        --lengths /tmp/events.jsonl \\
        --output  catalog-data/stride/

    # Dry-run (print only, write nothing):
    python3 calibrate_stride.py --lengths /tmp/lengths.txt --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

# Hard bounds from spec (token counts, not Q16.16).
T_MIN: int = 96
T_MAX: int = 256

# Percentiles used for low and high thresholds.
P_LOW: float = 25.0
P_HIGH: float = 75.0

# Stride presets (token counts).
STRIDE_COMPACT: int = 192   # short requests — window stride (default)
STRIDE_DEFAULT: int = 192   # mid requests — same as compact in base config
STRIDE_LONG: int = 128      # long requests — more overlap


def load_lengths(path: Path) -> list[int]:
    """Load token-length samples from a file.

    Accepts two formats:
    - One integer per line.
    - JSONL with a ``token_count`` integer field.

    Returns a non-empty sorted list of positive integers.
    """
    lengths: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            if raw.startswith("{"):
                try:
                    obj = json.loads(raw)
                    n = int(obj.get("token_count", 0))
                except (json.JSONDecodeError, ValueError):
                    continue
            else:
                try:
                    n = int(raw)
                except ValueError:
                    continue
            if n > 0:
                lengths.append(n)
    return lengths


def percentile(values: Sequence[int], p: float) -> float:
    """Return the p-th percentile of a sequence using linear interpolation."""
    if not values:
        raise ValueError("empty sequence")
    sorted_v = sorted(values)
    n = len(sorted_v)
    if n == 1:
        return float(sorted_v[0])
    idx = (p / 100.0) * (n - 1)
    lo = int(math.floor(idx))
    hi = lo + 1
    if hi >= n:
        return float(sorted_v[-1])
    frac = idx - lo
    return sorted_v[lo] * (1.0 - frac) + sorted_v[hi] * frac


def clamp_threshold(value: float) -> int:
    """Round and clamp a threshold to the [T_MIN, T_MAX] range."""
    return max(T_MIN, min(T_MAX, int(round(value))))


def compute_thresholds(lengths: list[int]) -> tuple[int, int]:
    """Return (T_LOW, T_HIGH) clamped to [T_MIN, T_MAX]."""
    t_low = clamp_threshold(percentile(lengths, P_LOW))
    t_high = clamp_threshold(percentile(lengths, P_HIGH))
    # Ensure ordering even after clamping.
    if t_low >= t_high:
        t_high = min(T_MAX, t_low + 1)
    return t_low, t_high


def build_manifest(
    t_low: int,
    t_high: int,
    sample_count: int,
    source: str,
) -> dict:
    return {
        "schema_version": 1,
        "source": source,
        "sample_count": sample_count,
        "percentile_low": P_LOW,
        "percentile_high": P_HIGH,
        "stride": {
            "STRIDE_FP_T_LOW": t_low,
            "STRIDE_FP_T_HIGH": t_high,
            "STRIDE_COMPACT": STRIDE_COMPACT,
            "STRIDE_DEFAULT": STRIDE_DEFAULT,
            "STRIDE_LONG": STRIDE_LONG,
            "note": (
                "Thresholds are request token counts (integers), not Q16.16. "
                "Runtime reads these when ADAPTIVE_STRIDE_ENABLED = 1. "
                "Flipping ADAPTIVE_STRIDE_ENABLED is an architecture change "
                "requiring a new model_hash."
            ),
        },
    }


def build_toml_fragment(manifest: dict) -> str:
    """Emit a TOML ``[stride]`` fragment from the manifest."""
    s = manifest["stride"]
    lines = [
        "[stride]",
        f"STRIDE_FP_T_LOW   = {s['STRIDE_FP_T_LOW']}",
        f"STRIDE_FP_T_HIGH  = {s['STRIDE_FP_T_HIGH']}",
        f"STRIDE_COMPACT    = {s['STRIDE_COMPACT']}",
        f"STRIDE_DEFAULT    = {s['STRIDE_DEFAULT']}",
        f"STRIDE_LONG       = {s['STRIDE_LONG']}",
        f'note = "{s["note"]}"',
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Calibrate adaptive stride thresholds from request-length distribution"
    )
    ap.add_argument(
        "--lengths",
        type=Path,
        required=True,
        help="File with per-request token counts (one int per line or JSONL with token_count)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for manifest.toml + manifest.json",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print thresholds but write no files",
    )
    args = ap.parse_args()

    if not args.lengths.exists():
        sys.exit(f"lengths file not found: {args.lengths}")

    lengths = load_lengths(args.lengths)
    if not lengths:
        sys.exit(f"no valid token-count samples found in {args.lengths}")

    t_low, t_high = compute_thresholds(lengths)
    manifest = build_manifest(t_low, t_high, len(lengths), str(args.lengths))
    toml_fragment = build_toml_fragment(manifest)

    summary = {
        "sample_count": len(lengths),
        "STRIDE_FP_T_LOW": t_low,
        "STRIDE_FP_T_HIGH": t_high,
        "bounds": [T_MIN, T_MAX],
    }
    print(json.dumps(summary, indent=2))

    if args.dry_run:
        print("\n# TOML fragment (dry-run):")
        print(toml_fragment)
        return

    if args.output is None:
        sys.exit("--output required when not using --dry-run")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (args.output / "manifest.toml").write_text(toml_fragment)

    print(f"wrote stride manifest to {args.output}/", file=sys.stderr)


if __name__ == "__main__":
    main()
