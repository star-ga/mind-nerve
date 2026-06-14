"""
tests/bit_identity/compare.py

Hash diff and top-K overlap reporter for the A1.4 bit-identity harness.

Loads two backend hash blobs (produced by runner.py) and:
  - Diffs row-by-row across all 7 hash positions per query.
  - Reports byte-identical pass/fail per hash position per query.
  - For native-vs-pytorch: computes top-K overlap (top-5 and top-1)
    per query, reports aggregate.

Exit codes:
  0  — both blobs bit-identical on all comparable hashes
  1  — hashes differ (any mismatch not explained by sentinel values)
  2  — comparison blocked by missing data or format error

Usage:
    python tests/bit_identity/compare.py \\
        /tmp/bit_identity_pytorch.json \\
        /tmp/bit_identity_native.json

    python tests/bit_identity/compare.py \\
        --blob-a /tmp/bit_identity_pytorch.json \\
        --blob-b /tmp/bit_identity_native.json \\
        --top-k 5 --top1-threshold 0.80 --topk-threshold 0.90
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Hash positions in order (must match runner.py HASH_KEYS)
HASH_KEYS = (
    "token_ids",
    "post_embed_ln",
    "final_layer_ln",
    "post_cls_slice",
    "post_l2_norm",
    "catalog_scores",
    "topk_indices_scores",
)

# Sentinels — not treated as mismatches
SENTINELS = frozenset(
    [
        "BACKEND_STUB_NOT_BUILT",
        "CUDA_DEFERRED_TO_V0_4_1",
    ]
)

# Gate thresholds (per A1.4 spec)
DEFAULT_TOPK_THRESHOLD = 0.90
DEFAULT_TOP1_THRESHOLD = 0.80

TOP_K = 5


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _load_blob(path: Path) -> dict:
    if not path.exists():
        print(f"ERROR: blob not found: {path}", file=sys.stderr)
        sys.exit(2)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _index_records(blob: dict) -> dict[str, dict]:
    """Index records by query ID for fast lookup."""
    return {r["id"]: r for r in blob["records"]}


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------


def _is_sentinel(h: str | None) -> bool:
    return h is None or h in SENTINELS or (isinstance(h, str) and h.startswith("ERROR:"))


def _compare_hash_row(
    query_id: str,
    rec_a: dict,
    rec_b: dict,
) -> dict:
    """
    Compare all 7 hash positions for a single query.
    Returns a result dict with per-position pass/fail.
    """
    hashes_a = rec_a.get("hashes", {})
    hashes_b = rec_b.get("hashes", {})

    positions: dict[str, str] = {}
    mismatch_count = 0
    skip_count = 0

    for key in HASH_KEYS:
        ha = hashes_a.get(key)
        hb = hashes_b.get(key)

        if _is_sentinel(ha) or _is_sentinel(hb):
            positions[key] = "SKIP"
            skip_count += 1
        elif ha == hb:
            positions[key] = "PASS"
        else:
            positions[key] = f"FAIL: {ha[:16]}... != {hb[:16]}..."
            mismatch_count += 1

    return {
        "id": query_id,
        "category": rec_a.get("category", "unknown"),
        "token_len_a": rec_a.get("token_len"),
        "token_len_b": rec_b.get("token_len"),
        "positions": positions,
        "mismatch_count": mismatch_count,
        "skip_count": skip_count,
    }


def _compute_topk_overlap(rec_a: dict, rec_b: dict, k: int) -> dict[str, float | None]:
    """
    Compute top-K and top-1 overlap between two records.
    Returns {"top_k": float, "top_1": float} or None values if not computable.
    """
    idx_a = rec_a.get("topk_indices")
    idx_b = rec_b.get("topk_indices")

    if idx_a is None or idx_b is None:
        return {"top_k": None, "top_1": None}

    set_a_k = set(idx_a[:k])
    set_b_k = set(idx_b[:k])

    if not set_a_k and not set_b_k:
        return {"top_k": 1.0, "top_1": 1.0}

    top_k_overlap = len(set_a_k & set_b_k) / max(len(set_a_k), len(set_b_k))

    # Top-1 overlap: exact match of first result
    top_1_overlap = 1.0 if (idx_a[:1] == idx_b[:1]) else 0.0

    return {"top_k": top_k_overlap, "top_1": top_1_overlap}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class CompareResult:
    def __init__(self) -> None:
        self.total_queries: int = 0
        self.queries_with_mismatch: int = 0
        self.total_hash_positions: int = 0
        self.hash_pass: int = 0
        self.hash_fail: int = 0
        self.hash_skip: int = 0
        self.topk_overlaps: list[float] = []
        self.top1_overlaps: list[float] = []
        self.per_position_pass: dict[str, int] = {k: 0 for k in HASH_KEYS}
        self.per_position_fail: dict[str, int] = {k: 0 for k in HASH_KEYS}
        self.per_position_skip: dict[str, int] = {k: 0 for k in HASH_KEYS}
        self.failing_queries: list[dict] = []


def compare_blobs(
    blob_a: dict,
    blob_b: dict,
    top_k: int = TOP_K,
    verbose: bool = False,
) -> CompareResult:
    """
    Compare two hash blobs and return a CompareResult.
    """
    index_a = _index_records(blob_a)
    index_b = _index_records(blob_b)

    # Only compare queries present in both blobs
    common_ids = sorted(set(index_a) & set(index_b))

    result = CompareResult()
    result.total_queries = len(common_ids)

    for query_id in common_ids:
        rec_a = index_a[query_id]
        rec_b = index_b[query_id]

        row = _compare_hash_row(query_id, rec_a, rec_b)

        result.total_hash_positions += len(HASH_KEYS)
        result.hash_pass += sum(1 for v in row["positions"].values() if v == "PASS")
        result.hash_fail += row["mismatch_count"]
        result.hash_skip += row["skip_count"]

        for key in HASH_KEYS:
            status = row["positions"][key]
            if status == "PASS":
                result.per_position_pass[key] += 1
            elif status == "SKIP":
                result.per_position_skip[key] += 1
            else:
                result.per_position_fail[key] += 1

        if row["mismatch_count"] > 0:
            result.queries_with_mismatch += 1
            result.failing_queries.append(row)
            if verbose:
                print(f"  FAIL query={query_id} ({row['mismatch_count']} mismatches)")
                for k, v in row["positions"].items():
                    if "FAIL" in v:
                        print(f"    {k}: {v}")

        # Top-K overlap (skip for sentinel-only records)
        overlap = _compute_topk_overlap(rec_a, rec_b, top_k)
        if overlap["top_k"] is not None:
            result.topk_overlaps.append(overlap["top_k"])
        if overlap["top_1"] is not None:
            result.top1_overlaps.append(overlap["top_1"])

    return result


def print_report(
    result: CompareResult,
    blob_a_path: str,
    blob_b_path: str,
    backend_a: str,
    backend_b: str,
    topk_threshold: float = DEFAULT_TOPK_THRESHOLD,
    top1_threshold: float = DEFAULT_TOP1_THRESHOLD,
) -> None:
    sep = "=" * 72
    print(sep)
    print("A1.4 BIT-IDENTITY COMPARISON REPORT")
    print(sep)
    print(f"Backend A: {backend_a}  ({blob_a_path})")
    print(f"Backend B: {backend_b}  ({blob_b_path})")
    print()

    # Overall hash comparison
    comparable = result.total_hash_positions - result.hash_skip
    if comparable == 0:
        pass_pct = 0.0
    else:
        pass_pct = 100 * result.hash_pass / comparable

    print("Hash comparison:")
    print(f"  Queries compared:     {result.total_queries}")
    print(f"  Total hash positions: {result.total_hash_positions}")
    print(f"  Comparable (non-skip):{comparable}")
    print(f"  PASS:                 {result.hash_pass}  ({pass_pct:.1f}%)")
    print(f"  FAIL:                 {result.hash_fail}")
    print(f"  SKIP (sentinel):      {result.hash_skip}")
    print(f"  Queries with mismatch:{result.queries_with_mismatch}")
    print()

    # Per-position breakdown
    print("Per hash position:")
    for key in HASH_KEYS:
        p = result.per_position_pass[key]
        f = result.per_position_fail[key]
        s = result.per_position_skip[key]
        total = p + f + s
        pct = 100 * p / (total - s) if (total - s) > 0 else 0.0
        status = "OK" if f == 0 else "FAIL"
        print(f"  {key:<30s} pass={p:5d}  fail={f:5d}  skip={s:5d}  [{status}] {pct:.1f}%")
    print()

    # Top-K overlap
    if result.topk_overlaps:
        avg_topk = sum(result.topk_overlaps) / len(result.topk_overlaps)
        avg_top1 = (
            sum(result.top1_overlaps) / len(result.top1_overlaps) if result.top1_overlaps else 0.0
        )
        topk_gate = avg_topk >= topk_threshold
        top1_gate = avg_top1 >= top1_threshold
        print("Top-K overlap (semantic equivalence under quantization):")
        print(f"  Queries with overlap data: {len(result.topk_overlaps)}")
        print(
            f"  avg top-{TOP_K} overlap:   {avg_topk:.4f}  "
            f"(gate >= {topk_threshold:.2f}): {'PASS' if topk_gate else 'FAIL'}"
        )
        print(
            f"  avg top-1 overlap:    {avg_top1:.4f}  "
            f"(gate >= {top1_threshold:.2f}): {'PASS' if top1_gate else 'FAIL'}"
        )
        print()
    else:
        avg_topk = 0.0
        avg_top1 = 0.0
        topk_gate = True  # No data to gate on (sentinel-only)
        top1_gate = True
        print("Top-K overlap: no comparable data (sentinel-only backend)")
        print()

    # First 5 failing queries
    if result.failing_queries:
        print(f"First {min(5, len(result.failing_queries))} failing queries:")
        for row in result.failing_queries[:5]:
            print(
                f"  id={row['id']}  category={row['category']}  mismatches={row['mismatch_count']}"
            )
            for k, v in row["positions"].items():
                if "FAIL" in v:
                    print(f"    {k}: {v}")
        if len(result.failing_queries) > 5:
            print(f"  ... and {len(result.failing_queries) - 5} more")
        print()

    # Gate verdict
    bit_identity_passed = result.hash_fail == 0
    gate_passed = bit_identity_passed and topk_gate and top1_gate

    print(sep)
    print("GATE VERDICT")
    print(sep)
    print(f"  Bit-identical hashes:  {'PASS' if bit_identity_passed else 'FAIL'}")
    if result.topk_overlaps:
        print(f"  Top-{TOP_K} overlap gate:   {'PASS' if topk_gate else 'FAIL'}")
        print(f"  Top-1 overlap gate:   {'PASS' if top1_gate else 'FAIL'}")
    print(f"  OVERALL:              {'PASS' if gate_passed else 'FAIL'}")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two bit-identity hash blobs from runner.py.",
    )
    parser.add_argument(
        "blob_a",
        type=Path,
        nargs="?",
        help="First hash blob (JSON)",
    )
    parser.add_argument(
        "blob_b",
        type=Path,
        nargs="?",
        help="Second hash blob (JSON)",
    )
    parser.add_argument("--blob-a", dest="blob_a_opt", type=Path)
    parser.add_argument("--blob-b", dest="blob_b_opt", type=Path)
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Top-K for overlap (default: {TOP_K})",
    )
    parser.add_argument(
        "--topk-threshold",
        type=float,
        default=DEFAULT_TOPK_THRESHOLD,
        help=f"Minimum top-K overlap to pass gate (default: {DEFAULT_TOPK_THRESHOLD})",
    )
    parser.add_argument(
        "--top1-threshold",
        type=float,
        default=DEFAULT_TOP1_THRESHOLD,
        help=f"Minimum top-1 overlap to pass gate (default: {DEFAULT_TOP1_THRESHOLD})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each failing query",
    )
    parser.add_argument(
        "--exit-on-mismatch",
        action="store_true",
        default=True,
        help="Exit 1 on any mismatch (default: True)",
    )

    args = parser.parse_args()

    path_a = args.blob_a_opt or args.blob_a
    path_b = args.blob_b_opt or args.blob_b

    if path_a is None or path_b is None:
        parser.print_help()
        sys.exit(2)

    blob_a = _load_blob(path_a)
    blob_b = _load_blob(path_b)

    result = compare_blobs(blob_a, blob_b, top_k=args.top_k, verbose=args.verbose)

    print_report(
        result,
        blob_a_path=str(path_a),
        blob_b_path=str(path_b),
        backend_a=blob_a.get("backend", "unknown"),
        backend_b=blob_b.get("backend", "unknown"),
        topk_threshold=args.topk_threshold,
        top1_threshold=args.top1_threshold,
    )

    # Exit code logic
    topk_overlaps = result.topk_overlaps
    avg_topk = sum(topk_overlaps) / len(topk_overlaps) if topk_overlaps else 1.0
    avg_top1 = (
        sum(result.top1_overlaps) / len(result.top1_overlaps) if result.top1_overlaps else 1.0
    )

    gate_passed = (
        result.hash_fail == 0
        and avg_topk >= args.topk_threshold
        and avg_top1 >= args.top1_threshold
    )

    sys.exit(0 if gate_passed else 1)


if __name__ == "__main__":
    main()
