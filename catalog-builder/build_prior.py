#!/usr/bin/env python3
"""Compute empirical-Bayes log-prior per route from catalog co-occurrence stats.

Reads the catalog JSONL (items.jsonl or a frozen items.jsonl) and a
co-occurrence stats file produced during catalog-load profiling, then writes
a prior vector file consumed by build_index.py when emitting the v2 catalog.

Co-occurrence stats format (JSONL, one line per event):
    {"route_id": "<hex-or-name>", "count": <int>}

If no stats file is provided the priors default to uniform (freq_r = 1 for
every route, so log(1 + 1) = log(2) ≈ 0.693).

Output (JSON):
    {
      "schema_version": 1,
      "num_routes": <int>,
      "total_observations": <int>,
      "priors": {"<route_id>": <log_prior_float>, ...},
      "shrinkage_alpha": <float>
    }

Empirical-Bayes shrinkage
-------------------------
Raw frequency counts are sparse and dominated by the most popular routes.
We apply additive smoothing with alpha = 1 (Laplace prior) before computing
the log-prior:

    freq_r      = raw_count_r + alpha
    log_prior_r = log(1 + freq_r)

This keeps all priors in (log(2), log(2 + max_count)] which maps to a
Q16.16 range of (45426, bounded by i32).  At alpha=1 even a route never
seen in the co-occurrence log gets log(1 + 1) = log(2) ≈ 0.693.

Usage
-----
    python3 build_prior.py \\
        --catalog catalog-data/freeze/v1.0/items.jsonl \\
        --cooccurrence catalog-data/stats/cooccurrence.jsonl \\
        --output catalog-data/prior/route_prior.json

    # Dry-run with no stats (uniform prior):
    python3 build_prior.py \\
        --catalog catalog-data/freeze/v1.0/items.jsonl \\
        --output catalog-data/prior/route_prior.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SHRINKAGE_ALPHA: float = 1.0  # Laplace smoothing constant


def load_catalog_ids(catalog_path: Path) -> list[str]:
    """Return list of route IDs (sha256 field) in catalog order."""
    ids: list[str] = []
    with catalog_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rid = obj.get("sha256") or obj.get("id") or obj.get("route_id")
            if rid:
                ids.append(str(rid))
    return ids


def load_cooccurrence(stats_path: Path) -> dict[str, int]:
    """Return raw per-route hit counts from a co-occurrence JSONL file."""
    counts: dict[str, int] = {}
    with stats_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rid = str(obj.get("route_id", ""))
            cnt = int(obj.get("count", 0))
            if rid:
                counts[rid] = counts.get(rid, 0) + cnt
    return counts


def compute_log_priors(
    route_ids: list[str],
    raw_counts: dict[str, int],
    alpha: float = SHRINKAGE_ALPHA,
) -> dict[str, float]:
    """Return per-route log-prior mapping.

    Prior formula:  log(1 + raw_count_r + alpha)
    The +alpha additive smoothing ensures no route ever reaches log(1) = 0.
    """
    return {rid: math.log(1.0 + raw_counts.get(rid, 0) + alpha) for rid in route_ids}


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute empirical-Bayes route log-priors")
    ap.add_argument(
        "--catalog",
        required=True,
        type=Path,
        help="Path to items.jsonl (frozen or live catalog JSONL)",
    )
    ap.add_argument(
        "--cooccurrence",
        type=Path,
        default=None,
        help="Optional co-occurrence stats JSONL. If absent, uniform prior is used.",
    )
    ap.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination JSON file for the prior vector",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=SHRINKAGE_ALPHA,
        help=f"Laplace smoothing constant (default {SHRINKAGE_ALPHA})",
    )
    args = ap.parse_args()

    if not args.catalog.exists():
        sys.exit(f"catalog not found: {args.catalog}")

    route_ids = load_catalog_ids(args.catalog)
    if not route_ids:
        sys.exit("catalog produced no route IDs; is the file empty or wrong format?")

    raw_counts: dict[str, int] = {}
    if args.cooccurrence is not None:
        if not args.cooccurrence.exists():
            sys.exit(f"co-occurrence stats not found: {args.cooccurrence}")
        raw_counts = load_cooccurrence(args.cooccurrence)

    total_observations = sum(raw_counts.values())
    log_priors = compute_log_priors(route_ids, raw_counts, alpha=args.alpha)

    result: dict = {
        "schema_version": 1,
        "num_routes": len(route_ids),
        "total_observations": total_observations,
        "shrinkage_alpha": args.alpha,
        "priors": log_priors,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    # Summary to stdout so CI can capture it
    n_with_data = sum(1 for rid in route_ids if raw_counts.get(rid, 0) > 0)
    prior_values = list(log_priors.values())
    print(
        json.dumps(
            {
                "num_routes": len(route_ids),
                "routes_with_cooccurrence_data": n_with_data,
                "total_observations": total_observations,
                "prior_min": round(min(prior_values), 6),
                "prior_max": round(max(prior_values), 6),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
