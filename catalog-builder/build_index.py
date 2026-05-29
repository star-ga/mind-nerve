#!/usr/bin/env python3
"""Walk all cloned skill-collection repos and emit a flat JSONL index.

One JSONL line per skill-like artifact. Schema:

    {"id":           "<sha256(canonical_bytes)>",
     "source_repo":  "owner/name",
     "source_path":  "relative/path/inside/repo",
     "kind":         "skill" | "command" | "agent" | "rule" | "extension" | "prompt",
     "name":         "<filename or H1>",
     "size_bytes":   1234,
     "sha256":       "<full hex>",
     "tokens_est":   <bytes / 4>,
     "freq_r":       <float>}    # co-occurrence frequency; 1.0 if no stats

Intentionally a flat indexer at this stage — no content extraction yet.
The Phase 1 catalog builder will read content and normalise per the
schema in docs/catalog_and_training_plan.md.

v2 binary emit (--output / --prior-file flags):
  When --output is given the builder writes a v2 binary catalog (.bin) in
  addition to the JSONL index. Embedding rows are pre-scaled by
  max(0.5, 1 / sqrt(freq_r)) before INT8 quantisation (RFC-004). This is a
  zero-runtime-cost pre-computation: the table is already scaled when loaded.
  The catalog magic is MNC2 and the trailing PRIR prior block is written with
  log(1 + freq_r) per route.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path("catalog-data/sources")
OUT = Path("catalog-data/index/items.jsonl")
STATS = Path("catalog-data/index/stats.json")

# Embedding dimensionality — must match ROUTE_EMBEDDING_DIM in src/lib.mind.
ROUTE_EMBEDDING_DIM = 256

# Q16.16 scaling constant.
Q16_SCALE = 65536
I32_MAX = 2_147_483_647
I32_MIN = -2_147_483_648


SKILL_PATTERNS = [
    (re.compile(r"(^|/)SKILL\.md$",                       re.IGNORECASE), "skill"),
    (re.compile(r"(^|/)skills?/[^/]+\.md$",               re.IGNORECASE), "skill"),
    (re.compile(r"(^|/)skills?/[^/]+/[^/]+\.md$",         re.IGNORECASE), "skill"),
    (re.compile(r"(\.claude|^claude)/(skills|commands|agents|hooks)/.*\.md$", re.IGNORECASE), "skill"),
    (re.compile(r"(^|/)commands?/[^/]+\.md$",             re.IGNORECASE), "command"),
    (re.compile(r"(^|/)commands?/[^/]+/[^/]+\.md$",       re.IGNORECASE), "command"),
    (re.compile(r"(^|/)agents?/[^/]+\.md$",               re.IGNORECASE), "agent"),
    (re.compile(r"(^|/)agents?/[^/]+/[^/]+\.md$",         re.IGNORECASE), "agent"),
    (re.compile(r"(^|/)subagents?/[^/]+\.md$",            re.IGNORECASE), "agent"),
    (re.compile(r"(^|/)plugins?/[^/]+\.md$",              re.IGNORECASE), "skill"),
    (re.compile(r"(^|/)workflows?/[^/]+\.md$",            re.IGNORECASE), "skill"),
    (re.compile(r"(^|/)recipes?/[^/]+\.md$",              re.IGNORECASE), "skill"),
    (re.compile(r"(^|/)playbooks?/[^/]+\.md$",            re.IGNORECASE), "skill"),
    (re.compile(r"(^|/)hooks?/[^/]+\.md$",                re.IGNORECASE), "skill"),
    (re.compile(r"(^|/)instructions/[^/]+\.md$",          re.IGNORECASE), "prompt"),
    (re.compile(r"\.cursorrules$",                        re.IGNORECASE), "rule"),
    (re.compile(r"\.mdc$",                                re.IGNORECASE), "rule"),
    (re.compile(r"(^|/)\.cursor/rules/.*\.mdc?$",         re.IGNORECASE), "rule"),
    (re.compile(r"(^|/)extensions?/[^/]+/[^/]+\.md$",     re.IGNORECASE), "extension"),
    (re.compile(r"(^|/)extensions?/[^/]+\.md$",           re.IGNORECASE), "extension"),
    (re.compile(r"(^|/)prompts?/[^/]+\.md$",              re.IGNORECASE), "prompt"),
    (re.compile(r"(^|/)prompts?/[^/]+/[^/]+\.md$",        re.IGNORECASE), "prompt"),
    (re.compile(r"(^|/)system[-_]prompts?/[^/]+",         re.IGNORECASE), "prompt"),
    (re.compile(r"(^|/)tools/[^/]+/[^/]+\.md$",           re.IGNORECASE), "skill"),
    (re.compile(r"copilot-instructions\.md$",             re.IGNORECASE), "prompt"),
    (re.compile(r"AGENTS?\.md$",                          re.IGNORECASE), "prompt"),
    (re.compile(r"CLAUDE\.md$",                           re.IGNORECASE), "prompt"),
    # JSON variants
    (re.compile(r"(^|/)(skills|agents|commands|subagents|plugins)/[^/]+\.json$", re.IGNORECASE), "skill"),
]
SKIP_DIRS = {".git", "node_modules", "dist", "build", "target", "__pycache__", ".venv", "venv"}


def detect_kind(rel: str) -> str | None:
    for pat, kind in SKILL_PATTERNS:
        if pat.search(rel):
            return kind
    return None


def first_h1(body: bytes) -> str | None:
    for line in body.splitlines()[:50]:
        try:
            s = line.decode("utf-8", "replace").strip()
        except Exception:
            continue
        if s.startswith("# "):
            return s[2:].strip()
    return None


def repo_name_for(repo_dir: Path) -> str:
    return repo_dir.name.replace("__", "/", 1)


PROMPT_REPO_RE = re.compile(r"(prompt|leak|cl4r1t4s|heimdall|oss-system|cursor-rules)", re.IGNORECASE)
SKIP_FILES = {"README.md", "CONTRIBUTING.md", "LICENSE", "LICENSE.md", "CODE_OF_CONDUCT.md",
              "CHANGELOG.md", "SECURITY.md", "GOVERNANCE.md", "AUTHORS.md", "MAINTAINERS.md",
              "PULL_REQUEST_TEMPLATE.md", "ISSUE_TEMPLATE.md"}


def walk_repo(repo_dir: Path):
    is_prompt_repo = bool(PROMPT_REPO_RE.search(repo_dir.name))
    for p in repo_dir.rglob("*"):
        if not p.is_file():
            continue
        parts = set(p.relative_to(repo_dir).parts)
        if parts & SKIP_DIRS:
            continue
        rel = str(p.relative_to(repo_dir))
        kind = detect_kind(rel)
        if not kind and is_prompt_repo and p.suffix.lower() in {".md", ".txt"} and p.name not in SKIP_FILES:
            # Fallback: in a prompt/skill repo, treat any non-boilerplate .md/.txt as a prompt.
            kind = "prompt"
        if not kind:
            continue
        try:
            body = p.read_bytes()
        except OSError:
            continue
        if len(body) > 256 * 1024:        # skip > 256 KiB (likely not a skill)
            continue
        if len(body) < 32:                # too short to be meaningful
            continue
        h = hashlib.sha256(body).hexdigest()
        name = first_h1(body) or p.stem
        yield {
            "id": h[:16],
            "source_repo": repo_name_for(repo_dir),
            "source_path": rel,
            "kind": kind,
            "name": name,
            "size_bytes": len(body),
            "sha256": h,
            "tokens_est": max(1, len(body) // 4),
        }


def load_prior_file(prior_path: Path) -> dict[str, float]:
    """Load a route_prior.json produced by build_prior.py.

    Returns a mapping from route ID to log-prior float.
    """
    data = json.loads(prior_path.read_text(encoding="utf-8"))
    raw: dict = data.get("priors", {})
    return {k: float(v) for k, v in raw.items()}



def scale_embedding_q16(embedding: list[int], scale: float) -> list[int]:
    """Apply a float scale to a Q16.16 i32 embedding vector, re-clamping.

    Each element is multiplied by scale and rounded back to i32.
    """
    out: list[int] = []
    for v in embedding:
        scaled = round(v * scale)
        out.append(max(I32_MIN, min(I32_MAX, scaled)))
    return out


def emit_v2_catalog(
    items: list[dict],
    prior_map: dict[str, float],
    output_path: Path,
) -> None:
    """Write a v2 binary catalog to *output_path*.

    ``items`` is the list of JSONL dicts already enriched with ``freq_r``.
    Embeddings are synthesised as zero vectors here — this function is
    the emit skeleton; in the full training pipeline the real INT8 embeddings
    would be supplied from the trained route table.  The v2 format, prior
    block, and freq-adaptive scaling are exercised by the integration tests.
    """
    from format.cat_v2 import encode_v2, freq_adaptive_scale  # local import

    routes: list[dict] = []
    log_priors: list[float] = []

    for item in items:
        rid_hex: str = item.get("sha256", item.get("id", ""))
        # Derive a 32-byte route_id: SHA-256 of the rid hex string ensures
        # the route_id is exactly 32 bytes regardless of how the id was stored.
        route_id_bytes = hashlib.sha256(rid_hex.encode()).digest()

        freq_r = float(item.get("freq_r", 1.0))
        scale = freq_adaptive_scale(freq_r)

        # Embedding: zero vector as placeholder.  In production the trained
        # INT8 embedding table is substituted here.
        base_emb = [0] * ROUTE_EMBEDDING_DIM
        scaled_emb = scale_embedding_q16(base_emb, scale)

        routes.append({"route_id": route_id_bytes, "embedding": scaled_emb})

        # log-prior: prefer the pre-computed prior map; fall back to computing
        # from the item's own freq_r.
        lp = prior_map.get(rid_hex, math.log(1.0 + freq_r))
        log_priors.append(lp)

    catalog_bytes = encode_v2(routes, log_priors)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(catalog_bytes)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Walk skill repos and emit JSONL catalog index (+ optional v2 binary)"
    )
    ap.add_argument(
        "--sources", type=Path, default=ROOT,
        help="Root directory of cloned skill repos",
    )
    ap.add_argument(
        "--out-jsonl", type=Path, default=OUT,
        help="Destination JSONL index path",
    )
    ap.add_argument(
        "--out-stats", type=Path, default=STATS,
        help="Destination stats JSON path",
    )
    ap.add_argument(
        "--output", type=Path, default=None,
        help="If given, also write a v2 binary catalog to this path",
    )
    ap.add_argument(
        "--prior-file", type=Path, default=None,
        help="route_prior.json from build_prior.py; used for the v2 binary emit",
    )
    args = ap.parse_args()

    if not args.sources.exists():
        sys.exit(f"sources root not found: {args.sources}")

    prior_map: dict[str, float] = {}
    if args.prior_file is not None:
        if not args.prior_file.exists():
            sys.exit(f"prior file not found: {args.prior_file}")
        prior_map = load_prior_file(args.prior_file)

    counts: dict = {"total": 0, "by_kind": {}, "by_repo": {}, "unique_sha": set()}
    all_items: list[dict] = []

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as out:
        for repo_dir in sorted(args.sources.iterdir()):
            if not repo_dir.is_dir():
                continue
            n = 0
            for item in walk_repo(repo_dir):
                # Enrich with freq_r from prior map, defaulting to 1.0.
                rid = item.get("sha256", item.get("id", ""))
                if rid in prior_map:
                    # Convert log-prior back to freq_r: exp(lp) - 1 = freq_r + alpha
                    # where alpha = 1 (Laplace).  Since we stored log(1 + freq_r + 1),
                    # freq_r = exp(lp) - 2.  Use the simpler monotone proxy instead:
                    # freq_r is stored as exp(lp) - 1 so that downstream callers can
                    # re-derive the scale without the manifest.  If no prior map is
                    # given default 1.0 gives scale = 1/sqrt(1) = 1.0 (no-op).
                    item["freq_r"] = max(1.0, math.exp(prior_map[rid]) - 1.0)
                else:
                    item["freq_r"] = 1.0
                out.write(json.dumps(item, separators=(",", ":")) + "\n")
                all_items.append(item)
                n += 1
                counts["total"] += 1
                counts["by_kind"][item["kind"]] = counts["by_kind"].get(item["kind"], 0) + 1
                counts["unique_sha"].add(item["sha256"])
            if n:
                counts["by_repo"][repo_name_for(repo_dir)] = n

    counts["unique"] = len(counts["unique_sha"])
    counts.pop("unique_sha")
    counts["dedup_drops"] = counts["total"] - counts["unique"]
    args.out_stats.parent.mkdir(parents=True, exist_ok=True)
    args.out_stats.write_text(json.dumps(counts, indent=2, default=int))

    summary = {
        "total": counts["total"],
        "unique": counts["unique"],
        "dedup_drops": counts["dedup_drops"],
        "by_kind": counts["by_kind"],
        "by_repo_top10": dict(
            sorted(counts["by_repo"].items(), key=lambda kv: -kv[1])[:10]
        ),
    }

    if args.output is not None:
        # Import format module relative to this file's directory.
        sys.path.insert(0, str(Path(__file__).parent))
        emit_v2_catalog(all_items, prior_map, args.output)
        summary["v2_catalog"] = str(args.output)
        summary["v2_routes"] = len(all_items)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
