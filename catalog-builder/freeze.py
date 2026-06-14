#!/usr/bin/env python3
"""Freeze the live `items.jsonl` into a versioned catalog release.

Outputs a deterministic, content-addressed artifact under
`catalog-data/freeze/<version>/`:

  items.jsonl     — canonical: dedup'd, sorted by id, newline-terminated
  manifest.json   — version header, item count, sha256, provenance
  manifest.sig    — STARGA HMAC signature (placeholder until signed
                    with the real root key — see "Signing" below)
  upstream.txt    — list of source repos + their clone-log timestamps

The freeze is content-addressed: `freeze_id` is the SHA-256 of the
canonical `items.jsonl` bytes. Different content → different id.

Signing
-------
This script does NOT sign with the STARGA HMAC root key. It writes a
`manifest.sig` stub the operator overwrites with the real signature
out-of-band (the root key is not on this machine). The catalog is
*draft-frozen* until the signature is real.

Usage
-----
    python3 freeze.py --version v1.0 [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path("catalog-data")
ITEMS = ROOT / "index" / "items.jsonl"
FREEZE_DIR = ROOT / "freeze"


def canonicalise(raw_path: Path) -> tuple[bytes, int, int]:
    """Return (canonical_bytes, total_items, raw_lines) for the catalog."""
    items: dict[str, dict] = {}  # dedupe by sha256 (real content hash)
    raw_lines = 0
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_lines += 1
            obj = json.loads(line)
            sha = obj.get("sha256")
            if not sha:
                continue
            if sha not in items:
                items[sha] = obj
            else:
                # On collision, keep the entry from the highest-priority
                # source. Skill files beat tool entries beats raw prompts.
                kind_pri = {
                    "skill": 0,
                    "agent": 1,
                    "command": 2,
                    "rule": 3,
                    "extension": 4,
                    "tool": 5,
                    "prompt": 6,
                }
                old_pri = kind_pri.get(items[sha]["kind"], 9)
                new_pri = kind_pri.get(obj["kind"], 9)
                if new_pri < old_pri:
                    items[sha] = obj
    # Canonical order: lex-sorted by sha256, one item per line, no trailing
    # spaces, deterministic JSON (no NaN, sorted keys, compact separators).
    out_lines = []
    for sha in sorted(items):
        item = items[sha]
        out_lines.append(json.dumps(item, separators=(",", ":"), sort_keys=True, ensure_ascii=True))
    canon = ("\n".join(out_lines) + "\n").encode("utf-8")
    return canon, len(items), raw_lines


def hash_provenance() -> dict:
    """Hash every clone_manifest_v*.tsv + clone_log_v*.tsv as provenance."""
    prov = {}
    idx_dir = ROOT / "index"
    for p in sorted(idx_dir.glob("clone_manifest*.tsv")):
        prov[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(idx_dir.glob("clone_log*.tsv")):
        prov[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return prov


def upstream_summary() -> list[dict]:
    """Per-repo summary: name, sha256(README.md) when present."""
    out = []
    sources = ROOT / "sources"
    for d in sorted(sources.iterdir()):
        if not d.is_dir() and not d.is_symlink():
            continue
        entry: dict = {"name": d.name.replace("__", "/", 1)}
        # Best-effort: resolve a current HEAD commit if it's a real git clone.
        head = d / ".git" / "HEAD"
        if head.exists():
            try:
                ref = head.read_text().strip()
                if ref.startswith("ref: "):
                    refpath = d / ".git" / ref[5:]
                    if refpath.exists():
                        entry["head"] = refpath.read_text().strip()
                else:
                    entry["head"] = ref
            except OSError:
                pass
        out.append(entry)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help="e.g. v1.0")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not ITEMS.exists():
        sys.exit(f"items.jsonl not found at {ITEMS}; run build_index.py + extract_links.py first")

    print(f"canonicalising {ITEMS} ...", file=sys.stderr)
    canon, n_items, n_raw = canonicalise(ITEMS)
    freeze_id = hashlib.sha256(canon).hexdigest()
    fnv = _fnv1a_32(canon)

    manifest = {
        "schema_version": 1,
        "catalog_version": args.version,
        "frozen_at": int(time.time()),
        "frozen_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "items_count": n_items,
        "raw_lines_seen": n_raw,
        "dedup_drops": n_raw - n_items,
        "freeze_id": freeze_id,
        "fnv1a_32": f"{fnv:08x}",
        "canonical_bytes": len(canon),
        "tokens_est_total": _tokens_total(canon),
        "encoder": "utf-8",
        "line_endings": "LF",
        "sort": "lex(sha256) ascending",
        "json_form": "compact, sort_keys=true, ensure_ascii=true",
        "upstream_manifests": hash_provenance(),
        "signing": {
            "algorithm": "HMAC-SHA256",
            "key_id": "STARGA-ROOT-2026",
            "status": "draft-unsigned",
            "note": "Signature lives in manifest.sig and is applied "
            "out-of-band with the STARGA root key.",
        },
    }

    out_dir = FREEZE_DIR / args.version
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        print(f"(dry-run) would write to {out_dir}", file=sys.stderr)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "items.jsonl").write_bytes(canon)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (out_dir / "manifest.sig").write_text(
        "DRAFT-UNSIGNED\n"
        f"freeze_id: {freeze_id}\n"
        "To sign: overwrite this file with the HMAC-SHA256(manifest.json bytes) "
        "computed with the STARGA-ROOT-2026 key.\n"
    )
    upstream_path = out_dir / "upstream.txt"
    upstream_path.write_text(
        "\n".join(f"{e['name']}\t{e.get('head', '-')}" for e in upstream_summary()) + "\n"
    )

    print(json.dumps(manifest, indent=2))
    print(f"\nfrozen to: {out_dir}", file=sys.stderr)


def _fnv1a_32(data: bytes) -> int:
    h = 0x811C9DC5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def _tokens_total(data: bytes) -> int:
    # Naive estimate matches the per-item rule used in build_index.py.
    return max(1, len(data) // 4)


if __name__ == "__main__":
    main()
