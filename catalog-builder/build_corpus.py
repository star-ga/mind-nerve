#!/usr/bin/env python3
"""Materialise the BPE training corpus from the frozen catalog.

Reads catalog-data/freeze/v1.0/items.jsonl,
re-loads each item's underlying bytes (or synthesises them for tool
entries), and emits a flat text corpus with one item per line:

    <NAME>\t<KIND>\t<BODY-FIRST-N-CHARS>

Output: catalog-data/tokenizer/corpus.txt
Used by train_bpe.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

FREEZE = Path("catalog-data/freeze/v1.0/items.jsonl")
SOURCES = Path("catalog-data/sources")
OUT_DIR = Path("catalog-data/tokenizer")
OUT = OUT_DIR / "corpus.txt"

MAX_CHARS_PER_ITEM = 2048  # trim long bodies so the corpus stays mineable


def synth_tool_body(item: dict) -> str:
    name = item.get("name", "")
    url = item.get("url", "")
    # Tool entries don't have on-disk bodies; reconstruct minimally.
    return f"# {name}\n\nurl: {url}\n"


def load_body(item: dict) -> str:
    src_repo = item["source_repo"]
    src_path = item["source_path"]
    repo_dir_name = src_repo.replace("/", "__", 1)
    # fall back to single-underscore naming used by clone_all.sh
    candidates = [
        SOURCES / repo_dir_name,
        SOURCES / src_repo.replace("/", "_", 1),
    ]
    for repo_dir in candidates:
        if repo_dir.is_dir() or repo_dir.is_symlink():
            p = repo_dir / src_path
            if p.exists() and p.is_file():
                try:
                    return p.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    return ""
    return ""


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not FREEZE.exists():
        sys.exit(f"freeze items.jsonl not found at {FREEZE}; run freeze.py first")

    n_total = 0
    n_loaded = 0
    n_tool = 0
    n_empty = 0

    with FREEZE.open("r", encoding="utf-8") as fin, OUT.open("w", encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line)
            n_total += 1
            name = (item.get("name") or "").replace("\t", " ").replace("\n", " ").strip()
            kind = item.get("kind", "unknown")
            if item["kind"] == "tool":
                body = synth_tool_body(item)
                n_tool += 1
            else:
                body = load_body(item)
                if body:
                    n_loaded += 1
                else:
                    n_empty += 1
                    body = name  # last-ditch fallback so the item is still seen

            # Squash whitespace runs and trim
            body = " ".join(body.split())[:MAX_CHARS_PER_ITEM]
            fout.write(f"{name}\t{kind}\t{body}\n")

    print(
        json.dumps(
            {
                "items_total": n_total,
                "items_with_loaded_body": n_loaded,
                "items_tool_synthesised": n_tool,
                "items_with_empty_body": n_empty,
                "corpus_path": str(OUT),
                "corpus_bytes": OUT.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
