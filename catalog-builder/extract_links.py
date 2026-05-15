#!/usr/bin/env python3
"""Extract tool/agent entries from awesome-* list READMEs.

Each markdown link [Name](url) followed by a description is a tool
entry. mind-nerve is a routing preselector — these *are* its training
data.

Output: appends to items.jsonl with kind="tool".
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path("/data/datasets/mind-nerve-catalog/sources")
OUT = Path("/data/datasets/mind-nerve-catalog/index/items.jsonl")

# Repo must look like an awesome list, list, directory, or curated registry.
AWESOME_RE = re.compile(r"(awesome|directory|registry|collection|list|tools|toolkit)", re.IGNORECASE)

# Markdown link followed by description: `- [Name](url) [- — :] desc`
# Tolerant of bold, italics, badges before the link.
LINK_LINE_RE = re.compile(
    r"^[*\-+]\s+"                          # bullet
    r"(?:\*\*|\[!\[.*?\)\s*)*"             # optional bold or badge prefix
    r"\[([^\]\n]{2,120})\]"                # name in [ ]
    r"\(([^\)\s]+)\)"                      # url in ( )
    r"\s*[:\-—–]?\s*"                       # optional separator
    r"(.{10,400})?\s*$",                   # optional description
    re.MULTILINE,
)


def extract_from(readme: Path, source_repo: str):
    text = readme.read_text(encoding="utf-8", errors="replace")
    for m in LINK_LINE_RE.finditer(text):
        name = m.group(1).strip()
        url = m.group(2).strip()
        desc = (m.group(3) or "").strip()
        if url.startswith("#"):              # skip anchor-only ToC links
            continue
        if url.startswith("./") or url.startswith("/"):
            continue                          # skip internal repo links (already indexed elsewhere)
        if not url.startswith(("http://", "https://")):
            continue
        if len(name) < 2 or len(name) > 120:
            continue
        if name.lower() in {"home", "back to top", "table of contents", "contributing", "license", "github", "twitter", "x"}:
            continue
        body = f"# {name}\n\n{desc}\n\nurl: {url}\n".encode("utf-8")
        h = hashlib.sha256(body).hexdigest()
        yield {
            "id": h[:16],
            "source_repo": source_repo,
            "source_path": f"{readme.name}#{name}",
            "kind": "tool",
            "name": name,
            "size_bytes": len(body),
            "sha256": h,
            "tokens_est": max(1, len(body) // 4),
            "url": url,
        }


def main():
    added = 0
    repos_scanned = 0
    out_seen = set()
    if OUT.exists():
        out_seen = {json.loads(l).get("sha256") for l in OUT.open()}

    with OUT.open("a", encoding="utf-8") as out:
        for repo_dir in sorted(ROOT.iterdir()):
            if not repo_dir.is_dir():
                continue
            if not AWESOME_RE.search(repo_dir.name):
                continue
            readme = None
            for cand in ("README.md", "Readme.md", "readme.md", "README.MD"):
                p = repo_dir / cand
                if p.exists():
                    readme = p
                    break
            if not readme:
                continue
            repos_scanned += 1
            src = repo_dir.name.replace("__", "/", 1)
            for item in extract_from(readme, src):
                if item["sha256"] in out_seen:
                    continue
                out_seen.add(item["sha256"])
                out.write(json.dumps(item, separators=(",", ":")) + "\n")
                added += 1

    print(json.dumps({
        "repos_scanned": repos_scanned,
        "tool_entries_added": added,
        "items_jsonl_total_lines": sum(1 for _ in OUT.open()),
    }, indent=2))


if __name__ == "__main__":
    main()
