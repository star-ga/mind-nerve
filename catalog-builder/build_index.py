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
     "tokens_est":   <bytes / 4>}

Intentionally a flat indexer at this stage — no content extraction yet.
The Phase 1 catalog builder will read content and normalise per the
schema in docs/catalog_and_training_plan.md.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path("/data/datasets/mind-nerve-catalog/sources")
OUT = Path("/data/datasets/mind-nerve-catalog/index/items.jsonl")
STATS = Path("/data/datasets/mind-nerve-catalog/index/stats.json")


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


def main():
    if not ROOT.exists():
        sys.exit(f"sources root not found: {ROOT}")
    counts = {"total": 0, "by_kind": {}, "by_repo": {}, "unique_sha": set()}
    with OUT.open("w", encoding="utf-8") as out:
        for repo_dir in sorted(ROOT.iterdir()):
            if not repo_dir.is_dir():
                continue
            n = 0
            for item in walk_repo(repo_dir):
                out.write(json.dumps(item, separators=(",", ":")) + "\n")
                n += 1
                counts["total"] += 1
                counts["by_kind"][item["kind"]] = counts["by_kind"].get(item["kind"], 0) + 1
                counts["unique_sha"].add(item["sha256"])
            if n:
                counts["by_repo"][repo_name_for(repo_dir)] = n
    counts["unique"] = len(counts["unique_sha"])
    counts.pop("unique_sha")
    counts["dedup_drops"] = counts["total"] - counts["unique"]
    STATS.write_text(json.dumps(counts, indent=2, default=int))
    print(json.dumps({
        "total": counts["total"],
        "unique": counts["unique"],
        "dedup_drops": counts["dedup_drops"],
        "by_kind": counts["by_kind"],
        "by_repo_top10": dict(sorted(counts["by_repo"].items(), key=lambda kv: -kv[1])[:10]),
    }, indent=2))


if __name__ == "__main__":
    main()
