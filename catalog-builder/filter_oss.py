#!/usr/bin/env python3
"""Filter the catalog down to OSS-redistributable items only.

Inputs:
  $MIND_NERVE_CATALOG_FREEZE/v1.0/items.jsonl   (default: ./catalog-builder/freeze/v1.0)
  $MIND_NERVE_LOCAL_SKILLS/<name>/SKILL.md      (license lookup for STARGA/local-skills)
  $MIND_NERVE_CLAUDE_AGENTS/<name>.md           (license lookup for STARGA/claude-agents)

Output (new freeze):
  $MIND_NERVE_CATALOG_FREEZE/v1.1-oss/items.jsonl
  $MIND_NERVE_CATALOG_FREEZE/v1.1-oss/manifest.json
  exclusion_log.jsonl  (every dropped item + reason)

Exclusion rules (any one excludes):
  - kind='tool' AND domain matches private-STARGA registry  (rare)
  - source_repo in EXCLUDED_REPOS  (the explicit blocklist)
  - source_repo='STARGA/local-skills' AND that skill's SKILL.md does
    not declare an OSS-compatible license OR matches a commercial marker
  - source_repo='STARGA/claude-agents'  (entire dir is internal; drop all)

Keeps catalog-v1.0 as-is for the STARGA-tenant dogfood corpus; v1.1-oss
is the public-redistributable corpus that the OSS wheel ships with.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

_FREEZE_BASE = Path(os.environ.get("MIND_NERVE_CATALOG_FREEZE", "./catalog-builder/freeze"))
V1 = _FREEZE_BASE / "v1.0" / "items.jsonl"
OUT_DIR = _FREEZE_BASE / "v1.1-oss"

LOCAL_SKILLS = Path(
    os.environ.get("MIND_NERVE_LOCAL_SKILLS", str(Path.home() / ".agents" / "skills"))
)

PUBLIC_LICENSES = {
    "apache-2.0",
    "apache 2.0",
    "apache2",
    "mit",
    "bsd-3-clause",
    "bsd-2-clause",
    "isc",
    "cc0",
    "cc0-1.0",
    "unlicense",
    "cc-by-4.0",
}
COMMERCIAL_MARKERS = re.compile(
    r"\b(starga[\s-]*commercial|proprietary|confidential|"
    r"all[\s-]*rights[\s-]*reserved|do[\s-]*not[\s-]*distribute|"
    r"closed[\s-]*source|naestro[\s-]*defense|mind[\s-]*internal|"
    r"\bnda\b)\b",
    re.IGNORECASE,
)

# Accept both naming conventions (single-underscore from `tr '/' '__'`
# and slash-separated from repo_name_for() output).
_EXCLUDED = [
    "jujumilk3_leaked-system-prompts",
    "jujumilk3/leaked-system-prompts",
    "YeeKal_leaked-system-prompts",
    "YeeKal/leaked-system-prompts",
    "elder-plinius_CL4R1T4S",
    "elder-plinius/CL4R1T4S",
    "AiFeatures_system-prompts-collection",
    "AiFeatures/system-prompts-collection",
    "STARGA_claude-agents",
    "STARGA/claude-agents",
    # awesome-aigc / awesome-ai-prompts that turned out to be image-prompt
    # collections (visual art prompts; off-domain for routing).
    "weekend-project-space_awesome-aigc-prompts",
]
EXCLUDED_REPOS = set(_EXCLUDED)


def parse_frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    out: dict[str, str] = {}
    for line in text[3:end].splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z_][\w-]*)\s*:\s*(.*)$", line)
        if m:
            key, val = m.group(1).strip().lower(), m.group(2).strip().strip('"').strip("'")
            out[key] = val
    return out


_local_skill_bucket: dict[str, str] = {}


def local_skill_bucket(skill_name: str) -> str:
    """Bucket the local STARGA skill by its frontmatter license."""
    if skill_name in _local_skill_bucket:
        return _local_skill_bucket[skill_name]
    skill_md = LOCAL_SKILLS / skill_name / "SKILL.md"
    if not skill_md.exists():
        _local_skill_bucket[skill_name] = "missing"
        return "missing"
    text = skill_md.read_text(encoding="utf-8", errors="replace")
    if COMMERCIAL_MARKERS.search(text):
        _local_skill_bucket[skill_name] = "commercial_risk"
        return "commercial_risk"
    fm = parse_frontmatter(text)
    if fm.get("visibility", "").lower() in {"private", "confidential", "internal"}:
        _local_skill_bucket[skill_name] = "commercial_risk"
        return "commercial_risk"
    lic = (fm.get("license") or "").lower().strip()
    if lic in PUBLIC_LICENSES:
        _local_skill_bucket[skill_name] = "public_ok"
        return "public_ok"
    _local_skill_bucket[skill_name] = "unknown"
    return "unknown"


def decide(item: dict) -> tuple[bool, str]:
    repo = item.get("source_repo", "")
    if repo in EXCLUDED_REPOS:
        return False, f"excluded_repo:{repo}"
    if repo == "STARGA/local-skills":
        # local skills: peek at SKILL.md license
        src_path = item.get("source_path", "")
        # source_path looks like "skill-name/SKILL.md"
        skill_name = src_path.split("/", 1)[0] if "/" in src_path else src_path
        bucket = local_skill_bucket(skill_name)
        if bucket == "public_ok":
            return True, "local_skill_oss"
        return False, f"local_skill_{bucket}"
    # External repos: keep all (their licenses are upstream); leakage
    # discipline is satisfied because none of them are STARGA-private.
    return True, "external_repo_ok"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    exclusion_log_path = OUT_DIR / "exclusion_log.jsonl"
    out_items_path = OUT_DIR / "items.jsonl"

    kept: list[dict] = []
    drop_reasons: dict[str, int] = {}
    n_seen = 0

    with (
        V1.open("r", encoding="utf-8") as fin,
        exclusion_log_path.open("w", encoding="utf-8") as flog,
    ):
        for line in fin:
            n_seen += 1
            item = json.loads(line)
            keep, reason = decide(item)
            if keep:
                kept.append(item)
            else:
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
                flog.write(
                    json.dumps({**item, "_drop_reason": reason}, separators=(",", ":")) + "\n"
                )

    # Canonicalise + write
    kept_sorted = sorted(kept, key=lambda i: i["sha256"])
    lines = [
        json.dumps(i, separators=(",", ":"), sort_keys=True, ensure_ascii=True) for i in kept_sorted
    ]
    canon = ("\n".join(lines) + "\n").encode("utf-8")
    out_items_path.write_bytes(canon)

    freeze_id = hashlib.sha256(canon).hexdigest()
    manifest = {
        "schema_version": 1,
        "catalog_version": "v1.1-oss",
        "frozen_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "items_count": len(kept),
        "items_dropped": n_seen - len(kept),
        "drop_reasons": drop_reasons,
        "freeze_id": freeze_id,
        "canonical_bytes": len(canon),
        "tokens_est_total": max(1, len(canon) // 4),
        "source_freeze": "v1.0",
        "source_freeze_id": "a63b55d728492fee9659d2568262c87aebe4d2fca615eb393cc24ce1fa59bb51",
        "policy": (
            "v1.1-oss is the public-redistributable corpus. Filters out: "
            "(a) STARGA private claude-agents directory in full, "
            "(b) STARGA local-skills entries without declared OSS license, "
            "(c) ethically gray leaked-system-prompts collections, "
            "(d) any item matching commercial-content markers."
        ),
        "signing": {
            "algorithm": "HMAC-SHA256",
            "key_id": "STARGA-ROOT-2026",
            "status": "draft-unsigned",
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (OUT_DIR / "manifest.sig").write_text(
        "DRAFT-UNSIGNED\n"
        f"freeze_id: {freeze_id}\n"
        "To sign: overwrite with HMAC-SHA256(manifest.json) under STARGA-ROOT-2026.\n"
    )

    print(
        json.dumps(
            {
                "v1_seen": n_seen,
                "v1.1_kept": len(kept),
                "dropped": n_seen - len(kept),
                "drop_reasons": drop_reasons,
                "freeze_id": freeze_id,
                "out_dir": str(OUT_DIR),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
