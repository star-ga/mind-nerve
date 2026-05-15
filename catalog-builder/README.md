# catalog-builder

Tools for mining the public skill/agent/rule/prompt corpus that becomes
the mind-nerve catalog. Output lands on the data SSD at
`/data/datasets/mind-nerve-catalog/`, not in this repo, because the
corpus is hundreds of megabytes of third-party content under varied
licenses.

## What gets built

A flat `items.jsonl` where each line is one routing candidate.
Schema:

```json
{
  "id": "<sha256[:16]>",
  "source_repo": "owner/name",
  "source_path": "relative/path",
  "kind": "skill|command|agent|rule|extension|prompt|tool",
  "name": "<file H1 or stem>",
  "size_bytes": 1234,
  "sha256": "<hex>",
  "tokens_est": <bytes / 4>,
  "url": "<for tool entries only>"
}
```

`kind` taxonomy:

| Kind | Source |
|---|---|
| `skill` | `SKILL.md`, `skills/*.md`, `claude/skills/**`, `workflows/`, `recipes/`, `playbooks/`, `hooks/`, JSON manifests under `skills/` / `plugins/` |
| `command` | `commands/*.md`, `claude/commands/**` |
| `agent` | `agents/*.md`, `subagents/*.md`, `claude/agents/**` |
| `rule` | `.cursorrules`, `*.mdc`, `.cursor/rules/**` |
| `extension` | Gemini CLI `extensions/*.md` |
| `prompt` | `prompts/*.md`, `system-prompts/*`, `AGENTS.md`, `CLAUDE.md`, `copilot-instructions.md`, plus loose `.md`/`.txt` in repos whose name contains prompt/leak/cursor-rules |
| `tool` | Markdown link entries `[Name](url) — description` extracted from awesome-list READMEs |

## Files

- `clone_all.sh` — shallow-clone every repo in `clone_manifest_vN.tsv`
  into `/data/datasets/mind-nerve-catalog/sources/`, log to TSV.
- `clone_manifest_v1.tsv` through `clone_manifest_v4.tsv` — curated
  TSVs of `(stars, size_kb, full_name, description)` selected from
  `gh search repos` queries.
- `build_index.py` — walk every cloned repo, emit `items.jsonl` with
  one entry per skill-like file, plus dedup and kind tagging.
- `extract_links.py` — for repos whose name matches awesome / list /
  registry / directory / toolkit, parse the README and emit one
  `kind: tool` entry per markdown link.

## How to rerun from scratch

```bash
# 1. clone every batch
for v in 1 2 3 4; do
  MANIFEST=clone_manifest_v${v}.tsv bash clone_all.sh
done

# 2. (optional) symlink local STARGA-curated skills so they're indexed
ln -sfn ~/.agents/skills /data/datasets/mind-nerve-catalog/sources/STARGA__local-skills
ln -sfn ~/.claude/agents /data/datasets/mind-nerve-catalog/sources/STARGA__claude-agents

# 3. build the flat index from file-level skill/command/agent/rule artefacts
python3 build_index.py

# 4. extract awesome-list link entries (kind=tool) into the same JSONL
python3 extract_links.py

# 5. stats
python3 -c "
import json
from collections import Counter
items = [json.loads(l) for l in open('/data/datasets/mind-nerve-catalog/index/items.jsonl')]
print('total', len(items))
print('kinds', Counter(i['kind'] for i in items).most_common())
"
```

## Catalog state (2026-05-14, first build)

| Stream | Sources | Items |
|---|---|---|
| skill-file extraction | 110 repos | 7,438 raw / 6,864 sha-unique |
| awesome-list link extraction | 40 repos | 5,604 raw / 5,140 url-unique |
| **combined unique routing candidates** | **111 repos + 2 STARGA symlinks** | **~12,004** |

Past the 8k Phase-1 training threshold declared in
`docs/catalog_and_training_plan.md`. The catalog is *not* frozen — see
ROADMAP blocker #1 and task #53.

## Provenance discipline

Every cloned repo is recorded with its `(stars, size_kb, description,
clone_log_timestamp)` tuple in `clone_manifest_vN.tsv` +
`clone_log_vN.tsv`. This is what the Phase 1 freeze (task #53) hashes
into the v1.0 manifest header. Re-runs against a different commit of
an upstream repo will produce different items; the manifest tracks
that.

## Out of scope

- Heavy file content extraction. `build_index.py` records sha + first
  H1; full content normalisation (per the schema in
  `docs/catalog_and_training_plan.md`) is the Phase 1 catalog-builder
  job, not this scanner.
- License vetting. Each source repo carries its own license; building
  *this* catalog is fair-use research scraping. Re-distributing items
  later will require per-license review (task #62 territory).
