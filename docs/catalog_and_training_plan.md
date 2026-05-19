# Catalog + Training Plan — Phase 1

Captures the architectural decisions made during the 2026-05-14 design pass.
Lives next to `ROADMAP.md` as the operational plan for the Phase 1 build.

## Decisions locked in this pass

| Decision | Choice | Why |
|---|---|---|
| **Sequencing** | Catalog-first, training after | The catalog IS the moat. Training on STARGA's local 1,000 skills only would yield a parochial model. Real public agent ecosystem has 8-15k+ skills/tools/MCPs/agents. |
| **Training path** | PyTorch (BF16/FP32) → `torch_to_bundle` Q16.16 export → native MIND inference | The future native-MIND training pipeline (gated on Phase 6 + dedicated training hardware) is 18-36mo out. Native MIND **inference** end-to-end is shippable today via the torch-to-bundle conversion path. |
| **Brand claim** | "Native MIND inference. Q16.16 deterministic. Sub-30ms p95." | All three are true with the practical path. We do NOT claim "native MIND training" until the native-MIND training pipeline ships. |
| **Tokenizer** | Custom BPE, 32k vocab, trained on the merged catalog + query corpus | Tool/skill names have unique vocabulary (`mcp__server__tool_name`, `RFC_NNN_NAME`, etc.) — generic BPE wastes tokens here. |
| **Architecture (frozen)** | Encoder-only (drop-the-decoder), sliding-window self-attention (window=256, stride=192), direct scoring head over catalog | Scaffold landed 2026-05-13. Frozen per ROADMAP.md. |
| **Quantization** | INT8 weights, Q16.16 activations, cross-arch bit-identity gate | Non-negotiable Phase 1 exit criterion. |
| **Integration approach** | Wrapper-process daemon + filesystem overlay + MCP gateway hybrid | Multi-LLM consensus (grok-4.3 + deepseek-v4-pro + internal). Covers 17 CLIs without per-CLI plugin matrix. |
| **Distribution** | One core artifact (`nerve` Python package on PyPI), thin wrappers for brew / npm / Claude Code marketplace / VS Code marketplace | Same artifact pulled by every channel. |
| **Protection** | Open Python orchestrator + wrappers; closed `nerve-runtime.so` built via STARGA protection toolchain | Mirrors mind-mem's public/protected split. |
| **Brand** | `nerve.md` domain (available, registry NIC.MD) | Drops the `mind-` prefix on purpose — free/open product with its own identity. |

## Phase 0 — Catalog mining (no GPU, fully autonomous)

Target: 8,000-15,000 unique entries after dedup, normalized into a typed mic@2 dataset.

### Sources (priority order)

| # | Source | Method | Expected volume |
|---|---|---|---|
| 1 | `awesome-claude-skills` + variants on GitHub | clone, parse README, scan `/skills/` dirs | 500-2,000 skills |
| 2 | `awesome-mcp-servers` aggregations | crawl, install in sandbox, query `list_tools` | 1,000-3,000 MCP tools |
| 3 | Claude Code marketplaces already cached locally (`~/.claude/plugins/marketplaces/`) | parse manifests | 1,000-5,000 skills/agents |
| 4 | Anthropic-published skill repos (if any) | clone + frontmatter scan | 50-500 |
| 5 | VS Code Marketplace (extensions tagged `agent`/`ai`/`copilot`) | API query | 500-2,000 |
| 6 | npm packages tagged `mcp-server` / `agent` | npm registry API | 500-1,500 |
| 7 | PyPI packages similarly | PyPI JSON API | 200-800 |
| 8 | HuggingFace Spaces with `agent` tags | HF API | 500-2,000 |
| 9 | Cursor / Windsurf / Zed extension stores | per-store API | 200-500 each |

### Catalog item schema (one entry)

```
CatalogItem {
    id: u64,                  // FNV-1a hash of canonical name
    name: String,             // canonical name (e.g. "mind-mem.recall")
    source_kind: SourceKind,  // GitHubSkill | MCPTool | VSCodeExt | NpmPkg | PyPIPkg | HFSpace | ...
    source_url: String,
    license: String,
    description: String,      // primary trigger phrase (first paragraph)
    trigger_phrases: Vec<String>,  // synthetic + extracted paraphrases (10-20 per item)
    category: Category,       // dev | data | writing | research | security | ops | ...
    canonical_cli: Option<String>,  // claude-code | codex | gemini | cursor | ... | universal
    sample_queries: Vec<String>,
    embedding_seed_hash: i32, // for incremental retrain detection
}
```

### Dedup & canonicalization rules

- Hash by **canonical name** (case-insensitive, namespace-collapsed)
- Same tool published in multiple marketplaces → keep the one with the best license (open > permissive > restrictive)
- If descriptions differ substantially, keep both as `aliases` of one canonical entry
- License-incompatible entries (proprietary, no-redistribute) get flagged but kept for ranking; cannot be cloned into the public catalog dataset
- Source attribution preserved in `source_url` per item

### Storage

- Canonical store: `catalog/items.mic` (mic@2 binary format per STARGA-native interchange rule)
- Pretty export: `catalog/items.json` (gitignored, regenerated from mic@2)
- Per-source raw dumps: `catalog/raw/<source>/...` (gitignored)
- Public release artifact: `catalog/nerve-catalog-vN.mic` (one immutable bundle per nerve release)

## Phase 1 — Training (after catalog ≥ 8k entries)

Single target: **≥92% top-5 accuracy** on held-out test set, **p95 ≤30ms** on 4-core CPU.

### Training pipeline (PyTorch internal, native MIND export)

```
1. Tokenize merged catalog + query corpus  →  custom 32k BPE
2. Build catalog embedding index            →  lex-sorted Vec<(id, vec)>
3. Train encoder (PyTorch, RTX):
   - Loss: contrastive InfoNCE (query ↔ chosen-tool)
   - Optimizer: AdamW, cosine LR, warmup 1k steps
   - Hard negative mining every 5k steps
   - Curriculum: random → BM25-mined → cross-encoder filtered
4. ~50-100k steps to convergence
5. Best checkpoint by validation top-5 accuracy
6. INT8 quantization (per-channel symmetric)
7. Q16.16 activation export via the torch-to-bundle conversion path
8. Cross-arch bit-identity verification gate (x86 vs CUDA)
```

### Data splits

- 80% train (catalog × synthetic paraphrases × real session-log queries)
- 10% validation (early stopping)
- 10% **held-out test** (hand-curated from real traffic, never seen during training, this is the 92% target)

## Phase 2 — Integration & distribution (after weights ship)

Wrapper-daemon architecture per multi-LLM consensus:

1. `nerve setup --auto` detects installed CLIs across PATH
2. Per CLI, applies the right adapter:
   - MCP-protocol CLIs → gateway URL injection
   - Filesystem-scan CLIs → curated overlay at `/tmp/nerve-N/skills/`
   - Closed CLIs → wrapper command (`nerve claude`) + env injection
3. Daemon listens on Unix socket, serves ranking requests
4. Protected `nerve-runtime.so` loaded once, in-memory, license-key gated

### Distribution channels (same artifact behind all of them)

```
curl nerve.md/install | sh   →  bash installer, auto-detect, all-channel fallback
pip install nerve            →  primary
brew install starga/tap/nerve →  macOS
npm install -g @starga/nerve →  JS ecosystems (cursor/cline/continue/copilot)
Claude Code marketplace      →  /plugin install @starga/nerve
VS Code Marketplace          →  one extension covers 6+ CLIs
```

## Schedule (revised, honest)

| Phase | Work | Duration | GPU required? |
|---|---|---|---|
| 0a | Scraper scaffolding + catalog/schema.mind + first source (awesome-claude-skills) | 3-5 days | No |
| 0b | All 9 sources crawled + dedup pipeline | 1-2 weeks | No |
| 0c | Trigger-phrase generation (LLM-assisted; needs Claude/Codex quota) | 1 week | No |
| 1a | Tokenizer training (custom 32k BPE) | 2-3 days | No |
| 1b | First training run (PyTorch) → 75-85% top-5 | 1-2 weeks | YES |
| 1c | Iteration (hard negatives, curriculum) → 85-92% top-5 | 1-2 weeks | YES |
| 1d | INT8 quantization + Q16.16 export + cross-arch verify | 1 week | No |
| 1e | Latency profile + optimization to ≤30ms p95 | 1 week | No |
| 2 | Wrapper-daemon + per-CLI adapters + installer + protected runtime build | 2-3 weeks | No |

**Total: 9-13 weeks from this commit to shipped v0.1.0.**

Phase 0 (catalog) is **all-autonomous, no GPU needed**, and runs in foreground/background of normal work. Phase 1b-1c are the only GPU-blocked windows.

## What ships in v0.1.0

- `nerve` CLI on PyPI
- `nerve setup --auto` works for at least: claude-code, codex, gemini, vibe (the 4 STARGA-supported CLIs)
- MCP gateway for any MCP-speaking CLI
- BM25 fallback ranking (sub-2ms latency) — kicks in when the trained encoder is unavailable
- Public catalog dataset (8-15k entries) at `catalog/nerve-catalog-v1.mic`
- Open weights at HuggingFace `star-ga/nerve-base`
- Protected runtime `nerve-runtime.so` at install-time download
- Marketing site at `nerve.md`

## Open items NOT decided this pass

- Catalog license model — every source has its own license; we may need to ship the dataset under a permissive license that respects sources, OR a hybrid where users must pull source-level metadata at install time
- Server-side license-key infra for the protected runtime — first-run validation flow needs designing
- Hard-negative-mining strategy in detail (BM25 / cross-encoder / cosine-distance thresholds)
- Trigger-phrase generation prompt template (LLM costs scale with catalog size)
- Test set curation methodology (who decides what's in the held-out 10%)
- Cross-CLI installer protocol for environments where `pip` / `brew` / `npm` don't exist (e.g. air-gapped or container-only)

These can be resolved during Phase 0 as the catalog matures and we see real distributions.
