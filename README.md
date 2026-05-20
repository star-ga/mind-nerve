<h1 align="center">mind-nerve</h1>

<p align="center">
  <strong>Intent-classification preselector for agent runtimes.</strong><br>
  <em>Open the library, hide the cost.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/mind-nerve/"><img alt="PyPI" src="https://img.shields.io/pypi/v/mind-nerve.svg?color=blue"></a>
  <a href="https://pypi.org/project/mind-nerve/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/mind-nerve.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/pypi/l/mind-nerve.svg?color=4c1"></a>
  <a href="https://github.com/star-ga/mind-nerve/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/star-ga/mind-nerve/actions/workflows/ci.yml/badge.svg?branch=main"></a>
  <a href="https://pypi.org/project/mind-nerve/"><img alt="Downloads" src="https://img.shields.io/pypi/dm/mind-nerve.svg"></a>
  <a href="https://huggingface.co/star-ga/mind-nerve"><img alt="Hugging Face" src="https://img.shields.io/badge/weights-Hugging%20Face-FFD21E"></a>
  <a href="https://github.com/star-ga/mind-nerve/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/star-ga/mind-nerve?style=social"></a>
</p>

---

mind-nerve sits between a user prompt and the host runtime. It reads the
prompt, decides which subset of the available skills, tools, and MCP servers
is relevant, and hands the host a short list — so the downstream LLM never
sees the full library in its system prompt.

Library size decouples from token cost. Hosting **4,400 skills** costs the
same prompt budget as hosting **44**, because only the top-K are ever loaded
per turn.

```bash
pip install mind-nerve
```

```python
from mind_nerve import route
result = route("deploy the staging build", top_k=5)
for r in result.routes:
    print(f"{r.score:.3f}  {r.name}")
```

```
0.912  deploy-pipeline
0.847  staging-environment
0.812  ci-cd
0.778  release-checklist
0.741  rollback-strategy
```

---

## Highlights

| | |
| :--- | :--- |
| **96.06% top-5 accuracy** | against 11,922 routing candidates (v1.1-oss catalog) |
| **Phase 1 latency (shipped)** | warm-daemon p95 ~23 ms on GPU and ~90 ms on a 4-core CPU (PyTorch SentenceTransformers backend) |
| **Phase 2 latency (target)** | ≤30 ms p95 on a 4-core CPU via the native MIND Q16.16 encoder — not yet end-to-end; see Phase 2 status below |
| **~95% token reduction** | on a 440-skill Claude Code catalog per turn |
| **One-line install** | `mind-nerve-install install --cli claude-code --with-preselect` |
| **Public integrations today** | Claude Code, Claude Desktop, Cursor, Codex, Gemini CLI, plus a stdio MCP server for any MCP-aware client — see [Integrations](#integrations) |

> **Dual-license note.** The repo source, the Python wheel surface, and the
> Phase-1 weights are Apache-2.0. The wheel additionally bundles
> `libmindnerve.so`, a compiled native runtime component under a separate
> STARGA license. The Phase-1 PyTorch inference path runs entirely under
> Apache-2.0 and does not require that binary. See [License](#license) and
> [`LICENSE.md`](LICENSE.md) for the full split.

## The problem

Agent runtimes today load every available skill / tool / MCP server into the
LLM's system prompt on every turn. At small scale this is fine. At hundreds
of skills, the prompt-cache and per-call token cost become the binding
constraint on library growth.

| Approach              | Correctness   | Latency           | Token cost |
| --------------------- | ------------- | ----------------- | ---------- |
| Load the whole library | strong        | fast              | O(N) skills, every turn |
| Vector-only retrieval  | weak on intent | fast              | low |
| LLM-as-router          | strong        | a full LLM call   | a full LLM call |
| **mind-nerve (Phase 1, GPU daemon)** | 96.06% top-5 | ~23 ms p95 (warm daemon, GPU) | a few hundred tokens |
| **mind-nerve (Phase 1, 4-core CPU)** | 96.06% top-5 | ~90 ms p95 (warm daemon, CPU) | a few hundred tokens |

## Quickstart

### 1. Install

```bash
pip install mind-nerve
```

The first `route()` call auto-downloads the Phase-1 weights (~150 MB) from
[`star-ga/mind-nerve`](https://huggingface.co/star-ga/mind-nerve)
into `~/.local/share/mind-nerve/runtime/`. To pre-seed or use a custom
location, set `MIND_NERVE_RUNTIME_DIR`.

### 2. Call it from Python

```python
from mind_nerve import route

result = route("debug a slow Postgres query", top_k=5)
for r in result.routes:
    print(r.score, r.name, r.kind)
```

### 3. Run as a daemon (recommended for hot paths)

For CLI hooks, the MCP server, or anything that hits `route()` many times
per minute, run the daemon and connect over a UNIX socket. It loads the
runtime once. After warmup the round trip is ~23 ms on GPU and ~90 ms on
4-core CPU. The model load (~250 ms) only happens once at daemon start,
so subsequent prompts never pay for it.

```bash
mind-nerve-routed &       # listens on $XDG_RUNTIME_DIR/mind-nerve.sock
```

```python
import json, os, socket

def route(prompt: str, top_k: int = 5) -> dict:
    sock_path = f"{os.environ.get('XDG_RUNTIME_DIR', f'/run/user/{os.getuid()}')}/mind-nerve.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.connect(sock_path)
        s.sendall(json.dumps({"prompt": prompt, "top_k": top_k}).encode() + b"\n")
        return json.loads(s.makefile("r").readline())
```

### 4. Wire it into Claude Code (one command)

```bash
mind-nerve-install install --cli claude-code --with-preselect
```

That writes two hooks into `~/.claude/settings.json`:

- **`SessionStart`** — spawns `mind-nerve-routed` if it's not already running
  (~7 s warmup; sub-30 ms responses afterwards).
- **`UserPromptSubmit`** — asks the daemon for the top-K matching skills and
  atomically rewrites `~/.claude/skills/` as a directory of symlinks into
  your real catalog.

The installer auto-detects your layout:

- **Default Claude Code install** (most users): your existing
  `~/.claude/skills/` directory is renamed once to `~/.claude/skills.full/`.
  After that the daemon projects a top-K subset back into
  `~/.claude/skills/` per turn.
- **Shared catalog** (multiple agent CLIs pointed at one directory, e.g.
  `~/.agents/skills/`): the shared catalog stays put; mind-nerve projects
  from there into `~/.claude/skills/` per turn.

Already use [`mind-mem`](https://pypi.org/project/mind-mem/) for durable
memory? Add the companion MCP:

```bash
mind-nerve-install install --cli claude-code --with-preselect --with-mind-mem
```

mind-nerve handles intent routing; mind-mem provides search-backed memory.
Together they bracket the prompt path.

## Integrations

| Host                        | Mechanism                                          | Status |
| --------------------------- | -------------------------------------------------- | ------ |
| Claude Code                 | MCP + optional `UserPromptSubmit`/`SessionStart` hooks | shipping |
| Claude Desktop              | MCP (`claude_desktop_config.json`)                 | shipping |
| Cursor                      | MCP (`~/.cursor/mcp.json`)                         | shipping |
| Codex                       | MCP (`~/.codex/config.toml`)                       | shipping |
| Gemini CLI                  | extension manifest (`~/.gemini/extensions/`)       | shipping |
| Any MCP-aware client        | stdio MCP server                                   | shipping |
| Aider, Windsurf             | shim integrations                                  | roadmap |

The CLI matrix is opt-in:

```bash
mind-nerve-install list      # see all supported targets
mind-nerve-install detect    # see what's installed on this machine
mind-nerve-install install --cli all
```

## Console scripts

| Script | What it does |
| --- | --- |
| `mind-nerve` | one-shot CLI router: `mind-nerve route "git status" --top-k 5` |
| `mind-nerve-mcp` | stdio MCP server exposing the `mind_nerve_route` tool |
| `mind-nerve-routed` | long-lived UNIX-socket route server (the hot path) |
| `mind-nerve-routed-ensure` | idempotent daemon starter, designed for SessionStart hooks |
| `mind-nerve-preselect` | UserPromptSubmit hook that atomically projects the skills dir |
| `mind-nerve-install` | wires the above into each CLI's config |

## Configuration

| Env var                       | Default                                     | What it controls |
| ----------------------------- | ------------------------------------------- | ---------------- |
| `MIND_NERVE_RUNTIME_DIR`      | `~/.local/share/mind-nerve/runtime/`        | model + catalog cache |
| `MIND_NERVE_DEVICE`           | auto (CUDA → MPS → CPU)                     | force device (e.g. `cpu` when sharing a GPU with another model — auto-fallback to CPU also happens on CUDA OOM) |
| `MIND_NERVE_SOCKET`           | `$XDG_RUNTIME_DIR/mind-nerve.sock`          | daemon UNIX socket |
| `MIND_NERVE_SOURCE_DIR`       | auto-detected (`~/.claude/skills.full` or `~/.agents/skills`) | preselect source catalog |
| `MIND_NERVE_PROJECTED_DIR`    | `~/.claude/skills`                          | preselect projection target |
| `MIND_NERVE_TOP_K`            | `20`                                        | how many skills to project per turn |
| `MIND_NERVE_OVERFETCH`        | `300`                                       | how many to ask the daemon for before dedup |
| `MIND_NERVE_SOCKET_TIMEOUT`   | `2.0`                                       | daemon socket timeout (s) |
| `MIND_NERVE_LOG`              | `~/.mind-nerve/hook.log`                    | jsonl log for the preselect hook |
| `MIND_NERVE_CORE_ALWAYS_ON`   | `diagnose:code-review:git-workflow:…`       | colon-separated names always added to the projection |
| `MIND_NERVE_HF_REVISION`      | pinned commit SHA in the package             | override the Hugging Face model revision to download; set to a specific commit SHA or tag for reproducible artifact pinning |

## How it works

The frozen design is **drop-the-decoder + sliding-window encoder + direct
scoring head**. The decoder is dropped entirely; the encoder uses
sliding-window self-attention (window 256 tokens, stride 192) and writes a
pooled query vector that is dot-producted against the precomputed catalog
embedding table to produce the top-K routes. Top-K extraction is
deterministic: ties break by ascending SHA-256(route_id), so the same input
on x86, ARM, and CUDA returns the same ranking every time. The authoritative
design is [`spec/architecture.md`](spec/architecture.md).

That single design has two backends. Phase 1 is the one users install today.
Phase 2 is being brought up incrementally and is not yet end-to-end.

### Phase 1 backend — shipped today

- **Implementation:** PyTorch + `sentence-transformers` (`BAAI/bge-small-en-v1.5`
  fine-tuned on the v1.1-oss catalog), loaded once into the
  `mind-nerve-routed` UNIX-socket daemon.
- **Routing path:** encoder forward → L2-normalised pooled query vector →
  dense dot product against the precomputed `route_table.npy` →
  deterministic top-K with SHA-256 tie-break and `top_k ∈ [1, 64]` bounds.
- **Weights:** auto-downloaded on first use from
  [`star-ga/mind-nerve`](https://huggingface.co/star-ga/mind-nerve)
  at the pinned revision recorded in the wheel (override via
  `MIND_NERVE_HF_REVISION`).
- **Latency:** warm-daemon p95 ~23 ms on GPU, ~90 ms on a 4-core CPU. The
  ≤30 ms-on-CPU target is the **Phase 2** target, not the Phase 1 result.
- **License:** Apache-2.0 end-to-end. Phase 1 does not load
  `libmindnerve.so`; it runs entirely on the Python wheel surface.

### Phase 2 backend — in progress (A1.5 PARTIAL)

The same drop-the-decoder + sliding-window encoder design, compiled to a
native MIND Q16.16 fixed-point `cdylib` that the wheel loads through a
C-ABI shim. Goals: remove the PyTorch dependency, close the ≤30 ms-on-CPU
budget, and prove cross-architecture bit-identity across x86, ARM, CUDA,
and WebGPU.

Status, as of commit
[`b9b6401`](https://github.com/star-ga/mind-nerve/commit/b9b6401)
(A1.5 PARTIAL):

- ✅ A1.1–A1.4 — Q16.16 corpus, encoder kernels, C-ABI export surface, and
  the SHA-256 bit-identity harness scaffold all landed.
- ✅ A1.5 — pure-MIND encoder `cdylib` builds. The native **score path**
  (matmul against the 11,922-row route table) measures
  **p50 14.4 ms / p95 15.1 ms** on a 4-core CPU at commit `b9b6401` —
  already inside the Phase 2 budget for that stage of the pipeline.
- 🚧 Blocked on **mindc Phase 6.2** quantizer + SIMD lowering for the
  full encoder forward; until that lands, the wheel still routes through
  the Phase 1 PyTorch backend by default.
- 🚧 Cross-architecture bit-identity hardware validation across x86, ARM,
  CUDA, and WebGPU is the gating step before Phase 2 becomes the
  default backend.

## Design constraints

- **Latency p95 ≤ 30 ms** on 4-core CPU — non-negotiable end target. Phase 1
  hits 23 ms via the GPU+daemon path and ~90 ms with a warm daemon on
  4-core CPU; the full ≤30 ms-on-CPU budget closes with the Phase 2 native
  MIND Q16.16 inference loop (toolchain-side mindc shipped through v0.4.2 /
  RFC 0005 Phase C; mind-nerve-side native encoder is the remaining work).
- **Cross-architecture bit-identity** — same request on x86, ARM, CUDA, and
  WebGPU returns the same top-K. Q16.16 fixed-point throughout, no IEEE-754
  fallback in the inference path. (Phase 2 gate; mindc-side cdylib emit
  landed in v0.3.0; mind-nerve-side hardware validation still pending.)
- **No training-data leakage at inference** — the classifier reveals only
  route names, never the training corpora content.
- **Tamper detection** — every inference emits an attestation envelope tying
  the request hash, model hash, and result hash into the evidence chain.

## Roadmap

**Phase 1 (now)** — Public alpha. PyTorch inference, HF-hosted weights, MCP
+ hooks integrations, six target CLIs, 96.06% top-5 accuracy on a 11,922-route
catalog.

**Phase 2 (next)** — Native MIND Q16.16 inference loop replaces PyTorch.
Cross-architecture bit-identity gate. p95 budget tightens. The HF artifact
will be `star-ga/mind-nerve-phase2` (parallel to the current
[`star-ga/mind-nerve`](https://huggingface.co/star-ga/mind-nerve)) —
same corpus + tokenizer + model hash contract, different inference path.
Toolchain prerequisites all shipped: `mindc` 0.2.6 (C-ABI export),
[`mindc` 0.3.0](https://github.com/star-ga/mind/releases/tag/v0.3.0)
(cdylib emit + Phase 0/1/1.5 std-surface intrinsics + RFC 0005 P0e/P0f
struct + FieldAccess ABI), and
[`mindc` 0.4.4](https://github.com/star-ga/mind/releases/tag/v0.4.4)
(RFC 0005 Phase 2 + B + C + D₁ + D₂a — pure-MIND std.vec/string/map/io
bundled into the binary, with a `$MIND_STDLIB_PATH` env-var
fork-without-recompile escape hatch, and Named-struct parameter names
preserved in arity/type error messages).  Remaining work is the
mind-nerve-side native encoder kernel that links against the toolchain.

**Phase 3** — Catalog v2: license-aware ingest at scale, evidence-chain
proofs, per-tenant route tables.

Full roadmap: [`ROADMAP.md`](./ROADMAP.md).

## Repository layout

```
mind-nerve/
  python/mind_nerve/        Python wheel (Phase 1 inference + CLI)
    cli.py                  `mind-nerve` entrypoint
    daemon.py               `mind-nerve-routed` UNIX-socket server
    ensure.py               `mind-nerve-routed-ensure` idempotent starter
    preselect_hook.py       `mind-nerve-preselect` UserPromptSubmit hook
    installer.py            `mind-nerve-install` cross-CLI installer
    mcp_server.py           `mind-nerve-mcp` MCP stdio server
    inference.py            PyTorch route() implementation
    discovery.py            route catalog discovery + atomic writes
  src/                      pure-MIND implementation (Phase 2 target)
  spec/                     authoritative design documents
  tests/python/             unit tests for the wheel
  .github/workflows/        CI: ruff lint + build + smoke + pytest matrix
```

## License

mind-nerve ships under a **dual license**:

- **Apache-2.0** — the repository source (`python/`, `src/`, `spec/`,
  `cli/`, `integrations/`, `tests/`), the Python wheel surface, and the
  Phase-1 trained weights at
  [`star-ga/mind-nerve`](https://huggingface.co/star-ga/mind-nerve)
  are Apache-2.0. Phase-1 PyTorch inference runs entirely under Apache-2.0
  and does **not** load any STARGA-licensed binary.
- **STARGA Commercial** — the wheel additionally bundles
  `libmindnerve.so`, a compiled native runtime component whose source is
  not part of this repository. That binary carries a separate STARGA
  license. Redistribution outside the published wheel is not granted by
  the Apache-2.0 file.

Full split is documented in [`LICENSE.md`](LICENSE.md). For commercial
enquiries, contact [`license@star.ga`](mailto:license@star.ga).

## Governance and support

- **Contributing:** [`CONTRIBUTING.md`](CONTRIBUTING.md) — build, test, and PR flow.
- **Security disclosures:** [`SECURITY.md`](SECURITY.md) — please do not file
  public issues for vulnerabilities; report to
  [`info@star.ga`](mailto:info@star.ga).
- **Privacy:** [`docs/privacy.md`](docs/privacy.md) — local-only routing,
  opt-in logging, no telemetry by default.
- **Model card:** [`docs/model_card.md`](docs/model_card.md) — Phase-1 base
  model, training data, intended use, and known limitations.
- **Dataset and governance:** [`docs/dataset.md`](docs/dataset.md) and
  [`docs/data_governance.md`](docs/data_governance.md) — corpus schema,
  provenance, retention, and license posture.

## Citation

If mind-nerve helps your work, a citation is appreciated:

```bibtex
@software{mind_nerve_2026,
  author  = {STARGA, Inc.},
  title   = {mind-nerve: Intent-classification preselector for agent runtimes},
  year    = {2026},
  url     = {https://github.com/star-ga/mind-nerve},
  version = {0.3.0-beta.1}
}
```

## Links

- **PyPI**: <https://pypi.org/project/mind-nerve/>
- **Phase-1 weights**: <https://huggingface.co/star-ga/mind-nerve>
- **MIND language**: <https://mindlang.dev>
- **Changelog**: [`CHANGELOG.md`](./CHANGELOG.md)
- **Roadmap**: [`ROADMAP.md`](./ROADMAP.md)
- **Issues**: <https://github.com/star-ga/mind-nerve/issues>

<!-- mind-profile: default -->
