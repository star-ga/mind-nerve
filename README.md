<p align="center">
  <img src="assets/brand/mind-mark.svg" alt="" width="110">
</p>

<h1 align="center">mind-nerve</h1>

<p align="center">
  <strong>Intent-classification preselector for agent runtimes.</strong><br>
  <em>Every skill available. Few in context.</em>
</p>

<p align="center">
  <strong>Mind-Nerve</strong> implements a drop-the-decoder + sliding-window encoder design compiled to native Q16.16 fixed-point — the same deterministic architecture as MIND, designed for byte-identical routing output across substrates. The native backend reproduces a pinned bit-identity reference on x86_64, and the underlying MIND Q16.16 substrate is verified byte-identical across x86_64 (AVX2) and ARM64 (NEON) on real hardware. The open-source release ships CPU backends only;
a GPU tier (CUDA/WebGPU) is reserved for a potential private/enterprise
offering, mirroring the MIND compiler's licensing split.
</p>

<p align="center">
  <a href="https://pypi.org/project/mind-nerve/"><img alt="PyPI" src="https://img.shields.io/pypi/v/mind-nerve.svg?color=blue&style=flat-square"></a>
  <a href="https://pypi.org/project/mind-nerve/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/mind-nerve.svg?style=flat-square"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue?style=flat-square"></a>
  <a href="https://github.com/star-ga/mind-nerve/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/star-ga/mind-nerve/ci.yml?branch=main&style=flat-square&label=CI"></a>
  <a href="https://github.com/star-ga/mind-nerve/releases"><img alt="Release" src="https://img.shields.io/github/v/release/star-ga/mind-nerve?style=flat-square&color=green&label=Release"></a>
  <img alt="Deterministic" src="https://img.shields.io/badge/deterministic-Q16.16-brightgreen?style=flat-square">
  <a href="https://huggingface.co/star-ga/mind-nerve"><img alt="Hugging Face" src="https://img.shields.io/badge/weights-HuggingFace-FFD21E?style=flat-square"></a>
</p>

---

mind-nerve sits between a user prompt and the host runtime. It reads the
prompt, decides which subset of the available skills, tools, and MCP servers
is relevant, and hands the host a short list — so the downstream LLM never
sees the full library in its system prompt.

Library size decouples from token cost. Point it at every SKILL file
published on GitHub — **~1.6M** of them — and the standing cost is the prompt
budget of **44**, because only the top-K are ever loaded per turn.

The catalog does not sit in the context window; the router does, and it is a
fixed cost. There is no ceiling in the design: the number of routable
artifacts is bounded by disk, not by context. Measured today: **96.06%
top-5 accuracy across 11,922 candidates**.

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

Runs on **Linux, macOS and Windows** from the same universal (`py3-none-any`)
wheel. The native Q16.16 encoder is a Linux-only speed-up that ships inside the
wheel as optional data; on macOS/Windows (or any box without the native library)
the router transparently falls back to the pure-Python backend — same results,
slightly slower per query, with a one-line notice on first use. The one-shot
`mind-nerve route` CLI needs no daemon and is fully OS-agnostic.

> **Current stable is `0.3.0`.** It ships the rebuilt native runtime, the
> offline quantizer, and the Phase 2 encoder rewire behind the plain install
> above — no `--pre` flag needed.

The first `route()` call auto-downloads the Phase-1 weights (~150 MB) from
[`star-ga/mind-nerve`](https://huggingface.co/star-ga/mind-nerve)
into `~/.local/share/mind-nerve/runtime/`. To pre-seed or use a custom
location, set `MIND_NERVE_RUNTIME_DIR`.

> **The runtime-dir pin is load-bearing.** If you maintain a curated route
> table, **export `MIND_NERVE_RUNTIME_DIR` to point at it** (for a daemon,
> pin it in the systemd unit / shared env). When the variable is unset,
> resolution falls through to the default location and serves the *generic*
> catalog — the routes will look plausible but be far less relevant. As of
> the fix that added this note, an unset pin prints a one-time
> `WARNING — MIND_NERVE_RUNTIME_DIR is not set` on stderr so the fallback is
> never silent. If `route_table.npy` and `route_table.jsonl` ever fall out of
> sync (a load-time *"Route table embeddings/meta length mismatch"*), run
> `mind-nerve prune` to realign them.

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

> **Recommended companion:** [`mind-mem`](https://github.com/star-ga/mind-mem)
> is our open-source (Apache-2.0) governed memory engine for agent CLIs —
> hybrid BM25+vector recall, a tamper-evident evidence chain, and an MCP
> server that plugs into the same installer. If you take one other tool from
> this ecosystem, take that one.

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

The full 20-client matrix (grok, kimi, qwen, windsurf, continue, cline,
roo, zed, aider, copilot, cody, qodo, …) plus the `verify` verb — per-client
checks of config, hooks, env pins, MCP entry, and daemon socket — lives in
the TypeScript installer under `integrations/installer/`
(`npm install && npm run build`, then `node dist/src/index.js install
<client>` / `verify --cli all`; bin name `mind-nerve-installer`, npm
publication pending).

## Console scripts

| Script | What it does |
| --- | --- |
| `mind-nerve` | one-shot CLI router: `mind-nerve route "git status" --top-k 5` |
| `mind-nerve-mcp` | stdio MCP server exposing the `mind_nerve_route` tool |
| `mind-nerve-routed` | long-lived UNIX-socket route server (the hot path) |
| `mind-nerve-routed-ensure` | idempotent daemon starter, designed for SessionStart hooks |
| `mind-nerve-preselect` | UserPromptSubmit hook that atomically projects the skills dir |
| `mind-nerve-install` | wires the above into each CLI's config |

## Acquiring skills

The catalog is not fixed. `mind-nerve acquire` searches the public ecosystem
across nine sources — Anthropic's skills repo, the official MCP servers repo,
the MCP registry API, Glama, Smithery, community skill/agent/template
libraries (`obra/superpowers`, `wshobson/agents`,
`davila7/claude-code-templates`), plus generic GitHub repository search as a
fallback. Skills, agents and MCP servers, all three. There is no curated
whitelist you are confined to: if it is published, it is reachable.

Anything found is vetted with a deterministic fail-closed static scanner (shell-pipe installers, reverse shells, exfiltration
collectors, prompt injection, archive escapes, obfuscation, credential
access, persistence hooks), and installs the clean ones into the hub:

```bash
mind-nerve acquire search "pdf"
mind-nerve acquire install <url> [--accept-warnings]
mind-nerve acquire list
mind-nerve acquire remove <name>
```

Fetches land in a size/file-count-capped quarantine dir first; a FAIL
verdict never reaches the hub. Installs write a per-file SHA-256 manifest,
reindex the route table through the license gate, and restart the routing
daemon so every hooked CLI sees the new skill immediately. Acquired content is
always reindexed as untrusted, even though the local hub is a first-party
trust root — a skill you did not write does not inherit your trust.

Acquisition is the only network path in the system, and it is explicit and
operator-invoked. Routing itself never opens a socket: local encoder, local
table, read off local disk. Full threat model and source-registry format:
[docs/acquisition.md](docs/acquisition.md).

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
| `MIND_NERVE_ENV_FILE`         | `~/.mind-nerve/env`                         | shared env file: `KEY=VALUE` lines (`#` comments) read by the hook, the MCP server, and the daemon spawner; an explicitly exported var or CLI-config pin ALWAYS wins — the file only fills vars that are unset |

## How it works

The frozen design is **drop-the-decoder + sliding-window encoder + direct
scoring head**. The decoder is dropped entirely; the encoder uses
sliding-window self-attention (window 256 tokens, stride 192) and writes a
pooled query vector that is dot-producted against the precomputed catalog
embedding table to produce the top-K routes. Top-K extraction is
deterministic: both backends break ties by ascending SHA-256(route_id),
matching the spec contract — the Python scoring path
(`python/mind_nerve/inference.py`) and the native Q16.16 top-K
(`src/top_k.mind`) share the same score-descending,
SHA-256(route_id)-ascending ordering. The underlying MIND Q16.16 substrate's
cross-architecture (x86_64 / ARM64 CPU) identity is verified on real
hardware; mind-nerve's own ranking pipeline reproduces the pinned x86_64
reference today, and ARM64 reproduction of that same pipeline is the
task #57 gate — not yet hardware-validated (see `docs/benchmarks.md` §1).
The authoritative design is [`spec/architecture.md`](spec/architecture.md).

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
- **License:** Apache-2.0 end-to-end. The wheel runs entirely on its own
  Apache-2.0 surface (the bundled native Q16.16 encoder `cdylib`
  included); it never loads the separately-licensed `libmindnerve.so`
  runtime.

### Phase 2 backend — native encoder shipping (default since 0.3.0b9)

The same drop-the-decoder + sliding-window encoder design, compiled to a
native MIND Q16.16 fixed-point `cdylib` that ships inside the wheel and is
the **default backend** (`MIND_NERVE_BACKEND=pytorch` selects the PyTorch
fallback). Goals: remove the PyTorch dependency, close the ≤30 ms-on-CPU
budget, and reach cross-architecture bit-identity across the shipped CPU
backends. The underlying MIND Q16.16 substrate is verified byte-identical
on x86_64 (AVX2) and ARM64 (NEON) real hardware; mind-nerve's own
encoder/route pipeline reproduces the pinned x86_64 reference today, and
ARM64 reproduction of that pipeline is the task #57 gate — not yet
hardware-validated. The open-source release ships CPU backends only; a
GPU tier (CUDA/WebGPU) is reserved for a potential private/enterprise
offering (scope decision 2026-08-15).

The pure-MIND front end also ships a **native MCP server**
(`src/mcp.mind`, compiled into the same binary): its JSON-RPC message
framing (`initialize`, error shapes, notification suppression) is
byte-identical to the Python `mind-nerve-mcp` server on the frozen golden
transcript (`tests/harness/mcp_golden.sh`). `tools/call` on the native
server is **fail-closed by default** with an explicit `unavailable` payload:
the binary-bundle producer plumbing is landed and loader-round-trip tested,
but the only bundle producible today uses placeholder zero embeddings (no
256-dim trained checkpoint exists; the 384-dim BGE route table is
incompatible with the native loader), which would rank by SHA-256 tie-break
rather than relevance — so auto-seed is intentionally off until a real
checkpoint lands (see the honest limits at the top of `src/mcp.mind`). The
Python `mind-nerve-mcp` server remains the one that actually routes today.
A native Windows PE build is declared in `Mind.toml`
(`[targets.windows]`, cross-compiled via MinGW-w64) but is not yet
CI-verified or distributed — Windows installs run the pure-Python
fallback, same as the encoder path above.

Status, as of v0.3.0:

- ✅ A1.1–A1.4 — Q16.16 corpus, encoder kernels, C-ABI export surface, and
  the SHA-256 bit-identity harness scaffold all landed.
- ✅ A1.5 — pure-MIND encoder `cdylib` builds and ships in the wheel. The
  native **score path** (matmul against the 11,922-row route table, now
  the pure-MIND MT `__mind_blas_gemv_q16_mt`) measures **p50 ≈0.58 ms /
  p95 ≈0.9 ms** across all 12 hardware threads (i7-5930K) — already
  inside the Phase 2 budget for that stage of the pipeline.
- ✅ Full `.mind` tree ported to mindc 0.10.2 (2026-08-15) — the 17-module
  kernel tree AND all 13 front-end files (sha256, q16_16, tokenizer,
  evidence, encoder_kernels, model, loader, inference, …) compile and
  execute, gated by the fail-closed `tests/mindc_gate.sh` (262
  exact-count tests + a native-ELF end-to-end harness byte-verified
  against CPython hashlib).
- ✅ x86_64 bit-identity: the native score path reproduces the pinned
  x86_64 Q16.16 reference byte-for-byte (AVX2 == scalar oracle == the
  pinned hash, `docs/benchmarks.md` §1). The underlying MIND Q16.16
  substrate is separately verified byte-identical on x86_64 (AVX2) and
  ARM64 (NEON) real hardware; ARM64 reproduction of **mind-nerve's own**
  encoder/route pipeline against that same pinned hash is the task #57
  gate and is **not yet hardware-validated**. The open-source release is
  CPU-only; no OSS GPU tier to validate.

## Design constraints

- **Latency p95 ≤ 30 ms** on 4-core CPU — non-negotiable end target. Phase 1
  hits 23 ms via the GPU+daemon path and ~90 ms with a warm daemon on
  4-core CPU; the ≤30 ms-on-CPU budget closes with the native
  MIND Q16.16 inference loop (the mindc-side prerequisites and the
  mind-nerve-side encoder both shipped; the encoder cdylib is the default
  backend since 0.3.0b9).
- **Cross-architecture bit-identity** — same request on x86_64 and ARM64
  CPU returns the same top-K. Q16.16 fixed-point throughout, no IEEE-754
  fallback in the inference path. The underlying MIND Q16.16 substrate is
  verified on real x86_64 + ARM64 hardware; mind-nerve's own pipeline is
  verified on x86_64 today, with ARM64 reproduction gated on task #57
  (not yet hardware-validated). There is no GPU backend (scope decision
  2026-08-15).
- **No training-data leakage at inference** — the classifier reveals only
  route names, never the training corpora content.
- **Tamper detection** — every inference can emit an attestation envelope
  tying the request hash, model hash, and result hash into the evidence
  chain (opt-in; see `python/mind_nerve/attestation.py`).

## Roadmap

**Phase 1 (shipping)** — Native Q16.16 encoder by default
(PyTorch fallback), HF-hosted weights, MCP + hooks integrations, 20
installer targets, 96.06% top-5 accuracy on a 11,922-route catalog.

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
preserved in arity/type error messages). The mind-nerve-side encoder
kernel has since shipped in the wheel (default backend), and the whole
`.mind` tree — kernel surface plus the 13 front-end files — compiles and
executes on the current toolchain.
The CI gate for the `mind/` kernel tree is pinned to
[`mindc` 0.10.2](https://github.com/star-ga/mind/releases/tag/v0.10.2)
(built with `std-surface,cross-module-imports,mlir-build` — `mlir-build`
produces the real native objects the gate's symbol leg checks): 0.10.2
name-checks
function bodies (mind#23) and resolves the tree's cross-module imports
in project mode via `mind/Mind.toml`. Run the same gate locally with
`bash tests/mindc_gate.sh` — it is fail-closed and never trusts a bare
exit code.

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
  src/                      pure-MIND implementation (native front-end, shipped)
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
  version = {0.3.0}
}
```

## Links

- **PyPI**: <https://pypi.org/project/mind-nerve/>
- **Phase-1 weights**: <https://huggingface.co/star-ga/mind-nerve>
- **mind-mem** (companion memory engine, Apache-2.0): <https://github.com/star-ga/mind-mem>
- **MIND language**: <https://mindlang.dev>
- **Changelog**: [`CHANGELOG.md`](./CHANGELOG.md)
- **Roadmap**: [`ROADMAP.md`](./ROADMAP.md)
- **Issues**: <https://github.com/star-ga/mind-nerve/issues>

<!-- mind-profile: default -->
