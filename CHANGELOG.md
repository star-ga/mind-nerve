# Changelog

All notable changes to mind-nerve. Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0-alpha.11] — 2026-05-16

### Changed
- README latency claims rewritten to be unambiguous. The "23 ms p95" figure
  is the **warm daemon, GPU** number; CPU cold-start is ~90 ms; the
  ≤30 ms-on-4-core-CPU end target lands with the Phase 2 native MIND
  inference loop. Both numbers are now stated together in the highlights,
  comparison table, daemon-mode section, and design-constraints section.
- Rebuilt and reverified the FORTRESS-protected `libmindnerve.so` (51,280
  bytes; 8 exports; 7/7 leak-verifier checks pass) before the wheel build.

## [0.1.0-alpha.10] — 2026-05-16

### Changed
- README rewritten for the public alpha: centered hero, status badges
  (PyPI / Python / License / CI / Downloads / HF / Stars), table-driven
  comparison against the standard responses to library-size growth,
  four-step Quickstart, integration matrix, console-script reference,
  full env-var table, citation block. No code changes.
- Hardware references in the README, CHANGELOG, and historical training
  docs are now generic (`RTX`, `RTX-class hw`) instead of naming a
  specific consumer card. Supersedes the 0.1.0-alpha.9 README, which was
  never uploaded to PyPI.

## [0.1.0-alpha.9] — 2026-05-16

### Changed
- README rewritten for the public alpha (superseded by 0.1.0-alpha.10
  before PyPI upload).

## [0.1.0-alpha.8] — 2026-05-16

### Changed
- Layout-detection labels in `mind-nerve-install detect` / install JSON
  output are now neutral: `starga_symlink` → `symlinked_catalog`,
  `starga_shared_catalog` → `shared_catalog_dir`. The detection logic
  is unchanged; only the user-facing strings are cleaned up.
- README + installer docstrings lead with the typical Claude Code
  install case (`~/.claude/skills` → `~/.claude/skills.full`); the
  shared-catalog path is described as one of several alternatives
  rather than the canonical layout.

## [0.1.0-alpha.7] — 2026-05-16

### Added
- **`mind-nerve-routed` daemon** — long-lived route server over a UNIX
  socket. Loads the runtime once at startup, answers single-line JSON
  requests forever. Round-trip after warmup is sub-30 ms (typical 23 ms
  on RTX-class hardware), 12× faster than the cold subprocess path
  and inside the Phase 2 p95 ≤ 30 ms target even on Phase 1 PyTorch.
  Socket defaults to `$XDG_RUNTIME_DIR/mind-nerve.sock` with a `/tmp`
  fallback. Console script: `mind-nerve-routed`. Module: `mind_nerve.daemon`.
- **`mind-nerve-routed-ensure`** — idempotent daemon starter, designed
  to be wired into a Claude Code `SessionStart` hook. Probes the socket;
  spawns the daemon detached if not responsive; always exits 0.
- **`mind-nerve-preselect`** — Claude Code `UserPromptSubmit` hook that
  reads the prompt, asks the daemon for the top-K matching skills, and
  atomically rewrites the projected skills directory. Auto-detects three
  install layouts (regular `~/.claude/skills.full/`, STARGA shared
  `~/.agents/skills/`, or symlink) and falls open on every error.
- **`mind-nerve-install install --with-preselect`** — wires the
  SessionStart + UserPromptSubmit hooks above into `~/.claude/settings.json`
  for the user's actual layout. For regular users this renames their
  existing `~/.claude/skills/` to `~/.claude/skills.full/` once.
- **`mind-nerve-install install --with-mind-mem`** — optional companion
  that registers the `mind-mem-mcp` MCP server alongside `mind-nerve-mcp`
  in the same CLI configs (claude-code / claude-desktop / cursor / codex).
  mind-nerve routes intent; mind-mem provides search-backed memory.
- PyPI metadata: `keywords = [agent, llm, mcp, preselector, ...]` +
  `Topic :: Scientific/Engineering :: Artificial Intelligence` classifier
  for better discoverability.

### Fixed
- CI: ruff format applied across `python/mind_nerve/`; smoke step no
  longer asserts the proprietary `libmindnerve.so` is in the wheel
  (CI builds an OSS-surface wheel by design; production wheels are
  built locally with the protected runtime before PyPI upload).

## [0.1.0-alpha.6] — 2026-05-16

### Added
- **`pip install mind-nerve` works out of the box.** First `route()` call
  auto-downloads the Phase-1 weights (~150 MB) from
  [`star-ga/mind-nerve-phase1`](https://huggingface.co/star-ga/mind-nerve-phase1)
  into `~/.local/share/mind-nerve/runtime/`. No more manual
  `huggingface-cli download` + `MIND_NERVE_RUNTIME_DIR` setup.
- GitHub Actions CI: ruff lint + wheel build + `libmindnerve.so`-in-wheel
  check + multi-Python smoke (3.10 / 3.11 / 3.12) + pytest gate.
- Regression tests for the 0.1.0a4 fixes in `tests/python/test_runtime_dir_env.py`:
  atomic save no longer leaks `.tmp.npy`; CLI learn/watch honor
  `MIND_NERVE_RUNTIME_DIR`.

### Changed
- `huggingface_hub>=0.20` is now a direct dependency (was indirect via
  `sentence-transformers`).
- `inference._DEFAULT_RUNTIME_DIR` is a lazy proxy now; the runtime path
  resolves on first use rather than at import time, so the HF download
  isn't triggered just by `import mind_nerve`.
- `precompute_routes` default `runtime_dir` and `catalog_path` are
  `None`; resolution mirrors the new auto-seed flow.

### Removed
- Hardcoded `/data/datasets/mind-nerve-catalog/...` default. Replaced by
  the user-local + HF-auto-seeded path. STARGA-internal use sets
  `MIND_NERVE_RUNTIME_DIR` explicitly.

## [0.1.0-alpha.5] — 2026-05-16

### Fixed
- README hygiene: `pip install` line in Quickstart no longer pins a stale version (was advising `pip install mind-nerve==0.1.0a3` even after later releases shipped). Status line updated to current version.

### Changed
- No code change. Same wheel surface as 0.1.0a4.

## [0.1.0-alpha.4] — 2026-05-16

### Fixed
- `mind-nerve learn` and `mind-nerve watch` now honor `MIND_NERVE_RUNTIME_DIR`. Previously, `cli.cmd_learn` / `cli.cmd_watch` hardcoded a fallback runtime path and bypassed `_DEFAULT_RUNTIME_DIR`, causing env-var users to hit `ENOENT` writing back to a path they don't own.
- `discovery._save_table_atomic` no longer leaves `route_table.npy.tmp.npy` on disk. The temp filename used `route_table.npy.tmp`, which NumPy auto-extended to `.tmp.npy`, breaking the subsequent atomic `os.replace`. Switched to passing an open file handle and a `route_table.tmp.npy` suffix.

## [0.1.0-alpha.3] — 2026-05-16

### Added
- **First PyPI release.** `pip install mind-nerve`. Project page live at <https://pypi.org/project/mind-nerve/>.
- Phase-1 weights uploaded to <https://huggingface.co/star-ga/mind-nerve-phase1> under Apache-2.0. 152 MB total (`checkpoint/` + `manifest.json` + `route_table.npy` + `route_table.jsonl`).

### Changed
- Wheel `package-data` includes `lib/*.so` / `lib/*.dylib` / `lib/*.dll` so the FORTRESS-protected `libmindnerve.so` actually ships inside the wheel (was being silently dropped by the prior `data/*.json,bin` glob).
- README / LICENSE / ROADMAP rewritten for the public alpha + dual-license framing (Apache code + weights; FORTRESS-protected runtime binary inside the wheel).

## [0.1.0-alpha.2] — 2026-05-16 (private alpha)

First private alpha tag. Phase 1 (Python-side inference) is complete; Phase 2 (native MIND Q16.16 inference) is the next milestone.

### Added
- **Catalog v1.1-oss** — 11,922 routing-candidate skills mined from public registries (npm, PyPI, crates.io, HF, GitHub). Frozen with content hash. License-gated (PUBLIC_LICENSES allowlist + COMMERCIAL_MARKERS regex).
- **Custom BPE tokenizer v1.0** — 16k vocab, byte-level, NFC, byte_fallback. Locked special tokens.
- **Phase 1 encoder + scoring head** — fine-tuned `BAAI/bge-small-en-v1.5` with MultipleNegativesRankingLoss. Top-5 = 96.06% against the full corpus pool.
- **Python wheel (`pip install mind-nerve`)** — `route()` / `precompute_routes()` API + `mind-nerve` CLI (`route`, `info`, `precompute-routes`, `learn`, `watch`).
- **MCP server façade** — stdio JSON-RPC, exposes `mind_nerve_route` tool to any MCP-capable client.
- **17-CLI installer** — MCP-first (`claude-code`, `claude-desktop`, `cursor`, `codex`) + `claude-code-hook` fallback + 10 stub adapters for vendor CLIs that don't speak MCP yet.
- **Discovery layer** — `scan()`, `Watcher`, `add_route()` with license-gated ingest (refuses `commercial_risk`, requires `--include-unknown` for unknown-license sources).
- **Protected runtime shell** — `libmindnerve.so` (51 KB) bundled in the wheel; 8 FORTRESS C-side primitives (`mindnerve_protection_init`, `mindnerve_heartbeat`, `mindnerve_auth_challenge`, `mindnerve_auth_verify`, `mindnerve_auth_is_verified`, `mindnerve_is_protected`, `mindnerve_get_version`, `mindnerve_shutdown_protection`). Build pipeline + protection sources live in private `star-ga/mind-nerve-protected`.

### Security
- Public mind-nerve repo history scrubbed of proprietary protection sources via `git filter-repo` on 2026-05-16. The FORTRESS toolchain (846-line `protection.mind`, 1199-line `protection.c`, build pipeline, exports.map, verify_leak.sh) lives only in the private sibling repo and never enters this tree.
- `.gitignore` hardened to block re-introduction of `protected-build/`, `dist/`, `*.so`, `*.dylib`, `*.dll`.
- Shipped `libmindnerve.so` passes 7-check leak verifier: 8 expected exports, no STARGA-private markers, no developer-machine paths, no API-key fingerprints, no embedded MIND source / mindc-getter symbols, `.comment` is the STARGA toolchain stamp only, and `ptrace` is referenced (anti-debug present).

### Known limitations
- Inference path runs Python-side (PyTorch via the wheel). Native MIND Q16.16 inference is Phase 2.
- Cross-architecture bit-identity gate (x86 CPU vs CUDA) — Phase 2 only; requires the native inference path.
- Latency p95 ≤ 30 ms target on a 4-core CPU — Phase 2 only; currently measured Python-side.
- `mindc` 0.2.5 parses `Mind.toml [protection]` / `[exports]` but does not yet act on them. Protection is delivered by the C bridge + build-pipeline post-processing.

[0.1.0-alpha.11]: https://github.com/star-ga/mind-nerve/releases/tag/v0.1.0-alpha.11
[0.1.0-alpha.10]: https://github.com/star-ga/mind-nerve/releases/tag/v0.1.0-alpha.10
[0.1.0-alpha.9]: https://github.com/star-ga/mind-nerve/releases/tag/v0.1.0-alpha.9
[0.1.0-alpha.8]: https://github.com/star-ga/mind-nerve/releases/tag/v0.1.0-alpha.8
[0.1.0-alpha.7]: https://github.com/star-ga/mind-nerve/releases/tag/v0.1.0-alpha.7
[0.1.0-alpha.6]: https://github.com/star-ga/mind-nerve/releases/tag/v0.1.0-alpha.6
[0.1.0-alpha.5]: https://github.com/star-ga/mind-nerve/releases/tag/v0.1.0-alpha.5
[0.1.0-alpha.4]: https://github.com/star-ga/mind-nerve/releases/tag/v0.1.0-alpha.4
[0.1.0-alpha.3]: https://github.com/star-ga/mind-nerve/releases/tag/v0.1.0-alpha.3
[0.1.0-alpha.2]: https://github.com/star-ga/mind-nerve/releases/tag/v0.1.0-alpha.2
