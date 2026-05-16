# Changelog

All notable changes to mind-nerve. Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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

[0.1.0-alpha.6]: https://github.com/star-ga/mind-nerve/releases/tag/v0.1.0-alpha.6
[0.1.0-alpha.5]: https://github.com/star-ga/mind-nerve/releases/tag/v0.1.0-alpha.5
[0.1.0-alpha.4]: https://github.com/star-ga/mind-nerve/releases/tag/v0.1.0-alpha.4
[0.1.0-alpha.3]: https://github.com/star-ga/mind-nerve/releases/tag/v0.1.0-alpha.3
[0.1.0-alpha.2]: https://github.com/star-ga/mind-nerve/releases/tag/v0.1.0-alpha.2
