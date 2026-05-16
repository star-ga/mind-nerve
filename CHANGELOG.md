# Changelog

All notable changes to mind-nerve. Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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

[0.1.0-alpha.2]: https://github.com/star-ga/mind-nerve/releases/tag/v0.1.0-alpha.2
