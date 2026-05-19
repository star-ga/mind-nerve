# Changelog

All notable changes to mind-nerve. Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] — v0.3.0 preparation

## [0.3.0-beta.5] - 2026-05-18

Independent-audit hygiene pass: removed public references to internal
build infrastructure and tightened the socket/lockfile path used by the
daemon.

### Security

- Removed mentions of internal toolchain processes from README, ROADMAP,
  CHANGELOG, LICENSE, spec/, and docs/.
- Daemon and ensure() now prefer `$XDG_RUNTIME_DIR` (or
  `~/.cache/mind-nerve/run/` at mode 0700) over a predictable
  `/tmp/mind-nerve-<uid>` path, closing a local symlink-attack DoS
  surface on shared systems.
- New `python/mind_nerve/_runtime_dir.py` + `tests/python/test_runtime_dir.py`.

No surface-API behaviour change.

## [0.3.0-beta.4] — 2026-05-18

Audit response (deep-research 2026-05-18): deterministic SHA-256 tie-break
in route(), top_k/request-length bounds, atomic installer writes with .bak,
HF revision pin, README architecture+latency+license clarifications.

### Fixed — deterministic SHA-256 tie-break in `route()`

- `_route_pytorch()` now uses `_deterministic_topk()` which sorts equal-score
  routes by ascending `SHA-256(route_id)` digest, matching the spec contract
  in `spec/architecture.md`. Previously equal-score routes could reorder
  across platforms, undermining the cross-arch bit-identity guarantee.
- Added `_tie_key(route_id)` and `_deterministic_topk(scores, route_ids, k)`
  helpers in `inference.py`.

### Added — `top_k` and request-length bounds in `route()`

- `route()` now raises `ValueError("top_k must be in [1, 64]")` if `top_k`
  is outside the spec-mandated range.
- `route()` now raises `ValueError("RequestTooLong: query exceeds 1024 tokens")`
  if the BPE token count of the query exceeds 1024, per spec `architecture.md`.
- `_count_bpe_tokens()` helper handles both pytorch (SentenceTransformer.tokenize)
  and native (AutoTokenizer) backends.

### Fixed — atomic installer writes with `.bak` safety copy

- Added `safe_write(path, content)` to `installer.py`: backs up the existing
  file to `<path>.bak` before writing, uses `tempfile + os.replace` for
  atomicity. All 12 `write_text` call-sites in `installer.py` now use this
  helper, eliminating partial-write data loss on crash or power failure.

### Fixed — HF revision pin in `_seed_from_hf()`

- `snapshot_download` now passes `revision="71221fd435f119cc50c92df4786352ac594efa17"`
  (the current `star-ga/mind-nerve-phase1` HEAD) and `allow_patterns` to
  limit downloaded artifacts to the inference-required files.
- Override with `MIND_NERVE_HF_REVISION=<sha-or-tag>` for reproducible builds.

### Changed — README architecture, latency, and license clarifications

- "How it works" now correctly describes the shipped architecture: encoder +
  direct scoring head (drop-the-decoder, sliding-window encoder, window=256
  stride=192). The stale "asymmetric encoder/decoder with a classifier head"
  description has been removed.
- Phase-1 Python latency (~90 ms warm-daemon CPU) and Phase-2 native target
  (≤30 ms CPU) are now stated separately and clearly.
- License section now has a single concise paragraph explaining the Apache-2.0
  Python/weights surface and the separately licensed bundled native runtime.
- `MIND_NERVE_HF_REVISION` env var added to the Configuration table.

## [0.3.0-beta.3] — 2026-05-18

### Changed — documentation polish

- Replaced internal-jargon phrasing in `ensure.py` and `installer.py`
  docstrings, the bundled `templates/mind-nerve-routed.service` unit
  comment, and the v0.3.0-beta.2 CHANGELOG entry with neutral
  product-facing language. The shipped behaviour is unchanged from
  v0.3.0-beta.2; only comments and prose were edited. Bumping the
  release so the language in the PyPI wheel matches the language in
  the source tree.

### Added — `spec/` reference to downstream orchestrators

- `spec/mind_mem_v4_integration.md` now references "downstream
  orchestrators" generically instead of naming a specific consumer.

## [0.3.0-beta.2] — 2026-05-18

### Fixed — concurrent ensure() spawning multiple daemons

- `python/mind_nerve/ensure.py` now serialises the spawn decision under
  a sibling `mind-nerve.sock.lock` `flock`. Prior to this fix, parallel
  CLI invocations during the daemon's ~5 s weight-load window all saw
  an unresponsive socket from their fast-path probe and each spawned a
  fresh `mind-nerve-routed` process. In a high-concurrency workload
  this accumulates 10+ zombie daemons (~1.3 GB each), pinning memory
  until the parent process is restarted.
- The flock guard makes the spawn decision exclusive: one ensure()
  caller wins, spawns, then **holds the lock while waiting for the
  socket to come up** (`WAIT_SECONDS = 20`). All other parallel
  callers either lose the flock and poll the socket for the winner's
  daemon, or acquire the lock after the winner exits and re-check the
  socket before deciding whether to re-spawn. Net effect: at most one
  spawn per WAIT_SECONDS window, regardless of caller concurrency.
- New `tests/python/test_ensure_concurrency.py` — 6 regression tests
  including a 16-thread race that asserts `spawn_count == 1` and a
  4-thread fall-through scenario proving the script still exits 0
  fail-open when the daemon never comes up.

### Added — Tier-3 script-floor CI gate (multilingual policy)

- `tests/python/test_tier3_script_floor.py` enforces the Tier-3
  contract from `spec/quality_targets.md` §"Multilingual language
  policy": the byte-level tokenizer round-trips losslessly
  (`decode(encode(x)) == x`) for every script, so no language
  silently breaks at the tokenizer layer. 42 checks across a
  UDHR-Article-1 / FLORES-200-proxy corpus covering Latin, Cyrillic,
  Han (Simplified), Japanese, Hangul, Devanagari, Bengali, Arabic,
  Greek, Hebrew, Thai, Tamil, Telugu + emoji / combining-mark /
  mixed-script edge cases.
- Hermetic: a pure-Python reference of the exact GPT-2 / `ByteLevel`
  byte↔unicode bijection. No torch / tokenizers / network — runs in
  CI's `tests` job (`pytest tests/python`) in ~0.2 s.
- Includes a structural bijection proof over all 256 byte values
  (guarantees *any* UTF-8 language round-trips, not only the sampled
  ones) and a corpus-coverage guard so the fixture cannot silently
  lose a required script.

## [0.3.0-beta.1] — 2026-05-17

### Added — public `mind_train` surface (bring-up trainer)

- **New `mind_nerve.mind_train` module.** Publishes the trainer as a
  stable, typed Python API:
  - `TrainConfig(catalog_path, output_dir, base_model, epochs,
    batch_size, lr, max_len, seed, eval_frac, smoke_test, backend)`
    — frozen dataclass.
  - `TrainResult(checkpoint_dir, manifest_path, model_hash,
    epochs_completed, train_pairs, eval_pairs, metrics,
    baseline_metrics, elapsed_seconds, backend_used, extras)` —
    frozen dataclass.
  - `train(config) -> TrainResult`
  - `config_to_dict(config)` — JSON-safe view.
- **Backend resolution.**
  - `backend="python"` (default): PyTorch + sentence-transformers MNR
    loss recipe ported from `catalog-builder/train_phase1.py`. Trains
    on a `name\\tkind\\tbody` TSV corpus; produces a
    sentence-transformers checkpoint + manifest with deterministic
    `model_hash` (SHA-256 over the sorted file tree, paths bound in).
  - `backend="native"`: raises `NotImplementedError` until mindc 0.3.0
    `--emit-shared` cdylib + the Q16.16 native kernel land.
- **New `mind-nerve train` CLI subcommand.** Same flags as the
  underlying `TrainConfig`, including `--smoke-test` (500 pairs / 1
  epoch / ~1 min) for pipeline validation without a full run.

### Added — tests

- `tests/integration/test_mind_train_contract.py` (9 invariants):
  frozen-dataclass shape, parser handles malformed rows, deterministic
  seeded split, checkpoint hash is deterministic and path-bound,
  `native` raises `NotImplementedError`, unknown backend raises
  `ValueError`, `config_to_dict` is JSON-safe, missing catalog fails
  fast before downloading any models.

### Note

- v0.3.0-beta.1 is the **bring-up ship**: the trainer is real and
  reproducible, but runs under PyTorch. v0.3.0 (final) replaces the
  Python backend with the native MIND Q16.16 kernel via the mindc
  0.3.0 cdylib emit foundation already landed in mindc 0.2.11
  (`--emit-shared`). The Python backend stays available behind the
  same `TrainConfig.backend` switch for reproducibility and
  cross-backend bit-identity comparison.

## [0.2.0] — 2026-05-17

### Added — Tier 3 attestation cross-binding (public Python surface)

- **New `mind_nerve.attestation` module.** Publishes the MindLLM
  cross-binding handshake primitives as stable Python API:
  `binding_message`, `sign_binding`, `verify_binding`,
  `application_verify_binding` (with ZeroField guard),
  `serialize_binding_record`, `deserialize_binding_record`,
  `manifest_export_bytes`, `neuron_hash_hex`, and a frozen
  `BindingRecord` dataclass. Mirrors `integrations/mindllm_attestation.mind`
  exactly — same wire format (200 bytes, magic `MNBA`, version 1),
  same Ed25519 over `SHA-256(mind_nerve ‖ mindllm ‖ nonce)`.
- **New `mind-nerve attest` CLI subcommand.**
  - `mind-nerve attest sign --mind-nerve-hash HEX --mindllm-hash HEX
    --nonce HEX --private-key-hex HEX` → prints the 200-byte
    `BindingRecord` as JSON `{record_hex, signer_pubkey_hex,
    binding_msg_hex}`.
  - `mind-nerve attest verify --record-hex HEX [--pubkey-hex HEX]`
    → exits 0 on `result == "ok"`, non-zero otherwise. Optional
    `--pubkey-hex` enforces a trust anchor.
- **Refactored tests.** `tests/integration/test_mindllm_handshake.py`
  now imports the public module instead of inlining reference
  primitives, so the published Python surface IS the contract. Added
  3 new invariants (round-trip, magic mismatch, short buffer).

### Language policy

- Singling out one non-English language as a release gate is now
  explicitly out of scope. The Russian-specific top-5 target referenced
  in earlier plans is replaced by the **multilingual policy** in
  [`spec/quality_targets.md`](spec/quality_targets.md): Tier 1 (twelve
  major languages, measured and gated), Tier 2 (next ~20, measured and
  monitored, not gated), Tier 3 (script floor — tokenizer round-trip
  CI gate over FLORES-200, no language silently breaks). v0.2.0 ships
  English as the certified target; Tier 1–3 land in their own
  multilingual workstream.

## [0.2.0-beta.1] — 2026-05-17

### Added — Tier 2 catalog-builder v2 (SOTA-tracks #3 + #4)

- **SOTA-track #4 — frequency-adaptive route scaling.**
  `precompute_routes(emit_freq_scale=True, ...)` (or any run with a
  `cooccurrence_path`) now writes `route_table_freq_scale.npy`, a
  per-route `float32` scalar equal to `max(1/sqrt(freq), 0.5)` under
  Laplace smoothing (`freq = raw_count + 1`). At catalog load time, the
  runtime multiplies each L2-normalized embedding row by this scale
  in place — zero runtime cost. Addresses the long-tail drown-out
  problem of rare-but-critical routes.
- **SOTA-track #3 — entropy → stride threshold table.**
  `precompute_routes(emit_stride_thresholds=True)` writes
  `stride_thresholds.json` with a calibrated entropy → stride map
  (`{<0.4: 256, <0.7: 192, else: 96}`, default 192). Consumed by the
  native-MIND windowed encoder once mindc 0.3.0 cdylib lands;
  forward-compatible bookkeeping in the Phase-1 sentence-transformers
  path.
- New `mind-nerve precompute-routes` flags: `--emit-freq-scale` and
  `--emit-stride-thresholds`. Both files are also emitted automatically
  when `--cooccurrence` is supplied, so a single catalog-builder run
  fills all v2 columns.

### Added — runtime consumers

- Catalog-v2 freq_scale loader on the `_Runtime` constructor. Absent
  file → unchanged v1 behavior; shape mismatch → early `RuntimeError`.
- Stride-threshold loader exposes `_Runtime.stride_thresholds` as a
  dict; ignored by the Phase-1 encoder, ready for the native path.

### Added — tests

- `tests/integration/test_route_freq_scale.py` (7 properties: absent
  file, present file multiplies rows, shape mismatch raises, near-zero
  scale suppresses a route, unit scale fallback, 0.5 floor for common
  routes, stride table well-formedness).

### Note

- Public HF Phase-1 weights remain catalog-v1; flipping a runtime to
  v2 still requires the matching catalog emit. v2 weights with the
  full model_hash bump arrive on the next training cadence; per-
  language eval coverage rolls in via the multilingual workstream
  (see `spec/quality_targets.md` §"Multilingual language policy").

## [0.1.0-beta.2] — 2026-05-17

### Added — catalog-v2 runtime readiness (SOTA-track #1)

- **Optional `route_table_prior.npy` column.** When present in the
  runtime dir, the runtime loads it as a per-route log-prior and adds
  it to the dot-product score before top-k selection. Bayesian
  combination: `P(route|query) ∝ P(query|route) · P(route)`.
- Absent prior file leaves the scoring path unchanged — v1 catalogs
  continue to work without modification.
- Shape mismatch at load time raises `RuntimeError` early rather than
  producing wrong results.
- The catalog-builder side already emits the v2 wire format
  (`catalog-builder/format/cat_v2.py`, magic `MNC2` + `PRIR` tail)
  with `freq_adaptive_scale` applied per-route. This release wires
  the runtime to consume it.

### Added — tests

- `tests/integration/test_route_prior.py` (4 properties: absent file,
  present file, shape mismatch, prior changes top-1 result).

### Note

The publicly shipped HF Phase-1 weights remain catalog-v1. This
release makes the runtime forward-compatible with v2; the v2 weights
arrive on the next multilingual training cadence (see
`spec/quality_targets.md` §"Multilingual language policy" for the
Tier-1 / Tier-2 / Tier-3 coverage commitments).

## [0.1.0-beta.1] — 2026-05-17

First beta. Closes Tier 1 + Tier 2 + Tier 3 of the locked Phase 2 +
Phase 3 ship plan (`docs/plans/FINAL_SHIP_PLAN_2026_05_17.md`). The
remaining `v1.0.0` blockers are external (mindc 0.3.0 cdylib emit,
mind-mem v4 cognitive kernel, ARM CI runner) and remain deferred.

### Added — Tier 1: installer matrix expansion

- **Native Gemini CLI extension installer** — registers mind-nerve as a
  first-class extension under `~/.gemini/extensions/` with manifest,
  shim, and uninstall path. Activates `--with-gemini`.
- **Vibe MCP installer** — wires mind-nerve into Vibe's MCP server
  registry with the standard MIND-MEM-style entry. Activates
  `--with-vibe`.
- **Claw-family installers** — five sibling targets (`--with-codeclaw`,
  `--with-cursorclaw`, `--with-graviton`, `--with-tirex`,
  `--with-claudeclaw`) using a shared shim writer. Each forwards
  request strings to `mind_nerve.route()` and prints the JSON result.

### Added — Tier 1: evidence-chain hardening

- **Reject envelopes with zero `request_hash`** in the verifier
  (`request_hash != 0` is now a verifier invariant, not advisory).
  Closes SOTA-track #2 from the ship plan: input-fingerprinted
  attestation. Mirrors the `model_hash == 0` rule already enforced.

### Added — Tier 2: adaptive window stride

- **Content-fingerprinted stride** — window/stride is now computed
  from a content fingerprint per request, replacing the hard-coded
  `stride=192`. Calibrated thresholds shipped in
  `tools/calibrate_stride.py`. Closes SOTA-track #3.

### Added — Tier 3: Phase 3 scaffolds

- **Skill-marketplace adapter** — typed interface (`adapter.mind` +
  Python stub) for routing requests to external skill registries.
  Stub-only ship; functional ship awaits the rest of Phase 2.
- **Federated cross-host routing** — typed-port design + stub for
  routing requests across mind-nerve instances on different hosts.
  Stub-only ship; functional ship awaits the future typed-edges composition layer.
- **mind-mem v4 cognitive-kernel binding spec** — the published
  contract for plugging mind-nerve into mind-mem's cognitive kernel
  (route-history as memory class). Spec ships now; functional ship
  awaits mind-mem v4 (external).

### Added — Tier 3: attestation cross-binding

- **Per-tensor weight manifest** (`src/loader.mind`): each weight tensor
  now carries a `neuron_hash: [u8; 32]` — the SHA-256 of its Q16.16 byte
  layout (i32 LE, row-major). The hashes are accumulated via
  `build_manifest_preimage` (magic `"MNPM"`, canonical alphabetical-name
  sort) into a manifest aggregate that supersedes the opaque `model_hash`
  field for consumers that need per-tensor traceability. New public
  types: `TensorManifestEntry`, `TensorManifest`. New public functions:
  `build_tensor_manifest()`, `manifest_export()`.

- **`manifest_export()`** emits a deterministic UTF-8 JSON document from
  a `TensorManifest`. Output is byte-identical across runs and platforms
  (no dict-iteration ordering, no timestamps). The `aggregate` field is
  the SHA-256 of the canonical preimage, hex-encoded lowercase.

- **MindLLM cross-binding handshake spec**
  (`integrations/mindllm_attestation.mind`): typed protocol
  `(mind_nerve_model_hash, mindllm_model_hash, shared_nonce) ->
  BindingSignature` using `SHA-256` for the binding message and
  `Ed25519` (RFC 8032) for signing. The spec is fully self-contained
  for external consumers — no STARGA-internal toolchain required to
  implement a verifier. New public types: `BindingRecord`,
  `BindingVerifyError`. New public functions: `binding_message()`,
  `sign_binding()`, `verify_binding()`, `serialize_binding()`. Wire
  format: 200-byte packed record, magic `"MNBA"`.

- **`spec/architecture.md`** extended with two new sections:
  §"Per-neuron manifest" and §"MindLLM cross-binding handshake"
  (protocol summary, verifier algorithm, `BindingRecord` wire format,
  chain discipline).

- **`cryptography>=41.0`** added to `pyproject.toml` runtime
  dependencies to support Ed25519 operations in Python test and
  integration code.

- **Unit tests** `tests/unit/test_manifest_export.mind` (7 properties).
- **Integration tests** `tests/integration/test_mindllm_handshake.py`
  (10 properties).

### Added — docs

- `docs/plans/FINAL_SHIP_PLAN_2026_05_17.md` — locked block-status
  matrix and version cadence from a13 → 1.0.0.

### Deferred (gated)

Tracked but not in this release; see `docs/plans/FINAL_SHIP_PLAN_2026_05_17.md`:

- 18-backend cross-arch bit-identity — needs mindc 0.3.0 cdylib emit.
- Native MIND inference replacing PyTorch — needs mindc 0.3.0.
- p95 ≤ 30 ms on 4-core CPU (native) — needs mindc 0.3.0.
- p95 ≤ 30 ms on ARM — needs mindc 0.3.0 + ARM CI runner.
- Tier-1 multilingual coverage (12 languages, gated) — compute-bound
  training run on the merged multilingual corpus; lives in the
  multilingual workstream (see `spec/quality_targets.md`).
- Native MIND `mind-train` pipeline — standalone bring-up shippable,
  deep work continues in v0.3.0.
- Per-head learned drop masks (SOTA-track #5) — depends on
  `mind-train`.

## [0.1.0-alpha.13] — 2026-05-16

  carries a `neuron_hash: [u8; 32]` — the SHA-256 of its Q16.16 byte layout
  (i32 LE, row-major). The hashes are accumulated via `build_manifest_preimage`
  (magic `"MNPM"`, canonical alphabetical-name sort) into a manifest aggregate
  that supersedes the opaque `model_hash` field for consumers that need
  per-tensor traceability. New public types: `TensorManifestEntry`,
  `TensorManifest`. New public functions: `build_tensor_manifest()`,
  `manifest_export()`.

- **`manifest_export()`** emits a deterministic UTF-8 JSON document from a
  `TensorManifest`. Output is byte-identical across runs and platforms (no
  dict-iteration ordering, no timestamps). The `aggregate` field is the
  SHA-256 of the canonical preimage, hex-encoded lowercase.

- **MindLLM cross-binding handshake spec** (`integrations/mindllm_attestation.mind`):
  typed protocol `(mind_nerve_model_hash, mindllm_model_hash, shared_nonce) ->
  BindingSignature` using `SHA-256` for the binding message and `Ed25519`
  (RFC 8032) for signing. The spec is fully self-contained for external
  consumers — no STARGA-internal toolchain required to implement a verifier.
  New public types: `BindingRecord`, `BindingVerifyError`. New public
  functions: `binding_message()`, `sign_binding()`, `verify_binding()`,
  `serialize_binding()`. Wire format: 200-byte packed record, magic `"MNBA"`.

- **`spec/architecture.md`** extended with two new sections: §"Per-neuron
  manifest" (tensor naming convention, `manifest_export` JSON format,
  neuron-hash aggregation rule) and §"MindLLM cross-binding handshake"
  (protocol summary, verifier algorithm, `BindingRecord` wire format, chain
  discipline).

- **`cryptography>=41.0`** added to `pyproject.toml` runtime dependencies
  to support Ed25519 operations in Python test and integration code.

- **Unit tests** `tests/unit/test_manifest_export.mind` (7 properties: P1
  determinism, P2 known-vector reference, P3 order-sensitivity, P4 sort
  stability, P5 JSON structure, P6 all-zero anchor, P7 tamper detection).

- **Integration tests** `tests/integration/test_mindllm_handshake.py` (10
  properties: H1–H10 covering binding_message determinism, input sensitivity,
  sign/verify round-trip, signature corruption, hash/nonce alteration, zero-
  field guard, serialization determinism + size, and manifest_export
  determinism with SHA-256 byte-identity check).

## [0.1.0-alpha.13] — 2026-05-16

> Final alpha. Beta cut as 0.1.0-beta.1 on 2026-05-17.

### Fixed
- **CUDA OOM no longer crashes `route()`.** Hit while installing 0.1.0a12
  alongside another GPU-resident model (e.g. a local LLM in Ollama): the
  default sentence-transformers device pick (CUDA) raised
  `torch.AcceleratorError: CUDA error: out of memory` on first inference
  and the whole call failed. `_Runtime.__init__` now catches GPU-init
  failures and falls back to CPU with a one-line stderr notice. Users
  who want to force CPU unconditionally can set `MIND_NERVE_DEVICE=cpu`.

## [0.1.0-alpha.12] — 2026-05-16

### Changed
- README precision fix: the "~90 ms" CPU number is the **warm-daemon** path
  (encoder reused, model already loaded) not a cold-start. The actual cold
  subprocess number is ~270–340 ms because of the one-time model load.
  Now that the daemon is in-package the warm path is what users hit
  every turn after `SessionStart`, so the README states that explicitly.
- Rebuilt and reverified the bundled `libmindnerve.so` — same build, leak
  verifier passes.
- Supersedes the 0.1.0-alpha.11 tag, which was never uploaded to PyPI.

## [0.1.0-alpha.11] — 2026-05-16

### Changed
- README latency claims rewritten to be unambiguous. The "23 ms p95" figure
  is the **warm daemon, GPU** number; CPU cold-start is ~90 ms; the
  ≤30 ms-on-4-core-CPU end target lands with the Phase 2 native MIND
  inference loop. Both numbers are now stated together in the highlights,
  comparison table, daemon-mode section, and design-constraints section.
- Rebuilt and reverified the bundled `libmindnerve.so` before the wheel build.

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
- Wheel `package-data` includes `lib/*.so` / `lib/*.dylib` / `lib/*.dll` so the bundled `libmindnerve.so` actually ships inside the wheel (was being silently dropped by the prior `data/*.json,bin` glob).
- README / LICENSE / ROADMAP rewritten for the public alpha + dual-license framing (Apache code + weights; bundled runtime binary inside the wheel).

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
- **Bundled runtime component** — `libmindnerve.so` ships inside the wheel under a separate STARGA license; see `LICENSE.md`.

### Security

Each release of the wheel ships only the documented public API surface.

### Known limitations
- Inference path runs Python-side (PyTorch via the wheel). Native MIND Q16.16 inference is Phase 2.
- Cross-architecture bit-identity gate (x86 CPU vs CUDA) — Phase 2 only; requires the native inference path.
- Latency p95 ≤ 30 ms target on a 4-core CPU — Phase 2 only; currently measured Python-side.
- `mindc` 0.2.5 parses `Mind.toml [protection]` / `[exports]` but does not yet act on them. Protection is delivered by the C bridge + build-pipeline post-processing.

[0.1.0-alpha.13]: https://github.com/star-ga/mind-nerve/releases/tag/v0.1.0-alpha.13
[0.1.0-alpha.12]: https://github.com/star-ga/mind-nerve/releases/tag/v0.1.0-alpha.12
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
