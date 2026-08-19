# Changelog

All notable changes to mind-nerve. Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.3.0] — 2026-08-19

Stable release, finalizing the `0.3.0b1`–`0.3.0b10` beta line. Headline:
mind-nerve ships a real pure-MIND native front end — a native-ELF binary
and a native MCP server, not just the Python wheel's bundled encoder
`cdylib`. This entry consolidates the native front-end (`56ca3cb`) and the
`0.3.0b10` score-path/bundle work below it into one release note; see the
`[0.3.0b*]` entries in this file for the full beta-by-beta history.

### Feat — pure-MIND ELF-native front-end + native MCP server (headline)

The mind-nerve kernel tree and all 13 front-end `.mind` modules now
compile and link into a real native ELF binary on mindc 0.10.2 with zero
JIT/launcher fallback (`56ca3cb`).

- **Native MCP server** (`src/mcp.mind`): a pure-MIND stdio JSON-RPC MCP
  server compiled into the same binary. Message framing (`initialize`,
  error shapes `{id,code}`, notification suppression) is byte-identical to
  the Python `mcp_server.py` on the gated subset, verified against a
  frozen golden transcript (`tests/harness/mcp_golden.sh`). Ranking runs
  in-process — no daemon socket.
- **Windows (PE) host-shim variant**: `mind/runtime/nerve_rt_shims_win.c`
  defines all nine `__mind_nerve_rt_*` symbols (incl. `getenv`) and forces
  binary-mode stdio so the attestation frame stays byte-identical across
  substrates. Declared as `[targets.windows]` in `Mind.toml`, cross-built
  from Linux via the MinGW-w64 sysroot.
- **Loader hardening**: `src/loader.mind` now reads a versioned
  catalog/weights header (v1/v2; reserved bytes must be zero; unknown
  versions rejected) with allocation-bomb DoS caps — 64 MiB catalog /
  256 MiB weights / 65536 routes — enforced *before* allocation.
- **Route-id uniqueness is O(n log n)**: `src/loader.mind` replaced the
  O(n²) route-id uniqueness scan with a merge-sort over an index copy plus
  an adjacent-pair check; a ~2 MB catalog could previously pin a core for
  ~a minute. Loaded rows keep file order, so the catalog bytes and
  canonical hash are unchanged.
- **UTF-8 Windows compatibility**: UTF-8-explicit text I/O across the
  Python surface (Windows defaults to cp1252, which raised on UTF-8
  skill/config content), TOML path backslash-escaping, and discovery
  path-separator normalization so nested skills classify correctly on
  Windows.

### Fix — 10 findings closed alongside the native front end (`56ca3cb`)

1. **JSON-RPC id overflow (MCP_ID_CAP)**: an unbounded id echoed verbatim
   into the 16 KiB response buffer was an overflow reachable on the first
   stdin line. An over-long id now echoes `null` (a legal id); guarded by
   a pathological-id line in the golden transcript.
2. **Loader allocation-bomb DoS**: the versioned v1/v2 header's caps (64
   MiB catalog / 256 MiB weights / 65536 routes) are checked before any
   allocation happens, not after.
3. **O(n²) route-id uniqueness scan**: replaced with the O(n log n)
   merge-sort described above — closes a core-pinning DoS on a
   pathologically large catalog.
4. **Unbounded `n_rows` on the embedder surface**: `mind/exports/c_abi.mind`
   now bounds `n_rows` before the packed-catalog allocations it drives.
5. **Windows UTF-8 crash**: text I/O across the Python surface is now
   UTF-8-explicit; Windows' cp1252 default previously raised on UTF-8
   skill/config content.
6. **TOML path escaping on Windows**: Windows paths are now
   backslash-escaped before being written into TOML.
7. **Nested-skill misclassification on Windows**: discovery
   path-separator normalization so nested skills classify correctly on
   Windows (was POSIX-separator-only).
8. **Installer runtime-dir pin on an empty dir**: `installer.py` pins
   `MIND_NERVE_RUNTIME_DIR` only when the runtime dir is actually
   populated (matching the TypeScript installer) — a fresh install (and
   the Windows in-process MCP fallback) no longer crashes on a
   not-yet-seeded dir; the pin is also now scoped to the mind-nerve entry
   instead of leaking globally.
9. **Duplicate MCP server blocks on re-install**: `installer.py` now
   collapses duplicate MCP server blocks so re-running install is
   idempotent and self-heals a corrupted config.
10. **Partial route table silently accepted**: `discovery.py` now refuses
    to load a route table when only one of the two on-disk files
    (`route_table.jsonl` / `route_table.npy`) is present, instead of
    silently discarding the surviving table.

Gates: mindc byte-identity 19/19 modules + 262 module tests; MCP golden
3/3; bit-identity 9/9 (regenerated fixtures reproduce the goldens); Python
mypy/ruff/bandit clean + 484 tests; TypeScript installer 372 tests.

### Perf — scoring path is pure-MIND MT gemv; legacy C score-path retired (U2)

The score path runs the deterministic `__mind_blas_gemv_q16_mt` (MT Q16.16
gemv + persistent thread pool). Retired the now-dead score-path symbols
from `mind/runtime/blas_shims_i64.c` (nm caller-free proof; encode-path
matmul/qkt/attnv kept). Measured on U1 (i7-5930K, 12 threads): score
**p50 ≈0.58 ms / p95 ≈0.9 ms** on the 11,922-route catalog. On a fair
equal-thread-count comparison (both sides on all 12 hardware threads) it
beats numpy+OpenBLAS at p50 (≥1.4×) and is ~10× lower at the p95 tail;
byte-identical run-to-run and cross-substrate by construction — the
load-bearing property a float BLAS cannot offer.

### Native `mcp` bundle format + producer plumbing (U4a + U4b)

`src/mcp.mind` constructs `route_table.cat` (MNC1) + `encoder_weights.mnw`
(MNW1) instead of format-incoherent `.jsonl`/`.q16.bin` (U4a). The binary
producer — the `catalog-builder` MNC1 writer + a `native_bundle` seed module
— landed and is byte-verified against the loader's round trip (U4b). But see
Known limitations: it can only emit placeholder embeddings today, so native
`tools/call` stays fail-closed by default and auto-seed is intentionally off.

### Known limitations

- **Native `tools/call` is fail-closed by default.** Its routing path is
  fully wired (getenv-resolved model/catalog paths → tokenize → load → infer
  → format) and the binary-bundle producer (U4b) is landed and
  loader-round-trip tested. But the only bundle producible today uses
  **placeholder zero embeddings** — no 256-dim trained checkpoint exists, and
  the shipped 384-dim BGE route table is incompatible with the native loader's
  2-layer/256-dim contract — which would rank routes by SHA-256 tie-break, not
  relevance. So auto-seed is intentionally OFF and `tools/call` returns an
  explicit `unavailable` payload until a real checkpoint lands; the placeholder
  producer is reachable only via the `mind-nerve seed-native-bundle` dev
  command (which warns loudly). The Python `mind-nerve-mcp` server is the real
  router.
- **Windows runs the pure-Python fallback.** A native Windows PE target is
  declared in `Mind.toml` (`[targets.windows]`, MinGW-w64 cross-compile) —
  the native ELF binary is not a PE binary and is not what ships on
  Windows today; Windows installs use the pure-Python wheel path.
- **int8 routing tier deferred to the roadmap.** Q16.16 stays the default
  router (best-in-world for a *deterministic* router at 2× under the p95
  budget); a two-stage-exact int8 tier (coarse-scan → Q16.16 rescore,
  recall@16 100%) is gated on a `gemv_i8` SIMD intrinsic maturing in the
  `mind` compiler. See `ROADMAP.md` Phase 2 SOTA-track.

## [0.3.0b10] — 2026-08-19

### Perf — scoring path is pure-MIND MT gemv; legacy C score-path retired (U2)

The score path runs the deterministic `__mind_blas_gemv_q16_mt` (MT Q16.16 gemv
+ persistent thread pool). Retired the now-dead score-path symbols from
`mind/runtime/blas_shims_i64.c` (nm caller-free proof; encode-path matmul/qkt/attnv
kept). Measured on U1: score p95 **≈1.1 ms** on the 11,922-route catalog (2× under
the 2 ms budget); on a fair equal-thread comparison it beats numpy+OpenBLAS at p50
(≥1.4×) and is far lower at the p95 tail (see `[0.3.0]` / `docs/benchmarks.md` for
the corrected fair-comparison numbers), deterministic + cross-substrate byte-identical.

### Fix — native `mcp` bundle filenames declare their format (U4a)

`src/mcp.mind` constructs `route_table.cat` (MNC1) + `encoder_weights.mnw` (MNW1)
instead of format-incoherent `.jsonl`/`.q16.bin`; native `tools/call` still
fails-closed on real dirs (superseded — see `[0.3.0]`: the producer since landed
but is placeholder-only, so `tools/call` stays fail-closed by default). The
Python MCP server routes today.

### Roadmap — deterministic int8 routing tier (deferred)

Fable-measured decision: Q16.16 stays the default router (perfect precision +
wedge, already best-in-world for a deterministic router). int8 is a future
edge/batch tier via a two-stage-exact design (int8 coarse-scan → Q16.16 rescore,
recall@16 100%), gated on a `gemv_i8` SIMD intrinsic maturing in the mind compiler.
See `ROADMAP.md` Phase 2 SOTA-track.

## [0.3.0b9] — 2026-08-15

### Fix — hook: malformed catalog rows can no longer kill a prompt's routing

A catalog row carrying a whole document as its `name` (observed live: a 300+
char runbook with embedded newlines) was turned into a candidate agent
filename by `agent_path()`; `stat()` on it raised `ENAMETOOLONG` —
`Path.is_file()` does **not** swallow it — the hook's fatal handler fired,
and the prompt lost ALL of its routes, valid ones included. Route names are
catalog-controlled text, so `agent_path()` now skips names that could never
be agent files (path separators, NUL, > 200 bytes) and treats `OSError` from
`stat` as "not an agent". The malformed row degrades to an unresolved route;
the rest of the table is unaffected. Regression test drives the real hook
subprocess with a monster-name route and asserts the valid sibling still
routes (verified: fails against the unfixed hook, passes against the fix).

### Fix — hook: SessionStart banner throttled

CLIs that fire SessionStart on every resume re-injected the full
`## mind-nerve ready` banner into every restored transcript. The banner is
now emitted at most once per 12 h per state dir (sentinel next to
`MIND_NERVE_LOG`; `MIND_NERVE_SESSION_BANNER_MINUTES` configures, `0`
restores always-show). The skill projection still runs on every SessionStart
— only the announcement is throttled.

### Fix — audit sweep (2026-08-10 CI-mirror)

### Fix — acquire security-audit hardening, round 2 (codex audit, 2026-08)

Six further findings against acquire/security_scan, all fixed with
regression tests:

- **Symlink-to-skipped-dir bypass (HIGH)**: `SKILL.md -> dist/payload.md`
  vetted clean because `dist/` is never walked and internal symlinks were
  trusted without scanning their content. Internal symlinks whose resolved
  target sits in a skipped dir now FAIL (`symlink-to-skipped-dir`), and
  `_copy_local` preserves symlinks for the scanner to judge instead of
  silently dropping them into an empty package.
- **Shell-continuation rule bypass (HIGH)**: per-line regexes missed
  `curl … \` <newline> `| bash` and prompt-injection phrases split across
  lines. The line rules now also run on a backslash-folded view and a
  sliding 2-line window (deduped against per-line hits).
- **Tree-URL ref ignored + unconfined subdir (MEDIUM)**:
  `tree/<ref>/<subdir>` clones honour `ref` via `--branch` (fail loudly on
  un-cloneable refs; the manifest records the actual resolved commit), and
  subdirs containing `..`/absolute components are rejected.
- **Zero-route installs reported success (MEDIUM)**: when the reindex adds 0
  routes (license gate excluded everything / no routable content), install
  returns a `no-routable-content` warning and the CLI exits 3 with a stderr
  message. Benign reinstalls (all rows already indexed) stay quiet. Also
  documented: `kind="mcp"` candidates are vetted/installed as content but
  NOT registered into any CLI's MCP config (follow-up).
- **Tar FIFO/device members (MEDIUM)**: `_safe_members` now rejects any
  member that is not a regular file, dir, or safe link — on every Python
  version (a FIFO member could block a reader indefinitely).
- **ClamAV determinism (LOW)**: clamscan is now opt-in
  (`use_clamav=True` / `--clamav`) since its verdicts depend on the host
  signature DB, and clamav finding excerpts carry the signature name only,
  never host paths.

### Fix — acquire security-audit hardening (cross-CLI audit, 2026-08)

Twelve findings against the new acquire subsystem, all fixed with regression
tests:

- **NUL content-scan bypass (CRITICAL)**: any NUL in the first 8 KiB used to
  skip all line rules, so a NUL-prefixed `SKILL.md` carrying `curl | sh`
  installed clean. Now only known-binary extensions skip; other files are
  scanned with NULs stripped plus a `binary-content-in-text-file` WARN.
- **git clone option injection**: URLs starting with `-` are refused, and
  `--` terminates option parsing.
- **git clone skipped the quarantine caps**: byte/file caps are now
  re-measured post-clone, fail-closed.
- **Search-result tree URLs were uninstallable**:
  `https://github.com/<o>/<r>/tree/<ref>/<subdir>` now clones the parent repo
  and installs `<subdir>` as the package root; the manifest records
  `repo_url` + `subdir`.
- **Prefix prune deleted sibling routes**: `remove pdf-tools` no longer
  prunes `pdf-tools-2`'s route rows (exact-or-`os.sep`-boundary match).
- **Tar bomb**: uncompressed byte/file caps enforced pre-extraction.
- **Symlink boundary check**: resolved-path parents comparison instead of
  `startswith`.
- **Reindex failure now exits 1** with a stderr message (the install itself
  stands; the signal matters).
- HTTP read cap aligned to the 25 MB quarantine cap (was 16 MiB, truncating
  legal tarballs).
- `tarfile filter="data"` is feature-detected (`hasattr(tarfile,
  "data_filter")`) — the pyproject floor is Python 3.10, where early patch
  releases lack the kwarg.
- Nested archive *member content* is line-scanned (bounded: 1 MiB/member,
  8 MiB/archive; nested archives flagged `nested-archive-opaque` WARN).
- `remove()` no longer routes through `inference._resolve_runtime_dir`, which
  could auto-seed from Hugging Face — remove stays fully local.

- `mind_train.py`: import `losses` from the non-deprecated
  `sentence_transformers.sentence_transformer.losses` path (the old import
  warns under sentence-transformers ≥ 5.5 and breaks in a future release).
- Integration tests: the env-gated skip reason now names the actual opt-in
  (`MIND_NERVE_RUN_INTEGRATION=1`) instead of "needs mind-nerve checkout".

### Feat — `mind-nerve acquire`: vetted external skill/agent/MCP acquisition

A new `acquire` subsystem closes the "router knows the hub, but the hub only
grows by hand" gap: operators can now search curated external sources, vet
candidates deeply, and install the clean ones into the hub with a live
reindex.

- **Sources** (`acquire.sources` / `acquire search`). In-code defaults —
  `github:anthropics/skills`, `github:modelcontextprotocol/servers`, the MCP
  registry API, and a GitHub repo-search fallback (`GITHUB_TOKEN`-aware,
  graceful on the 403 rate limit) — plus a user extension file
  `<runtime_dir>/acquire_sources.json`. Per-source network failures degrade
  to empty results, never a crash.
- **Quarantine fetch** (`acquire.fetch`). Local/`file://` copies, shallow
  git clones (commit SHA recorded), and capped tarball extraction all land
  in `<runtime_dir>/quarantine/<sha256(url)[:16]>/`; 25 MB / 2000-file caps
  are enforced during extraction, and archive path/symlink escapes are
  refused before a single byte lands.
- **Vetting** (`security_scan.scan_path` / `acquire vet`). A deterministic
  static vetter — pure function of file bytes, no network, no clock, no
  execution — with PASS/WARN/FAIL verdicts and per-finding
  `file:line/rule/excerpt`. FAIL: shell-pipe installers, reverse shells,
  crypto miners, exfiltration collectors, prompt injection (incl. MCP tool
  descriptions), archive escapes. WARN: dynamic exec, credential paths,
  persistence hooks, obfuscation (installable only with
  `--accept-warnings`). Fail-closed on any scanner error; optional
  `clamscan` fold-in.
- **Install/remove** (`acquire install|list|remove`). PASS/WARN packages
  copy into the hub under a slugified unique name with a
  `.mind-nerve-install.json` manifest (per-file SHA-256, source URL, commit
  SHA — no timestamps), then reindex via `discovery.scan(trusted=False)` so
  the license gate applies, and restart `mind-nerve-routed` (new
  `ensure.restart_daemon`) so hooked CLIs route to the new skill on the
  next prompt. `remove` refuses any directory without an install manifest
  and prunes `route_table.jsonl`/`route_table.npy` row-aligned, atomically.
- **Hook**: the no-match context now points at
  `mind-nerve acquire search "<query>"` when nothing local routes.

### Feat — CLI integrations: per-prompt routing hook (reachable hub without the announce cost)

A hub of ~1,374 skills symlinked into a CLI's skills directory is bulk-announced
into every session. Measured on the STARGA hub: 1,374 indexable `SKILL.md` files,
378.5–382.9 kB of `name` + `description` frontmatter (the spread is whitespace
handling — folding continuation lines vs. keeping them), **≈95k tokens per
announce, per CLI, per session**. Loading every skill *body* instead would be
~3.3 M tokens, roughly 16x a 200k context window — so routing is the only viable
design, not merely the cheaper one. The new `integrations/` subsystem makes a hub
of that size *reachable* without it being *announced*, in three parts — all three
are needed, each alone leaves a real gap.

> Measurement note: count the description by folding continuation lines. A
> naive `^description:\s*(.+)$` truncates multi-line descriptions at the first
> newline and under-reports the announce cost by ~40% (≈56k, not ≈95k).

- **Structural** (`installer/src/skills_dir.ts`). The CLI's skills directory
  becomes a real directory holding only `mind-nerve-router`, dropping announce
  from ≈95k to ≈2k tokens. Without it the hub is announced in full.
- **Automatic** (`hook/mind-nerve-hook`). A `UserPromptSubmit` hook queries the
  routing daemon over its UNIX socket per prompt, projects the relevant skills,
  and injects a ranked route table carrying absolute `SKILL.md` paths. Without
  it the router is announced but nothing routes, and the model has to guess.
- **Hygiene** (`installer/src/hygiene.ts`, `installer/src/npy.ts`). Dead and
  renamed routes are pruned from `route_table.jsonl` **and** its row-aligned
  `route_table.npy` together. Pruning one without the other leaves the embedding
  matrix off by a row, silently mis-scoring every route after the deletion.

**Score floor is a backstop, not the primary filter.** `MIND_NERVE_MIN_SCORE`
defaults to `0.40`; the intent gate (which rejects harness-shaped and contentless
turns before embedding) does the real work. An earlier n=78 calibration argued
for `0.476`, but it built its positives from each route's own indexed text with
the name stripped — a near self-match that upper-bounds the score and
manufactures a separation gap real queries do not have. Measured against
hand-written paraphrases, three of six rank-1-**correct** hits fall below `0.476`,
including `diagnose` at `0.456`. The cost of `0.40` is recorded rather than
hidden: two of five observed noise cases (`yeet` 0.45, `check-work` 0.44) survive
this gate and are caught downstream. Both populations are pinned in
`test/hook_gates.test.ts` so any recalibration starts from evidence.

### Fix — route identity: `.system` name-shadowing, and `source_path` on every surface

Four `.system/` hub entries shadow real skills by name — `plugin-creator`,
`openai-docs`, `skill-creator` and `imagegen` each exist as both
`.system/<name>/SKILL.md` and `<name>/SKILL.md`. Projection symlinks are keyed on
name, so whichever row the router happened to return silently overwrote the
other. `.system` is now excluded at **ingest** rather than at projection, so the
ambiguity cannot reach any consumer.

`Route` gains an optional `source_path`, so route identity can be compared on
`(name, source_path)` instead of name alone and a future collision is visible
rather than silent. A shared `daemon_client` resolves routes through one path for
the socket, MCP and CLI surfaces, with a parity test asserting they cannot drift.

### Docs — catalog v1.0 is tamper-evident, not signed

The dataset card and model card both described `route_table.jsonl` v1.0 as "frozen
and signed". The `manifest.sig` HMAC is a placeholder pending an operator
signature with the real root key, so the accurate claim is **tamper-evident**:
the freeze is reproducible and hash-anchored (`SHA256SUMS` + manifest), but not
signed. Also corrects the documented `MIND_NERVE_MIN_SCORE` default (README said
`0.35` while the hook shipped `0.40`) and the ROADMAP hub count (~1300 → ~1,374).

### Feat — cross-platform: Windows + macOS + Linux universal router (v0.3.0b9)

mind-nerve now runs as the universal skill-router on Windows, macOS and Linux,
not Linux-only. The native Q16.16 encoder remains a Linux-only optimisation; on
every other platform (and on any box without the `.so`) the router degrades to
the pure-Python backend automatically.

- **Robust pure-Python fallback.** `load_default_runtime()` (the actual loader)
  now catches a missing/unloadable native encoder and falls back to the
  pure-Python backend with a one-line stderr notice, instead of crashing on the
  default `native` backend. Previously the fallback logic lived only in an
  unused helper, so a missing `.so` raised `FileNotFoundError`. Caught cases:
  `FileNotFoundError` (no `.so`), `ImportError` (ctypes/binding gap), and
  `OSError` (present-but-unloadable `.so`, e.g. the Linux library on Windows/
  macOS or the wrong CPU arch). `route()` now dispatches on the resolved runtime
  *instance type*, so a fallback is ranked through the correct path.
  `_NativeEncoderRuntime.__del__` is hardened against a half-built object.
- **Cross-platform daemon imports.** `mind_nerve.ensure` no longer hard-imports
  `fcntl` at module load (absent on Windows); the flock spawn-guard is skipped
  where `fcntl` is unavailable. `daemon`, `ensure` and `preselect_hook` guard
  `socket.AF_UNIX` so they exit cleanly (or pass through) on platforms that lack
  UNIX-domain sockets. `ensure` detaches the daemon with the right flags per OS
  (`start_new_session` on POSIX, `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`
  on Windows). The one-shot `mind-nerve route` CLI is fully OS-agnostic and
  needs no daemon.
- **macOS launchd template.** Added `templates/ga.star.mind-nerve-routed.plist`
  alongside the existing systemd unit; both ship in the wheel as package data.
- **Universal wheel.** The wheel is `py3-none-any`. The native `.so` is packaged
  as OPTIONAL data; absent or unloadable on a given platform it simply triggers
  the fallback. Install on any OS: `pip install mind-nerve`.

Determinism note: the **native** Q16.16 path is the strict-determinism one
(integer-only, no float; the native top-K breaks ties by ascending SHA-256(route_id)).
It reproduces a pinned bit-identity reference on x86; cross-substrate (ARM/GPU)
bit-identity is the Phase 2 gate and is not yet hardware-validated. The
pure-Python fallback uses float32 and the SHA-256 deterministic top-K, so it is
run-to-run deterministic on a given machine but its float scores are not
guaranteed bit-identical to the native path or across differing CPU/BLAS arch.
Use the native backend where strict determinism is required.

### Fix — MCP server: non-blocking model warmup (instant handshake for strict MCP clients)

`mind-nerve-mcp` warmed the embedding model inline before entering the stdin
loop, so the JSON-RPC `initialize` handshake stayed unanswered for the duration
of the cold model load (several seconds). MCP clients that enforce a short
startup deadline marked the server as "failed" even though it would have come up
moments later. Warmup now runs in a background thread: `initialize` /
`tools/list` respond immediately while the model still starts loading right
away, and a `tools/call` arriving before warmup finishes blocks until the model
is ready (thread-safe, one-time load). No API or behaviour change to the
`mind_nerve_route` tool — it just works out of the box across MCP clients.

### Feat — pure-MIND front-end port complete (13/13 dormant files on mindc 0.10.2)

The entire dormant front-end tree (`src/*.mind`, ~6.6k lines that had never
compiled under any shipping mindc) is ported to the real 0.10.2 dialect and
EXECUTES: sha256 16/16, q16_16 86/86, lib 12/12, top_k 30/30, tokenizer
18/18, chain_log 5/5, runtime_ffi 3/3, clock 5/5, evidence 36/36,
encoder_kernels 17/17, model 8/8, loader 12/12, inference 9/9 — 267 tests
wired as exact-count legs 3–15 of `tests/mindc_gate.sh`, plus a native-ELF
end-to-end harness (stateful oracle + chain link, full-path bytes oracle vs
CPython hashlib). The wave-3 host ABI was redesigned for the shipped extern
surface: pointer+length out-params, caller-owned buffers, explicit
host-vs-counter clock, caller-supplied entropy seeds — no wall-clock, no
thread-locals, no cfg. Numerics delegate to the pinned `mind/luts` tables
("delegate, never duplicate"). Bit-oracles pinned against CPython hashlib /
a reference Python BPE throughout. Three compiler findings with minimal
repros were filed upstream (lazy assert evaluation in `mindc test`,
misleading unresolved-ref diagnostics, missing interpreter bounds checks).
Honest flags preserved in file headers: loader's INT8→Q16.16 dequantize
keeps the reference formula verbatim (suspected 2^16-fold scale error —
decision belongs to the checkpoint pipeline); dormant
`preselect_pre_tokenized` intentionally lacks the model-hash pin.

### Fix — pre-release cross-CLI audit, round 3 (grok + claude-fable + codex, 2026-08)

Three independent CLI auditors reviewed the full changeset; all returned
BLOCK. 24 distinct findings fixed across two passes, each with a
regression test proven red first:

- **Directory-symlink exfiltration (CRITICAL, reproduced by all three)**:
  a package `docs -> ~/.ssh` vetted PASS (dir symlinks never reached the
  file-only symlink checks) and `copytree(symlinks=False)` then
  dereferenced it into the hub. Directory symlinks now get the same
  escape/skip-dir/dangling rules as files, install copies `symlinks=True`
  so vetted links stay links, dangling links FAIL at vet, and a failed
  copy rmtrees the partial hub dir.
- **Skipped-dir payload (CRITICAL)**: `dist/guide.md` carrying
  `curl … | bash` installed unscanned via the tarball/git-clone fetch
  paths. All fetch paths now filter `SKIP_DIRS` to match the scanner
  contract (plus defense-in-depth `ignore=` at install).
- **NUL-byte scan bypasses (CRITICAL ×2)**: NUL-prefixed archive members
  skipped line rules, and the NUL probe only covered the first 8 KiB
  (`cu\0rl` mid-buffer passed even though bash strips NULs). NUL handling
  now spans the whole scanned buffer, top-level and archive members alike.
- **Windows drive-qualified tar members (CRITICAL)**: `C:\…` member/link
  forms rejected version-independently of `data_filter`.
- **`--register-mcp` executes unvetted artifacts (CRITICAL)**: `npx -y
  <ident>` from `server.json` allowed leading-dash option injection and
  ran never-scanned registry artifacts. Registration now only accepts
  local, path-checked entry points inside the vetted package.
- Shell-pipe FAIL rule widened (`/bin/sh`, `python[0-9.]*`, `ruby`,
  `perl`, `node`, `sudo -E`, `eval $(…)`, process substitution);
  prompt-injection detection gained a whole-file whitespace-normalized
  pass (3+-line splits); archive dispatch by magic bytes with an
  `unscanned-binary` WARN for opaque executables; streaming member caps +
  mid-flight clone size monitor (DoS); local/`file://` fetch restricted
  to operator-typed targets (registry candidates are https-only); clone
  URLs redacted (userinfo/query/fragment) in manifests, CLI output, and
  error text; MCP config replacement is mode-preserving + fsynced;
  reinstall prunes stale route rows transactionally; curated GitHub
  agent hits emit per-artifact `kind="agent"` candidates that survive
  fetch with their relative path; the hook honours daemon-provided
  `kind`/`source_path` when root-confined (acquired agents no longer
  drop as unknown).
- Installer: documented flag-first order (`install --mcp-launcher uvx
  <client>`) parsed correctly; `--mcp` (MCP-only mode) no longer
  suppresses the MCP write itself (inverted guard); Vibe entries emit
  the required `transport = "stdio"` with a `mcp-schema` verify check;
  Vibe/Grok idempotency compares the complete managed entry; Cody/Aider
  instruction blocks are format-valid JSON/YAML and verify parses them;
  `verify` gained the `mcp-env-pin` check (WARN when a populated local
  runtime dir exists but the entry lacks `MIND_NERVE_RUNTIME_DIR`) and
  install pins only populated runtime dirs.
- CI: the `mindc-gate` job builds mindc with `mlir-build` (without it the
  native-symbol leg necessarily fails on embedded-fallback objects) and
  installs `mlir-20-tools`/`clang-20`; gate install hints and docs match.

### Feat — installer: `verify` verb + `--mcp-launcher venv|uvx` + corrected MCP entry

`mind-nerve-install verify [--cli X]` checks every wired surface per client
(config parses, hook block present, hook script executable, env pins, hook
fail-open probe, MCP entry/command, daemon socket) and FAILs legacy
`mind-nerve mcp-facade` entries (the Python package never had that
subcommand — those entries were dead). MCP entries now point at the
`mind-nerve-mcp` console script (venv launcher, default) or
`uvx --from mind-nerve mind-nerve-mcp` (`--mcp-launcher uvx`), with
`TRANSFORMERS_NO_TORCHVISION=1` always pinned and `MIND_NERVE_RUNTIME_DIR`
pinned when a populated local runtime dir exists (fleet incident
2026-07-10). Kimi wiring uses the documented shapes: flat `[[hooks]]` in
`config.toml` + `~/.kimi-code/mcp.json` (`mcpServers`). Registry covers 20
clients.

## [0.3.0b8] — 2026-05-20 — hotfix: revert #233(a), yanks v0.3.0b7

### Fix — yank v0.3.0b7, restore Track A C-shim encoder matmul

The v0.3.0b7 "thesis-pure encode" path (#233 a) shipped a silent
correctness regression. Root cause: `__mind_blas_dot_q16_v` expects
i32-stride-4 dense buffers (`elem_bytes = sizeof::<i32>() = 4`,
`vector<8xi32>` load + `extsi to vector<8xi64>` per the mindc
lowering), but mind-nerve's encoder weight blob is i64-stride-8.
Passing i64-stride pointers to dot_q16_v made every iteration read 4
real values + 4 sign-extension halves; the sign-ext lanes' products
shift to 0 after `>>16`, so each dot accumulated only the first K/2
input dims of every linear matmul, then the encoder's terminal
L2-normalize hid the magnitude loss. The A1.5 cosine ≥ 0.92 /
top-5 ≥ 0.92 thresholds and the L2-normalize-then-cosine harness
were both too loose to catch the residual direction error.

Verified directly via ctypes: dot_q16_v(n=8) returns 13369344 (full
1²+2²+…+8² Q16.16) on an i32 buffer but only 1966080 (first half
1²+2²+3²+4²) on an i64 buffer.

**Action**: revert the `#233(a)` merge (`f1f7f0b`) on `main`
(`0333666`). matmul_384_384/384_1536/1536_384 are back on Track A
`matmul_q16_blas` → C-shim `__mind_nerve_blas_matmul_q16_i64` (the
correct path that matches the i64-stride blob). Quantizer reverts
to the `(in, out) = (K, N)` layout that Track A expects. Encoder
weight blob on HF `star-ga/mind-nerve` is deleted (was only
consumed by v0.3.0b7); users continue to build their blob locally
via `tools/quantize_encoder_to_q16.py` against the HF checkpoint as
they did pre-v0.3.0b7.

**Correct-path future work**: either (a) add a new mindc intrinsic
`__mind_blas_dot_q16_v_i64` whose lowering reads i64-stride-8
(small, targeted compiler change in `mindc/src/mlir/lowering.rs`)
so thesis-pure works with the existing blob, **OR** (b) full
dense-int32 storage path (offline blob change to 4-byte stride +
keeps using existing dot_q16_v as-is). Either is multi-session
deliberate work and **must include a strict bit-identity gate**
(not cosine-after-normalize) before any future re-release. Tracked
on task #233.

`mind-nerve@0333666`. mindc / matmul_rmajor_q16_v in `std/blas.mind`
(mind `641e6cb`) is unaffected — the primitive itself is correct
for its i32-stride ABI; the bug was in the mind-nerve consumer.

## [0.3.0b7] — 2026-05-20 — thesis-pure encode

### Feat — thesis-pure encode matmul via dot_q16_v intrinsic (#233 a)

- `mind/kernels/matmul_q16.mind`: the per-token linear matmul now
  composes the mindc-emitted `__mind_blas_dot_q16_v` MLIR vector
  dialect intrinsic in pure MIND, with no Track-A C-shim involvement.
  Loop order is outer-row, inner-token so each `W[r,:]` (3-12 KB)
  stays hot in L1 across all T inner iterations (closing the inc5
  weight-cache equivalent gap that the initial per-token order had).
- `tools/quantize_encoder_to_q16.py`: encoder weight blob now ships
  in `(out, in) = (N, K)` row-major (PyTorch nn.Linear native
  layout); the historical `.T` is gone. Re-quantize locally from the
  Hugging Face checkpoint to produce the new blob.
- `python/mind_nerve/inference.py`: fail-fast `_verify_encoder_weights_sha256`
  check on runtime construction. Old cached blob + new cdylib (or
  vice versa) raises immediately with a clear remediation hint,
  preventing the silent wrong-embedding mode an unchecked mismatch
  would produce.

Byte-identity (cosine 0.999996, top-5 0.9975 vs pytorch
SentenceTransformer) preserved. Perf A/B vs v0.3.0b6 / 45eabdd
baseline (50 iters, native): p50 -2.2 / -5.7 / +3.5 pct,
p95 +1.2 / -5.8 / +1.2 pct at approx T 64 / 128 / 256. Pure-MIND
BEATS the Track-A C-shim + inc4 + inc5 path at T=128; at-parity
within run-to-run noise at T=64 / T=256. Thesis-pure goal met: zero
C-shim dependency in the encoder linear matmul path. vs MIND's own
prior path.

The increments below (#228, #236 inc 1-5, the criterion bench harness, the
A1.5 mind-blas score-path SIMD rewire, the A1.5 pure-MIND LUT port #218,
and the Phase 6.2 offline quantizer) are the preparatory work this release
built on top of; re-homed here (was a stray second `[Unreleased]` header)
since they shipped as part of reaching 0.3.0b7, not as a separate release.

### Fix — native tokenize truncates to model max_seq_length (#228)

- `python/mind_nerve/inference.py` `_tokenize` used `max_length=512`
  while the reference SentenceTransformer truncates to
  `sentence_bert_config.json` `max_seq_length=256` then CLS-pools. Any
  input >256 tokens therefore reached the native encoder's
  sliding-window ("later-window-wins") path and **silently produced a
  different embedding than pytorch** — the A1.5 gate never caught it (its
  harness tokenizes at 256). Now `max_length=256`: native route()/encode
  is pytorch-SentenceTransformer-identical for all inputs; the
  sliding-window kernel stays for explicit long-document use, never
  silently on the contract path. New skip-guarded regression
  `tests/python/test_tokenize_maxseq.py`. Python-only; no encoder
  rebuild / no quantizer/blob/A1.5 change. `mind@d87c4c1`.

### Perf — process-lifetime transposed-weight-panel cache (#236 inc 5)

- `mind/runtime/blas_shims_i64.c`: profiling showed the linear GEMMs are
  ~59% of encode wall and **memory-bandwidth-bound** (the dense path
  re-ran `pack_i32_transpose` on the SAME immutable weight matrix every
  encode call — ~42 MB/encode of repack DRAM traffic). The encoder
  weights load once into a stable blob at fixed offsets, so the B
  pointer to `matmul_q16` is identical across every call. Added a
  process-lifetime cache of transposed int32 weight panels keyed by
  (B addr, K, N) + a 3-word content probe (the probe rules out a
  freed-then-reused-address collision without re-reading the panel —
  preserves byte-identity, does not re-incur the eliminated traffic).
  Cache miss/full/`MIND_NERVE_BLAS_WCACHE=0` falls back to the
  per-call malloc+pack. Same data, memoised → **byte-identical**.
  A/B vs prior commit (50 iters, native): p50 **−12.6% / −7.1% /
  −6.7%**, p95 **−9.5% / −7.3% / −6.9%** at ~T 64 / 128 / 256,
  monotone, no sign-flips. A1.5 cosine 0.999996 / top-5 0.9975
  byte-for-byte unchanged; blas/LUT 6/6. (A regime-incorrect MR=6×NR=2
  register-widening attempt was measured, regressed, and discarded —
  the linear GEMM is memory-bound, not compute-bound.) vs MIND's own
  prior path.

### Perf — attention GEMMs densified (qkt · attn·V) (#236 inc 4)

- `mind/runtime/blas_shims_i64.c`: `__mind_nerve_blas_qkt_q16_i64` and
  `__mind_nerve_blas_attnv_q16_i64` now take the dense-int32 path
  instead of the i64-stride-8 `dot_q32_accum` / k-outer-accrow
  reference. qkt packs each head's Q/K to dense int32 and dots 8-wide
  via a raw Q32.32 `gemm_dot_i32_accum` (no narrow; qkt applies its own
  `>>16` then `*scale>>16`). attn·V is exactly the `matmul_q16` shape
  (M=T, K=T, N=D) — attn row-major, V transposed — so it reuses the
  proven MR=4×NR=2 / mr4 / dot microkernels verbatim. Every output is
  the same i32 products + ascending-k associative i64 sum + identical
  narrow as the reference → **byte-identical**; the original i64-stride
  paths are retained as the malloc-fail fallback. Controlled
  before/after A/B (50 iters, native): p50 **−4.9% / −6.3% / −7.4%**,
  p95 **−3.8% / −7.8% / −12.7%** at ~T 64 / 128 / 256 (gain grows with
  T — attention is O(T²D)), monotone, no sign-flips. A1.5 cosine
  0.999996 / top-5 0.9975 byte-for-byte unchanged across both A/B
  builds (re-verified, fresh rebuild); blas/LUT/tokenize 8/8.
  Profiling note: attention is ~10% of encode wall (linear GEMMs
  ~59%), so the end-to-end effect is bounded; the next levers target
  the linear GEMM. vs MIND's own prior path.

### Perf — MR=4 × NR=2 register tile on the dense GEMM (#236 inc 3)

- `mind/runtime/blas_shims_i64.c`: each A-row vector load is now reused
  across 2 Bt rows AND each Bt across 4 A rows (8 i64 accumulators, safe
  AVX2 register budget). Each output is the same i64 K-dot as inc 1/2
  (even/odd `_mm256_mul_epi32`, associative i64 sum, single final
  `>>16`) → byte-identical; N%2 tail uses the proven mr4 microkernel.
  Native encode p95 vs main: **T=64 441→225 ms, T=128 898→405 ms,
  T=256 1928→854 ms (~1.96–2.26× cumulative, ~1.2× over inc 2)**,
  monotone, no regression. A1.5 cosine 0.999996 / top-5 0.9975
  byte-for-byte unchanged (re-verified, fresh rebuild); LUT 3/3; no
  quantizer/blob change. vs MIND's own prior path. `mind@e520a9b`.

### Perf — MR=4 register microkernel on the dense GEMM (#236 inc 2)

- `mind/runtime/blas_shims_i64.c`: 4 contiguous A rows are now computed
  against one transposed-B row together, reusing each `Bt[k]` vector
  load across all 4 rows (4× fewer B streams — the dominant Goto/BLIS
  reuse). Each output is the same i64 K-dot as inc 1 (even/odd
  `_mm256_mul_epi32`, associative i64 sum, single final `>>16`) →
  byte-identical; the M%4 tail and scalar path keep the proven single
  dot. Native encode p95 vs main: **T=64 441→266 ms, T=128 898→497 ms,
  T=256 1928→1076 ms (~1.66–1.81× cumulative, ~1.25–1.3× over inc 1)**,
  monotone, no regression. A1.5 cosine 0.999996 / top-5 0.9975
  byte-for-byte unchanged (re-verified, fresh rebuild); LUT 3/3; no
  quantizer/blob change. vs MIND's own prior path. `mind@be4c53d`.
  Increment 3 (full 6×8 register tile + B-panel L1 packing + qkt/attnv
  same treatment) is the remaining FBGEMM-class gap.

### Perf — dense-int32 transposed-B Q16.16 GEMM (#236 inc 1)

- `mind/runtime/blas_shims_i64.c`: the linear-GEMM path now repacks A
  (M×K) to dense `int32` and B (K×N) to dense **transposed** `int32`
  (N×K) once per call, so every `C[i,j]` is a contiguous length-K dot
  using the proven 8-wide even/odd `_mm256_mul_epi32` technique (exact
  full-range i32×i32→i64; single final `>>16`). Kills the i64-stride-8
  2× memory / half-SIMD penalty and the strided `B[:,j]` access. The
  i64-stride reference is retained as the byte-identity oracle +
  malloc-fail fallback.
- Native encode p95: **T=64 441→338 ms, T=128 898→680 ms,
  T=256 1928→1295 ms (~1.30–1.49×)**, monotone, no regression at any
  shape. Correctness byte-for-byte unchanged (A1.5 cosine 0.999996 /
  top-5 0.9975, independently re-verified on a fresh encoder rebuild;
  LUT bit-identity 3/3; no quantizer/blob change). Speedup is vs MIND's
  own prior path. `mind@30c3fd9`. Increment 2 (BLIS panel packing + 6×8
  register microkernel; qkt/attnv same treatment) is the follow-on.

### Perf — attention GEMMs vectorised through byte-identical SIMD Q16.16 (#230 Step 2)

- `mind/runtime/blas_shims_i64.c` (new `__mind_nerve_blas_qkt_q16_i64`,
  `__mind_nerve_blas_attnv_q16_i64`, `dot_q32_accum_{scalar,avx2}`):
  byte-identical AVX2 Q16.16 kernels for the per-head attention
  contractions. Critical correctness detail preserved: `qkt` accumulates
  Q32.32 with **no intermediate `>> 16`** (unlike the linear-GEMM
  `dot_q16` which shifts per product) — a dedicated `dot_q32_accum`
  helper keeps that distinction. `qkt_matmul`/`attnv_matmul` re-pointed
  to the SIMD path (`qkt_blas`/`attnv_blas` in `matmul_blas.mind`);
  scalar `qkt_dot_k`/`attnv_dot_k` retained as the byte-identity oracle.
  `mind@14f1444`.
- Native encode latency (i7-5930K, on top of Step 1): **T=64 551→441 ms
  p95 (1.28×), T=128 1377→898 ms (1.55×), T=256 3718→1928 ms (1.96×)** —
  speedup grows with T as the O(T²·D) attention block becomes dominant
  and the per-element MIND recursion-frame overhead is eliminated.
  Correctness gate **byte-for-byte unchanged** (cosine 0.999996 /
  top-5 0.9975), independently re-verified on the mindc-v0.6.7 encoder
  rebuild; LUT bit-identity 3/3 hashes untouched; 415 pytest pass; no
  quantizer/blob/layout change. `add_bias_row` measured a non-bottleneck
  (sub-10 ms at T=256) and left scalar. Combined with Step 1 the encode
  GEMM path is fully byte-identical SIMD; the remaining thesis-purity
  item (replace the Track-A C shim with Track-B `dot_q16_v`, needs an
  (N,K) weight layout) is tracked separately — it is an architectural
  purity follow-on, not a latency need.

### Perf — native encode GEMMs routed through byte-identical SIMD Q16.16 matmul (#230 Step 1)

- `mind/runtime/blas_shims_i64.c` (new symbol `__mind_nerve_blas_matmul_q16_i64`):
  a k-outer/j-inner AVX2 Q16.16 GEMM (`_mm256_mul_epi32` widening 32×32→64
  on the low-32 of each i64 stride-8 slot, Q32.32 accumulate, single
  `>> 16` narrow at write) + scalar fallback. **Byte-identical** to the
  scalar `matmul_q16` oracle by construction (integer-domain reduction is
  associative; same guarantee as the task-#57 score-path AVX2 dot) — and
  empirically: the full 12-layer encode embedding cosine vs pytorch is
  **unchanged at 0.999996 / top-5 0.9975** (a lane bug would have
  collapsed it across the stacked GEMMs, as the prior precision
  regression did). `matmul_blas.mind` exposes it as `matmul_q16_blas`;
  `matmul_384_384` / `matmul_384_1536` / `matmul_1536_384` now route
  through it instead of the tail-recursive scalar `matmul_q16` (scalar
  primitives retained as the byte-identity oracle).
- Native encode latency (i7-5930K, same blob): **T=10 1047→77 ms p95
  (13.9×), T=64 7077→553 ms (11.9×), T=128 19632→1365 ms (10.6×)**.
  Correctness gate unchanged (independently re-verified on the
  mindc-v0.6.7 rebuild); LUT bit-identity 3/3, hashes untouched; no
  quantizer/blob/layout change. Remaining cost is now bias-add /
  layernorm / attention (qkt/attnv still scalar) — that is #230 Step 2
  (incl. the thesis-pure Track-B `dot_q16_v` path, which needs an (N,K)
  weight layout). `mind@4d25383`.

### Bench — criterion speed + efficiency harness

- `tests/perf/bench_criterion.py` (new): score-only speed bench over the
  seeded synthetic 11,922 × 384 Q16.16 catalog (1000 queries, 64 distinct,
  single thread, warm), matrixing MIND + mind-blas-A (AVX2), MIND + scalar
  oracle, numpy + BLAS reference, and pytorch CPU. Emits
  `bench_criterion.json` + a human table. Measured on i7-5930K:
  mind-blas-A `p50 = 1.42 ms · p95 = 1.61 ms · p99 = 1.73 ms` (~696 QPS),
  scalar oracle `p50 = 1.69 ms · p95 = 1.94 ms`, numpy+BLAS `p50 = 0.24 ms`,
  pytorch CPU `p50 = 1.03 ms · p95 = 1.24 ms`; peak RSS ≈460–475 MiB.
  mind-blas-A is ≈1/6 the idealised numpy+BLAS p50 path and 9.3× faster
  than the pre-A1.5 scalar baseline, while preserving cross-arch Q16.16
  bit-identity (a property BLAS does not offer). Encode-only + end-to-end
  are explicitly `PENDING` (blocked on the Phase 6.2 full-catalog run with
  the real Phase 1 checkpoint). Pytest entry point hard-fails iff
  mind-blas-A p95 > 2.0 ms (regression detector).
- `tests/perf/bench_efficiency.py` (new): the substrate bench. (1) Cross-arch
  Q16.16 bit-identity (task #57) — SHA-256 of the top-5 `(idx, score)`
  stream over the 100-query corpus on BOTH dispatch paths; AVX2 == scalar
  == the pinned x86 reference
  `f4524bd56fd74e9dfbfb17b5b1f56fafda0e7e99321ef75ebce777219cda45fc`, the
  cross-arch oracle for future ARM / CUDA / photonic backends. (2) L1/L2/L∞
  metric matrix on the same catalog — L1-vs-L2 top-5 rank-overlap 37.4%
  (Jaccard 0.24), L∞-vs-L2 2.4% (Jaccard 0.01): L1/L∞ are distinct metrics,
  not cosine approximations, on synthetic data. (3) Joules/query via Intel
  RAPL — `null` (`rapl_unreadable`, root-only sysfs on this host); GPU path
  `PENDING` (no GPU score path). Emits `bench_efficiency.json` + a table.
- `tests/perf/_bench_common.py` (new): single source of truth for the seeded
  synthetic catalog/query builders, percentile helper, native-runtime
  resolver, and BLAS dispatch binding shared by the two benches.
- `docs/benchmarks.md` (new): publication writeup with honest framing —
  mind-blas-A reaches a fraction of idealised BLAS while preserving
  cross-arch Q16.16 bit-identity, memory-bandwidth-limited regime stated
  explicitly, encode path marked PENDING.
- Both benches are runnable standalone (`python tests/perf/bench_*.py`) and
  under pytest (gated, self-skip under `MIND_NERVE_PERF_SKIP=1`). Bench JSON
  artefacts are git-ignored (machine-specific timings).

### A1.5 — score-path rewire to mind-blas SIMD

- `mind/kernels/matmul_blas.mind` (new) + `mind/runtime/blas_shims_i64.c`
  (new): the `mn_encoder_score` path now routes through a SIMD-backed
  i64-layout Q16.16 dot/matmul shim instead of the tail-recursive scalar
  reduction in `mind/kernels/matmul_q16.mind`. The shim exposes scalar +
  AVX2 paths with a `.so`-load-time dispatcher honouring
  `MIND_NERVE_BLAS` (`1`/`avx2` = auto-detect AVX2, `0`/`scalar` = force
  the byte-identical oracle). `mind/exports/c_abi.mind` calls
  `matmul_score_blas`; `tools/build_encoder_cdylib.py` compiles the shim
  with `-mavx2 -mfma` and links it into the encoder cdylib.
- Q16.16 SIMD reduction with explicit per-lane i64 widening is
  associative, so the AVX2 path is **byte-identical to scalar** — the
  cross-arch determinism gate (task #57) is preserved. A reference
  SHA-256 of the top-5 `(idx, score)` stream over 100 deterministic
  queries is pinned in `tests/python/test_blas_byte_identity.py` for
  future ARM / CUDA / photonic comparison.
- Measured on i7-5930K, synthetic 11,922-route × 384-dim catalog,
  1000 queries, single thread, score-only:
  `p50 = 1.44 ms · p95 = 1.60 ms · p99 = 1.73 ms` — **9.3× faster**
  than the 15 ms pre-A1.5 scalar baseline. Memory-bandwidth-limited
  (the i64 stride-8 catalog is ~36 MB and saturates single-channel
  DDR4); a future i32 stride-4 repack would approach the ~0.4 ms
  compute floor.
- New gates: `tests/python/test_blas_byte_identity.py` (scalar vs AVX2
  byte-identity + cross-arch reference hash) and
  `tests/perf/test_score_latency.py` (p95 < 2 ms hard gate,
  `MIND_NERVE_PERF_SKIP=1` opt-out).
- A1.5 verdict: **score path PASS**. Encode path remains tracked
  separately (depends on the Phase 6.2 quantizer artifact, below).

### A1.5 — pure-MIND tanh/rsqrt/softmax Q16.16 LUTs replace C shim (#218)

- The encode path's activation LUTs are now **pure MIND**, not libm.
  `mind/runtime/lut_shims.c` (libm `tanh`/`exp`/`sqrt` float math, NOT
  cross-arch bit-identical, "MEASUREMENT ONLY") is **deleted** and
  superseded by `mind/runtime/lut_cache.c` — a thin handle cache that
  performs **zero arithmetic**: it lazily calls each pure-MIND `*_init()`
  once and caches the i64 table handle (matching the prior shim's
  implicit table-resident caching contract, so kernel call sites are
  unchanged). Every numeric value — table entries and lookups — is now
  produced by the deterministic integer Q16.16 sources in `mind/luts/`.
- New single-arg pure-MIND wrappers, called by bare name (the
  MIND↔C link convention used by `__mind_alloc` and the score-path
  shim — no `extern fn` needed):
  - `tanh_q16(x)` in `mind/luts/tanh_q16.mind` — 4096-entry table lookup.
  - `softmax_q16(buf, n_rows, row_len)` in `mind/luts/softmax_q16.mind` —
    row-loops the existing 5-stage pinned `softmax_q16_run` (exp + recip
    LUTs); the pinned integer-D normalisation is unchanged (it is a
    cross-arch-deterministic design, not an accuracy target).
  - `rsqrt_q16(x)` in `mind/luts/rsqrt_q16.mind` (new) — composes the
    `sqrt_q16.mind` 2048-entry rsqrt table seed with **one Newton step**
    (`y1 = y0·(3 − x·y0²)/2`, pure integer Q16.16). Non-positive input
    returns the `Q16_MAX` sentinel. Max error over the representable
    domain `x∈[0.125, 256.0]`: **≤ 1.12e-4 abs / ≤ 1.74e-3 rel**.
  - The wrappers are emitted by `tools/gen_luts.py` (the LUT source of
    truth) so a regenerate keeps them; `rsqrt_q16.mind` is the only
    hand-written LUT file.
- Latent compiler-interaction bug surfaced + closed: `tools/gen_luts.py`
  emitted bare negative integer literals (`let domain_lo: i64 = -524288;`
  and `__mind_store_i64(buf+0, -65536)`) which the current mindc
  front-end lowers to the constant `0` — this silently zeroed the tanh
  table interior and the LUT domain offsets. It was masked by the C shim
  (the pure MIND lookup path was never exercised end-to-end). The
  generator now emits the subtraction form `(0 - N)` for all negative
  constants and table entries (lowers to a correct `arith.subi`,
  bit-identical, diff-stable for non-negatives); all 5 LUT sources
  regenerated. Post-fix: `tanh_q16` max abs error ≤ 3.74e-3 vs libm over
  `|x|≤8` (≈2e-5 in the GELU interior).
- New gate `tests/python/test_lut_bit_identity.py`: for tanh / rsqrt /
  softmax — determinism across calls, idempotent handle-cache (a freshly
  built table reproduces lookups bit-for-bit), accuracy bounds, and a
  pinned cross-arch SHA-256 reference per wrapper (task #57, same pattern
  as `test_blas_byte_identity.py`). x86_64 references:
  - tanh   `190e488bd5a0f67fcc7a2ca60df688d98b53fe1643b9cbe485e8859740bf4bb8`
  - rsqrt  `c7e2791a73ad234187c00f0d2a918c86826ea509346c37b448ade18379b06a2d`
  - softmax `e39ad4ec913ae5b0a77add0c0d1ec1526f00f4c729fc3c8fbc4b04525a2621ae`
  (the integer-only path cannot drift across substrate / compiler / SIMD).
- `tools/build_encoder_cdylib.py`: drops `lut_shims.c`, compiles
  `lut_cache.c`, adds `mind/luts/rsqrt_q16.mind` to the source list. The
  rebuilt cdylib resolves `tanh_q16` / `rsqrt_q16` / `softmax_q16` as
  pure-MIND `T` symbols; libm `tanh`/`exp`/`sqrt` are no longer
  referenced by the LUT path. Full suite green (356 passed); the
  score-path byte-identity + p95<2ms gates are unchanged.
- A1.5 verdict: encode-path LUT bit-identity prerequisite **CLOSED**.
  The remaining encode-path blocker is unchanged (the end-to-end
  `mn_encoder_encode` measurement still depends on the Phase 6.2
  quantizer artifact, below).

### Phase 6.2 — offline Q16.16 quantizer

- `tools/quantize_phase1_to_q16.py` (new): offline FP32 → Q16.16
  quantizer. Reads a precomputed catalog `route_table.npy` (float32,
  shape `(N_rows, hidden_dim)`) and emits the runtime artifact
  `route_table.q16.bin` (row-major Q16.16, `int64` LE per element to
  match the MIND heap ABI's i64-only loads in
  `mind/kernels/encode.mind`) plus `route_table.q16.meta.json` with
  the spec-mandated reproducibility metadata (quantizer version,
  catalog SHA-256, optional checkpoint SHA-256, blob SHA-256,
  saturation count).
- `spec/quantization.md` (new): normative spec. Scale = `2^16`,
  rounding = round-half-to-even, saturation = clamp to `[INT32_MIN,
  INT32_MAX]`, on-disk encoding = `int64` LE. Determinism is
  achieved by `float64` intermediate × explicit `numpy.round` ×
  `int64` cast — no platform-specific FP ordering.
- `tests/python/test_quantize_phase1.py` (new): round-trip
  (`< 2 * 2^-16` ≈ 3.05e-5 max abs error over 1000 random
  floats), bit-identity gate (same input → byte-identical `.bin`
  across two runs), saturation, dtype, meta key order, CLI smoke,
  cross-check against `mind_nerve._native._f32_to_q16`.
- `python/mind_nerve/cli.py`: new `mind-nerve quantize` subcommand.
  `mind-nerve quantize --catalog <route_table.npy> --output <dir>`
  defaults `--output` to `$MIND_NERVE_RUNTIME_DIR` or
  `~/.cache/mind-nerve/q16/`.

The full-catalog (~4400 routes × 384 dim ≈ 13 MB) blob is produced
on demand by the user. The `.bin` is never committed to git
(matches the existing `*.bin` gitignore rule). Full-catalog
quantization is gated on the Phase 1 PyTorch checkpoint being
present locally; the quantizer + tests run against a synthetic
NumPy fixture in the absence of a real checkpoint.

This unblocks the A1.5 `mn_encoder_encode` end-to-end measurement
path: the native encoder kernel now has a real Q16.16 weight blob
to consume.

## [0.3.0-beta.6] — 2026-05-19

Audit-response wave responding to the second external deep-research
audit (10 scopes, baseline 4.8/10 → target 8+/10). Four parallel
work-streams landed independently with disjoint file-sets and no merge
conflicts.

### Repository structure & documentation (Stream A — `6ed923c`)

- Added `CONTRIBUTING.md`, `SECURITY.md`, `CODEOWNERS`,
  `.github/ISSUE_TEMPLATE/{bug_report,feature_request,config}.yml`,
  `.github/pull_request_template.md`.
- New `docs/dataset.md`, `docs/privacy.md`, `docs/model_card.md`,
  `docs/data_governance.md` covering the public dataset contract,
  privacy posture, model card with headline metrics, and
  responsible-use governance.
- README: reconciled architecture description to the frozen
  drop-the-decoder + sliding-window encoder design; separated Phase-1
  shipped (PyTorch SentenceTransformer, ~90 ms 4-core CPU p95) from
  Phase-2 native (mind-nerve A1.5 partial measurement: 14.4 ms p50 /
  15.1 ms p95 score path); honest perf framing in Highlights and
  comparison table; dual-license callout pulled into Highlights;
  Governance section linking new docs.
- `pyproject.toml` long-description now states the Apache-2.0 +
  separately-licensed-runtime split.
- `python/mind_nerve/cli.py` stale `an absolute dataset path` help text
  replaced with `$MIND_NERVE_RUNTIME_DIR` + auto-seeded HF cache.

### Tests, CI, and dependency hardening (Stream B — `f3c79ba`)

- `requirements.in` + `requirements.lock` (hash-locked via
  `uv pip compile --generate-hashes`, 1359 lines covering full
  dev + mcp closure).
- `.github/dependabot.yml` (pip + github-actions, weekly grouped).
- `.github/workflows/dependency-audit.yml` (pip-audit + Bandit +
  Semgrep p/ci+p/python + CycloneDX SBOM, weekly + on-PR).
- `.github/workflows/ci.yml` extended with `qa-gates` and
  `perf-budget` jobs (existing 6 jobs preserved).
- New integration tests at `tests/integration/`:
  - `test_route_determinism.py` (100× byte-identical
    `route_id` + score list assertions).
  - `test_installer_roundtrip.py` (6 tests covering `.bak`
    preservation when `os.replace` fails).
  - `test_daemon_socket.py` (spawns the daemon, probes the
    UNIX socket).
- `tests/perf/test_route_latency_budget.py` (warm p50/p95/p99 over
  100 queries, honest budget vs README/spec, 200 ms regression
  ceiling, `MIND_NERVE_PERF_SKIP=1` opt-out).
- `pyproject.toml` dev extras (`pytest-cov`, `mypy`, `bandit`,
  `pip-audit`) + `[tool.coverage]` + `[tool.mypy]` with strict mode
  on `inference.py`, `installer.py`, `_runtime_dir.py` (49 errors
  surfaced; path-by-path expansion plan documented in
  `pyproject.toml`).
- Coverage 42% on `tests/python + tests/integration` with ratchet
  plan to 55% → 70% → 85% documented in workflow.

### Trainer determinism, MRR/nDCG/ECE, run.json (Stream C — `a6d6ede`)

- New `python/mind_nerve/eval_metrics.py` (pure-NumPy MRR, nDCG@k,
  ECE, 168 LOC).
- `python/mind_nerve/mind_train.py`: `TrainConfig.deterministic`
  (default true), `_apply_deterministic_flags`
  (`torch.use_deterministic_algorithms`, cuDNN deterministic,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`), `_emit_run_json` with
  `git_sha`, `dataset_manifest_sha256`,
  `requirements_lock_sha256`, `hf_revision`, host facts;
  extended `_evaluate_top_k` to emit MRR + nDCG@{1,5,10} + ECE
  alongside existing top-1/5/10.
- `tests/python/test_eval_metrics.py` (18 unit tests, hand-computed
  fixtures + edge cases).
- `tests/python/test_trainer_determinism.py` (2 integration tests
  under `@pytest.mark.slow`, runs trainer twice with same seed,
  asserts identical metrics within 1e-9).
- `docs/reproducibility.md` documents the `run.json` schema,
  deterministic-mode guarantees and non-guarantees, and a
  reproduce-a-published-run recipe.

### Deployment, packaging, release provenance (Stream D — `ab2aa43`)

- `installer.py`: `_target_config_paths` helper, `rollback_last`,
  `cmd_rollback` + new `rollback` subparser in
  `mind-nerve-install`; CLI surface
  `mind-nerve rollback --target <name>` also added in `cli.py`.
- `tests/python/test_installer_atomic_writes.py` (17 tests covering
  per-target `safe_write` coverage, `rollback_last` round-trips,
  idempotency, and a source-level invariant test that fails if any
  installer regresses).
- New `Dockerfile` (multi-stage `python:3.12-slim`, non-root user,
  real socket-probe `HEALTHCHECK`), `docker-compose.yml`
  (single `mind-nerve-daemon` service with persistent volumes),
  `.dockerignore`.
- `docs/deployment.md` (Docker quickstart, daemon mode, MCP mode,
  health/readiness, logging, rollback procedure,
  supply-chain provenance pointer).
- `.github/workflows/release.yml` switches PyPI publishing to
  OIDC trusted-publishing + `actions/attest-build-provenance@v2`
  + `SHA256SUMS` attached to the GitHub Release.

### Verified

- 324 passing tests (83 unit + 240 integration + 1 daemon socket).
- Coverage 42% (strongly covered modules: `_runtime_dir.py` 88%,
  `eval_metrics.py` 98%, `mind_train.py` 90%, `daemon.py` 86%).
- All 13 installer writes use `safe_write`; source-level invariant
  test prevents regression.
- All four stream commits compose cleanly with no merge conflicts.

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
- Hardcoded `catalog-data/...` default. Replaced by
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
