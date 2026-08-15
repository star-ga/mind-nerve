# src/dormant — quarantined mind-nerve front-end sources (never-shipped dialect)

**Status: DORMANT. These 13 files (~6,600 lines) have never compiled under
any shipping mindc** — not the CI-pinned v0.4.4, not the current 0.10.2.
Quarantined 2026-08-10 (CI-mirror audit, finding D2). The operator direction
is "mind-nerve goes full pure MIND": these files are the port backlog, not
dead code. They were moved here so that no walk of `src/` can silently
report them as part of the compilable surface.

The CI-gated, numerically pinned production MIND surface is `mind/`
(luts + kernels + exports), which IS fully ported to 0.10.2 and gated by
`tests/mindc_gate.sh`. Nothing below affects it.

## Why a wholesale port is NOT a mechanical find-and-replace

The tree was written against a Rust-flavoured MIND dialect that was designed
but never shipped, plus a host runtime (`mind-runtime` `crate::std::time` /
`crate::std::cell` / stdin-stdout byte-slice externs) that also never
shipped. Constructs with no 0.10.2 equivalent:

- `use crate::mind_nerve::{ A, B }` grouped braced imports (real dialect:
  `import module.path;`, one module per statement, file-derived paths).
- `@[determinism(...)]` / `@[reduction_order(Pinned)]` / `@[arithmetic(Wrapping)]`
  lint attributes — the E_NERVE_001..005 contract from `spec/numerics.md`;
  no shipped mindc implements them (0.10.2 attribute form is `#[name]`).
- `#[cfg(has_runtime_clock)]` conditional compilation — no cfg surface.
- `static COUNTER: ThreadLocal<i64>` + `fetch_add` — no thread-local statics.
- `crate::std::time::monotonic_steady_ns()` etc. — bundled std is
  vec/string/map/io/fs/sha256/json/cli; no time/cell modules.
- `extern fn f(...) -> [u8]` — 0.10.2 externs are `extern "C" { unsafe fn
  ...; }` over C-ABI scalar/pointer types; byte-slice returns (`[u8]`,
  `[u8; 32]`) are not expressible, so the runtime_ffi ABI contract itself
  must be redesigned (pointer+length out-params or the like).
- `[Q16_16; ENCODER_HEADS as usize]` const-generic array lengths — 0.10.2
  requires a literal positive-integer length.
- `65536_i32` underscore-suffixed literals (hex `0x...u32` suffixes DO parse;
  decimal `_i32` does not).

Two port-blockers are DESIGN decisions, not syntax:

1. **q16_16.mind's embedded LUTs conflict with the pinned tables.** Its
   `q16_exp`/`q16_rsqrt` build 257/256-entry tables via aspirational
   const-fn evaluation — different tables from the pinned, generated
   `mind/luts/exp_q16.mind` (4096 entries) and `sqrt_q16.mind` (16384)
   whose bit-identity canaries are the CI contract. A faithful port would
   create a second, conflicting numeric surface; a correct port DELEGATES
   to `mind/luts`. That choice owns to the architecture spec, not to a
   mechanical translation.
2. **runtime_ffi.mind / clock.mind define a host contract that does not
   exist.** Any 0.10.2 rendering invents ABI decisions (buffer ownership,
   pointer+length conventions, clock fallback policy without cfg/statics).

## Validated port pattern (piloted against 0.10.2, 2026-08-10)

These constructs WERE verified to parse, check, and execute correctly under
`mindc test` 0.10.2:

- `pub type Q16_16 = i32;` + `pub const ONE: Q16_16 = 65536;`
- `v as i64` / `x as i32` widening/narrowing casts; arithmetic `>>`; `if`
  expressions as `let` RHS; `while` loops (std-surface).
- Fixed-literal-length arrays (`[u32; 8]`, `[0u32; 64]`) and `u32` hex
  suffix literals (`0x6a09e667u32`).
- `#[test]` fns calling in-file helpers — the interpreter executes memory
  intrinsics (`__mind_alloc` / `__mind_store_i64` / `__mind_load_i64` /
  `__mind_free`) natively, runtime-free.

## Per-file port order + effort estimates

Smallest-dependency-first. S ≈ hours, M ≈ a day, L ≈ multi-day (each
including oracle-based validation, never "it parses").

| file | lines | blockers beyond syntax | effort | status (2026-08-12) |
|---|---|---|---|---|
| sha256.mind | 411 | `&[u8; 64]` sigs → slices/values; `usize`; `data.len()`; verify vs FIPS KATs (std.sha256 cross-check available) | M | **PORTED** (wave 1) — delegates to std.sha256; 16/16 tests, gate leg 3 |
| q16_16.mind | 683 | **table-dedup decision (above)**; local consts replace `use crate::{…}`; then mechanical (piloted) | M after decision | **PORTED** (wave 1) — delegates to pinned mind/luts tables; 86/86 tests, gate leg 4 |
| top_k.mind | 639 | enums + match (0.10.2 supports both); tie-break SHA binding to ported sha256 | M | **PORTED** (wave 2) — handle-ABI heap; 30/30 tests incl. 8 hashlib-verified oracles, gate leg 6 |
| tokenizer.mind | 330 | byte-slice I/O contract | M | **PORTED** (wave 2) — handle ABI, signed error returns; 18/18 tests incl. hashlib-pinned tokenizer_hash oracle, gate leg 7 |
| chain_log.mind | 221 | `#[cfg]` removal (pick a branch — policy decision); entropy extern | M | **PORTED** (wave 2) — 56-byte handle; cfg decision: entropy is a caller-supplied seed (`session_id = SHA-256(seed)[0..16]`), the pure-MIND surface never reads clocks/entropy; 5/5 tests, gate leg 8 |
| evidence.mind | 604 | envelope byte packing over `[u8; N]`; depends on sha256 + chain_log + clock | L | **PORTED** (wave 4) — 212-byte wire built byte-wise over caller buffers; Result → signed EV_ERR_* returns with dormant gate order preserved; +2 sequence-API tests the dormant suite lacked; 36/36 tests, gate leg 11 |
| runtime_ffi.mind | 102 | **ABI redesign (above)** — small file, large decision | S after decision | **PORTED** (wave 3) — pointer+length out-param ABI, caller-owns-all-buffers, size-probe reads, negative-status errors; extern "C" block + thin wrappers; 3/3 tests (status logic; extern bodies gate at link time), gate leg 9 |
| clock.mind | 133 | cfg + static + ThreadLocal replacements; needs runtime_ffi decision | S after decision | **PORTED** (wave 3) — cfg probe replaced by explicit host-vs-counter selection; ThreadLocal replaced by per-session 8-byte counter handle; 5/5 tests, gate leg 10 |
| model.mind | 446 | weights layout + loader contract | M | **PORTED** (wave 4) — 18-slot weight-table handle (MODEL_WTAB_*); 8/8 tests, gate leg 13 |
| loader.mind | 1,536 | file I/O over the redesigned extern surface; heaviest suffix-literal sweep | L | **PORTED** (wave 4) — probe-then-read file ABI; dequantize formula preserved verbatim with an anomaly flag (2^16-fold scale suspicion documented in the file header); 12/12 tests incl. hashlib-pinned catalog-hash / model_hash / manifest oracles, gate leg 14 |
| encoder_kernels.mind | 1,288 | const-generic array lengths; overlaps mind/kernels (dedupe question) | L | **PORTED** (wave 4) — flat row-major i64-lane buffers, LUT handles threaded, all RFC const gates preserved as runtime-evaluated const branches; scores/probs consumed row-at-a-time (same pinned reduction order, bounded memory); 17/17 tests, gate leg 12 |
| inference.mind | 548 | ties model+encoder+evidence; `match` on payload enums OK | L | **PORTED** (wave 4) — timestamp_ms is a caller parameter (wave-3 clock decision); no re-export wrappers (merged-image global namespace); 9/9 tests + native e2e run-harness (real ELF, hashlib-pinned envelope oracles), gate leg 15 |
| lib.mind | 196 | `module mind_nerve {}` wrapper parses; const-generic lengths need literals; becomes the const re-export hub | S after q16_16 decision | **PORTED** (wave 1) — 12/12 tests, gate leg 5 |

The companion test suite is quarantined at `tests/unit/dormant/` with its
own README; port tests alongside their target module, never ahead of it.

Do not move files back to `src/` until they pass `mindc check --no-fmt
--no-lint` under the CI-pinned mindc AND have executing `#[test]` coverage
wired into `tests/mindc_gate.sh` with exact-count assertions.

## Port complete (2026-08-15)

All 13 files are ported and gated (legs 3-15 of `tests/mindc_gate.sh`, all
green under mindc 0.10.2). This tree remains the pre-port archive.

Compiler/harness findings filed during the port (repros under
/home/n/mind-flow/target/repro-0.10.2/):

- **r9** (`r9_test_assert_deferred.mind`): `mindc test` evaluates `assert`
  conditions lazily — a condition reading state mutated later in the same
  test is mis-evaluated. Let-bind conditions that need eager reads.
- **r10** (`r10_test_unresolved_callee_env.mind`): an unresolved reference
  inside a CALLED function aborts it and surfaces a misleading
  `unknown variable` error in the caller, not the undefined symbol.
- **r11** (`r11_oob_interpreter_corruption/`): the test interpreter
  performs no bounds checking on the memory intrinsics; OOB reads silently
  corrupt its evaluation state (native execution segfaults at the fault —
  the honest failure). Root-caused the wave-4 fixture segfault.
- **Toolchain note:** an intermediate 0.10.2 rebuild (2026-08-14) briefly
  lost project-mode cross-module resolution (E2003 on every cross-module
  call; the `cross-module-imports` feature gates module-table seeding —
  src/check/mod.rs:216). The 2026-08-15 rebuild restores it; the gate's
  documented install flags (`--features std-surface,cross-module-imports`)
  are load-bearing.
