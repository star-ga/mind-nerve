# tests/unit/dormant — quarantined MIND unit tests (never-shipped dialect)

**Status: DORMANT. These 8 files (232 `#[test]` fns) do not parse under any
shipping mindc — including the v0.4.4 the CI pinned and the v0.10.2 the gate
now pins — and they never did.** Quarantined 2026-08-10 (CI-mirror audit,
finding D2).

## Why they are here

These are RED-phase TDD files written against `src/*.mind` modules in a
Rust-flavoured MIND dialect that was designed but never shipped in any mindc
release. Constructs used here that no shipping compiler accepts:

- `use crate::mind_nerve::{ ... }` — grouped, braced `crate::`-rooted imports
  (a Rust-ism; real MIND uses `import module.path;` per the v1.0 spec, and
  project-mode module paths are file-derived, not `crate::`-prefixed).
- `65536_i32` / `0u8` underscore/letter-suffixed literals (note: `0x...u32`
  hex suffixes *do* parse under 0.10.2 — the audit's claim was over-broad;
  the decimal `_i32` form does not).
- `&[u8; 64]` reference-to-fixed-array parameters (0.10.2 parses `&[u8]`
  slices but not `&[u8; N]`).
- `@[determinism(BitIdentical)]` / `@[reduction_order(Pinned)]` — the
  E_NERVE_001..005 lint-attribute surface from `spec/numerics.md`, which no
  shipped mindc implements (0.10.2 attributes are `#[name]`).

They also target the quarantined `src/dormant/*.mind` modules, so even a
syntactic port leaves them testing modules that do not compile.

## The trap they created

`mindc test tests/unit/` under every shipping mindc prints the parse errors,
then `running 0 tests` / `test result: ok`, and **exits 0** — a silent-green
trap that masked this suite's total non-execution. `tests/mindc_gate.sh`
leg 2 now hard-fails on any `running 0 tests` output, so a suite in this
state can never read as green again.

## Upgrade path (per-file effort estimates)

A port is a REWRITE against the real 0.10.2 dialect plus the (equally
dormant) `src/` modules they exercise — see `src/dormant/README.md` for the
module-side port order. Sensible sequencing, smallest-first:

| file | tests | target module | effort |
|---|---|---|---|
| test_q16_16.mind | 66 | src/dormant/q16_16.mind | M — but blocked on the exp/rsqrt table-dedup decision (see src README) |
| test_sha256.mind | 24 | src/dormant/sha256.mind | M — mechanical once sha256.mind is ported; KATs are objective |
| test_top_k.mind | 34 | src/dormant/top_k.mind | M |
| test_tokenizer.mind | 16 | src/dormant/tokenizer.mind | M |
| test_manifest_export.mind | 7 | root Mind.toml surface | S — rewrite against mind/Mind.toml |
| test_evidence.mind | 54 | src/dormant/evidence.mind | L (envelope packing + chain log deps) |
| test_encoder.mind | 20 | src/dormant/encoder_kernels.mind | L (12-layer driver; most kernel coverage already exists as the LUT smoke + bit-identity harness) |
| test_inference.mind | 11 | src/dormant/inference.mind | L (needs runtime_ffi contract decisions) |

(S/M/L ≈ hours / a day / multi-day per file, including oracle validation.)

Do not move files back out of `dormant/` until they parse under the
CI-pinned mindc AND are wired into `tests/mindc_gate.sh` with an exact-count
assertion.
