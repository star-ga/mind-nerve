# Pre-merge audit follow-ups (2026-05-14)

Two findings from the four-agent audit of the autoresearch IMPLEMENT
phase (branch `autoresearch/may13c`) are deferred rather than fixed
in the merge commit, because both predate the autoresearch loop and
both are latent (do nothing under the default backwards-soft config).

Tracked here so they don't get lost.

---

## F1 — `model_hash` is never actually computed

**Status:** Pre-existing scope, not introduced by autoresearch.

**Finding:** `parse_weights` (`src/loader.mind:608-615`) reads the
32-byte `model_hash` field straight from the weights file and treats
it as opaque. There is no `compute_model_hash`, `manifest_preimage`,
or analogous function anywhere in `src/` or `cli/`. So every commit
message that says "binds into `model_hash` via the manifest header"
is asserting a property mind-nerve cannot enforce. A reference-
checkpoint producer can flip any compile-time constant and the
loader will not detect the mismatch.

**Implication:** The 12 backwards-soft architecture switches landed
this round all claim model_hash binding. Until a real
`compute_model_hash(constants, weights_manifest) -> [u8; 32]` exists,
the "silent flip ⇒ HashMismatch" defence is fictional. The switches
still work as gates (default values make the new code paths dead),
but the safety net is paper.

**Fix sketch:**
1. Add `pub fn compute_model_hash() -> [u8; 32]` to `src/loader.mind`
   that SHA-256s a canonical byte preimage built from every
   `pub const` named in `spec/architecture.md` "Backwards-soft
   architecture switches" plus the weights manifest header.
2. Add a `manifest_hash: [u8; 32]` field to the v3 weights file
   format (no v1/v2 break — the field is appended).
3. In `parse_weights`, compare `compute_model_hash()` against the
   on-disk `manifest_hash` field and return `HashMismatch` on
   inequality. Existing v1/v2 files lack the field → emit a
   `ManifestHashMissing` warning, accept anyway.
4. Tests in `tests/bit_identity/test_model_hash.mind`.

**Scope:** New work, ~300 LOC. Touches loader.mind + weights format
+ spec/architecture.md "Model-hash discipline" section + tests.
Not blocking the autoresearch merge because nothing in the current
phase actually changed `model_hash`'s contract — every RFC just
inherited the existing (fictional) contract.

---

## F2 — `request_hash` is computed over pre-prepend tokens

**Status:** Latent — only matters once `QUERY_PREFIX_LEN > 0`.

**Finding:** `src/inference.mind:434` calls
`request_hash_from_tokens(tokens)` with the **pre-prepend** `tokens`
slice while the encoder consumes `effective_tokens` (which may
include up to 8 `QUERY_PREFIX_TOKENS` slots prepended per RFC-012).
A third-party verifier holding only the 212-byte envelope cannot
reproduce the encoder input from `request_hash` alone — they would
also need access to mind-nerve's compiled `QUERY_PREFIX_TOKENS`
constant.

**Implication:** Replay reproducibility holds *internally*
(`model_hash` covers `QUERY_PREFIX_TOKENS`, assuming F1 lands).
But the envelope's external auditability contract bends once the
prefix is non-empty.

**Fix sketch (when RFC-012 flips on):** Either
- Hash `effective_tokens` AND emit a separate `prefix_hash` field
  in the envelope, OR
- Document explicitly in `spec/architecture.md` that
  `request_hash` covers user-supplied tokens only, with
  prefix-aware replay requiring out-of-band knowledge of the
  compiled `QUERY_PREFIX_TOKENS` (which is bound into
  `model_hash`).

**Scope:** Either a v3 envelope (semver bump) or a one-paragraph
spec clarification. Defer until a calibrated prefix-trained
checkpoint actually ships.

---

## Agent-by-agent verdict pointers (for the next reviewer)

- `mind-code-reviewer` agent — flagged 3 surgical fixes (RouteCatalog
  field, ALiBi alloc, stride fingerprint gate) — **all applied in
  this merge commit**. Plus the type-system concerns around
  `[Q16_16; n]` and `concat.push` which were determined to be
  pre-existing scaffold idioms that already ship in `q16_layernorm`
  / `q16_softmax`.
- `mind-architect` agent — flagged F1 (model_hash fiction) and
  D1 (encoder_kernels constants visibility). D1 was already
  half-resolved by my patch to `src/lib.mind` that exposed the 7
  remaining backwards-soft switches at the crate root.
- `mind-security-reviewer` agent — caught the request_hash/prefix
  issue (F2) and a potential `num_routes * 4 KiB` overflow on
  32-bit `usize` at attacker-controlled catalog sizes. The overflow
  is gated by the existing `validate_catalog_size` check earlier
  in `parse_catalog`.
- `mind-auditor` agent — confirmed every new kernel composes only
  pinned primitives and carries the right reduction-order
  annotations. No clock reads, no randomness, no atomic-RMW on
  shared state.
