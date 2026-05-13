# Bit-identity test suite

The load-bearing correctness gate. Runs the same fixed-input vector against
every supported backend and verifies all backends produce the same
`result_hash` byte-for-byte.

## Scope by phase

| Phase | Backends in scope |
|---|---|
| Phase 1 | x86 CPU, CUDA |
| Phase 2 | x86 CPU, ARM CPU, CUDA, WebGPU, NPU |

A backend not in scope for the current phase is allowed to diverge. A
backend in scope that diverges by even one byte fails the build.

## Fixture inputs

The fixture set lives in `fixtures/` and contains:

1. 100 known English requests labeled to specific routes (covers happy path)
2. 50 ambiguous requests (covers tie-break determinism)
3. 30 adversarial requests (long inputs, emoji, mixed languages, encoded
   binary, attempts to break tokenization)
4. The reference catalog (`fixtures/catalog.json`) with 200 routes

Every fixture has an expected `result_hash` and `chain_curr` for the
reference checkpoint. Updating the reference checkpoint requires
regenerating these and is a non-trivial process documented in
`PROTOCOL.md` (Phase 1 deliverable).

## Phase 0 status

No fixtures yet. Fixture generation is Phase 1 work, alongside the reference
checkpoint.
