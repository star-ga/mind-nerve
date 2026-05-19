# mind-mem v4 Cognitive Kernel Integration

Design document for the Phase 3 binding between mind-nerve route history and the
mind-mem v4 cognitive memory kernel. This spec publishes the interface contract so
external consumers (downstream contract scoring consumers) can validate
the binding shape while Phase 2 completes.

Status: **design-only** — functional ship is BLOCKED on mind-mem v4 cognitive kernel
(see ROADMAP.md §"Phase 3", external dependency).

---

## Context

mind-mem v4.1 §7 ("Tool-routing preselector intersection") introduces a cognitive memory
kernel that can treat agent tool-routing decisions as a first-class memory class. The
roadmap item states:

> Route history becomes a memory class: mind-nerve preselections are stored as
> episodic memories indexed by (request_hash, catalog_hash, model_hash) and queried
> during future routing to bias the preselector toward patterns that have been
> reinforced by successful task completions downstream.

This document defines the exact interface that mind-nerve must expose for mind-mem v4
to consume, and the interface that mind-mem v4 must expose for mind-nerve to write
routing events into.

---

## Route History as a Memory Class

Every inference that mind-nerve produces carries an attestation envelope
(spec/architecture.md §"Attestation envelope"). The envelope already contains:

- `request_hash` — SHA-256 of the input request bytes
- `result_hash` — SHA-256 of the canonical top-K route list
- `catalog_hash` — SHA-256 of the RouteCatalog used
- `model_hash` — SHA-256 of the weights manifest

These four fields are sufficient to uniquely identify any routing event across time,
hosts, and model versions. The cognitive kernel binding adds a fifth dimension:
**outcome signal** — whether the selected routes led to a successful task completion
as reported by the downstream host.

The memory class shape is:

```
RouteMemoryEntry {
    request_hash:  bytes[32]   # from attestation envelope
    result_hash:   bytes[32]   # from attestation envelope
    catalog_hash:  bytes[32]   # from attestation envelope
    model_hash:    bytes[32]   # from attestation envelope
    timestamp_ms:  i64         # from attestation envelope (monotonic)
    outcome:       f32 | null  # host-reported; null if outcome not yet known
    query_text:    str | null  # plaintext request; null if host opts out
}
```

`outcome` is a float in `[0.0, 1.0]`: 0.0 = complete failure, 1.0 = full success.
`null` means the host has not yet reported an outcome for this routing event. The
cognitive kernel accumulates `null` entries and resolves them when the host reports
back (within a configurable TTL; default 300 s, after which unresolved entries are
discarded from the pending queue).

`query_text` is optional. Hosts that store plaintext in memory must ensure their
privacy policy permits it. Hosts that omit it lose the semantic search capability
but retain the hash-based reinforcement path.

---

## mind-nerve → mind-mem Write Interface

mind-nerve writes routing events via the mind-mem MCP tool interface. The target tool
name is `store_route_event` (to be registered by mind-mem v4 at MCP server startup).

Proposed tool call shape (JSON):

```json
{
  "tool": "store_route_event",
  "params": {
    "request_hash": "<64 hex chars>",
    "result_hash": "<64 hex chars>",
    "catalog_hash": "<64 hex chars>",
    "model_hash": "<64 hex chars>",
    "timestamp_ms": 1748000000000,
    "outcome": null,
    "query_text": "git status"
  }
}
```

The write is fire-and-forget from mind-nerve's perspective: write failures are logged
to the local evidence stream and do not affect inference results. mind-nerve MUST NOT
block on the write or retry synchronously. The cognitive kernel is an optional
enhancement; its unavailability must not degrade routing latency.

Writes occur:
- Immediately after every successful inference (outcome = null at this point).
- Again when the host calls `report_route_outcome` (below) with a resolved outcome.

---

## Host → mind-nerve Outcome Reporting Interface

Hosts report task outcomes via the mind-nerve daemon's outcome endpoint:

```
POST /v1/outcome
Content-Type: application/json

{
  "result_hash": "<64 hex chars>",
  "outcome": 0.85
}
```

`result_hash` is the value from the attestation envelope emitted for the routing
event. The daemon matches it to the pending `RouteMemoryEntry`, sets `outcome`,
and re-writes the entry to mind-mem via `store_route_event`.

The daemon maintains a pending queue of at most `OUTCOME_QUEUE_MAX` entries (default
4096). Entries older than `OUTCOME_TTL_S` (default 300 s) are evicted without
resolution.

---

## mind-mem → mind-nerve Read Interface (Route Bias)

When mind-mem v4 cognitive kernel is active, mind-nerve may request a bias vector
before performing top-K scoring. The bias vector is a `(|RouteCatalog|,)` f32 array
where each element represents the reinforcement score for the corresponding route
based on historical outcomes for similar requests.

The read uses the mind-mem MCP tool `query_route_bias`:

```json
{
  "tool": "query_route_bias",
  "params": {
    "request_hash": "<64 hex chars>",
    "catalog_hash": "<64 hex chars>",
    "top_k": 64
  }
}
```

Response:

```json
{
  "biases": [
    {"route_id": "abc123", "bias": 0.15},
    {"route_id": "def456", "bias": -0.05}
  ],
  "confidence": 0.72
}
```

`bias` is added to the logit score before top-K extraction. Positive bias lifts
routes that have historically led to successful outcomes for similar requests. The
`confidence` field (range `[0.0, 1.0]`) is used to scale the bias contribution:
effective_bias = `bias × confidence`. At `confidence = 0` the bias has no effect,
making the system safe to use when mind-mem has insufficient history.

The bias query is optional: if mind-mem is unavailable, mind-nerve proceeds with
unbiased scoring. The read adds at most `BIAS_QUERY_BUDGET_MS` (default 5 ms) to
the latency budget; if the query exceeds the budget it is aborted and unbiased
scoring is used.

---

## Interaction with the Attestation Chain

Route bias injection does NOT alter the attestation envelope. The envelope records
the result of scoring after bias application, not the pre-bias scores. This means:

- `result_hash` reflects the actual output, including any bias effect.
- Two runs with identical `(request_hash, catalog_hash, model_hash)` but different
  cognitive kernel histories will produce different `result_hash` values. This is
  intentional: the evidence chain records the full causal chain of how the result
  was produced.
- `model_hash` does NOT include the cognitive kernel state. It covers only the
  static weights manifest. The cognitive kernel state is tracked separately via
  mind-mem's own evidence log.

---

## Versioning and Rollout

The binding is controlled by two feature flags in the mind-nerve config:

```toml
[mind_mem_integration]
enabled = false       # set true when mind-mem v4 cognitive kernel ships
write_events = true   # write route events; requires enabled = true
read_bias = false     # read bias vector; requires write_events = true and
                      # sufficient history (≥ BIAS_MIN_EVENTS = 100 events)
```

`enabled = false` is the safe default. The inference path is byte-identical to the
non-integrated path when disabled. Enabling write-only (`write_events = true,
read_bias = false`) accumulates history without affecting routing decisions.
Enabling read (`read_bias = true`) activates the bias path; this is the full
integration mode and requires prior history.

---

## Cross-references

- ROADMAP.md §"Phase 3 — Ecosystem" — "mind-mem v4 cognitive-kernel integration"
  exit criterion.
- mind-mem v4.1 ROADMAP §7 — "Tool-routing preselector intersection" — the
  corresponding entry on the mind-mem side of this binding.
- spec/architecture.md §"Attestation envelope" — the 212-byte layout that provides
  `request_hash`, `result_hash`, `catalog_hash`, `model_hash`, `timestamp_ms`.
- spec/integration_surface.md §"Host 2: MCP servers" — the MCP façade that
  mind-nerve uses to communicate with mind-mem.
