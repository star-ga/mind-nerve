# Federated Routing

Design document for the Phase 3 federated routing extension. This spec governs how
multiple mind-nerve instances running on different hosts collaborate to share route
decisions, reconcile their attestation chains, and maintain a consistent evidence record
across a cluster.

Status: **design-only** — functional ship requires Phase 2 completion and
the future typed-edges composition layer.

---

## Overview

A single mind-nerve instance is stateless per call. Federated routing introduces a
lightweight peer mesh where instances can:

1. Exchange their current attestation chain tip (the SHA-256 of the most recent
   envelope) to detect divergence early.
2. Reconcile chains when two or more peers have processed different request histories
   using a vector-clock protocol (Lamport 1978).
3. Share route-table updates so a catalog delta ingested by one peer propagates to
   the mesh without each peer independently fetching from the marketplace.

Federation is opt-in. A standalone mind-nerve instance is unaffected by this spec until
`[federation]` is present in its config.

---

## Peer Discovery

Two mechanisms are supported:

### mDNS (zero-config LAN)

mind-nerve advertises itself as `_mind-nerve._tcp.local.` via mDNS/DNS-SD (RFC 6762 /
RFC 6763). Each instance broadcasts:

- `hostname` — the mDNS hostname (e.g. `devbox.local`)
- `port` — the federation listener port (default 47361)
- `pubkey` — the instance's 32-byte ed25519 public key, hex-encoded, as a TXT record

Peers discovered via mDNS are added to the peer list with `trust_level = "lan"`.
mDNS discovery is active only when `federation.mdns = true` in config.

### Explicit peer config

Any number of peers may be listed explicitly in the config:

```toml
[federation]
enabled = true
mdns = false

[[federation.peers]]
host = "10.0.0.5"
port = 47361
pubkey_hex = "<64 hex chars>"
```

Explicit peers are added with `trust_level = "configured"`.

---

## Vector-clock Protocol

Each mind-nerve instance maintains a vector clock over its known peer set. The vector
clock follows Lamport 1978: a logical timestamp that advances on every local event
(inference, catalog load, peer sync) and on receipt of any peer message whose component
exceeds the local value.

The vector clock is used to establish a partial order on events for chain reconciliation.
It does NOT replace the cryptographic chain — the chain provides tamper detection; the
vector clock provides ordering context for divergence resolution.

### Clock structure

```
VectorClock = { peer_id → u64 }
```

`peer_id` is `SHA-256(pubkey_bytes)[:16]` (hex). The local instance's own clock slot
advances on every envelope it emits.

### Sync message

When two peers connect, they exchange a sync message:

```json
{
  "protocol": "mind-nerve-federation/1",
  "peer_id": "<16 hex chars>",
  "chain_tip": "<32-byte SHA-256 hex>",
  "clock": { "<peer_id>": <u64>, ... },
  "signature": "<ed25519 signature of (chain_tip_bytes || clock_canonical_bytes)>"
}
```

The signature is verified before any state update. An unverifiable sync message is
discarded.

### Divergence detection

Two peers have diverged when neither peer's `chain_tip` is an ancestor of the other's.
Divergence is detected by walking `chain_prev` links backward from each tip. A common
ancestor within a configurable depth (default 1024 envelopes) confirms the divergence
scope. If no common ancestor is found within the depth limit, the peers are considered
to be on incompatible chains and reconciliation is abandoned — both peers log the event
and continue independently.

---

## Chain-reconciliation Algorithm

When divergence is detected and the common ancestor is within the search depth:

1. Both peers exchange the envelope sequences between the common ancestor and their
   respective tips.
2. Each received envelope is verified (signature, `chain_prev` linkage, reserved-byte
   check, architecture-enum validity) against the spec in `spec/architecture.md
   §"Attestation envelope"`.
3. Envelopes that pass verification are merged into a shared reconciled chain. The
   merge strategy is **tip-union**: both tip sequences are appended in vector-clock
   order (earlier logical time first; ties broken by `SHA-256(envelope_bytes)
   ascending`).
4. The reconciled chain's new tip is `SHA-256` of the last merged envelope.
5. All participating peers adopt the reconciled tip. Each peer emits a synthetic
   `CatalogLoad` entry into its local evidence log noting the reconciliation event and
   the contributing peer IDs.

Reconciliation does not alter inference results. It only updates the evidence chain for
audit purposes. Route decisions are always made from the local route table.

---

## Route-table Propagation

When a peer ingests a marketplace delta (from `spec/skill_marketplace.md`) it
broadcasts a compact propagation message to connected peers:

```json
{
  "kind": "route_table_delta",
  "catalog_hash_after": "<32-byte hex>",
  "added_route_ids": ["<16 hex>", ...],
  "removed_route_ids": ["<16 hex>", ...],
  "provider_url": "https://example.com/skills",
  "signature": "<ed25519 signature of (catalog_hash_after_bytes || nonce_bytes)>"
}
```

Receiving peers:

1. Verify the signature with the sender's pinned public key.
2. Check whether `added_route_ids` contain routes already present locally (skip if so).
3. Fetch the full skill entries for missing routes from the marketplace provider
   directly (not from the peer). This preserves the cryptographic chain of custody:
   routes arrive signed by their provider, not re-signed by a peer.
4. Apply the license gate.
5. Update the local route table.

A peer that cannot reach the marketplace provider directly marks the missing routes as
`pending_fetch` and retries on the next scheduled fetch cycle.

---

## Security Model

- **Peer authentication**: every peer message is signed with ed25519. Public keys are
  pinned on first contact (TOFU — Trust On First Use) and verified on every subsequent
  message. Key changes on an established peer are treated as a security event: the peer
  is quarantined and the operator is notified via the evidence log.
- **Chain integrity**: reconciliation only merges envelopes that pass full signature
  and structural verification per `spec/architecture.md §"Attestation envelope"`.
  A single invalid byte in any envelope causes the entire received sequence to be
  rejected.
- **Route table trust**: routes are never accepted directly from peers. A peer may
  broadcast that a route exists; the local instance always fetches and verifies the
  route from its originating marketplace provider.
- **No elevation of privilege**: `trust_level = "lan"` and `trust_level =
  "configured"` peers have identical capability sets. The distinction is informational
  only (for logging and operator audit).

---

## Cross-references

- `spec/architecture.md §"Attestation envelope"` — the 212-byte envelope layout that
  `chain_tip` hashes reference. `chain_curr = SHA-256(212-byte serialization)`.
- `spec/skill_marketplace.md` — delta propagation from marketplace to peer mesh.
- `python/mind_nerve/federation.py` — typed stub for the interfaces above.
- Lamport, L. (1978). "Time, clocks, and the ordering of events in a distributed
  system." *Communications of the ACM*, 21(7), 558–565.
  https://doi.org/10.1145/359545.359563
- RFC 6762: Multicast DNS — https://www.rfc-editor.org/rfc/rfc6762
- RFC 6763: DNS-Based Service Discovery — https://www.rfc-editor.org/rfc/rfc6763
- RFC 8032: Edwards-Curve Digital Signature Algorithm (ed25519) —
  https://www.rfc-editor.org/rfc/rfc8032
