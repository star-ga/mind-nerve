# Skill Marketplace Adapter

Design document for the Phase 3 skill marketplace integration. This spec governs how
third-party skill libraries register themselves with mind-nerve, how the route table is
updated without retraining, and how every registration is cryptographically authenticated.

Status: **design-only** — functional ship requires Phase 2 completion (native MIND
inference, `mindc` 0.3.0 cdylib emit).

---

## Overview

The skill marketplace adapter extends the existing `discovery.py` scan-and-embed model
to cover remotely hosted skill providers. A provider is any HTTP(S) endpoint that
exposes a catalog of skills conforming to the JSON-RPC 2.0 registration schema defined
below. mind-nerve fetches, authenticates, and ingests the catalog delta without requiring
the user to clone a repository or restart the daemon.

Key properties:

- **Cryptographic authentication.** Every provider registration carries an ed25519
  signature (RFC 8032) over `name || url || nonce`. mind-nerve verifies the signature
  before touching the route table. An unverifiable registration is silently discarded
  and logged to the evidence stream.
- **License-gate inheritance.** The same `PUBLIC_LICENSES` allowlist and
  `COMMERCIAL_MARKERS` regex from `discovery.py` applies to every remotely fetched
  skill entry. Remote providers cannot bypass the license gate.
- **Delta semantics.** Providers expose a `delta` endpoint that returns only the skills
  added or removed since a caller-supplied `since_version` cursor. The caller supplies
  the cursor obtained from the previous successful fetch. This keeps incremental updates
  cheap regardless of provider catalog size.
- **Namespace conflict resolution.** Route IDs are globally unique by `SHA-256(url +
  skill_id)[:16]`. Two providers may list skills with identical `name` strings; they
  will receive distinct route IDs. Conflicts on the natural-language `name` field are
  resolved by appending `@provider_domain` to the display name.
- **Rate limiting.** The adapter enforces a per-provider request budget: at most 1
  fetch per `MARKETPLACE_MIN_INTERVAL` seconds (default 300 s). Repeated polling below
  this threshold returns the cached last-delta without a network call. The limit is
  configurable per provider in the provider record.

---

## JSON-RPC 2.0 Registration Schema

All provider endpoints speak JSON-RPC 2.0 over HTTPS. HTTP is rejected; TLS 1.2+ is
required.

### `marketplace.describe` — provider metadata

Request:

```json
{
  "jsonrpc": "2.0",
  "method": "marketplace.describe",
  "id": "1"
}
```

Response:

```json
{
  "jsonrpc": "2.0",
  "id": "1",
  "result": {
    "name": "example-skills",
    "version": "1.4.2",
    "license": "apache-2.0",
    "pubkey_ed25519_b64": "<base64 DER>",
    "catalog_size": 312,
    "min_fetch_interval_s": 300,
    "delta_supported": true
  }
}
```

Fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | yes | Canonical provider name (slug, no spaces) |
| `version` | string | yes | SemVer provider catalog version |
| `license` | string | yes | SPDX license identifier for all emitted skills |
| `pubkey_ed25519_b64` | string | yes | Base64-encoded 32-byte ed25519 public key |
| `catalog_size` | integer | yes | Total number of skills in the full catalog |
| `min_fetch_interval_s` | integer | yes | Minimum seconds between fetches the provider accepts |
| `delta_supported` | boolean | yes | Whether `/marketplace.delta` is available |

### `marketplace.delta` — incremental route updates

Request:

```json
{
  "jsonrpc": "2.0",
  "method": "marketplace.delta",
  "params": {
    "since_version": "1.3.0",
    "nonce": "<32 random bytes, hex>"
  },
  "id": "2"
}
```

Response:

```json
{
  "jsonrpc": "2.0",
  "id": "2",
  "result": {
    "as_of_version": "1.4.2",
    "added": [
      {
        "skill_id": "grep-search",
        "name": "grep: search file contents",
        "description": "Search for patterns in files using grep-style matching",
        "kind": "skill",
        "license": "apache-2.0",
        "url": "https://example.com/skills/grep-search",
        "signature_b64": "<base64 64-byte ed25519 signature>"
      }
    ],
    "removed": ["old-skill-id"]
  }
}
```

The `signature_b64` field in each skill entry is the ed25519 signature of:

```
name_bytes || url_bytes || nonce_bytes
```

where `||` is byte concatenation, `name_bytes` is the UTF-8 encoding of `name`,
`url_bytes` is the UTF-8 encoding of `url`, and `nonce_bytes` is the 32-byte
nonce hex-decoded from the request. The provider signs with its private key; the
caller verifies with `pubkey_ed25519_b64` from the `describe` response.

A skill entry whose signature does not verify is discarded and logged.

### Error codes

Standard JSON-RPC error codes apply. Additional marketplace-specific codes:

| Code | Meaning |
|---|---|
| -32001 | `RateLimitExceeded` — caller below `min_fetch_interval_s` |
| -32002 | `UnknownVersion` — `since_version` not in provider history |
| -32003 | `CatalogTooLarge` — delta would exceed 10 MB; use full re-fetch |

---

## Authentication Model

ed25519 (RFC 8032) is the sole signature algorithm accepted. Rationale:

- 64-byte signatures and 32-byte keys are compact relative to RSA alternatives.
- Deterministic signing; no per-operation randomness requirement beyond the caller
  nonce.
- Supported natively in Python ≥ 3.6 via `cryptography` and in the standard library
  via `hashlib` for the pre-image; full signature ops require `cryptography` or
  `PyNaCl`.

The ed25519 public key is pinned on first successful `marketplace.describe` call and
stored in the provider record. A provider that returns a different `pubkey_ed25519_b64`
on a subsequent `describe` call is treated as a key rotation event: mind-nerve logs the
rotation, invalidates the cached signatures, and requires re-verification of the full
catalog before accepting new deltas.

Nonce binding prevents replay: the nonce sent in the delta request is included in the
signed pre-image. A replayed signature from a previous delta call will not verify
against a new nonce.

---

## Route-table Delta Semantics

`marketplace.delta` returns `added` and `removed` sets relative to `since_version`.

On receipt of a delta:

1. Verify each `added` entry's signature. Discard entries that fail.
2. Apply the `discovery.py` license gate to each passing entry.
3. For each passing entry, compute the global route ID as
   `SHA-256(provider_url + "/" + skill_id)[:16]` (hex).
4. Append new embeddings to the route table via `_save_table_atomic`.
5. Remove `removed` route IDs from the table.
6. Update the stored `since_version` cursor to `as_of_version`.
7. Emit a `CatalogLoad` attestation envelope with the new catalog hash.

Steps 4–6 are performed under a write lock to prevent concurrent daemon fetches from
interleaving partial states.

---

## Namespace Conflict Resolution

Two providers may define skills with the same human-readable `name`. The resolution
strategy:

1. Route IDs are `SHA-256(provider_url + "/" + skill_id)[:16]` — globally unique by
   construction.
2. Display names (`name` field in `Route`) receive a `@provider_domain` suffix when a
   collision exists in the active route table: `"grep: search" → "grep: search
   @example.com"`. The suffix uses the registered URL's second-level domain.
3. Deduplication by description content hash is NOT performed across providers — two
   providers may legitimately offer different implementations of a skill with the same
   description text.

---

## Rate Limiting

The rate limiter operates per-provider-URL and is implemented as a simple timestamp
gate in the daemon's fetch loop:

- The last successful fetch time is stored in the provider record.
- A fetch is skipped (returns cached delta cursor) if `now - last_fetch_time <
  max(MARKETPLACE_MIN_INTERVAL, provider.min_fetch_interval_s)`.
- `MARKETPLACE_MIN_INTERVAL` defaults to 300 seconds and is configurable via the
  `MIND_NERVE_MARKETPLACE_INTERVAL` environment variable.
- Rate limit state is not persisted across daemon restarts. On restart, the first
  fetch for each provider is always permitted.

---

## Cross-references

- `discovery.py` — license-gate constants (`PUBLIC_LICENSES`, `COMMERCIAL_MARKERS`)
  and `_save_table_atomic` are reused directly.
- `spec/architecture.md §"Attestation envelope"` — every successful delta application
  emits a `CatalogLoad` envelope with the updated catalog hash.
- `python/mind_nerve/marketplace.py` — typed stub implementing the interfaces above.
- RFC 8032 (ed25519 signature scheme): https://www.rfc-editor.org/rfc/rfc8032
- JSON-RPC 2.0 specification: https://www.jsonrpc.org/specification
