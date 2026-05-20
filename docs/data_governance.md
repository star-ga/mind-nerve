# Data Governance

This document describes the data-governance posture for the public
mind-nerve repository, the published Phase-1 weights, and the routing
catalog assembled by `mind_nerve.discovery.scan`.

## Scope

In scope:

- The Phase-1 v1.1-oss training corpus and the
  [`star-ga/mind-nerve`](https://huggingface.co/star-ga/mind-nerve)
  Hugging Face artefact.
- The `route_table.jsonl` shipped inside the runtime directory.
- Catalogs produced locally by users running
  `mind-nerve learn` / `mind_nerve.discovery.scan(...)`.

Out of scope:

- Training corpora used to produce non-public STARGA-licensed weights
  (these are released, when released at all, under separate STARGA
  agreements and are clearly labelled at release time per
  [`LICENSE.md`](../LICENSE.md)).
- Catalogs assembled by third parties outside this repository.

## License posture

The Phase-1 v1.1-oss training corpus is curated to be **public-clean**.
Every ingested artefact must satisfy at least one of:

- An SPDX-identifiable open-source license known to be compatible with
  Apache-2.0 redistribution.
- An explicit public-domain / `CC0-1.0` declaration.

The accepted license whitelist (case-insensitive SPDX match):

- `Apache-2.0`
- `MIT`
- `BSD-2-Clause`, `BSD-3-Clause`
- `MPL-2.0`
- `ISC`
- `CC0-1.0`

Anything else — including unknown / missing license metadata — is
**excluded by default**. `mind_nerve.discovery.scan(...)` enforces this
through its `include_unknown=False` default; flipping it to `True` opts
the local catalog (not the public Phase-1 corpus) into a more permissive
mode and is the operator's responsibility.

## Commercial-risk exclusions

The following content categories are excluded from the public Phase-1
catalog regardless of stated license:

- Anything marked **"commercial use restricted"**, **"non-commercial"**,
  **"no redistribution"**, or **"evaluation only"**.
- Proprietary vendor SDK documentation that is not separately licensed
  for redistribution.
- Internal-only or unpublished work products.
- Content that names individuals in a way that would qualify as personal
  data under GDPR / CCPA-style regulations without their explicit public
  consent.

## Provenance fields

Each row in `route_table.jsonl` carries provenance fields. Required:

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Stable route identifier. |
| `name` | string | Human-readable name (used as the surface for top-K results). |
| `kind` | string | One of `skill`, `tool`, `mcp`, `agent`, `command`. |
| `source_repo` | string | Origin label (e.g. `"local"`, `"public-skills-v1"`). |

Optional (set when known):

| Field | Description |
| --- | --- |
| `license` | SPDX identifier of the ingested artefact's license. |
| `license_url` | Canonical URL for the license text. |
| `source_url` | Upstream URL of the artefact. |
| `description` | Short description used as the encoder input body. |

Rows that fail the license whitelist or that are missing the
`source_repo` field are rejected at write time.

## Retention

Local catalogs and runtime directories are retained at the user's
discretion. mind-nerve does not run a background expiry process.

For operators integrating mind-nerve in shared or regulated
environments:

- **Per-user runtime directories** — keep `MIND_NERVE_RUNTIME_DIR`
  scoped to per-user paths; do not share runtime dirs across users on
  multi-tenant hosts.
- **Hook log files** — opt-in only; if enabled, retain only what your
  organisation's policy allows. See
  [`docs/privacy.md`](privacy.md#logging) for what the log records
  (timestamps, projected route IDs, round-trip latency; **not** prompt
  text).
- **Attestation envelopes** — produced in-memory only by mind-nerve.
  If downstream tooling persists them, retention is that tooling's
  responsibility.

## Personal data

The Phase-1 v1.1-oss catalog is curated to avoid personal data. Skill
and tool descriptions are technical text; names of individuals appear
only in publicly-attributed authorship contexts (e.g. `@cputer` in a
public README), not as data subjects. The dataset validator
(`scripts/validate_dataset.py`) checks for obvious email regexes as a
final guard.

If you discover personal data in the published catalog, please report
it to [`info@star.ga`](mailto:info@star.ga) and we will redact and
re-release.

## Data-subject requests

Because mind-nerve is local-only, requests against locally-collected
data are addressed to the operator (the user who ran `mind-nerve learn`
on their host). For the published Phase-1 corpus, contact
[`info@star.ga`](mailto:info@star.ga).

## Audit trail

The published catalog carries:

- A `corpus_hash` (SHA-256 over the deduplicated training TSV) inside
  `manifest.json`.
- A `model_hash` (SHA-256 over the serialised checkpoint).
- A pinned Hugging Face revision recorded by the wheel.

Together these form the integrity chain used by attestation envelopes.
Replaying the same `(corpus_hash, model_hash, hf_revision)` tuple is
the audit primitive for the public Phase-1 artefact.

## Updates and re-releases

If a catalog row must be removed (license dispute, personal-data
report, factual correction), STARGA will:

1. Remove the row from the source TSV.
2. Rebuild `route_table.jsonl` and `route_table.npy`.
3. Recompute `corpus_hash` and `model_hash`.
4. Publish a new Hugging Face revision and bump
   `MIND_NERVE_HF_REVISION`'s pinned default in the next wheel.
5. Note the change in [`CHANGELOG.md`](../CHANGELOG.md) under a
   `Removed` heading.

## Contact

- General governance enquiries: [`info@star.ga`](mailto:info@star.ga)
- License enquiries: [`license@star.ga`](mailto:license@star.ga)
- Security disclosures: see [`SECURITY.md`](../SECURITY.md)
