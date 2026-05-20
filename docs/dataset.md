# Dataset

This document describes the routing-catalog dataset that mind-nerve trains
and routes against. It is the contract between the corpus, the trainer
(`mind_train.train`), and the runtime (`mind_nerve.route`).

The Phase-1 public-clean catalog shipped with the
[`star-ga/mind-nerve`](https://huggingface.co/star-ga/mind-nerve)
weights is **`route_table.jsonl` v1.0**: **11,922 routes**, frozen and
signed.

## Files

A frozen runtime directory contains three load-bearing artefacts plus
optional priors:

| File                          | Required | Description |
| ----------------------------- | -------- | ----------- |
| `manifest.json`               | yes      | Catalog version, model version, base model, training hyperparameters, `model_hash`, corpus hash, build timestamp. |
| `route_table.jsonl`           | yes      | One JSON record per route (id, name, kind, source_repo, license, description). Ordered; line index ↔ embedding row. |
| `route_table.npy`             | yes      | `float32` matrix of L2-normalised route embeddings; shape `(N_routes, hidden)`. Row order matches `route_table.jsonl`. |
| `route_table_prior.npy`       | no       | Optional Laplace prior over routes (catalog-v2 / SOTA-track #1). |
| `route_table_freq_scale.npy`  | no       | Optional frequency-adaptive per-route scale (SOTA-track #4). |
| `stride_thresholds.json`      | no       | Entropy → stride map for the native MIND encoder (SOTA-track #3). |

The runtime resolves the runtime directory in this order:

1. The explicit `runtime_dir=` argument to `route(...)` or `load_default_runtime(...)`.
2. The `MIND_NERVE_RUNTIME_DIR` environment variable.
3. `~/.local/share/mind-nerve/runtime/` (auto-seeded from Hugging Face on
   first use; revision pinned by `MIND_NERVE_HF_REVISION`).

## Corpus schema (training input)

The trainer (`mind-nerve train`) expects a TSV file: one record per line,
exactly three tab-separated fields, no header.

```
name<TAB>kind<TAB>body
```

| Field  | Type   | Notes |
| ------ | ------ | ----- |
| `name` | string | Human-readable identifier — used as the route name surfaced to users. |
| `kind` | string | One of `skill`, `tool`, `mcp`, `agent`, `command`. |
| `body` | string | The free-text description that the encoder is trained against. |

### Normalisation and filtering

The trainer applies the following passes before pair construction (see
`mind_train._load_pairs`):

- **Strip surrounding whitespace** from each field; lines that do not have
  exactly three fields are skipped.
- **Clip `body` to 1024 characters.** Anything beyond that is dropped to
  match the encoder's worst-case 1024-token request budget.
- **Skip short examples.** Rows whose `body` is shorter than the
  minimum-length threshold are dropped — they do not produce useful
  positive pairs.
- **Lowercase normalisation is _not_ applied** — the encoder's BPE
  tokenizer handles casing.

### Deduplication

Catalog construction de-duplicates by the SHA-256 of
`f"{name}\t{kind}\t{body}"`. Two routes that differ only by whitespace or
casing collapse to one; the first occurrence wins. The deduplicated row
count is recorded in `manifest.json` under `train_pairs` and `eval_pairs`.

### Validation

Before training, run the dataset validator:

```bash
python scripts/validate_dataset.py data/catalog.tsv
```

The validator checks:

- Exactly three TSV fields per line.
- `name` length ≥ 2 characters.
- `body` length ≥ 16 characters and ≤ 100 000 characters.
- No obvious PII (email regex) in `body` — emails should be redacted
  before training.
- No duplicate rows (by full-row SHA-256).

It emits a JSON summary and exits non-zero on the first failure.

## Source collection rules

The Phase-1 v1.1-oss catalog is collected from:

- The `~/.agents/skills/` directory contributed by published
  STARGA-public skill packs.
- Open-source MCP server READMEs (description, capabilities sections).
- Open-source CLI tool documentation surfaces (one-line description plus
  long description).

Discovery defaults to OSS-compatible licenses only (`Apache-2.0`, `MIT`,
`BSD-2-Clause`, `BSD-3-Clause`, `MPL-2.0`, `ISC`, `CC0-1.0`). The
`include_unknown=False` default in `mind_nerve.discovery.scan` excludes any
artefact whose license cannot be identified.

## Train / eval split

The trainer splits the deduplicated row set into train and eval pools
using `seed` from `TrainConfig` (default `1337`). The split fraction is
`eval_frac` (default `0.1`). The procedure:

1. Sort rows by their SHA-256 hash to get a deterministic ordering
   independent of file ingestion order.
2. Take the last `round(eval_frac * N)` rows as the eval pool; the
   remainder is the train pool.
3. Build positive pairs `(name, body)` from each pool.

Both pool sizes are recorded in the per-run manifest under `train_pairs`
and `eval_pairs`. Evaluation reports top-1 / top-5 / top-10 over the eval
queries against the full positives pool. (MRR and per-slice calibration
are on the Phase-2 roadmap; see [`ROADMAP.md`](../ROADMAP.md).)

## Provenance fields in `manifest.json`

Every produced manifest carries the following fields, at minimum:

- `catalog_version` — e.g. `"v1.1-oss"`
- `model_version` — git-style tag of the trained checkpoint
- `base_model` — e.g. `"BAAI/bge-small-en-v1.5"`
- `epochs`, `batch_size`, `lr`, `max_len`, `seed`, `eval_frac`
- `model_hash` — SHA-256 over the serialised checkpoint
- `corpus_hash` — SHA-256 over the full deduplicated input TSV
- `device`, `torch_version`, `elapsed_seconds`, `timestamp`

The `model_hash` and `corpus_hash` together form the integrity chain used
by the attestation envelope (see `python/mind_nerve/attestation.py`).

## License posture

The Phase-1 v1.1-oss catalog and its trained weights are **Apache-2.0**.
The training corpus is curated to be **public-clean** (no STARGA-private
content). See [`docs/data_governance.md`](data_governance.md) for the
full license posture and exclusion rules.

## Reproducing the catalog

The full reproduction path will land once the dependency lockfile and the
catalog-builder published manifest are pinned (see audit P0
"Dependencies and security" and "Reproducibility and data"). Today, the
catalog can be re-derived from the published `route_table.jsonl` plus the
training TSV referenced in `manifest.json`. The
[`catalog-builder/`](../catalog-builder) directory contains the discovery
and assembly scripts.
