# BPE tokenizer v1.0

- **Trained:** 2026-05-16T02:28:07Z (UTC)
- **Status:** draft-unsigned. `manifest.sig` is a placeholder.
- **Location:** `/data/datasets/mind-nerve-catalog/tokenizer/v1.0/`
- **Manifest:** [`tokenizer-v1.0.manifest.json`](./tokenizer-v1.0.manifest.json)
  (copied here for provenance).
- **Bound to:** catalog [`v1.0`](./v1.0.md)
  (corpus derived from catalog-v1.0 `items.jsonl`).

## Identity

| Field | Value |
|---|---|
| `tokenizer_sha256` | `1b9ebc24b712e10fdc71e44dafe9882a18c255dd7e46793be35439d72fad71f3` |
| `corpus_sha256` (derived from freeze) | (see manifest) |
| `vocab_size` (target) | 16,384 |
| `vocab_size` (actual after training) | (see manifest — typically 16,384) |
| `model_kind` | BPE (byte-level) |
| `byte_fallback` | true (no `<unk>` ever emitted) |
| `pre_tokenizer` | ByteLevel (GPT-2-style) |
| `normalizer` | NFC |
| `training_seconds` | ~1.7 |
| `corpus_bytes` | 12,704,103 |

## Reserved special tokens (locked at v1.0)

| ID | Token | Use |
|---|---|---|
| 0 | `[PAD]` | padding |
| 1 | `[BOS]` | beginning of sequence |
| 2 | `[EOS]` | end of sequence |
| 3 | `[QUERY]` | query-side marker (mind-nerve InfoNCE pairing) |
| 4 | `[POSITIVE]` | positive-pair marker |
| 5 | `[SINK]` | RFC-007 sink position |
| 6 | `[MASK]` | RFC-021 / RFC-022 reserved |
| 7 | `[SEP]` | separator (reserved) |

Token IDs 0-7 are frozen. Any future re-train must preserve these.

## Smoke test

Input: `search for all repos that offer skill collections and add them to data SSD`

Output: 15 tokens; canonical-looking word splits with byte-level
continuation markers; no `<unk>`.

## Why these choices

- **Byte-level BPE.** Guarantees no unknown tokens; every UTF-8 byte
  is encodable. Critical for an agent-CLI distribution that includes
  paths, code, and multi-language fragments.
- **Vocab 16,384.** Larger than a code-only tokenizer (BPE 8k is
  typical there), smaller than a general LLM tokenizer (50-100k).
  Chosen for the agent-CLI request distribution where tool names,
  short descriptions, and code references dominate.
- **NFC normalizer.** Ensures the same visual string always tokenises
  the same way — preserves cross-arch byte-identity at the input
  layer.
- **`byte_fallback=true`.** Hard guarantee that encode → decode is
  lossless for any UTF-8 input.

## Binding into `model_hash`

mind-nerve binds `tokenizer_sha256` into its `model_hash` via the
model manifest header (alongside `corpus_hash` and `architecture_hash`).
Different tokenizer = different model_hash = different evidence
envelope. The training pipeline must refuse to load a tokenizer whose
sha256 doesn't match the model's bound hash.

## Reproducibility

```bash
# from ./catalog-builder/
python3 build_corpus.py          # rebuild corpus.txt from frozen catalog
python3 train_bpe.py --version v1.0 --vocab-size 16384
```

Same corpus + same vocab + same min_frequency = byte-identical
`tokenizer.json`. The `tokenizers` library trains
deterministically given a single thread; multi-thread training can
introduce non-determinism — keep that disabled at release time.

## Open caveats

- Sub-2-second training implies the corpus is small. Phase 2 will
  enlarge the corpus across the 12 Tier-1 languages (see
  [`spec/quality_targets.md`](../../spec/quality_targets.md)
  §"Multilingual language policy"), additional skill repos, and mined
  CLI traces, then retrain to `v1.1` or `v2.0`. The 32k vocab is
  likely insufficient for proper CJK + Devanagari + Arabic-script
  coverage; expansion to 48k or 64k is a tracked decision in the
  Phase 2 catalog-builder workstream.
- Special-token IDs are frozen even if the vocab grows; new
  special tokens enter at high IDs only.
- The byte-level GPT-2 mark (`Ġ`) is part of the tokenizer's encoded
  vocab; downstream consumers must use the decoder rather than
  string-slicing to recover the original text.
