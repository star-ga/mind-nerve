# mind-nerve distribution plan

## The problem solved tonight

`pip install mind-nerve` ships the wheel but the runtime expects weights
at a local path (`catalog-data/phase1/...`). For an
end-user, this is a broken UX. We need a weights-delivery story.

## Three-tier delivery (preferred)

| Tier | Where the weights live | When used |
|---|---|---|
| **1. Bundled (small)** | Inside the wheel under `mind_nerve/data/` | Default for `pip install mind-nerve` |
| **2. Lazy-fetched (HF)** | `huggingface.co/star-ga/mind-nerve-encoder-v1.1-oss` | When the user opts into a larger / newer model |
| **3. Local override** | `MIND_NERVE_RUNTIME_DIR` env var | Developers / STARGA dogfood — the tier 1+2 paths are bypassed |

### Tier 1 — Bundled

What ships in the wheel:

- `mind_nerve/data/tokenizer.json` (~1.1 MB)
- `mind_nerve/data/route_table.npy` (~18 MB)
- `mind_nerve/data/route_table.jsonl` (~3.7 MB)
- `mind_nerve/data/manifest.json` (~2 KB)
- Encoder weights: see below.

The BGE-small encoder is 33M params × FP32 = ~130 MB. Too big for
PyPI hospitality. Two options for the bundled tier:

- **A. INT8 quantize the encoder** (33 MB) — ships in the wheel.
  Quality loss expected to be < 1 point top-5. Phase 1 patches.
- **B. Skip bundling encoder weights**; on first `route()`, lazy-download
  from HuggingFace and cache to `~/.cache/mind-nerve/`.

Default for v0.1.0-alpha.2: **B**. The wheel is small (~25 MB total),
the first call is slow (~10 s download), every later call is fast.

### Tier 2 — HuggingFace upload

Public repo: `huggingface.co/star-ga/mind-nerve-encoder-v1.1-oss`

Required artefacts at HF:

```
pytorch_model.bin / model.safetensors    (BGE-small fine-tuned)
config.json
tokenizer.json                              (catalog-v1.1-oss tokenizer)
tokenizer_config.json                       (HF wrapper config)
special_tokens_map.json
modules.json                                (sentence-transformers metadata)
sentence_bert_config.json
1_Pooling/config.json
README.md (model card)
```

Plus a sibling **dataset** repo `huggingface.co/datasets/star-ga/
mind-nerve-catalog-v1.1-oss` containing `items.jsonl` and the manifest.
Per-source-repo license attribution lives in `LICENSE-SOURCES.md` (the
121 cloned repos plus the OSS-cleared local skills).

Both repos: license **Apache-2.0**.

### Tier 3 — Local override

`MIND_NERVE_RUNTIME_DIR=/path/to/runtime` is already honoured by
`mind_nerve.inference.load_default_runtime`. No code change needed.

## Upload steps (operator action — not automated tonight)

```bash
# Once HF token is on host:
export HF_TOKEN="..."
pip install huggingface_hub

# 1. Encoder repo
huggingface-cli upload star-ga/mind-nerve-encoder-v1.1-oss \
  catalog-data/phase1/v1.1-oss/checkpoint/ \
  --repo-type model --commit-message "v1.1-oss initial"

# 2. Catalog dataset repo
huggingface-cli upload-large-folder star-ga/mind-nerve-catalog-v1.1-oss \
  catalog-data/freeze/v1.1-oss/ \
  --repo-type dataset --commit-message "v1.1-oss initial"

# 3. Tokenizer
huggingface-cli upload star-ga/mind-nerve-encoder-v1.1-oss \
  catalog-data/tokenizer/v1.1-oss/tokenizer.json \
  tokenizer.json --repo-type model
```

## Lazy-fetch implementation (Python side)

`mind_nerve.inference._Runtime` learns a fallback:

```python
def _Runtime.__init__(self, runtime_dir):
    if not (runtime_dir / "checkpoint").exists():
        runtime_dir = _materialise_from_hf("star-ga/mind-nerve-encoder-v1.1-oss")
    ...
```

`_materialise_from_hf` uses `huggingface_hub.snapshot_download` and
caches to `~/.cache/mind-nerve/v1.1-oss/`. First call: download + warm.
Subsequent calls: zero network.

## License + provenance discipline at the HF level

- **Wheel itself**: Apache-2.0 (the Python source).
- **Encoder weights on HF**: Apache-2.0 (derivative of BGE-small Apache-2.0).
- **Catalog dataset on HF**: per-source-repo licensing tracked in
  `LICENSE-SOURCES.md`. The whole catalog is redistributable under the
  union of upstream licenses (we verified at v1.1-oss filter time that
  no STARGA-private content remains).
- **Excluded from the public release**: STARGA-private skills (222),
  commercial-marker skills (12), leaked-system-prompts collections (300),
  off-domain image-prompt collections (7). All retained in the private
  catalog-v1.0 dogfood corpus.

## What does NOT get distributed

- `libmindnerve.so` (the protected native runtime) — STARGA Commercial,
  not part of the public OSS wheel. Phase 2.
- Private MIND source (`src/*.mind`) — never in the wheel.
- The STARGA-Inc skill catalog in private form (`catalog-v1.0`) — only
  the OSS-cleaned `catalog-v1.1-oss` ships.
- Internal-marker docs (autoresearch RFCs, protected-build internals) —
  stay private.

## Status

- Tier 1 + Tier 3 work today (with `MIND_NERVE_RUNTIME_DIR`).
- Tier 2 lazy-fetch is *spec-only* until an operator uploads to HF.
- The wheel currently does NOT include `mind_nerve/data/*` — that lands
  with the Tier-1 INT8 quantization work in the next iteration.
