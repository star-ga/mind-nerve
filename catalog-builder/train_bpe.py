#!/usr/bin/env python3
"""Train the mind-nerve custom BPE tokenizer.

Uses HuggingFace `tokenizers` with byte-level pre-tokenization so the
vocabulary covers any UTF-8 byte (no unknown tokens). Sized for the
agent-CLI request distribution — vocab=16384, not LLM-scale.

Reserves 8 special token IDs that mind-nerve's encoder relies on:

    0  [PAD]      padding
    1  [BOS]      beginning of sequence
    2  [EOS]      end of sequence
    3  [QUERY]    query-side marker
    4  [POSITIVE] positive-pair marker
    5  [SINK]     RFC-007 sink token
    6  [MASK]     RFC-021/022 reserved
    7  [SEP]      reserved separator

Outputs:
  /data/datasets/mind-nerve-catalog/tokenizer/v1.0/
    ├── tokenizer.json     (HF tokenizers format)
    ├── manifest.json      (vocab size, hash, training config)
    └── manifest.sig       (STARGA HMAC placeholder)
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path("/data/datasets/mind-nerve-catalog/tokenizer")
CORPUS = ROOT / "corpus.txt"

# Default vocab; can be overridden by CLI.
DEFAULT_VOCAB = 16384

SPECIAL_TOKENS = [
    "[PAD]", "[BOS]", "[EOS]",
    "[QUERY]", "[POSITIVE]", "[SINK]",
    "[MASK]", "[SEP]",
]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1.0")
    ap.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB)
    ap.add_argument("--min-frequency", type=int, default=2)
    args = ap.parse_args()

    if not CORPUS.exists():
        sys.exit(f"corpus.txt not found at {CORPUS}; run build_corpus.py first")

    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, normalizers

    tok = Tokenizer(models.BPE(unk_token=None, byte_fallback=True))
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print(f"training BPE: vocab={args.vocab_size} min_freq={args.min_frequency}", file=sys.stderr)
    started = time.time()
    tok.train(files=[str(CORPUS)], trainer=trainer)
    elapsed = time.time() - started

    out_dir = ROOT / args.version
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_path = out_dir / "tokenizer.json"
    tok.save(str(tok_path))

    tok_bytes = tok_path.read_bytes()
    tok_hash = hashlib.sha256(tok_bytes).hexdigest()
    corpus_hash = hashlib.sha256(CORPUS.read_bytes()).hexdigest()

    manifest = {
        "schema_version": 1,
        "tokenizer_version": args.version,
        "vocab_size": args.vocab_size,
        "actual_vocab_size": tok.get_vocab_size(),
        "special_tokens": SPECIAL_TOKENS,
        "min_frequency": args.min_frequency,
        "model_kind": "BPE",
        "pre_tokenizer": "ByteLevel",
        "byte_fallback": True,
        "corpus_path": str(CORPUS),
        "corpus_sha256": corpus_hash,
        "corpus_bytes": CORPUS.stat().st_size,
        "tokenizer_sha256": tok_hash,
        "tokenizer_bytes": len(tok_bytes),
        "trained_at": int(time.time()),
        "trained_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "training_seconds": round(elapsed, 2),
        "signing": {
            "algorithm": "HMAC-SHA256",
            "key_id": "STARGA-ROOT-2026",
            "status": "draft-unsigned",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (out_dir / "manifest.sig").write_text(
        "DRAFT-UNSIGNED\n"
        f"tokenizer_sha256: {tok_hash}\n"
        "To sign: overwrite this file with the HMAC-SHA256 of manifest.json "
        "computed with STARGA-ROOT-2026.\n"
    )

    sample = "search for all repos that offer skill collections and add them to data SSD"
    encoded = tok.encode(sample)

    print(json.dumps({
        **manifest,
        "smoke_test": {
            "input":  sample,
            "ids":    encoded.ids[:20],
            "tokens": encoded.tokens[:20],
            "n_tokens": len(encoded.ids),
        },
    }, indent=2))


if __name__ == "__main__":
    main()
