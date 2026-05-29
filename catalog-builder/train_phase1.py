#!/usr/bin/env python3
"""mind-nerve Phase 1 — PyTorch fine-tune on catalog-v1.0.

Per docs/catalog_and_training_plan.md: Phase 1 trains in PyTorch
internally. Phase 2 retrains under native MIND once mind-train v0.1
ships. This script is the Phase 1 path.

Recipe
------
- Base encoder: BAAI/bge-small-en-v1.5 (33M params, fits the
  sub-50M envelope, well-tested sentence-transformer base).
- Loss: MultipleNegativesRankingLoss (HF ST's InfoNCE variant).
- Pairs: (query=item name, positive=item body) for every entry in
  catalog-v1.0 frozen items.jsonl. In-batch negatives via MNR loss.
- 90/10 train/eval split, deterministic seed.
- Eval: top-1 / top-5 / top-10 on the held-out 10%.

Outputs
-------
  catalog-data/phase1/
    ├── checkpoint/       (sentence-transformers save_directory)
    ├── train.log
    ├── eval.json
    └── manifest.json     (corpus_hash + tokenizer_hash + model_hash binding)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

CORPUS = Path("catalog-data/tokenizer/corpus.txt")
FREEZE_MANIFEST = Path("catalog-data/freeze/v1.0/manifest.json")
TOKENIZER_MANIFEST = Path("catalog-data/tokenizer/v1.0/manifest.json")
OUT_ROOT = Path("catalog-data/phase1")

DEFAULT_BASE = "BAAI/bge-small-en-v1.5"
DEFAULT_EPOCHS = 3
DEFAULT_BATCH = 32
DEFAULT_LR = 2e-5
DEFAULT_MAX_LEN = 256
SEED = 1337


def load_pairs() -> list[tuple[str, str]]:
    """One (query, positive) pair per catalog item."""
    pairs: list[tuple[str, str]] = []
    with CORPUS.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) != 3:
                continue
            name, kind, body = parts
            name = name.strip()
            body = body.strip()
            if len(name) < 2 or len(body) < 16:
                continue
            # Keep body reasonable for fine-tuning context.
            body = body[:1024]
            pairs.append((name, body))
    return pairs


def split(pairs, eval_frac=0.1, seed=SEED):
    rng = random.Random(seed)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    cut = int(len(idx) * (1 - eval_frac))
    train = [pairs[i] for i in idx[:cut]]
    evald = [pairs[i] for i in idx[cut:]]
    return train, evald


def evaluate(model, eval_pairs, all_positives, device, k_list=(1, 5, 10)):
    """Top-k accuracy: for each held-out query, rank its true positive
    against the *entire* catalog's positives (not just the eval batch).

    `eval_pairs`     : list of (query, positive) for the held-out set
    `all_positives`  : list of *every* positive in the catalog (corpus-wide)
                       The first len(eval_pairs) entries must be the eval
                       positives so the index match works.
    """
    import torch
    queries = [q for q, _ in eval_pairs]
    q_emb = model.encode(queries, batch_size=128, convert_to_tensor=True,
                         show_progress_bar=False, device=device)
    p_emb = model.encode(all_positives, batch_size=128, convert_to_tensor=True,
                         show_progress_bar=False, device=device)
    q_emb = torch.nn.functional.normalize(q_emb, dim=-1)
    p_emb = torch.nn.functional.normalize(p_emb, dim=-1)
    sims = q_emb @ p_emb.T                     # [Q, |corpus|]
    correct_idx = torch.arange(len(eval_pairs), device=device)
    out = {"candidate_pool": len(all_positives)}
    for k in k_list:
        topk = sims.topk(min(k, sims.size(1)), dim=-1).indices
        hit = (topk == correct_idx.unsqueeze(1)).any(dim=-1).float().mean().item()
        out[f"top{k}"] = round(hit, 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=DEFAULT_BASE)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    ap.add_argument("--out-version", default="v1.0")
    ap.add_argument("--smoke-test", action="store_true",
                    help="Use 500 pairs and 1 epoch; ~1 min total runtime.")
    args = ap.parse_args()

    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)

    import torch
    from torch.utils.data import DataLoader
    from sentence_transformers import SentenceTransformer, InputExample, losses

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[phase1] device={device} torch={torch.__version__}", file=sys.stderr)

    pairs = load_pairs()
    print(f"[phase1] loaded {len(pairs)} (query, positive) pairs", file=sys.stderr)

    train, evald = split(pairs)
    if args.smoke_test:
        train = train[:500]
        evald = evald[:200]
        args.epochs = 1
    print(f"[phase1] train={len(train)} eval={len(evald)} epochs={args.epochs}", file=sys.stderr)

    print(f"[phase1] loading base model: {args.base_model}", file=sys.stderr)
    model = SentenceTransformer(args.base_model, device=device)
    model.max_seq_length = args.max_len

    train_examples = [InputExample(texts=[q, p]) for q, p in train]
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=args.batch)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    out_dir = OUT_ROOT / args.out_version
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reorder so eval positives sit at the first len(evald) indices of
    # the full-corpus positives list — needed for top-k bookkeeping.
    eval_positives = [p for _, p in evald]
    train_positives = [p for _, p in train]
    all_positives = eval_positives + train_positives

    print(f"[phase1] baseline eval (full-catalog candidate pool of {len(all_positives)}) ...", file=sys.stderr)
    base_metrics = evaluate(model, evald, all_positives, device)
    print(f"[phase1] baseline: {base_metrics}", file=sys.stderr)

    print(f"[phase1] fine-tuning ...", file=sys.stderr)
    t0 = time.time()
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=args.epochs,
        warmup_steps=max(1, int(len(train_loader) * 0.1)),
        optimizer_params={"lr": args.lr},
        show_progress_bar=True,
        output_path=str(out_dir / "checkpoint"),
        use_amp=True,
    )
    train_seconds = time.time() - t0

    print(f"[phase1] post-train eval (full-catalog pool) ...", file=sys.stderr)
    final_metrics = evaluate(model, evald, all_positives, device)
    print(f"[phase1] final: {final_metrics}", file=sys.stderr)

    freeze_meta = json.loads(FREEZE_MANIFEST.read_text())
    tok_meta = json.loads(TOKENIZER_MANIFEST.read_text()) if TOKENIZER_MANIFEST.exists() else {}

    # Hash the saved checkpoint dir's tokenizer + model files for an
    # approximate model_hash. Not bundle-v3 yet — that's a later step.
    ckpt_dir = out_dir / "checkpoint"
    model_files = sorted(p for p in ckpt_dir.rglob("*") if p.is_file())
    h = hashlib.sha256()
    for p in model_files:
        h.update(p.read_bytes())
    model_hash = h.hexdigest()

    manifest = {
        "schema_version": 1,
        "phase": 1,
        "phase1_version": args.out_version,
        "base_model": args.base_model,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "max_len": args.max_len,
        "seed": SEED,
        "train_pairs": len(train),
        "eval_pairs": len(evald),
        "train_seconds": round(train_seconds, 1),
        "baseline_metrics": base_metrics,
        "final_metrics": final_metrics,
        "improvement": {k: round(final_metrics[k] - base_metrics[k], 4) for k in final_metrics},
        "corpus_hash": freeze_meta.get("freeze_id"),
        "catalog_version": freeze_meta.get("catalog_version"),
        "tokenizer_hash": tok_meta.get("tokenizer_sha256"),
        "model_hash": model_hash,
        "trained_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "torch_version": torch.__version__,
        "note": "Phase 1 = PyTorch internal training per docs/catalog_and_training_plan.md. "
                "Phase 2 retrains under native MIND via mind-train v0.1.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (out_dir / "eval.json").write_text(json.dumps(final_metrics, indent=2) + "\n")

    print()
    print(json.dumps({
        "phase1_version": args.out_version,
        "out_dir": str(out_dir),
        "model_hash": model_hash[:16] + "…",
        "metrics": final_metrics,
        "baseline": base_metrics,
        "improvement": manifest["improvement"],
        "train_seconds": manifest["train_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
