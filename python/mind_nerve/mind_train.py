"""Public training surface — `mind-nerve train` and `mind_nerve.mind_train`.

v0.3.0-beta.1 ships the **bring-up Python backend** (PyTorch + the
`sentence-transformers` MNR-loss recipe used in Phase 1) under a typed
contract that is forward-compatible with the **native MIND backend**
arriving once `mindc 0.3.0 --emit-shared` produces a callable cdylib.

Public API:

    from mind_nerve.mind_train import TrainConfig, TrainResult, train

    result = train(
        TrainConfig(
            catalog_path=Path("corpus.tsv"),
            output_dir=Path("./run"),
            epochs=3,
            backend="python",
        )
    )
    print(result.model_hash, result.metrics)

Stability promises:

  - `TrainConfig` and `TrainResult` are frozen dataclasses. New fields
    are added with defaults; existing fields are not renamed.
  - `train()` is deterministic given the same `(config, corpus bytes,
    seed)` tuple — the recipe is fixed; only the backend may vary.
  - `model_hash` is the SHA-256 of the saved checkpoint's file bytes,
    bound into the manifest. Same bytes → same hash. Changing recipes
    → new `model_hash`.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

DEFAULT_BASE_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 2e-5
DEFAULT_MAX_LEN = 256
DEFAULT_SEED = 1337
DEFAULT_EVAL_FRAC = 0.1

Backend = Literal["python", "native"]


@dataclass(frozen=True)
class TrainConfig:
    """All knobs the caller can set. Defaults match the Phase 1 recipe."""

    catalog_path: Path
    output_dir: Path
    base_model: str = DEFAULT_BASE_MODEL
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    lr: float = DEFAULT_LR
    max_len: int = DEFAULT_MAX_LEN
    seed: int = DEFAULT_SEED
    eval_frac: float = DEFAULT_EVAL_FRAC
    smoke_test: bool = False
    backend: Backend = "python"


@dataclass(frozen=True)
class TrainResult:
    """What the caller gets back. All numeric fields are JSON-safe."""

    checkpoint_dir: Path
    manifest_path: Path
    model_hash: str
    epochs_completed: int
    train_pairs: int
    eval_pairs: int
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    elapsed_seconds: float
    backend_used: str
    extras: dict[str, Any] = field(default_factory=dict)


def _load_pairs(catalog_path: Path) -> list[tuple[str, str]]:
    """Read `name\\tkind\\tbody` triples; return `(query, positive)` pairs.

    Same parsing rules as the Phase 1 trainer: tab-separated, skip rows
    with empty fields, clip body to 1024 chars to keep MNR contexts
    bounded.
    """
    pairs: list[tuple[str, str]] = []
    with catalog_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) != 3:
                continue
            name, _kind, body = parts
            name = name.strip()
            body = body.strip()
            if len(name) < 2 or len(body) < 16:
                continue
            pairs.append((name, body[:1024]))
    return pairs


def _split_pairs(
    pairs: list[tuple[str, str]],
    eval_frac: float,
    seed: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    rng = random.Random(seed)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    cut = int(len(idx) * (1.0 - eval_frac))
    train = [pairs[i] for i in idx[:cut]]
    evald = [pairs[i] for i in idx[cut:]]
    return train, evald


def _compute_checkpoint_hash(checkpoint_dir: Path) -> str:
    """SHA-256 over every file in checkpoint_dir, in sorted relative-path order."""
    h = hashlib.sha256()
    for p in sorted(checkpoint_dir.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(checkpoint_dir).as_posix().encode("utf-8"))
            h.update(b"\x00")
            h.update(p.read_bytes())
    return h.hexdigest()


def _evaluate_top_k(
    model: Any,
    eval_pairs: list[tuple[str, str]],
    all_positives: list[str],
    device: str,
    k_list: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Top-k accuracy of held-out queries against the full positives pool.

    `all_positives` MUST be ordered so the first `len(eval_pairs)` entries
    are the eval positives (so column ``i`` is the correct answer for
    query ``i``).
    """
    import torch
    import torch.nn.functional as F

    queries = [q for q, _ in eval_pairs]
    q_emb = model.encode(
        queries, batch_size=128, convert_to_tensor=True, show_progress_bar=False, device=device
    )
    p_emb = model.encode(
        all_positives,
        batch_size=128,
        convert_to_tensor=True,
        show_progress_bar=False,
        device=device,
    )
    q_emb = F.normalize(q_emb, dim=-1)
    p_emb = F.normalize(p_emb, dim=-1)
    sims = q_emb @ p_emb.T  # (Q, |corpus|)
    correct = torch.arange(len(eval_pairs), device=device)
    metrics: dict[str, float] = {"candidate_pool": float(len(all_positives))}
    for k in k_list:
        topk = sims.topk(min(k, sims.size(1)), dim=-1).indices
        hit = (topk == correct.unsqueeze(1)).any(dim=-1).float().mean().item()
        metrics[f"top{k}"] = round(hit, 4)
    return metrics


def _train_python_backend(config: TrainConfig) -> TrainResult:
    """Bring-up backend: PyTorch + sentence-transformers MNR loss.

    Faithful port of `catalog-builder/train_phase1.py` to a public,
    typed surface. Will be retired once the native MIND backend can
    produce an equivalent checkpoint.
    """
    import torch
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    random.seed(config.seed)
    os.environ["PYTHONHASHSEED"] = str(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = _load_pairs(config.catalog_path)
    if not pairs:
        raise RuntimeError(
            f"catalog at {config.catalog_path} produced 0 usable pairs; "
            "check the tab-separated name/kind/body schema"
        )
    train_set, eval_set = _split_pairs(pairs, config.eval_frac, config.seed)

    epochs = config.epochs
    if config.smoke_test:
        train_set = train_set[:500]
        eval_set = eval_set[:200]
        epochs = 1

    model = SentenceTransformer(config.base_model, device=device)
    model.max_seq_length = config.max_len

    train_examples = [InputExample(texts=[q, p]) for q, p in train_set]
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=config.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = config.output_dir / "checkpoint"

    # Eval positives sit at the head of all_positives so the column
    # index of each eval query's correct answer equals its row index.
    all_positives = [p for _, p in eval_set] + [p for _, p in train_set]
    baseline = _evaluate_top_k(model, eval_set, all_positives, device)

    t0 = time.time()
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=epochs,
        warmup_steps=max(1, int(len(train_loader) * 0.1)),
        optimizer_params={"lr": config.lr},
        show_progress_bar=False,
        output_path=str(checkpoint_dir),
        use_amp=True,
    )
    elapsed = time.time() - t0

    final = _evaluate_top_k(model, eval_set, all_positives, device)
    model_hash = _compute_checkpoint_hash(checkpoint_dir)

    manifest = {
        "schema_version": 2,
        "phase": "0.3.0-beta.1",
        "backend": "python",
        "base_model": config.base_model,
        "epochs": epochs,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "max_len": config.max_len,
        "seed": config.seed,
        "eval_frac": config.eval_frac,
        "train_pairs": len(train_set),
        "eval_pairs": len(eval_set),
        "baseline_metrics": baseline,
        "final_metrics": final,
        "elapsed_seconds": round(elapsed, 2),
        "model_hash": model_hash,
        "trained_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "torch_version": torch.__version__,
    }
    manifest_path = config.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (config.output_dir / "eval.json").write_text(json.dumps(final, indent=2) + "\n")

    return TrainResult(
        checkpoint_dir=checkpoint_dir,
        manifest_path=manifest_path,
        model_hash=model_hash,
        epochs_completed=epochs,
        train_pairs=len(train_set),
        eval_pairs=len(eval_set),
        metrics=final,
        baseline_metrics=baseline,
        elapsed_seconds=round(elapsed, 2),
        backend_used="python",
        extras={"device": device, "torch_version": torch.__version__},
    )


def _train_native_backend(config: TrainConfig) -> TrainResult:
    raise NotImplementedError(
        "Native MIND backend lands with mindc 0.3.0 --emit-shared cdylib + the "
        "Q16.16 native kernel. v0.3.0-beta.1 ships the Python (PyTorch) bring-up "
        "backend; set TrainConfig(backend='python')."
    )


def train(config: TrainConfig) -> TrainResult:
    """Train a mind-nerve encoder according to ``config``.

    Backend resolution:
      - ``python`` (default) — PyTorch + sentence-transformers bring-up.
        Available now.
      - ``native``           — Q16.16 native MIND kernel via mindc 0.3.0
        cdylib. Raises ``NotImplementedError`` until that lands.
    """
    if config.backend == "python":
        return _train_python_backend(config)
    if config.backend == "native":
        return _train_native_backend(config)
    raise ValueError(f"unknown backend: {config.backend!r}")


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    """JSON-safe view of a TrainConfig (Path → str)."""
    d = asdict(config)
    d["catalog_path"] = str(d["catalog_path"])
    d["output_dir"] = str(d["output_dir"])
    return d


__all__ = [
    "Backend",
    "TrainConfig",
    "TrainResult",
    "DEFAULT_BASE_MODEL",
    "DEFAULT_EPOCHS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_LR",
    "DEFAULT_MAX_LEN",
    "DEFAULT_SEED",
    "DEFAULT_EVAL_FRAC",
    "train",
    "config_to_dict",
]
