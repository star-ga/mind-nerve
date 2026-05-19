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
import platform
import random
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .eval_metrics import expected_calibration_error, mrr, ndcg_at_k

DEFAULT_BASE_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 2e-5
DEFAULT_MAX_LEN = 256
DEFAULT_SEED = 1337
DEFAULT_EVAL_FRAC = 0.1
DEFAULT_DETERMINISTIC = True

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
    deterministic: bool = DEFAULT_DETERMINISTIC


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
    """Retrieval metrics for held-out queries against the full positives pool.

    `all_positives` MUST be ordered so the first ``len(eval_pairs)`` entries
    are the eval positives (so column ``i`` is the correct answer for query
    ``i``).

    Emits, in addition to historical top-k accuracy:

      * ``mrr``                  — mean reciprocal rank over the full pool
      * ``ndcg@1``/``ndcg@5``/``ndcg@10`` — mean nDCG@k with binary
        relevance
      * ``ece``                  — expected calibration error using the
        top-1 softmax-normalized similarity score as a confidence proxy

    The accuracy fields ``top1``/``top5``/``top10`` and ``candidate_pool``
    remain present and unchanged so downstream consumers do not regress.
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
    n_eval = len(eval_pairs)
    correct = torch.arange(n_eval, device=device)
    metrics: dict[str, float] = {"candidate_pool": float(len(all_positives))}
    max_k = min(max(k_list + (10,)), sims.size(1))
    full_topk = sims.topk(max_k, dim=-1).indices  # (Q, max_k)

    # Classic top-k hit-rate (preserved verbatim).
    for k in k_list:
        kk = min(k, sims.size(1))
        topk = full_topk[:, :kk]
        hit = (topk == correct.unsqueeze(1)).any(dim=-1).float().mean().item()
        metrics[f"top{k}"] = round(hit, 4)

    # MRR + nDCG@k via the pure-Python reference implementations.
    ranked_lists = full_topk.detach().cpu().tolist()
    ground_truth = list(range(n_eval))
    metrics["mrr"] = round(mrr(ranked_lists, ground_truth), 6)
    for k in k_list:
        metrics[f"ndcg@{k}"] = round(ndcg_at_k(ranked_lists, ground_truth, k=k), 6)

    # ECE on the top-1 softmax confidence vs whether the top-1 row index
    # matches the ground-truth column. Cosine similarities live in
    # [-1, 1]; softmax normalizes them into a confidence distribution and
    # we keep only the max per query.
    probs = torch.softmax(sims.float(), dim=-1)
    top1_conf, top1_idx = probs.max(dim=-1)
    scores_np = top1_conf.detach().cpu().numpy()
    correct_np = (top1_idx == correct).detach().cpu().numpy().astype("int64")
    metrics["ece"] = round(expected_calibration_error(scores_np, correct_np, n_bins=10), 6)
    return metrics


def _git_sha(repo_root: Path | None = None) -> str | None:
    """Return ``HEAD`` git SHA for ``repo_root`` (cwd by default) or None."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root) if repo_root is not None else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    sha = proc.stdout.strip()
    return sha or None


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_requirements_lock(start: Path) -> Path | None:
    """Walk upward from ``start`` looking for a ``requirements.lock`` sibling."""
    cur = start.resolve()
    for parent in (cur, *cur.parents):
        candidate = parent / "requirements.lock"
        if candidate.is_file():
            return candidate
    return None


def _cpu_info_line() -> str:
    """Best-effort one-line CPU descriptor (`platform.processor()` fallback)."""
    info = platform.processor() or ""
    try:
        if Path("/proc/cpuinfo").is_file():
            with Path("/proc/cpuinfo").open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        _, _, value = line.partition(":")
                        info = value.strip() or info
                        break
    except OSError:
        pass
    return info or "unknown"


def _runtime_environment_facts() -> dict[str, Any]:
    """Collect host-level facts useful for reproducing a training run."""
    import torch  # local import to keep this module importable without torch

    cuda_version: str | None = None
    try:
        cuda_version = torch.version.cuda  # type: ignore[attr-defined]
    except Exception:
        cuda_version = None

    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": cuda_version,
        "cpu_info": _cpu_info_line(),
    }


def _apply_deterministic_flags(seed: int) -> None:
    """Seed every RNG mind-nerve depends on and pin deterministic algorithms.

    Documented cost: ``torch.use_deterministic_algorithms(True, warn_only=True)``
    plus disabling cuDNN autotuning typically slows GPU training by 10-25% on
    the BGE-small recipe; the CPU path is unaffected. Set
    ``TrainConfig(deterministic=False)`` to opt out (and lose run-to-run
    bit-equality).
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, RuntimeError):
        # Older Torch builds may not expose the flag.
        pass
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except AttributeError:
        pass
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _emit_run_json(
    output_dir: Path,
    config: TrainConfig,
    dataset_manifest_sha256: str | None,
    manifest_payload: dict[str, Any],
    started_at_iso: str,
    finished_at_iso: str,
) -> Path:
    """Write ``run.json`` next to the training artifact.

    ``run.json`` is the single reproducibility manifest required by the
    public audit gate. It captures the source commit, the dependency lock
    hash, the dataset hash, the runtime model revision, every hyper-
    parameter, and the host/CPU/CUDA fingerprint of the producer.
    """
    runtime = _runtime_environment_facts()
    lock_path = _find_requirements_lock(Path(__file__).parent)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "mind_nerve.train.run",
        # Source identity
        "git_sha": _git_sha(),
        "requirements_lock_sha256": _sha256_file(lock_path) if lock_path else None,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "hf_revision": os.environ.get("MIND_NERVE_HF_REVISION") or None,
        # Hyper-parameters
        "seed": config.seed,
        "epochs": manifest_payload.get("epochs", config.epochs),
        "batch_size": config.batch_size,
        "lr": config.lr,
        "max_length": config.max_len,
        "eval_fraction": config.eval_frac,
        "deterministic": config.deterministic,
        "backend": manifest_payload.get("backend", config.backend),
        "base_model": config.base_model,
        # Host facts
        "hostname": runtime["hostname"],
        "platform": runtime["platform"],
        "python_version": runtime["python_version"],
        "torch_version": runtime["torch_version"],
        "cuda_available": runtime["cuda_available"],
        "cuda_version": runtime["cuda_version"],
        "cpu_info": runtime["cpu_info"],
        # Time
        "started_at": started_at_iso,
        "finished_at": finished_at_iso,
        # Augmentation of manifest.json (existing fields preserved verbatim)
        "manifest": manifest_payload,
    }
    run_path = output_dir / "run.json"
    run_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return run_path


def _train_python_backend(config: TrainConfig) -> TrainResult:
    """Bring-up backend: PyTorch + sentence-transformers MNR loss.

    Faithful port of `catalog-builder/train_phase1.py` to a public,
    typed surface. Will be retired once the native MIND backend can
    produce an equivalent checkpoint.
    """
    import torch
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    started_at_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.environ["PYTHONHASHSEED"] = str(config.seed)
    if config.deterministic:
        _apply_deterministic_flags(config.seed)
    else:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_manifest_sha256 = _sha256_file(config.catalog_path)

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

    finished_at_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
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
        "deterministic": config.deterministic,
        "train_pairs": len(train_set),
        "eval_pairs": len(eval_set),
        "baseline_metrics": baseline,
        "final_metrics": final,
        "elapsed_seconds": round(elapsed, 2),
        "model_hash": model_hash,
        "trained_at_iso": finished_at_iso,
        "started_at": started_at_iso,
        "finished_at": finished_at_iso,
        "device": device,
        "torch_version": torch.__version__,
        "dataset_manifest_sha256": dataset_manifest_sha256,
    }
    manifest_path = config.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (config.output_dir / "eval.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    _emit_run_json(
        output_dir=config.output_dir,
        config=config,
        dataset_manifest_sha256=dataset_manifest_sha256,
        manifest_payload=manifest,
        started_at_iso=started_at_iso,
        finished_at_iso=finished_at_iso,
    )

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
