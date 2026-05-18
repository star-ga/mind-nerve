"""Contract tests for mind_nerve.mind_train.

Exercises the typed surface without invoking PyTorch — the bring-up
trainer is heavy and runs out-of-CI on a GPU pod. CI validates the
public API contract: dataclass shape, parsing helpers, hash
determinism, backend resolution.

Invariants:
  T1 — TrainConfig is a frozen dataclass with the documented defaults.
  T2 — TrainResult is a frozen dataclass with the documented fields.
  T3 — _load_pairs handles malformed rows safely.
  T4 — _split_pairs respects eval_frac and is deterministic in seed.
  T5 — _compute_checkpoint_hash is order-independent on filename traversal
       (sorted internally) and stable across calls.
  T6 — train(backend='native') raises NotImplementedError (until mindc 0.3.0).
  T7 — train(backend='bogus') raises ValueError.
  T8 — config_to_dict produces JSON-serialisable output.
"""

from __future__ import annotations

import dataclasses
import json
import os

import pytest
from mind_nerve.mind_train import (
    DEFAULT_BASE_MODEL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_EVAL_FRAC,
    DEFAULT_LR,
    DEFAULT_MAX_LEN,
    DEFAULT_SEED,
    TrainConfig,
    TrainResult,
    _compute_checkpoint_hash,
    _load_pairs,
    _split_pairs,
    config_to_dict,
    train,
)


def test_t1_train_config_is_frozen_with_defaults(tmp_path):
    cfg = TrainConfig(catalog_path=tmp_path / "corpus.tsv", output_dir=tmp_path / "out")
    assert dataclasses.is_dataclass(cfg)
    # frozen=True means we can't reassign fields
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.epochs = 99  # type: ignore[misc]
    assert cfg.base_model == DEFAULT_BASE_MODEL
    assert cfg.epochs == DEFAULT_EPOCHS
    assert cfg.batch_size == DEFAULT_BATCH_SIZE
    assert cfg.lr == DEFAULT_LR
    assert cfg.max_len == DEFAULT_MAX_LEN
    assert cfg.seed == DEFAULT_SEED
    assert cfg.eval_frac == DEFAULT_EVAL_FRAC
    assert cfg.smoke_test is False
    assert cfg.backend == "python"


def test_t2_train_result_dataclass_shape(tmp_path):
    # Synthetic instance — we only verify the shape, not produce real training.
    r = TrainResult(
        checkpoint_dir=tmp_path / "ckpt",
        manifest_path=tmp_path / "manifest.json",
        model_hash="0" * 64,
        epochs_completed=1,
        train_pairs=10,
        eval_pairs=2,
        metrics={"top1": 0.5, "top5": 0.9, "top10": 1.0, "candidate_pool": 12.0},
        baseline_metrics={"top1": 0.1, "top5": 0.5, "top10": 0.8, "candidate_pool": 12.0},
        elapsed_seconds=3.14,
        backend_used="python",
    )
    assert dataclasses.is_dataclass(r)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.epochs_completed = 0  # type: ignore[misc]
    # extras defaults to empty dict, not shared across instances
    r2 = TrainResult(
        checkpoint_dir=tmp_path / "ckpt2",
        manifest_path=tmp_path / "m2.json",
        model_hash="1" * 64,
        epochs_completed=1,
        train_pairs=1,
        eval_pairs=1,
        metrics={},
        baseline_metrics={},
        elapsed_seconds=0.0,
        backend_used="python",
    )
    assert r.extras == {}
    assert r2.extras == {}
    assert r.extras is not r2.extras


def test_t3_load_pairs_skips_malformed(tmp_path):
    corpus = tmp_path / "corpus.tsv"
    corpus.write_text(
        "good\tskill\tThis is a long enough body string\n"
        "ok2\ttool\tAnother body string that is long enough\n"
        "tooShort\tskill\tshort\n"  # body < 16
        "\tskill\tEmpty name should be skipped due to len<2 check\n"
        "two-cols-only\tskill\n"  # not three tabs
        "x\tskill\tFinal valid body row, long enough\n"
    )
    pairs = _load_pairs(corpus)
    names = [p[0] for p in pairs]
    assert "good" in names
    assert "ok2" in names
    assert "tooShort" not in names
    assert "two-cols-only" not in names
    # only the 2 good rows + the final (x has name len 1 -> rejected)
    assert len(pairs) == 2


def test_t4_split_pairs_deterministic_in_seed():
    pairs = [(f"q{i}", f"p{i} body long enough") for i in range(100)]
    a_train, a_eval = _split_pairs(pairs, eval_frac=0.1, seed=42)
    b_train, b_eval = _split_pairs(pairs, eval_frac=0.1, seed=42)
    assert a_train == b_train
    assert a_eval == b_eval
    # different seed -> different split
    c_train, c_eval = _split_pairs(pairs, eval_frac=0.1, seed=43)
    assert c_eval != a_eval
    # eval_frac respected within rounding
    assert len(a_eval) == 10
    assert len(a_train) == 90


def test_t5_checkpoint_hash_deterministic_and_path_bound(tmp_path):
    ckpt_a = tmp_path / "a"
    ckpt_a.mkdir()
    (ckpt_a / "model.bin").write_bytes(b"weights-bytes")
    (ckpt_a / "tokenizer.json").write_bytes(b'{"vocab":"..."}')
    h1 = _compute_checkpoint_hash(ckpt_a)
    h2 = _compute_checkpoint_hash(ckpt_a)
    assert h1 == h2, "hash must be deterministic"
    assert len(h1) == 64

    # Same bytes under different filename → different hash (path is bound in).
    ckpt_b = tmp_path / "b"
    ckpt_b.mkdir()
    (ckpt_b / "model.bin").write_bytes(b"weights-bytes")
    (ckpt_b / "tokenizer_renamed.json").write_bytes(b'{"vocab":"..."}')
    h3 = _compute_checkpoint_hash(ckpt_b)
    assert h3 != h1, "renaming a file must change the checkpoint hash"


def test_t6_native_backend_raises_not_implemented(tmp_path):
    cfg = TrainConfig(
        catalog_path=tmp_path / "corpus.tsv",
        output_dir=tmp_path / "out",
        backend="native",
    )
    with pytest.raises(NotImplementedError, match="mindc 0.3.0"):
        train(cfg)


def test_t7_unknown_backend_raises_value_error(tmp_path):
    cfg = TrainConfig(
        catalog_path=tmp_path / "corpus.tsv",
        output_dir=tmp_path / "out",
        backend="bogus",  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="unknown backend"):
        train(cfg)


def test_t8_config_to_dict_is_json_safe(tmp_path):
    cfg = TrainConfig(
        catalog_path=tmp_path / "corpus.tsv",
        output_dir=tmp_path / "out",
        epochs=2,
        smoke_test=True,
    )
    d = config_to_dict(cfg)
    blob = json.dumps(d)
    parsed = json.loads(blob)
    assert parsed["epochs"] == 2
    assert parsed["smoke_test"] is True
    assert parsed["backend"] == "python"
    # Paths serialised to strings
    assert isinstance(parsed["catalog_path"], str)
    assert isinstance(parsed["output_dir"], str)


def test_t9_python_backend_missing_catalog_raises(tmp_path):
    """python backend with a missing catalog should fail fast, before
    loading SentenceTransformer (which pulls a multi-hundred-MB model
    on first call). We rely on _load_pairs hitting an empty/missing
    file path."""
    cfg = TrainConfig(
        catalog_path=tmp_path / "does_not_exist.tsv",
        output_dir=tmp_path / "out",
    )
    # Ensure we don't accidentally invoke an expensive model download on CI.
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    with pytest.raises((FileNotFoundError, RuntimeError)):
        train(cfg)
