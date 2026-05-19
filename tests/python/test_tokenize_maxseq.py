# Copyright 2025 STARGA Inc.
# Licensed under the Apache License, Version 2.0 (the "License").

"""#228 regression: native-backend tokenization must truncate to the
model's ``max_seq_length`` (256), exactly like the reference
SentenceTransformer — never feed >256 tokens to the native encoder
(which would silently take the sliding-window 'later-window-wins' path
and diverge from pytorch).

Skips cleanly when transformers / the runtime checkpoint is absent
(matches the repo's optional-artifact test convention).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RUNTIME = Path.home() / ".local/share/mind-nerve/runtime"
CKPT = RUNTIME / "checkpoint"


def _model_max_seq() -> int:
    cfg = CKPT / "sentence_bert_config.json"
    if not cfg.exists():
        pytest.skip("runtime checkpoint absent")
    return int(json.loads(cfg.read_text())["max_seq_length"])


def test_native_tokenize_truncates_to_model_max_seq() -> None:
    """A long document must tokenize to <= max_seq_length (256), and a
    clearly-overlong input must hit exactly that cap (truncation active).
    This is the #228 invariant: native == pytorch-SentenceTransformer
    (truncate-to-max_seq + CLS pool), no silent sliding-window path."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")
    if not CKPT.exists():
        pytest.skip("runtime checkpoint absent")

    max_seq = _model_max_seq()
    assert max_seq == 256, f"unexpected model max_seq_length {max_seq}"

    tok = AutoTokenizer.from_pretrained(str(CKPT), use_fast=True)
    long_text = "lorem ipsum dolor sit amet consectetur " * 400  # ~2000+ tok

    # Mirror InferenceEngine._tokenize exactly (the #228-fixed call).
    enc = tok(
        long_text,
        truncation=True,
        max_length=max_seq,
        return_tensors="np",
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    n = len(enc["input_ids"][0])
    assert n <= max_seq, f"native tokenize produced {n} > {max_seq} tokens"
    assert n == max_seq, (
        f"overlong input truncated to {n}, expected exactly {max_seq} "
        "(truncation not active — #228 silent-divergence regression)"
    )


def test_inference_tokenize_uses_max_seq_not_512() -> None:
    """Source guard: InferenceEngine._tokenize must not regress to
    max_length=512 (the #228 bug)."""
    src = (Path(__file__).resolve().parents[2]
           / "python/mind_nerve/inference.py").read_text()
    # Match the CODE form (kwarg with trailing comma) — prose/docstring
    # mentions of the old value (e.g. ``max_length=512``) are fine.
    assert "max_length=512," not in src, (
        "inference.py _tokenize call uses max_length=512 — #228 regression: "
        "native would silently take the sliding-window path for >256-token "
        "inputs and diverge from pytorch SentenceTransformer"
    )
    assert "max_length=256," in src, (
        "expected the _tokenize call to use max_length=256 "
        "(= model max_seq_length) in inference.py"
    )
