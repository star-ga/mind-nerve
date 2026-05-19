"""Retrieval and calibration metrics for the public training surface.

This module implements pure-Python (NumPy-only) reference versions of the
standard retrieval metrics the trainer needs to publish alongside the
existing ``top-k`` accuracy numbers:

  * ``mrr(...)``                       — Mean Reciprocal Rank.
  * ``ndcg_at_k(...)``                 — normalized DCG at cut-off ``k``.
  * ``expected_calibration_error(...)``— ECE with equal-width bins.

Design choices:

  * Implementations are vectorized but never allocate quadratic temporaries;
    each metric is ``O(N * K)`` in the number of queries.
  * Inputs are accepted as Python lists or NumPy arrays so callers do not
    need to pre-cast.
  * The functions are deterministic — given the same numerical inputs they
    return the same float on every architecture supported by NumPy.

All metrics handle the empty-input case by returning ``0.0`` so that smoke
tests on tiny synthetic catalogs do not raise.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "mrr",
    "ndcg_at_k",
    "expected_calibration_error",
    "reciprocal_ranks",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _coerce_ranked_lists(ranked_lists: Iterable[Sequence[object]]) -> list[list[object]]:
    out: list[list[object]] = []
    for row in ranked_lists:
        out.append(list(row))
    return out


def _index_of(row: Sequence[object], target: object) -> int:
    """Return the 1-based index of ``target`` in ``row``; 0 if absent."""
    for pos, item in enumerate(row, start=1):
        if item == target:
            return pos
    return 0


# ---------------------------------------------------------------------------
# Public metrics
# ---------------------------------------------------------------------------


def reciprocal_ranks(
    ranked_lists: Iterable[Sequence[object]],
    ground_truth: Sequence[object],
) -> list[float]:
    """Return the reciprocal rank for each query.

    A reciprocal rank of ``0.0`` means the ground-truth item never appears
    in the corresponding ranked list.
    """
    rows = _coerce_ranked_lists(ranked_lists)
    if len(rows) != len(ground_truth):
        raise ValueError(
            f"ranked_lists ({len(rows)}) and ground_truth ({len(ground_truth)}) "
            "must have the same length"
        )
    rrs: list[float] = []
    for row, gt in zip(rows, ground_truth, strict=False):
        rank = _index_of(row, gt)
        rrs.append(0.0 if rank == 0 else 1.0 / float(rank))
    return rrs


def mrr(
    ranked_lists: Iterable[Sequence[object]],
    ground_truth: Sequence[object],
) -> float:
    """Mean Reciprocal Rank across all queries.

    ``ranked_lists[i]`` is the ordered list of candidate ids returned for
    query ``i``; ``ground_truth[i]`` is that query's true label. Missing
    truths contribute ``0`` to the mean.
    """
    rrs = reciprocal_ranks(ranked_lists, ground_truth)
    if not rrs:
        return 0.0
    return float(sum(rrs) / len(rrs))


def ndcg_at_k(
    ranked_lists: Iterable[Sequence[object]],
    ground_truth: Sequence[object],
    k: int,
) -> float:
    """Mean nDCG@k over all queries with binary relevance.

    Each query has exactly one relevant item (``ground_truth[i]``). For
    binary relevance the ideal DCG is ``1.0`` (the relevant doc at rank 1),
    so ``nDCG@k`` collapses to ``1 / log2(1 + rank)`` if the truth is in
    the top-k and ``0`` otherwise.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k!r}")
    rows = _coerce_ranked_lists(ranked_lists)
    if len(rows) != len(ground_truth):
        raise ValueError(
            f"ranked_lists ({len(rows)}) and ground_truth ({len(ground_truth)}) "
            "must have the same length"
        )
    if not rows:
        return 0.0
    total = 0.0
    for row, gt in zip(rows, ground_truth, strict=False):
        rank = _index_of(row[:k], gt)
        if rank == 0:
            continue
        total += 1.0 / math.log2(1.0 + rank)
    return float(total / len(rows))


def expected_calibration_error(
    scores: Sequence[float] | np.ndarray,
    correct: Sequence[bool | int] | np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error with equal-width confidence bins.

    ``scores[i]`` is the model's confidence (in ``[0, 1]``) that prediction
    ``i`` is correct; ``correct[i]`` is the binary ground-truth label
    (``True`` / ``1`` if the prediction matched, ``False`` / ``0``
    otherwise). Returns the mean weighted deviation between predicted
    confidence and observed accuracy across all non-empty bins.
    """
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins!r}")
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    c = np.asarray(correct, dtype=np.float64).reshape(-1)
    if s.shape != c.shape:
        raise ValueError(f"scores shape {s.shape} != correct shape {c.shape}")
    if s.size == 0:
        return 0.0
    if np.any(s < 0.0) or np.any(s > 1.0):
        raise ValueError("scores must lie in [0, 1]")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = float(s.size)
    ece = 0.0
    for b in range(n_bins):
        lo = edges[b]
        hi = edges[b + 1]
        # Last bin is inclusive on the right so scores == 1.0 are counted.
        if b == n_bins - 1:
            mask = (s >= lo) & (s <= hi)
        else:
            mask = (s >= lo) & (s < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        bin_conf = float(s[mask].mean())
        bin_acc = float(c[mask].mean())
        ece += (count / total) * abs(bin_conf - bin_acc)
    return float(ece)
