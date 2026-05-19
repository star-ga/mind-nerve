"""Unit tests for ``mind_nerve.eval_metrics``.

Covers MRR, nDCG@k, and ECE against hand-computed fixtures so the public
training surface ships with provable, reproducible retrieval metrics.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from mind_nerve.eval_metrics import (
    expected_calibration_error,
    mrr,
    ndcg_at_k,
    reciprocal_ranks,
)

# ---------------------------------------------------------------------------
# MRR
# ---------------------------------------------------------------------------


def test_mrr_three_queries_with_known_ranks() -> None:
    # ranks 1, 2, 5 -> MRR = (1 + 1/2 + 1/5) / 3 = 0.5666...
    ranked = [
        ["a", "x", "y", "z", "w"],  # truth at rank 1
        ["x", "b", "y", "z", "w"],  # truth at rank 2
        ["x", "y", "z", "w", "c"],  # truth at rank 5
    ]
    truth = ["a", "b", "c"]
    expected = (1.0 + 0.5 + 0.2) / 3.0
    got = mrr(ranked, truth)
    assert math.isclose(got, expected, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(got, 0.5666666666, abs_tol=1e-9)


def test_mrr_missing_truth_contributes_zero() -> None:
    ranked = [["a", "b", "c"], ["x", "y", "z"]]
    truth = ["a", "q"]  # second query: truth not in list
    # rank 1 + rank 0 -> (1.0 + 0.0) / 2 = 0.5
    assert mrr(ranked, truth) == pytest.approx(0.5)


def test_mrr_empty_input_returns_zero() -> None:
    assert mrr([], []) == 0.0


def test_reciprocal_ranks_individual_values() -> None:
    ranked = [["a", "b"], ["x", "y", "a"]]
    truth = ["a", "a"]
    assert reciprocal_ranks(ranked, truth) == [1.0, 1.0 / 3.0]


def test_mrr_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        mrr([["a"], ["b"]], ["a"])


# ---------------------------------------------------------------------------
# nDCG@k
# ---------------------------------------------------------------------------


def test_ndcg_at_1_matches_top1_accuracy() -> None:
    ranked = [["a", "x"], ["x", "b"], ["c", "y"]]
    truth = ["a", "b", "c"]
    # truths at ranks 1, 2, 1 -> ndcg@1 only credits rank 1 -> 2/3
    assert ndcg_at_k(ranked, truth, k=1) == pytest.approx(2.0 / 3.0)


def test_ndcg_at_k_handcomputed() -> None:
    # ranks 1, 2, 5 (k=5) -> mean of 1/log2(2), 1/log2(3), 1/log2(6)
    ranked = [
        ["a", "x", "y", "z", "w"],
        ["x", "b", "y", "z", "w"],
        ["x", "y", "z", "w", "c"],
    ]
    truth = ["a", "b", "c"]
    expected = (1.0 / math.log2(2.0) + 1.0 / math.log2(3.0) + 1.0 / math.log2(6.0)) / 3.0
    assert ndcg_at_k(ranked, truth, k=5) == pytest.approx(expected, abs=1e-12)


def test_ndcg_at_k_truth_outside_topk_scores_zero_for_that_query() -> None:
    ranked = [["x", "y", "z", "w", "v", "a"]]  # truth "a" at rank 6
    truth = ["a"]
    # nDCG@5 excludes rank 6 entirely -> 0.0
    assert ndcg_at_k(ranked, truth, k=5) == 0.0
    # nDCG@10 includes it -> 1/log2(7)
    assert ndcg_at_k(ranked, truth, k=10) == pytest.approx(1.0 / math.log2(7.0))


def test_ndcg_rejects_nonpositive_k() -> None:
    with pytest.raises(ValueError):
        ndcg_at_k([["a"]], ["a"], k=0)
    with pytest.raises(ValueError):
        ndcg_at_k([["a"]], ["a"], k=-1)


def test_ndcg_empty_input_returns_zero() -> None:
    assert ndcg_at_k([], [], k=5) == 0.0


# ---------------------------------------------------------------------------
# ECE
# ---------------------------------------------------------------------------


def test_ece_perfectly_calibrated_is_zero() -> None:
    # 10 buckets of 100 samples each; in bucket b the accuracy exactly
    # equals the midpoint confidence -> ECE should be 0.0
    rng = np.random.default_rng(0)
    scores: list[float] = []
    correct: list[int] = []
    for b in range(10):
        midpoint = (b + 0.5) / 10.0
        bucket_size = 100
        scores.extend([midpoint] * bucket_size)
        # set exactly round(midpoint * bucket_size) ones
        n_correct = int(round(midpoint * bucket_size))
        labels = [1] * n_correct + [0] * (bucket_size - n_correct)
        rng.shuffle(labels)
        correct.extend(labels)
    ece = expected_calibration_error(scores, correct, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-12)


def test_ece_completely_miscalibrated_is_one() -> None:
    # All predictions claim 1.0 confidence but every label is 0 -> gap = 1
    scores = [1.0] * 50
    correct = [0] * 50
    assert expected_calibration_error(scores, correct, n_bins=10) == pytest.approx(1.0)


def test_ece_known_two_bucket_fixture() -> None:
    # 5 samples at conf 0.9, accuracy 3/5 -> bucket gap 0.3, weight 5/10
    # 5 samples at conf 0.1, accuracy 1/5 -> bucket gap 0.1, weight 5/10
    # ECE = 0.5 * 0.3 + 0.5 * 0.1 = 0.2
    scores = [0.9] * 5 + [0.1] * 5
    correct = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    assert expected_calibration_error(scores, correct, n_bins=10) == pytest.approx(0.2)


def test_ece_score_at_one_is_included_in_last_bin() -> None:
    scores = [1.0, 1.0, 1.0, 1.0]
    correct = [1, 1, 1, 1]
    # confidence 1.0 with accuracy 1.0 -> ECE 0
    assert expected_calibration_error(scores, correct, n_bins=10) == pytest.approx(0.0)


def test_ece_rejects_scores_out_of_range() -> None:
    with pytest.raises(ValueError):
        expected_calibration_error([1.5], [1], n_bins=10)
    with pytest.raises(ValueError):
        expected_calibration_error([-0.1], [0], n_bins=10)


def test_ece_rejects_nonpositive_bins() -> None:
    with pytest.raises(ValueError):
        expected_calibration_error([0.5], [1], n_bins=0)


def test_ece_empty_input_returns_zero() -> None:
    assert expected_calibration_error([], [], n_bins=10) == 0.0


def test_ece_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        expected_calibration_error([0.5, 0.5], [1], n_bins=10)
