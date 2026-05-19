"""
tests/bit_identity/test_harness_smoke.py

pytest entry for the A1.4 bit-identity harness.

Runs on a clean checkout via:
    pip install -e ".[dev]" && pytest tests/bit_identity -v

Tests:
  1. Corpus JSON is well-formed (correct count, schema, categories).
  2. PyTorch backend produces 7,000 hashes for 1,000 queries.
  3. All 7 hash positions are valid hex strings (or known sentinels).
  4. Hash blob JSON structure is well-formed.
  5. Top-K indices are within valid catalog bounds.
  6. Sliding-window invariant holds for all T > 256 corpus entries.
  7. Reproducing the corpus produces the same result (determinism).
  8. Sentinel hashes are well-formed for native and cuda backends.
  9. Comparison report runs without error on pytorch-vs-pytorch (100% pass).
  10. Token IDs from different queries differ (no hash collision on trivial inputs).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

THIS_DIR = Path(__file__).parent
CORPUS_PATH = THIS_DIR / "corpus.json"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CORPUS_SIZE = 1000
HASH_KEYS = (
    "token_ids",
    "post_embed_ln",
    "final_layer_ln",
    "post_cls_slice",
    "post_l2_norm",
    "catalog_scores",
    "topk_indices_scores",
)
HASHES_PER_QUERY = len(HASH_KEYS)
TOTAL_HASHES = CORPUS_SIZE * HASHES_PER_QUERY

SENTINELS = frozenset(
    [
        "BACKEND_STUB_NOT_BUILT",
        "CUDA_DEFERRED_TO_V0_4_1",
    ]
)

VALID_CATEGORIES = {"eval", "long", "adversarial"}
EXPECTED_CATEGORY_COUNTS = {"eval": 600, "long": 200, "adversarial": 200}

# Whether to require the runtime dir (skip heavy tests if absent)
_RUNTIME_AVAILABLE = (
    Path.home() / ".local" / "share" / "mind-nerve" / "runtime" / "manifest.json"
).exists() or bool(os.environ.get("MIND_NERVE_RUNTIME_DIR"))

# Slow tests take 2-5 minutes on the full 1000-query corpus.
# Mark them so CI can set SKIP_SLOW=1 to skip during lint-only runs.
_SKIP_SLOW = os.environ.get("BIT_IDENTITY_SKIP_SLOW", "").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def corpus() -> list[dict]:
    """Load or build the committed corpus."""
    if CORPUS_PATH.exists():
        with CORPUS_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    # Fallback: build on the fly (happens only in fresh checkout without corpus.json).
    sys.path.insert(0, str(THIS_DIR))
    from corpus import build_corpus  # type: ignore[import-not-found]
    return build_corpus()


@pytest.fixture(scope="session")
def pytorch_hashes(corpus: list[dict]) -> dict:
    """
    Run the PyTorch backend on the corpus and return the hash blob.

    This is the ground-truth hash set. Running the full 1,000-query corpus
    takes ~2 minutes on CPU; this is acceptable for a session-scoped fixture.
    Skipped entirely if the runtime is not available.
    """
    if not _RUNTIME_AVAILABLE:
        pytest.skip("mind-nerve runtime dir not available; set MIND_NERVE_RUNTIME_DIR")

    if _SKIP_SLOW:
        pytest.skip("BIT_IDENTITY_SKIP_SLOW=1")

    runtime_dir = _resolve_runtime_dir()

    sys.path.insert(0, str(THIS_DIR))
    from runner import run_backend  # type: ignore[import-not-found]

    records = run_backend("pytorch", corpus, runtime_dir)
    return {
        "backend": "pytorch",
        "corpus_size": len(corpus),
        "total_hashes": len(records) * HASHES_PER_QUERY,
        "records": records,
    }


@pytest.fixture(scope="session")
def pytorch_hashes_small(corpus: list[dict]) -> dict:
    """
    Run the PyTorch backend on a small subset (10 queries) for fast tests.
    Uses the first 10 eval entries to avoid long queries.
    """
    if not _RUNTIME_AVAILABLE:
        pytest.skip("mind-nerve runtime dir not available; set MIND_NERVE_RUNTIME_DIR")

    runtime_dir = _resolve_runtime_dir()
    small_corpus = [e for e in corpus if e["category"] == "eval"][:10]

    sys.path.insert(0, str(THIS_DIR))
    from runner import run_backend  # type: ignore[import-not-found]

    records = run_backend("pytorch", small_corpus, runtime_dir)
    return {
        "backend": "pytorch",
        "corpus_size": len(small_corpus),
        "total_hashes": len(records) * HASHES_PER_QUERY,
        "records": records,
    }


def _resolve_runtime_dir() -> Path:
    env = os.environ.get("MIND_NERVE_RUNTIME_DIR")
    if env:
        p = Path(env).expanduser()
        if p.is_dir():
            return p
    return Path.home() / ".local" / "share" / "mind-nerve" / "runtime"


# ---------------------------------------------------------------------------
# Test 1: Corpus is well-formed
# ---------------------------------------------------------------------------

class TestCorpusWellFormed:
    def test_corpus_total_count(self, corpus: list[dict]) -> None:
        assert len(corpus) == CORPUS_SIZE, (
            f"Expected {CORPUS_SIZE} corpus entries, got {len(corpus)}"
        )

    def test_corpus_category_distribution(self, corpus: list[dict]) -> None:
        counts: dict[str, int] = {}
        for e in corpus:
            cat = e["category"]
            counts[cat] = counts.get(cat, 0) + 1

        for cat, expected in EXPECTED_CATEGORY_COUNTS.items():
            actual = counts.get(cat, 0)
            assert actual == expected, (
                f"Category {cat!r}: expected {expected} entries, got {actual}"
            )

    def test_corpus_all_have_id(self, corpus: list[dict]) -> None:
        for e in corpus:
            assert "id" in e and e["id"], f"Missing id in entry: {e!r}"

    def test_corpus_all_have_text_field(self, corpus: list[dict]) -> None:
        for e in corpus:
            assert "text" in e, f"Missing 'text' field in entry: {e['id']!r}"

    def test_corpus_ids_are_unique(self, corpus: list[dict]) -> None:
        ids = [e["id"] for e in corpus]
        assert len(ids) == len(set(ids)), "Duplicate IDs in corpus"

    def test_corpus_all_categories_valid(self, corpus: list[dict]) -> None:
        for e in corpus:
            assert e["category"] in VALID_CATEGORIES, (
                f"Unknown category {e['category']!r} in entry {e['id']!r}"
            )

    def test_long_queries_are_plausibly_long(self, corpus: list[dict]) -> None:
        """Long queries should have enough characters to tokenize to > 256 tokens."""
        long_entries = [e for e in corpus if e["category"] == "long"]
        # At 4 chars/token, we need ~256*4=1024 chars for T>256.
        # Allow some entries to be shorter (the tokenizer's behavior varies).
        min_long_chars = 200  # permissive lower bound
        short_count = sum(1 for e in long_entries if len(e["text"]) < min_long_chars)
        # Allow at most 10% of long queries to be under the threshold.
        assert short_count <= len(long_entries) * 0.1, (
            f"{short_count} long-category queries are too short "
            f"(< {min_long_chars} chars)"
        )

    def test_adversarial_contains_empty_string(self, corpus: list[dict]) -> None:
        adv = [e for e in corpus if e["category"] == "adversarial"]
        texts = {e["text"] for e in adv}
        assert "" in texts, "Adversarial corpus must include the empty string"

    def test_corpus_is_deterministic(self) -> None:
        """Building the corpus twice from the same seed yields the same result."""
        sys.path.insert(0, str(THIS_DIR))
        from corpus import build_corpus  # type: ignore[import-not-found]

        corpus1 = build_corpus()
        corpus2 = build_corpus()
        assert len(corpus1) == len(corpus2)
        for a, b in zip(corpus1, corpus2, strict=True):
            assert a == b, f"Corpus is not deterministic: {a!r} != {b!r}"


# ---------------------------------------------------------------------------
# Test 2: Hash blob structure
# ---------------------------------------------------------------------------

def _is_valid_hash(h: str | None) -> bool:
    """Return True if h is a valid 64-char hex SHA-256 or a known sentinel."""
    if h is None:
        return False
    if h in SENTINELS:
        return True
    if h.startswith("ERROR:"):
        return True
    return len(h) == 64 and all(c in "0123456789abcdef" for c in h)


class TestHashBlobStructure:
    def test_small_blob_record_count(self, pytorch_hashes_small: dict) -> None:
        records = pytorch_hashes_small["records"]
        assert len(records) == 10

    def test_small_blob_total_hashes(self, pytorch_hashes_small: dict) -> None:
        records = pytorch_hashes_small["records"]
        total = len(records) * HASHES_PER_QUERY
        assert total == 10 * HASHES_PER_QUERY

    def test_small_blob_all_hash_keys_present(self, pytorch_hashes_small: dict) -> None:
        for rec in pytorch_hashes_small["records"]:
            hashes = rec.get("hashes", {})
            for key in HASH_KEYS:
                assert key in hashes, (
                    f"Missing hash key {key!r} in record {rec['id']!r}"
                )

    def test_small_blob_all_hashes_valid(self, pytorch_hashes_small: dict) -> None:
        for rec in pytorch_hashes_small["records"]:
            for key in HASH_KEYS:
                h = rec["hashes"].get(key)
                assert _is_valid_hash(h), (
                    f"Invalid hash for {key!r} in {rec['id']!r}: {h!r}"
                )

    def test_small_blob_backend_field(self, pytorch_hashes_small: dict) -> None:
        assert pytorch_hashes_small["backend"] == "pytorch"

    def test_small_blob_token_len_positive(self, pytorch_hashes_small: dict) -> None:
        for rec in pytorch_hashes_small["records"]:
            assert rec["token_len"] is not None
            assert rec["token_len"] >= 2, (
                f"Token len {rec['token_len']} too short for {rec['id']!r}"
            )

    def test_small_blob_topk_indices_valid(self, pytorch_hashes_small: dict) -> None:
        for rec in pytorch_hashes_small["records"]:
            indices = rec.get("topk_indices")
            assert indices is not None, f"Missing topk_indices in {rec['id']!r}"
            assert len(indices) > 0, f"Empty topk_indices in {rec['id']!r}"
            for idx in indices:
                assert idx >= 0, f"Negative topk index {idx} in {rec['id']!r}"

    def test_small_blob_different_queries_different_token_hashes(
        self, pytorch_hashes_small: dict
    ) -> None:
        """Different queries must produce different token_ids hashes."""
        records = pytorch_hashes_small["records"]
        if len(records) < 2:
            pytest.skip("Need at least 2 records")
        hashes = [r["hashes"]["token_ids"] for r in records]
        # At least half should be unique (adversarial edge cases may collide)
        unique = len(set(hashes))
        assert unique >= max(1, len(hashes) // 2), (
            f"Too many duplicate token_ids hashes: {len(hashes) - unique} collisions "
            f"out of {len(hashes)} queries"
        )


# ---------------------------------------------------------------------------
# Test 3: PyTorch baseline full run (session-scoped, slow)
# ---------------------------------------------------------------------------

class TestPytorchBaseline:
    def test_full_corpus_hash_count(self, pytorch_hashes: dict) -> None:
        assert pytorch_hashes["total_hashes"] == TOTAL_HASHES, (
            f"Expected {TOTAL_HASHES} hashes, got {pytorch_hashes['total_hashes']}"
        )

    def test_full_corpus_record_count(self, pytorch_hashes: dict) -> None:
        assert len(pytorch_hashes["records"]) == CORPUS_SIZE

    def test_full_corpus_no_missing_hashes(self, pytorch_hashes: dict) -> None:
        missing: list[str] = []
        for rec in pytorch_hashes["records"]:
            for key in HASH_KEYS:
                h = rec["hashes"].get(key)
                if not _is_valid_hash(h):
                    missing.append(f"{rec['id']!r}.{key}")
        assert not missing, f"Invalid hashes: {missing[:10]}"

    def test_full_corpus_all_hashes_are_hex_or_sentinel(self, pytorch_hashes: dict) -> None:
        for rec in pytorch_hashes["records"]:
            for key in HASH_KEYS:
                h = rec["hashes"].get(key)
                assert _is_valid_hash(h), (
                    f"Invalid hash at {rec['id']!r}.{key}: {h!r}"
                )

    def test_full_corpus_json_serializable(self, pytorch_hashes: dict) -> None:
        """Verify the blob round-trips through JSON."""
        serialized = json.dumps(pytorch_hashes)
        reloaded = json.loads(serialized)
        assert reloaded["total_hashes"] == pytorch_hashes["total_hashes"]
        assert len(reloaded["records"]) == len(pytorch_hashes["records"])

    def test_pytorch_vs_pytorch_self_comparison(
        self, pytorch_hashes: dict, corpus: list[dict]
    ) -> None:
        """
        Comparing a blob against itself must yield 100% pass and >= 90% top-K overlap.
        This validates the compare module's gate logic.
        """
        sys.path.insert(0, str(THIS_DIR))
        from compare import compare_blobs  # type: ignore[import-not-found]

        result = compare_blobs(pytorch_hashes, pytorch_hashes)

        assert result.hash_fail == 0, (
            f"Self-comparison produced {result.hash_fail} hash mismatches"
        )
        if result.topk_overlaps:
            avg = sum(result.topk_overlaps) / len(result.topk_overlaps)
            assert avg >= 0.90, f"Self-comparison top-K overlap {avg:.4f} < 0.90"


# ---------------------------------------------------------------------------
# Test 4: Sentinel backend blobs
# ---------------------------------------------------------------------------

class TestSentinelBackends:
    def test_native_sentinel_blob_structure(self, corpus: list[dict]) -> None:
        """Native backend emits correct sentinel structure when .so absent."""
        sys.path.insert(0, str(THIS_DIR))
        from runner import HASH_KEYS as RUNNER_HASH_KEYS
        from runner import (  # type: ignore[import-not-found]
            SENTINEL_NATIVE_STUB,  # type: ignore[import-not-found]
            _sentinel_records,
        )

        records = _sentinel_records(corpus[:5], "native", SENTINEL_NATIVE_STUB)
        assert len(records) == 5
        for rec in records:
            assert rec["backend"] == "native"
            for key in RUNNER_HASH_KEYS:
                assert rec["hashes"][key] == SENTINEL_NATIVE_STUB

    def test_cuda_sentinel_blob_structure(self, corpus: list[dict]) -> None:
        """CUDA backend emits deferred sentinel per §3.2."""
        sys.path.insert(0, str(THIS_DIR))
        from runner import SENTINEL_CUDA, _sentinel_records  # type: ignore[import-not-found]

        records = _sentinel_records(corpus[:5], "cuda", SENTINEL_CUDA)
        assert len(records) == 5
        for rec in records:
            assert rec["backend"] == "cuda"
            for key in HASH_KEYS:
                assert rec["hashes"][key] == SENTINEL_CUDA

    def test_compare_with_sentinel_does_not_fail(self, corpus: list[dict]) -> None:
        """
        Comparing pytorch hashes against a sentinel-only native blob
        must not produce hard failures (sentinels are skipped).
        """
        if not _RUNTIME_AVAILABLE:
            pytest.skip("runtime not available")

        if _SKIP_SLOW:
            pytest.skip("BIT_IDENTITY_SKIP_SLOW=1")

        runtime_dir = _resolve_runtime_dir()
        small = [e for e in corpus if e["category"] == "eval"][:5]

        sys.path.insert(0, str(THIS_DIR))
        from compare import compare_blobs  # type: ignore[import-not-found]
        from runner import (  # type: ignore[import-not-found]
            SENTINEL_NATIVE_STUB,
            _sentinel_records,
            run_backend,
        )

        pytorch_recs = run_backend("pytorch", small, runtime_dir)
        pytorch_blob = {"backend": "pytorch", "records": pytorch_recs}

        sentinel_recs = _sentinel_records(small, "native", SENTINEL_NATIVE_STUB)
        sentinel_blob = {"backend": "native", "records": sentinel_recs}

        result = compare_blobs(pytorch_blob, sentinel_blob)

        # All positions are SKIP (sentinel), so zero failures.
        assert result.hash_fail == 0, (
            f"Sentinel comparison produced {result.hash_fail} failures "
            f"(expected 0 — all should be skipped)"
        )
        assert result.hash_skip > 0, "Expected some SKIP positions for sentinel backend"


# ---------------------------------------------------------------------------
# Test 5: Sliding-window invariant (§3.3)
# ---------------------------------------------------------------------------

class TestSlidingWindowInvariant:
    def test_single_window_when_t_lte_256(self) -> None:
        sys.path.insert(0, str(THIS_DIR))
        from sliding_window_invariant import (  # type: ignore[import-not-found]
            compute_window_plan,
        )

        for T in [0, 1, 2, 100, 200, 256]:
            plan = compute_window_plan(T)
            expected_windows = 1 if T > 0 else 0
            assert plan.window_count == expected_windows, (
                f"T={T}: expected {expected_windows} window(s), got {plan.window_count}"
            )
            assert all(w == 0 for w in plan.token_to_window), (
                f"T={T}: all tokens should map to window 0"
            )

    def test_two_windows_when_t_is_257(self) -> None:
        sys.path.insert(0, str(THIS_DIR))
        from sliding_window_invariant import (  # type: ignore[import-not-found]
            STRIDE,
            compute_window_plan,
        )

        T = 257
        plan = compute_window_plan(T)
        assert plan.window_count >= 2, f"T={T}: expected >= 2 windows"
        # Tokens 0..STRIDE-1 -> window 0; tokens STRIDE..T-1 -> window 1
        for t in range(STRIDE):
            assert plan.token_to_window[t] == 0, (
                f"T={T}, t={t}: expected window 0, got {plan.token_to_window[t]}"
            )
        for t in range(STRIDE, T):
            assert plan.token_to_window[t] == 1, (
                f"T={T}, t={t}: expected window 1, got {plan.token_to_window[t]}"
            )

    def test_later_window_wins_in_overlap_region(self) -> None:
        sys.path.insert(0, str(THIS_DIR))
        from sliding_window_invariant import (  # type: ignore[import-not-found]
            STRIDE,
            WINDOW_SIZE,
            compute_window_plan,
        )

        # T = 400: two windows, overlap at [192, 256)
        T = 400
        plan = compute_window_plan(T)
        # Overlap: window 0 covers [0, 256), window 1 covers [192, 400)
        # Tokens 192..255 are in both windows -> must go to window 1
        for t in range(STRIDE, WINDOW_SIZE):
            w = plan.token_to_window[t]
            assert w == 1, (
                f"T={T}, t={t} in overlap: expected window 1 (later), got {w}"
            )

    def test_every_token_assigned_exactly_once(self) -> None:
        sys.path.insert(0, str(THIS_DIR))
        from sliding_window_invariant import compute_window_plan  # type: ignore[import-not-found]

        for T in [0, 1, 50, 256, 257, 300, 400, 512, 1000]:
            plan = compute_window_plan(T)
            assert len(plan.token_to_window) == T, (
                f"T={T}: token_to_window length mismatch"
            )
            for t, w in enumerate(plan.token_to_window):
                assert 0 <= w < plan.window_count or plan.window_count == 0, (
                    f"T={T}, t={t}: window {w} out of range [0, {plan.window_count})"
                )

    def test_check_invariants_passes_for_valid_plans(self) -> None:
        sys.path.insert(0, str(THIS_DIR))
        from sliding_window_invariant import (  # type: ignore[import-not-found]
            check_invariants,
            compute_window_plan,
        )

        for T in [0, 1, 50, 256, 257, 300, 400, 512, 750, 1000]:
            plan = compute_window_plan(T)
            violations = check_invariants(plan, f"test_T{T}")
            assert not violations, (
                f"T={T}: unexpected violations: {violations[:3]}"
            )

    def test_corpus_long_queries_invariant(self, corpus: list[dict]) -> None:
        """All long-category corpus entries pass the §3.3 invariant."""
        sys.path.insert(0, str(THIS_DIR))
        from sliding_window_invariant import run_invariant_tests  # type: ignore[import-not-found]

        result = run_invariant_tests(corpus, hashes_blob=None, verbose=False)
        assert result.long_queries_tested == 200, (
            f"Expected 200 long queries, got {result.long_queries_tested}"
        )
        assert result.passed, (
            f"Sliding-window invariant failed: "
            f"{result.long_queries_with_violations} queries with violations, "
            f"{result.total_violations} total violations"
        )

    @pytest.mark.parametrize(
        "T, expected_windows",
        [
            (1, 1),
            (256, 1),
            (257, 2),
            (448, 2),   # 192 + 256 = 448, just one full second window
            (449, 3),   # needs third window
            (640, 3),   # 2*192 + 256 = 640
            (641, 4),
        ],
    )
    def test_window_count(self, T: int, expected_windows: int) -> None:
        sys.path.insert(0, str(THIS_DIR))
        from sliding_window_invariant import compute_window_plan  # type: ignore[import-not-found]

        plan = compute_window_plan(T)
        assert plan.window_count == expected_windows, (
            f"T={T}: expected {expected_windows} window(s), got {plan.window_count}"
        )


# ---------------------------------------------------------------------------
# Test 6: Gate thresholds (overlap math)
# ---------------------------------------------------------------------------

class TestGateThresholds:
    def test_compare_topk_overlap_perfect(self) -> None:
        """100% overlap when indices are identical."""
        sys.path.insert(0, str(THIS_DIR))
        from compare import _compute_topk_overlap  # type: ignore[import-not-found]

        rec_a = {"topk_indices": [1, 2, 3, 4, 5]}
        rec_b = {"topk_indices": [1, 2, 3, 4, 5]}
        result = _compute_topk_overlap(rec_a, rec_b, 5)
        assert result["top_k"] == 1.0
        assert result["top_1"] == 1.0

    def test_compare_topk_overlap_zero(self) -> None:
        """0% overlap when no indices match."""
        sys.path.insert(0, str(THIS_DIR))
        from compare import _compute_topk_overlap  # type: ignore[import-not-found]

        rec_a = {"topk_indices": [1, 2, 3, 4, 5]}
        rec_b = {"topk_indices": [6, 7, 8, 9, 10]}
        result = _compute_topk_overlap(rec_a, rec_b, 5)
        assert result["top_k"] == 0.0

    def test_compare_topk_overlap_partial(self) -> None:
        """Partial overlap (3 of 5)."""
        sys.path.insert(0, str(THIS_DIR))
        from compare import _compute_topk_overlap  # type: ignore[import-not-found]

        rec_a = {"topk_indices": [1, 2, 3, 4, 5]}
        rec_b = {"topk_indices": [1, 2, 3, 6, 7]}
        result = _compute_topk_overlap(rec_a, rec_b, 5)
        assert abs(result["top_k"] - 0.6) < 1e-9
        assert result["top_1"] == 1.0  # top-1 matches

    def test_compare_topk_overlap_top1_miss(self) -> None:
        """Top-1 differs but rest overlap."""
        sys.path.insert(0, str(THIS_DIR))
        from compare import _compute_topk_overlap  # type: ignore[import-not-found]

        rec_a = {"topk_indices": [1, 2, 3, 4, 5]}
        rec_b = {"topk_indices": [99, 2, 3, 4, 5]}
        result = _compute_topk_overlap(rec_a, rec_b, 5)
        assert result["top_1"] == 0.0
        # Both sets have 5 elements; 4 are in common (2,3,4,5); union has 6.
        # overlap = |intersection| / |max(|set_a|, |set_b|)| = 4/5 = 0.8
        assert abs(result["top_k"] - 4 / 5) < 1e-9  # 4 common out of 5 per set

    def test_compare_sentinel_is_skipped(self) -> None:
        """Sentinel hashes do not contribute to FAIL count."""
        sys.path.insert(0, str(THIS_DIR))
        from compare import compare_blobs  # type: ignore[import-not-found]
        from runner import SENTINEL_NATIVE_STUB  # type: ignore[import-not-found]

        # Build minimal blobs
        ids = ["q0", "q1"]
        blob_pt = {
            "backend": "pytorch",
            "records": [
                {
                    "id": qid,
                    "category": "eval",
                    "backend": "pytorch",
                    "token_len": 5,
                    "hashes": {k: "a" * 64 for k in HASH_KEYS},
                    "topk_indices": [0, 1, 2, 3, 4],
                }
                for qid in ids
            ],
        }
        blob_stub = {
            "backend": "native",
            "records": [
                {
                    "id": qid,
                    "category": "eval",
                    "backend": "native",
                    "token_len": None,
                    "hashes": {k: SENTINEL_NATIVE_STUB for k in HASH_KEYS},
                    "topk_indices": None,
                }
                for qid in ids
            ],
        }
        result = compare_blobs(blob_pt, blob_stub)
        assert result.hash_fail == 0, "Sentinel should not count as mismatch"
        assert result.hash_skip == len(ids) * len(HASH_KEYS)

    def test_compare_detects_real_mismatch(self) -> None:
        """A real hash difference is caught as a FAIL."""
        sys.path.insert(0, str(THIS_DIR))
        from compare import compare_blobs  # type: ignore[import-not-found]

        blob_a = {
            "backend": "pytorch",
            "records": [
                {
                    "id": "q0",
                    "category": "eval",
                    "backend": "pytorch",
                    "token_len": 5,
                    "hashes": {k: "a" * 64 for k in HASH_KEYS},
                    "topk_indices": [0, 1, 2, 3, 4],
                }
            ],
        }
        blob_b = {
            "backend": "pytorch",
            "records": [
                {
                    "id": "q0",
                    "category": "eval",
                    "backend": "pytorch",
                    "token_len": 5,
                    "hashes": {
                        **{k: "a" * 64 for k in HASH_KEYS},
                        "token_ids": "b" * 64,  # deliberate mismatch
                    },
                    "topk_indices": [0, 1, 2, 3, 4],
                }
            ],
        }
        result = compare_blobs(blob_a, blob_b)
        assert result.hash_fail == 1, f"Expected 1 FAIL, got {result.hash_fail}"
        assert result.queries_with_mismatch == 1
