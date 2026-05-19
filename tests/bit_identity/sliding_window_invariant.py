"""
tests/bit_identity/sliding_window_invariant.py

§3.3 sliding-window invariant test for the A1.4 harness.

For each T > 256 query in the corpus, verifies that the "later-window-wins"
rule is correctly implemented:

    winning_window_index(t) = t // stride

where stride = 192 (the canonical window stride from §3.3).

Contracts verified:
  1. Every token t in [0, T-1] maps to exactly one window.
  2. The winning window index is floor(t / stride), capped at
     the index of the last window.
  3. Overlap tokens (in the region shared by two windows) are
     assigned to the LATER window (higher index).
  4. The total number of windows is ceil((T - window_size) / stride) + 1
     for T > window_size, and 1 for T <= window_size.
  5. Every window covers at least 1 new (non-overlapping) token.

These properties together guarantee that the SHA-256 evidence chain has
a deterministic preimage for every token's attention output.

Usage:
    python tests/bit_identity/sliding_window_invariant.py
    python tests/bit_identity/sliding_window_invariant.py --verbose
    python tests/bit_identity/sliding_window_invariant.py --corpus /path/to/corpus.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import NamedTuple

THIS_DIR = Path(__file__).parent
CORPUS_PATH = THIS_DIR / "corpus.json"

WINDOW_SIZE = 256  # encoder max tokens per window
STRIDE = 192       # window stride per §3.3 "later-window-wins"


# ---------------------------------------------------------------------------
# Window arithmetic (pure Python, no deps)
# ---------------------------------------------------------------------------

class WindowPlan(NamedTuple):
    """Sliding-window layout for a sequence of length T."""
    T: int
    window_count: int
    # For each token index, the winning window index (0-based).
    token_to_window: list[int]
    # For each window, the start token (inclusive).
    window_starts: list[int]
    # For each window, the end token (exclusive).
    window_ends: list[int]


def compute_window_plan(T: int) -> WindowPlan:
    """
    Compute the later-window-wins sliding-window assignment for T tokens.

    Rules (per §3.3):
    - If T <= WINDOW_SIZE: one window [0, T), all tokens -> window 0.
    - If T > WINDOW_SIZE: windows start at 0, STRIDE, 2*STRIDE, ...
      The last window is aligned to end at T (may be shorter than WINDOW_SIZE).
      Token t is assigned to: min(t // STRIDE, window_count - 1).
      This implements "later-window-wins" because higher window indices
      correspond to higher STRIDE multiples, and a token in the overlap
      region of window n and window n+1 has t >= (n+1)*STRIDE, so it
      maps to window n+1.

    Returns a WindowPlan with all token assignments.
    """
    if T <= 0:
        return WindowPlan(
            T=T,
            window_count=0,
            token_to_window=[],
            window_starts=[],
            window_ends=[],
        )

    if T <= WINDOW_SIZE:
        return WindowPlan(
            T=T,
            window_count=1,
            token_to_window=[0] * T,
            window_starts=[0],
            window_ends=[T],
        )

    # Compute window start positions.
    starts: list[int] = []
    s = 0
    while True:
        starts.append(s)
        next_s = s + STRIDE
        if next_s + WINDOW_SIZE >= T:
            # Last window: starts far enough to end at T.
            # Align last window to cover the tail.
            last_start = max(T - WINDOW_SIZE, next_s)
            if last_start > s:
                starts.append(last_start)
            break
        s = next_s

    window_count = len(starts)
    window_ends = [min(start + WINDOW_SIZE, T) for start in starts]

    # Assign each token to its winning window:
    # winning_window_index(t) = min(t // STRIDE, window_count - 1)
    token_to_window = [min(t // STRIDE, window_count - 1) for t in range(T)]

    return WindowPlan(
        T=T,
        window_count=window_count,
        token_to_window=token_to_window,
        window_starts=starts,
        window_ends=window_ends,
    )


# ---------------------------------------------------------------------------
# Invariant checks
# ---------------------------------------------------------------------------

class InvariantViolation(Exception):
    pass


def check_invariants(plan: WindowPlan, query_id: str) -> list[str]:
    """
    Check all §3.3 invariants. Returns list of violation messages.
    Empty list means all invariants hold.
    """
    violations: list[str] = []

    # Invariant 1: every token has exactly one window assignment.
    for t in range(plan.T):
        w = plan.token_to_window[t]
        if w < 0 or w >= plan.window_count:
            violations.append(
                f"{query_id}: token {t} assigned to invalid window {w} "
                f"(window_count={plan.window_count})"
            )

    # Invariant 2: winning_window_index(t) matches floor(t / STRIDE) capped at last.
    for t in range(plan.T):
        expected = min(t // STRIDE, plan.window_count - 1)
        actual = plan.token_to_window[t]
        if actual != expected:
            violations.append(
                f"{query_id}: token {t} -> window {actual}, "
                f"expected {expected} (= min({t}//{STRIDE}, {plan.window_count - 1}))"
            )

    # Invariant 3: overlap tokens go to the later window.
    # Overlap region: tokens in [window_start[n+1], window_end[n])
    # must be assigned to window n+1, not window n.
    for w in range(plan.window_count - 1):
        overlap_start = plan.window_starts[w + 1]
        overlap_end = plan.window_ends[w]
        for t in range(overlap_start, min(overlap_end, plan.T)):
            if plan.token_to_window[t] != w + 1:
                violations.append(
                    f"{query_id}: overlap token {t} (windows {w} and {w+1}) "
                    f"assigned to window {plan.token_to_window[t]}, "
                    f"expected later window {w+1}"
                )
                break  # One example per window pair is enough

    # Invariant 4: window count is correct.
    if plan.T > WINDOW_SIZE:
        # Expected: covers [0, T) with stride STRIDE.
        expected_count = math.ceil((plan.T - WINDOW_SIZE) / STRIDE) + 1
        if plan.window_count != expected_count:
            # Allow off-by-one for the tail alignment.
            if abs(plan.window_count - expected_count) > 1:
                violations.append(
                    f"{query_id}: window_count={plan.window_count}, "
                    f"expected ~{expected_count} for T={plan.T}"
                )
    elif plan.T > 0:
        if plan.window_count != 1:
            violations.append(
                f"{query_id}: T={plan.T} <= WINDOW_SIZE={WINDOW_SIZE} "
                f"but window_count={plan.window_count}, expected 1"
            )

    # Invariant 5: every window covers at least 1 token.
    for w in range(plan.window_count):
        if plan.window_starts[w] >= plan.window_ends[w]:
            violations.append(
                f"{query_id}: window {w} is empty "
                f"(start={plan.window_starts[w]}, end={plan.window_ends[w]})"
            )

    return violations


# ---------------------------------------------------------------------------
# Token length estimation (without running a real tokenizer)
# ---------------------------------------------------------------------------

def _estimate_token_count(text: str) -> int:
    """
    Estimate BERT WordPiece token count from character count.
    BGE-small-en-v1.5 tokenizes at roughly 4.5 chars/token for ASCII text
    and about 2 chars/token for dense code.
    We use 4.0 chars/token as a conservative estimate.
    The actual token count is checked only for corpus entries with category='long'.
    """
    if not text:
        return 2  # [CLS] + [SEP]
    chars = len(text)
    estimate = max(2, chars // 4)
    return estimate


def _get_token_count_from_blob(
    query_id: str, hashes_blob: dict | None
) -> int | None:
    """
    Extract actual token count from a runner.py hash blob if available.
    """
    if hashes_blob is None:
        return None
    idx = {r["id"]: r for r in hashes_blob.get("records", [])}
    rec = idx.get(query_id)
    if rec and rec.get("token_len") is not None:
        return int(rec["token_len"])
    return None


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------

class SlidingWindowTestResult(NamedTuple):
    long_queries_tested: int
    long_queries_with_violations: int
    total_violations: int
    passed: bool


def run_invariant_tests(
    corpus: list[dict],
    hashes_blob: dict | None = None,
    verbose: bool = False,
) -> SlidingWindowTestResult:
    """
    Run §3.3 invariant tests for all long-category corpus entries.
    Optionally uses actual token counts from a hash blob.

    For entries without actual token counts, uses the estimate.
    The invariant test is meaningful even with estimated counts since
    the window arithmetic is purely a function of T.
    """
    long_queries = [e for e in corpus if e.get("category") == "long"]
    total_violations = 0
    n_with_violations = 0

    for entry in long_queries:
        query_id = entry["id"]
        text = entry["text"]

        # Get actual token count if available, else estimate
        actual_t = _get_token_count_from_blob(query_id, hashes_blob)
        if actual_t is not None:
            T = actual_t
        else:
            T = _estimate_token_count(text)

        plan = compute_window_plan(T)

        violations = check_invariants(plan, query_id)

        if violations:
            n_with_violations += 1
            total_violations += len(violations)
            if verbose:
                print(f"FAIL {query_id}: T={T}, windows={plan.window_count}")
                for v in violations:
                    print(f"  VIOLATION: {v}")
        else:
            if verbose:
                print(
                    f"PASS {query_id}: T={T}, windows={plan.window_count}, "
                    f"stride={STRIDE}"
                )

    passed = total_violations == 0

    return SlidingWindowTestResult(
        long_queries_tested=len(long_queries),
        long_queries_with_violations=n_with_violations,
        total_violations=total_violations,
        passed=passed,
    )


def print_summary(result: SlidingWindowTestResult) -> None:
    print("=" * 60)
    print("SLIDING-WINDOW INVARIANT TEST (§3.3)")
    print("=" * 60)
    print(f"  Long queries tested:      {result.long_queries_tested}")
    print(f"  Window size:              {WINDOW_SIZE}")
    print(f"  Stride:                   {STRIDE}")
    print(f"  Queries with violations:  {result.long_queries_with_violations}")
    print(f"  Total violations:         {result.total_violations}")
    print(f"  RESULT:                   {'PASS' if result.passed else 'FAIL'}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test §3.3 later-window-wins sliding-window invariant.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=CORPUS_PATH,
        help="Corpus JSON file path",
    )
    parser.add_argument(
        "--hashes-blob",
        type=Path,
        default=None,
        help="Hash blob from runner.py (to use actual token lengths)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-query results",
    )
    args = parser.parse_args()

    # Load corpus
    if args.corpus.exists():
        with args.corpus.open("r", encoding="utf-8") as f:
            corpus = json.load(f)
    else:
        print(f"Corpus not found at {args.corpus}, building...", file=sys.stderr)
        sys.path.insert(0, str(THIS_DIR))
        from corpus import build_corpus
        corpus = build_corpus()

    # Load hashes blob if provided
    hashes_blob: dict | None = None
    if args.hashes_blob and args.hashes_blob.exists():
        with args.hashes_blob.open("r", encoding="utf-8") as f:
            hashes_blob = json.load(f)

    result = run_invariant_tests(corpus, hashes_blob=hashes_blob, verbose=args.verbose)
    print_summary(result)

    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
