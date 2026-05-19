"""BLAS byte-identity gate: scalar vs AVX2 produce identical top-5 results.

The mind-blas Track A AVX2 path for Q16.16 dot products is byte-identical to
the scalar oracle by construction (integer-domain SIMD reduction with explicit
i64 widening per lane is associative — the result depends only on the value
sequence, not the reduction tree). This test verifies the property end-to-end
through ``score(query, top_k=5)`` on a deterministic synthetic catalog.

Two paths are exercised:
  * MIND_NERVE_BLAS=1 (default): AVX2 dispatched at .so load.
  * MIND_NERVE_BLAS=0          : scalar oracle forced.

The dispatch flag is controlled at runtime via the ``__mind_nerve_blas_set_use_avx2``
test-only hook (exposed by ``mind/runtime/blas_shims_i64.c``).

The test also records the SHA-256 of the concatenated ``(route_idx, q16_score)``
top-5 stream for 100 deterministic queries. This becomes the cross-arch
reference hash for task #57 (Q16.16 cross-arch bit-identity gate) when ARM /
CUDA / photonic backends come online.
"""

from __future__ import annotations

import ctypes
import hashlib

import numpy as np
import pytest
from mind_nerve._native import _f32_to_q16, _NativeRuntime

# Reference SHA-256 of the (idx, score) top-5 stream over 100 synthetic queries.
# Generated on i7-5930K, x86_64-Linux, AVX2 path. Updated when the synthetic
# catalog seed or the score reduction order changes (both of which would
# legitimately break this hash).
#
# Cross-arch verification (task #57): ARM / CUDA / photonic backends must
# reproduce this exact hash when fed the same seed + catalog shape.
REFERENCE_HASH_X86_AVX2 = "f4524bd56fd74e9dfbfb17b5b1f56fafda0e7e99321ef75ebce777219cda45fc"

_N_ROWS = 11922
_DIM = 384
_N_QUERIES = 100
_TOP_K = 5
_SEED = 0xA1_5_BEEF


def _make_catalog(rng: np.random.Generator) -> np.ndarray:
    """Synthetic Q16.16 catalog with shape (_N_ROWS, _DIM).

    Generated from a seeded RNG so every host produces identical bytes.
    Values are L2-normalised in f32 then quantised — this matches the live
    catalog's value distribution closely enough for the reduction to exercise
    the same accumulation regime.
    """
    f = rng.standard_normal((_N_ROWS, _DIM)).astype(np.float32)
    norms = np.linalg.norm(f, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    f = f / norms
    return np.ascontiguousarray(_f32_to_q16(f), dtype=np.int64)


def _make_queries(rng: np.random.Generator) -> list[np.ndarray]:
    """_N_QUERIES synthetic Q16.16 query vectors of shape (_DIM,)."""
    queries: list[np.ndarray] = []
    for _ in range(_N_QUERIES):
        f = rng.standard_normal(_DIM).astype(np.float32)
        f = f / max(float(np.linalg.norm(f)), 1e-8)
        queries.append(np.ascontiguousarray(_f32_to_q16(f), dtype=np.int64))
    return queries


@pytest.fixture(scope="module")
def native_runtime() -> _NativeRuntime:
    rt = _NativeRuntime()
    return rt


@pytest.fixture(scope="module")
def blas_dispatch(native_runtime: _NativeRuntime) -> dict[str, ctypes._FuncPointer]:
    """Bind the BLAS dispatcher get/set hooks."""
    lib = native_runtime._lib
    get_fn = lib.__mind_nerve_blas_get_use_avx2
    get_fn.restype = ctypes.c_int
    get_fn.argtypes = []
    set_fn = lib.__mind_nerve_blas_set_use_avx2
    set_fn.restype = ctypes.c_int
    set_fn.argtypes = [ctypes.c_int]
    return {"get": get_fn, "set": set_fn}


def _topk_from_scores(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Stable top-K: argpartition + descending stable sort on score."""
    k = min(k, scores.shape[0])
    cand = np.argpartition(-scores, k - 1)[:k]
    order = np.argsort(-scores[cand], kind="stable")
    idx = cand[order]
    return idx.astype(np.int64), scores[idx]


def test_blas_dispatch_hooks_present(
    blas_dispatch: dict[str, ctypes._FuncPointer],
) -> None:
    """The dispatch get/set hooks are exported by the .so."""
    flag = blas_dispatch["get"]()
    assert flag in (0, 1)


def test_score_byte_identical_avx2_vs_scalar(
    native_runtime: _NativeRuntime,
    blas_dispatch: dict[str, ctypes._FuncPointer],
) -> None:
    """top-5 (idx, score) results match exactly between AVX2 and scalar paths."""
    rt = native_runtime
    handle = rt.init(0, 0)
    assert handle != 0

    try:
        rng_cat = np.random.default_rng(_SEED)
        rng_q = np.random.default_rng(_SEED + 1)
        catalog = _make_catalog(rng_cat)
        queries = _make_queries(rng_q)

        # AVX2 path (default after .so load on this host).
        prev = blas_dispatch["set"](1)
        actual_flag = blas_dispatch["get"]()
        if actual_flag != 1:
            pytest.skip("AVX2 path not available on this host — byte-identity test requires both")
        avx2_results: list[tuple[np.ndarray, np.ndarray]] = []
        for qv in queries:
            scores = rt.score(handle, qv, catalog)
            idx, top_sc = _topk_from_scores(scores, _TOP_K)
            avx2_results.append((idx, top_sc))

        # Scalar oracle.
        blas_dispatch["set"](0)
        assert blas_dispatch["get"]() == 0
        scalar_results: list[tuple[np.ndarray, np.ndarray]] = []
        for qv in queries:
            scores = rt.score(handle, qv, catalog)
            idx, top_sc = _topk_from_scores(scores, _TOP_K)
            scalar_results.append((idx, top_sc))

        # Restore prior dispatch.
        blas_dispatch["set"](prev)

        # Byte-identity check on every query.
        mismatches = 0
        for q_i, ((idx_a, sc_a), (idx_s, sc_s)) in enumerate(
            zip(avx2_results, scalar_results, strict=True)
        ):
            if not np.array_equal(idx_a, idx_s):
                mismatches += 1
                if mismatches <= 3:
                    print(f"  query {q_i}: idx mismatch AVX2={idx_a} vs scalar={idx_s}")
            elif not np.array_equal(sc_a, sc_s):
                mismatches += 1
                if mismatches <= 3:
                    print(f"  query {q_i}: score mismatch AVX2={sc_a} vs scalar={sc_s}")
        assert mismatches == 0, (
            f"{mismatches}/{_N_QUERIES} queries diverged between AVX2 and scalar"
        )
    finally:
        rt.free(handle)


def test_score_topk_reference_hash(
    native_runtime: _NativeRuntime,
    blas_dispatch: dict[str, ctypes._FuncPointer],
) -> None:
    """Record SHA-256 of the (idx, score) top-5 stream — cross-arch oracle.

    On the first run this test prints the computed hash so it can be pinned
    in REFERENCE_HASH_X86_AVX2 above. Subsequent runs assert the hash is
    stable across both AVX2 and scalar paths (and, eventually, across ARM /
    CUDA / photonic backends per task #57).
    """
    rt = native_runtime
    handle = rt.init(0, 0)
    assert handle != 0

    try:
        rng_cat = np.random.default_rng(_SEED)
        rng_q = np.random.default_rng(_SEED + 1)
        catalog = _make_catalog(rng_cat)
        queries = _make_queries(rng_q)

        # Use the scalar path so the hash is the cross-arch oracle reference.
        prev = blas_dispatch["set"](0)
        try:
            hasher = hashlib.sha256()
            for qv in queries:
                scores = rt.score(handle, qv, catalog)
                idx, top_sc = _topk_from_scores(scores, _TOP_K)
                # Pack idx (int64) and score (int64) into bytes for the hash.
                hasher.update(idx.astype(np.int64).tobytes())
                hasher.update(top_sc.astype(np.int64).tobytes())
            computed = hasher.hexdigest()
        finally:
            blas_dispatch["set"](prev)

        print()
        print(f"  q16 top-5 SHA-256 (scalar oracle): {computed}")
        print(f"  REFERENCE_HASH_X86_AVX2          : {REFERENCE_HASH_X86_AVX2}")

        if REFERENCE_HASH_X86_AVX2 != "<recorded-on-first-run>":
            assert computed == REFERENCE_HASH_X86_AVX2, (
                f"top-5 hash drifted: computed={computed} reference={REFERENCE_HASH_X86_AVX2}"
            )
        else:
            # First run: surface the hash but do not fail; orchestrator pins it
            # into REFERENCE_HASH_X86_AVX2 once measurement is complete.
            print(
                "  (REFERENCE_HASH_X86_AVX2 placeholder — pin this value into "
                "the constant at the top of this file before tagging.)"
            )
    finally:
        rt.free(handle)
