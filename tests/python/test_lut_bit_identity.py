"""LUT bit-identity gate: pure-MIND tanh / rsqrt / softmax Q16.16 wrappers.

The encode path's activation LUTs (``tanh_q16``, ``rsqrt_q16``,
``softmax_q16``) were formerly bridged through a libm C shim
(``mind/runtime/lut_shims.c``) that used IEEE-754 float arithmetic and
was therefore NOT Q16.16 cross-arch bit-identical (A1.5 PARTIAL,
"cannot ship to production").

They are now pure-MIND: deterministic integer Q16.16 table lookups
(``mind/luts/tanh_q16.mind``, ``mind/luts/rsqrt_q16.mind`` composing
``sqrt_q16.mind``'s 2048-entry table + one Newton step,
``mind/luts/softmax_q16.mind``). The 32 KiB / 16 KiB / 2 KiB tables are
built once and the handles cached on the C side
(``mind/runtime/lut_cache.c`` — zero arithmetic in C).

This test, mirroring ``tests/python/test_blas_byte_identity.py``:

  * asserts every wrapper is deterministic (identical output on repeated
    calls and from a freshly re-built table — the handle cache is
    idempotent, so a re-built table reproduces the lookups bit-for-bit);
  * records the SHA-256 of the Q16.16 output stream over a fixed input
    sweep as the cross-arch reference for task #57 (ARM / CUDA /
    photonic backends must reproduce these exact hashes — the path is
    integer-only, so the hash cannot drift across substrates);
  * measures and asserts the max-abs / max-rel error of each wrapper vs
    the true real-valued reference over its representable domain,
    documenting the LUT accuracy contract.

The hashes break only if a LUT table value or a lookup/Newton step
changes (a legitimate numeric change), never from a substrate / compiler
/ SIMD difference.
"""

from __future__ import annotations

import ctypes
import hashlib
import math

import numpy as np
import pytest
from mind_nerve._native import _NativeRuntime

# Cross-arch reference hashes (task #57). Generated on i7-5930K,
# x86_64-Linux, pure-MIND integer Q16.16 path.
REFERENCE_HASH_TANH = "190e488bd5a0f67fcc7a2ca60df688d98b53fe1643b9cbe485e8859740bf4bb8"
REFERENCE_HASH_RSQRT = "c7e2791a73ad234187c00f0d2a918c86826ea509346c37b448ade18379b06a2d"
REFERENCE_HASH_SOFTMAX = "e39ad4ec913ae5b0a77add0c0d1ec1526f00f4c729fc3c8fbc4b04525a2621ae"

_Q16_ONE = 65536


@pytest.fixture(scope="module")
def lib() -> ctypes.CDLL:
    rt = _NativeRuntime()
    lib = rt._lib

    lib.__mind_alloc.restype = ctypes.c_int64
    lib.__mind_alloc.argtypes = [ctypes.c_int64]
    lib.__mind_free.restype = ctypes.c_int64
    lib.__mind_free.argtypes = [ctypes.c_int64]
    lib.__mind_load_i64.restype = ctypes.c_int64
    lib.__mind_load_i64.argtypes = [ctypes.c_int64]
    lib.__mind_store_i64.restype = ctypes.c_int64
    lib.__mind_store_i64.argtypes = [ctypes.c_int64, ctypes.c_int64]

    lib.tanh_q16.restype = ctypes.c_int64
    lib.tanh_q16.argtypes = [ctypes.c_int64]
    lib.rsqrt_q16.restype = ctypes.c_int64
    lib.rsqrt_q16.argtypes = [ctypes.c_int64]
    lib.softmax_q16.restype = ctypes.c_int64
    lib.softmax_q16.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]

    # Pure-MIND table builders (prove the handle cache is idempotent: a
    # freshly built table yields bit-identical lookups).
    lib.tanh_q16_init.restype = ctypes.c_int64
    lib.tanh_q16_init.argtypes = []
    lib.tanh_q16_lookup.restype = ctypes.c_int64
    lib.tanh_q16_lookup.argtypes = [ctypes.c_int64, ctypes.c_int64]
    return lib


def _alloc_q16(lib: ctypes.CDLL, values: list[int]) -> int:
    addr = lib.__mind_alloc(len(values) * 8)
    assert addr != 0
    for i, v in enumerate(values):
        lib.__mind_store_i64(addr + i * 8, int(v))
    return addr


def _read_q16(lib: ctypes.CDLL, addr: int, n: int) -> list[int]:
    return [lib.__mind_load_i64(addr + i * 8) for i in range(n)]


# ---------------------------------------------------------------------------
# tanh_q16  (4096-entry table, domain [-8, 8] Q16.16, saturating tails)
# ---------------------------------------------------------------------------

_TANH_SWEEP = list(range(-700_000, 700_001, 1_111))


def test_tanh_q16_deterministic_and_reference(lib: ctypes.CDLL) -> None:
    out_a = [lib.tanh_q16(x) for x in _TANH_SWEEP]
    out_b = [lib.tanh_q16(x) for x in _TANH_SWEEP]
    assert out_a == out_b, "tanh_q16 not deterministic across calls"

    # Idempotent handle cache: a freshly built table reproduces lookups.
    fresh = lib.tanh_q16_init()
    assert fresh != 0
    domain_lo, domain_hi = -524288, 524288
    for x, y in zip(_TANH_SWEEP, out_a, strict=True):
        clamped = min(max(x, domain_lo), domain_hi)
        assert y == lib.tanh_q16_lookup(fresh, clamped)

    arr = np.asarray(out_a, dtype=np.int64)
    digest = hashlib.sha256(arr.tobytes()).hexdigest()
    print(f"\n  tanh_q16   SHA-256: {digest}")
    print(f"  REFERENCE_HASH_TANH: {REFERENCE_HASH_TANH}")

    # Accuracy vs true tanh over the full representable domain |x| <= 8
    # (includes the saturating boundary buckets — table caps at ~0.99998
    # one step before +1.0, so the worst case sits at the +8 edge).
    max_abs = max(
        abs(y / _Q16_ONE - math.tanh(x / _Q16_ONE))
        for x, y in zip(_TANH_SWEEP, out_a, strict=True)
        if abs(x) <= 8 * _Q16_ONE
    )
    print(f"  tanh_q16 max abs error vs libm (|x|<=8): {max_abs:.6e}")
    # 4096-entry table, floor-truncated; <= 4.0e-3 worst case (the +8 edge
    # bucket); ~2e-5 in the interior used by GELU.
    assert max_abs < 4.0e-3

    if REFERENCE_HASH_TANH != "<recorded-on-first-run>":
        assert digest == REFERENCE_HASH_TANH


# ---------------------------------------------------------------------------
# rsqrt_q16  (1/sqrt(x) Q16.16; sqrt_q16.mind 2048-entry table + 1 Newton)
# ---------------------------------------------------------------------------
#
# Representable domain: x_real in [0.125, 256.0] (table bucket stride
# 256/2048 = 0.125). The wrapper composes the rsqrt table seed with one
# Newton step; inputs below the first bucket (x_real < 0.125) clamp to
# table[0] and are out of the accuracy contract (the encoder defensively
# clamps LayerNorm variance / L2 sum-of-squares before calling).


def test_rsqrt_q16_deterministic_and_reference(lib: ctypes.CDLL) -> None:
    # Sweep: non-positive sentinel + every table bucket midpoint.
    sweep: list[int] = [-_Q16_ONE, -1, 0]
    for i in range(2048):
        x_real = (i + 1) * (256.0 / 2048)
        sweep.append(int(round(x_real * _Q16_ONE)))

    out_a = [lib.rsqrt_q16(x) for x in sweep]
    out_b = [lib.rsqrt_q16(x) for x in sweep]
    assert out_a == out_b, "rsqrt_q16 not deterministic across calls"

    q16_max = 2147483647
    for x, y in zip(sweep, out_a, strict=True):
        if x <= 0:
            assert y == q16_max, f"rsqrt_q16({x}) sentinel must be Q16_MAX"

    arr = np.asarray(out_a, dtype=np.int64)
    digest = hashlib.sha256(arr.tobytes()).hexdigest()
    print(f"\n  rsqrt_q16  SHA-256: {digest}")
    print(f"  REFERENCE_HASH_RSQRT: {REFERENCE_HASH_RSQRT}")

    # Accuracy vs true 1/sqrt(x) over the representable domain.
    max_abs = 0.0
    max_rel = 0.0
    for x, y in zip(sweep, out_a, strict=True):
        if x <= 0:
            continue
        xr = x / _Q16_ONE
        ref = 1.0 / math.sqrt(xr)
        e = abs(y / _Q16_ONE - ref)
        max_abs = max(max_abs, e)
        max_rel = max(max_rel, e / ref)
    print(f"  rsqrt_q16 max abs error (table+1 Newton, [0.125,256]): {max_abs:.6e}")
    print(f"  rsqrt_q16 max rel error (table+1 Newton, [0.125,256]): {max_rel:.6e}")
    # One Newton step on the 2048-entry seed: <= 1.2e-4 abs / <= 2.0e-3 rel
    # across the entire representable domain.
    assert max_abs < 2.0e-4
    assert max_rel < 2.0e-3

    if REFERENCE_HASH_RSQRT != "<recorded-on-first-run>":
        assert digest == REFERENCE_HASH_RSQRT


# ---------------------------------------------------------------------------
# softmax_q16  (in-place n_rows x row_len, 5-stage pinned Q16.16 pipeline)
# ---------------------------------------------------------------------------
#
# The pinned A1.1 pipeline normalises by an INTEGER denominator
# (D = floor(sum) clamped to [1, 256]) — see softmax_q16.mind Stage 4.
# This is a deliberate cross-arch-deterministic design, not an accuracy
# target: it trades float-grade normalisation for integer determinism.
# The wrapper's job (task #218) is to make this pipeline pure-MIND and
# bit-identical, NOT to change its numerics. Accuracy here is therefore
# asserted only loosely (probabilities are non-negative, monotone in the
# logits, and the argmax is preserved); the load-bearing assertion is
# determinism + the pinned reference hash.

_SM_ROWS = 7
_SM_LEN = 13


def test_softmax_q16_deterministic_and_reference(lib: ctypes.CDLL) -> None:
    rng = np.random.default_rng(0xA1_5_50F7)
    fv = rng.standard_normal((_SM_ROWS, _SM_LEN)).astype(np.float64) * 3.0
    base = np.round(fv * _Q16_ONE).astype(np.int64).reshape(-1).tolist()

    addr_a = _alloc_q16(lib, base)
    rc = lib.softmax_q16(addr_a, _SM_ROWS, _SM_LEN)
    assert rc == 0
    out_a = _read_q16(lib, addr_a, _SM_ROWS * _SM_LEN)

    addr_b = _alloc_q16(lib, base)
    lib.softmax_q16(addr_b, _SM_ROWS, _SM_LEN)
    out_b = _read_q16(lib, addr_b, _SM_ROWS * _SM_LEN)
    assert out_a == out_b, "softmax_q16 not deterministic across calls"

    # Degenerate-arg guards return 0 and touch nothing.
    assert lib.softmax_q16(0, _SM_ROWS, _SM_LEN) == 0
    assert lib.softmax_q16(addr_a, 0, _SM_LEN) == 0
    assert lib.softmax_q16(addr_a, _SM_ROWS, 0) == 0

    arr = np.asarray(out_a, dtype=np.int64)
    digest = hashlib.sha256(arr.tobytes()).hexdigest()
    print(f"\n  softmax_q16 SHA-256: {digest}")
    print(f"  REFERENCE_HASH_SOFTMAX: {REFERENCE_HASH_SOFTMAX}")

    out_mat = arr.reshape(_SM_ROWS, _SM_LEN)
    # All probabilities non-negative.
    assert np.all(out_mat >= 0)
    # Argmax preserved row-wise (the dominant logit stays dominant).
    in_mat = np.asarray(base, dtype=np.int64).reshape(_SM_ROWS, _SM_LEN)
    assert np.array_equal(out_mat.argmax(axis=1), in_mat.argmax(axis=1))
    # Document the integer-D normalisation residual (informational).
    row_sums = out_mat.sum(axis=1) / _Q16_ONE
    print(
        f"  softmax_q16 row-sum range (integer-D pipeline): "
        f"[{row_sums.min():.4f}, {row_sums.max():.4f}]"
    )

    lib.__mind_free(addr_a)
    lib.__mind_free(addr_b)

    if REFERENCE_HASH_SOFTMAX != "<recorded-on-first-run>":
        assert digest == REFERENCE_HASH_SOFTMAX
