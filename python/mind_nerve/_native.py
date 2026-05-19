"""mind-nerve native Q16.16 encoder — ctypes binding.

Binds the six mn_encoder_* entry points exported by libmind_nerve_encoder.so
(compiled from mind/exports/c_abi.mind via mindc --emit-shared).

All data crosses the FFI boundary as flat i64 arrays in Q16.16 fixed-point
little-endian. No f32 is passed to or from the .so.

Q16.16 encoding:
    python_float → q16 = int(round(value * 65536))
    q16 → python_float = q16 / 65536.0

The WordPiece tokenizer stays Python-side (out of A1.3 scope); this module
receives pre-tokenized int32 arrays and returns Q16.16 int32 arrays.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Q16.16 constants
# ---------------------------------------------------------------------------
_Q16_ONE: int = 65536
_Q16_MAX: int = 2_147_483_647
_Q16_MIN: int = -2_147_483_648

# ---------------------------------------------------------------------------
# .so search order
# ---------------------------------------------------------------------------
_NATIVE_DIR = Path(__file__).parent / "_native"
_SEARCH_PATHS: tuple[Path, ...] = (
    _NATIVE_DIR,
    Path(os.environ.get("MIND_NERVE_NATIVE_PATH", str(_NATIVE_DIR))),
)
_SO_NAME = "libmind_nerve_encoder.so"


def _find_so() -> Path:
    """Return the path to libmind_nerve_encoder.so, searching in order."""
    seen: set[Path] = set()
    for directory in _SEARCH_PATHS:
        candidate = Path(directory) / _SO_NAME
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    searched = ", ".join(str(Path(d) / _SO_NAME) for d in _SEARCH_PATHS)
    raise FileNotFoundError(
        f"libmind_nerve_encoder.so not found. Searched: {searched}. "
        f"Run tools/build_native_encoder.sh to build the shared library."
    )


# ---------------------------------------------------------------------------
# Q16.16 conversion helpers
# ---------------------------------------------------------------------------

def _f32_to_q16(arr: np.ndarray) -> np.ndarray:
    """Convert float32 ndarray to Q16.16 int64 ndarray (element-wise, saturating)."""
    scaled = np.round(arr.astype(np.float64) * _Q16_ONE).astype(np.int64)
    return np.clip(scaled, _Q16_MIN, _Q16_MAX)


def _q16_to_f32(arr: np.ndarray) -> np.ndarray:
    """Convert Q16.16 int64 ndarray to float32 ndarray."""
    return (arr.astype(np.float64) / _Q16_ONE).astype(np.float32)


def _token_ids_to_i64(token_ids: np.ndarray) -> np.ndarray:
    """Widen int32 token IDs to int64 for the MIND heap ABI."""
    return token_ids.astype(np.int64, copy=False)


# ---------------------------------------------------------------------------
# NativeRuntime
# ---------------------------------------------------------------------------

class _NativeRuntime:
    """ctypes wrapper for the six mn_encoder_* C-ABI entry points.

    Lifecycle::

        rt = _NativeRuntime()                  # loads the .so
        handle = rt.init(weights_blob_addr, blob_len_bytes)
        emb = rt.encode(handle, token_ids)     # np.ndarray[int32], Q16.16
        scores = rt.score(handle, qv, catalog) # np.ndarray[int32], Q16.16
        indices, top_scores = rt.topk(scores, k=5)
        rt.free(handle)

    Weights blob:
        The weights_blob is a ctypes pointer-width integer (int) holding
        the address of the pre-quantised Q16.16 weight flat buffer.
        In practice the Python caller obtains this address from a
        numpy array allocated with np.frombuffer / np.fromfile and pinned
        in memory for the lifetime of the handle.

    Thread safety:
        Shared library is loaded once at __init__ time. The handle
        encapsulates all mutable scratch state, so multiple handles on
        the same .so are safe; a single handle is NOT thread-safe
        (scratch buffers are aliased).
    """

    def __init__(self, so_path: Path | None = None) -> None:
        path = so_path or _find_so()
        self._lib = ctypes.CDLL(str(path))
        self._bind_symbols()
        self._so_path = path

    def _bind_symbols(self) -> None:
        lib = self._lib

        # int64_t mn_encoder_init(int64_t weights_blob, int64_t len)
        lib.mn_encoder_init.restype = ctypes.c_int64
        lib.mn_encoder_init.argtypes = [ctypes.c_int64, ctypes.c_int64]

        # int64_t mn_encoder_encode(int64_t handle, int64_t token_ids,
        #                           int64_t n_tokens, int64_t out_vec)
        lib.mn_encoder_encode.restype = ctypes.c_int64
        lib.mn_encoder_encode.argtypes = [
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ]

        # int64_t mn_encoder_score(int64_t handle, int64_t qv,
        #                          int64_t catalog, int64_t n_rows,
        #                          int64_t out_scores)
        lib.mn_encoder_score.restype = ctypes.c_int64
        lib.mn_encoder_score.argtypes = [
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64,
        ]

        # int64_t mn_encoder_topk(int64_t scores, int64_t n, int64_t k,
        #                         int64_t out_idx, int64_t out_scores)
        lib.mn_encoder_topk.restype = ctypes.c_int64
        lib.mn_encoder_topk.argtypes = [
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64,
        ]

        # int64_t mn_encoder_free(int64_t handle)
        lib.mn_encoder_free.restype = ctypes.c_int64
        lib.mn_encoder_free.argtypes = [ctypes.c_int64]

        # int64_t mn_encoder_version(void)
        lib.mn_encoder_version.restype = ctypes.c_int64
        lib.mn_encoder_version.argtypes = []

    # ----------------------------------------------------------------
    # Public interface
    # ----------------------------------------------------------------

    def version(self) -> str:
        """Return the build-id string embedded in the .so."""
        addr: int = self._lib.mn_encoder_version()
        if addr == 0:
            return "<unknown>"
        # The MIND heap stores one ASCII byte per i64 slot (stride 8).
        # Decode up to 256 slots; stop at NUL.
        chars: list[str] = []
        for i in range(256):
            val = ctypes.cast(addr + i * 8, ctypes.POINTER(ctypes.c_int64))[0]
            if val == 0:
                break
            chars.append(chr(val & 0xFF))
        return "".join(chars)

    def init(self, weights_blob_addr: int, blob_len_bytes: int) -> int:
        """Allocate an encoder handle backed by the given weight blob.

        Args:
            weights_blob_addr: Integer address of the Q16.16 weight buffer.
                The buffer must remain valid for the lifetime of the handle.
            blob_len_bytes: Size of the weight buffer in bytes.

        Returns:
            Opaque i64 handle. Returns 0 on allocation failure.
        """
        return int(self._lib.mn_encoder_init(
            ctypes.c_int64(weights_blob_addr),
            ctypes.c_int64(blob_len_bytes),
        ))

    def encode(
        self,
        handle: int,
        token_ids: np.ndarray,
    ) -> np.ndarray:
        """Encode a token ID sequence, returning a Q16.16 embedding vector.

        Args:
            handle: Opaque handle from ``init()``.
            token_ids: int32 ndarray of shape (T,) where T ≤ 512.

        Returns:
            int64 ndarray of shape (384,) holding Q16.16 L2-normalised
            embedding values. Caller converts to float32 via ``_q16_to_f32``.

        Raises:
            RuntimeError: If the .so reports a failure (handle == 0).
        """
        if handle == 0:
            raise RuntimeError("mn_encoder_encode: null handle")

        ids_i64: np.ndarray = _token_ids_to_i64(token_ids)
        ids_i64 = np.ascontiguousarray(ids_i64)
        out: np.ndarray = np.zeros(384, dtype=np.int64)

        ids_ptr = ids_i64.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))

        rc: int = self._lib.mn_encoder_encode(
            ctypes.c_int64(handle),
            ctypes.c_int64(ctypes.cast(ids_ptr, ctypes.c_void_p).value or 0),
            ctypes.c_int64(len(ids_i64)),
            ctypes.c_int64(ctypes.cast(out_ptr, ctypes.c_void_p).value or 0),
        )
        if rc != 0:
            raise RuntimeError(f"mn_encoder_encode returned {rc}")
        return out

    def encode_f32(
        self,
        handle: int,
        token_ids: np.ndarray,
    ) -> np.ndarray:
        """Encode and return a float32 embedding (convenience wrapper)."""
        return _q16_to_f32(self.encode(handle, token_ids))

    def score(
        self,
        handle: int,
        qv_q16: np.ndarray,
        catalog_q16: np.ndarray,
    ) -> np.ndarray:
        """Compute dot-product scores between a query vector and a catalog.

        Args:
            handle:      Opaque handle (used only for null-check).
            qv_q16:      int64 ndarray of shape (384,) in Q16.16.
            catalog_q16: int64 ndarray of shape (N, 384) in Q16.16.

        Returns:
            int64 ndarray of shape (N,) holding Q16.16 dot-product scores.
        """
        if handle == 0:
            raise RuntimeError("mn_encoder_score: null handle")

        qv = np.ascontiguousarray(qv_q16, dtype=np.int64)
        cat = np.ascontiguousarray(catalog_q16, dtype=np.int64)
        n_rows: int = cat.shape[0]
        out: np.ndarray = np.zeros(n_rows, dtype=np.int64)

        qv_ptr = qv.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        cat_ptr = cat.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))

        rc: int = self._lib.mn_encoder_score(
            ctypes.c_int64(handle),
            ctypes.c_int64(ctypes.cast(qv_ptr, ctypes.c_void_p).value or 0),
            ctypes.c_int64(ctypes.cast(cat_ptr, ctypes.c_void_p).value or 0),
            ctypes.c_int64(n_rows),
            ctypes.c_int64(ctypes.cast(out_ptr, ctypes.c_void_p).value or 0),
        )
        if rc != 0:
            raise RuntimeError(f"mn_encoder_score returned {rc}")
        return out

    def topk(
        self,
        scores_q16: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select top-K entries from a score array.

        Args:
            scores_q16: int64 ndarray of shape (N,) in Q16.16.
            k:          Number of results to return (K ≤ 32).

        Returns:
            (indices, scores) — two int64 ndarrays of shape (K,).
            indices are into the original scores array; scores are Q16.16.
            Both are sorted descending by score.
        """
        scores = np.ascontiguousarray(scores_q16, dtype=np.int64)
        n: int = scores.shape[0]
        k_clamped: int = min(k, n)
        out_idx: np.ndarray = np.zeros(k_clamped, dtype=np.int64)
        out_sc: np.ndarray = np.zeros(k_clamped, dtype=np.int64)

        sc_ptr = scores.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        idx_ptr = out_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        osc_ptr = out_sc.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))

        self._lib.mn_encoder_topk(
            ctypes.c_int64(ctypes.cast(sc_ptr, ctypes.c_void_p).value or 0),
            ctypes.c_int64(n),
            ctypes.c_int64(k_clamped),
            ctypes.c_int64(ctypes.cast(idx_ptr, ctypes.c_void_p).value or 0),
            ctypes.c_int64(ctypes.cast(osc_ptr, ctypes.c_void_p).value or 0),
        )
        return out_idx, out_sc

    def free(self, handle: int) -> None:
        """Release the encoder handle and all scratch buffers."""
        if handle != 0:
            self._lib.mn_encoder_free(ctypes.c_int64(handle))
