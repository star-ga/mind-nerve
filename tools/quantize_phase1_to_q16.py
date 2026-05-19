"""Offline Phase 6.2 quantizer — FP32 catalog → Q16.16 ``route_table.q16.bin``.

Reads a precomputed catalog (`route_table.npy`, dtype ``float32``, shape
``(N_rows, hidden_dim)``) and emits:

  * ``route_table.q16.bin`` — flat row-major Q16.16 blob, ``int64`` little-
    endian per element, no header.
  * ``route_table.q16.meta.json`` — deterministic reproducibility metadata
    including the SHA-256 of the binary blob.

The Phase 1 PyTorch checkpoint is optional; when supplied, its SHA-256
contributes to the meta JSON only. The quantizer never reads PyTorch
state dicts in Phase 1 — the catalog ``.npy`` already carries the encoder
output that participates in scoring.

Quantization scheme (authoritative spec: `spec/quantization.md`):

  * scale = ``2^16 = 65536``
  * rounding = round-half-to-even (``numpy.round``)
  * saturation = clamp to ``[INT32_MIN, INT32_MAX]``
  * on-disk encoding = ``int64`` little-endian (Q16.16 widened for the
    MIND heap ABI's i64-only loads in ``mind/kernels/encode.mind``)

The quantizer is deterministic: same input bytes → byte-identical output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

QUANTIZER_VERSION = "1.0"
SCHEMA_VERSION = 1
KIND = "mind_nerve.quantize.route_table"

# Q16.16 constants
SCALE: int = 1 << 16  # 65536
INT32_MAX: int = 2_147_483_647
INT32_MIN: int = -2_147_483_648

DEFAULT_HIDDEN_DIM: int = 384


# ---------------------------------------------------------------------------
# Pure-Python round-trip helpers
# ---------------------------------------------------------------------------


def f32_to_q16(value: float) -> int:
    """Quantize one ``float`` to Q16.16 ``int`` with round-half-to-even.

    The single-element form of the array quantizer; used by the round-trip
    unit tests. ``float64`` arithmetic; final clamp via ``min``/``max`` so
    out-of-range inputs saturate at ``INT32_MAX`` / ``INT32_MIN`` rather
    than wrapping.
    """
    if not np.isfinite(value):
        raise ValueError("f32_to_q16: non-finite input")
    scaled = float(value) * float(SCALE)
    # numpy.round implements IEEE-754 round-half-to-even; the Python-builtin
    # round() is banker's-rounding for floats only and exhibits float-repr
    # artifacts that the spec forbids.
    rounded = float(np.round(scaled))
    as_int = int(rounded)
    if as_int > INT32_MAX:
        return INT32_MAX
    if as_int < INT32_MIN:
        return INT32_MIN
    return as_int


def q16_to_f32(q: int) -> float:
    """Dequantize one Q16.16 ``int`` to a ``float``.

    Exact: every Q16.16 integer in ``[INT32_MIN, INT32_MAX]`` is exactly
    representable in IEEE-754 ``float64``, so ``int → float → int`` is
    bit-perfect.
    """
    return float(q) / float(SCALE)


# ---------------------------------------------------------------------------
# Vectorized quantization (the hot path; deterministic NumPy ops only)
# ---------------------------------------------------------------------------


def quantize_array(arr: np.ndarray) -> tuple[np.ndarray, int]:
    """Quantize an ``f32`` (or ``f64``) ndarray to Q16.16 ``int64`` (widened).

    Returns ``(q16_int64, saturated_count)``. The ``saturated_count`` is
    the number of elements that hit the ``[INT32_MIN, INT32_MAX]`` clamp.

    Deterministic across platforms:

      1. Cast to ``float64`` first. ``SCALE`` is exactly representable;
         the product carries 52 mantissa bits.
      2. ``np.round`` (round-half-to-even, ties to even). Result stays
         ``float64``.
      3. Saturation in the ``float64`` domain via ``np.clip`` against
         the int32 boundaries (themselves exactly representable as
         ``float64``).
      4. Cast to ``int64`` *after* the clamp. The final widen to i64
         is a no-op on value (it is widened by sign extension) and
         matches the MIND heap ABI's i64-only loads.
    """
    if arr.size == 0:
        return np.empty((0,), dtype=np.int64), 0

    as_f64 = arr.astype(np.float64, copy=False)
    scaled = as_f64 * np.float64(SCALE)
    # Allocate an explicit ``out`` buffer so the round happens in-place
    # against a deterministic memory layout. Avoids any reduction-style
    # path the np.round implementation might take.
    rounded = np.empty_like(scaled)
    np.round(scaled, out=rounded)
    # Count saturations BEFORE the clamp (clamp loses the information).
    saturated = int(np.count_nonzero(rounded > INT32_MAX)) + int(
        np.count_nonzero(rounded < INT32_MIN)
    )
    clamped = np.clip(rounded, np.float64(INT32_MIN), np.float64(INT32_MAX))
    # Cast to int64. ``casting='unsafe'`` because we know the value is in
    # the int32 range after the clamp; NumPy refuses to downcast f64 → i64
    # implicitly otherwise.
    q16 = clamped.astype(np.int64, casting="unsafe")
    return q16, saturated


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------


def _sha256_bytes(buf: bytes) -> str:
    return hashlib.sha256(buf).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_checkpoint_dir(checkpoint_dir: Path) -> str:
    """SHA-256 over every file in ``checkpoint_dir`` in sorted relative-path order.

    Matches the algorithm in ``python/mind_nerve/mind_train.py``
    ``_compute_checkpoint_hash`` so the same checkpoint produces the same
    hash whether it is consumed by the trainer or the quantizer.
    """
    h = hashlib.sha256()
    for p in sorted(checkpoint_dir.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(checkpoint_dir).as_posix().encode("utf-8"))
            h.update(b"\x00")
            h.update(p.read_bytes())
    return h.hexdigest()


def _checkpoint_hash(checkpoint_path: Path | None) -> str | None:
    if checkpoint_path is None:
        return None
    if checkpoint_path.is_dir():
        return _sha256_checkpoint_dir(checkpoint_path)
    if checkpoint_path.is_file():
        return _sha256_file(checkpoint_path)
    return None


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------


def _ordered_meta(
    *,
    n_rows: int,
    hidden_dim: int,
    byte_size: int,
    bin_sha256: str,
    catalog_path: Path,
    catalog_sha256: str,
    catalog_dtype: str,
    checkpoint_path: Path | None,
    checkpoint_hash: str | None,
    min_q16: int,
    max_q16: int,
    saturated_count: int,
    produced_at_iso: str,
) -> dict[str, Any]:
    """Build the meta JSON dict with keys in the spec-mandated order."""
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "quantizer_version": QUANTIZER_VERSION,
        "n_rows": n_rows,
        "hidden_dim": hidden_dim,
        "scale": SCALE,
        "rounding": "half_to_even",
        "saturation": "int32",
        "dtype_disk": "int64_le",
        "dtype_value": "int32_q16_16",
        "byte_size": byte_size,
        "sha256": bin_sha256,
        "source": {
            "catalog_npy_path": str(catalog_path),
            "catalog_npy_sha256": catalog_sha256,
            "catalog_npy_dtype": catalog_dtype,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "checkpoint_hash": checkpoint_hash,
        },
        "stats": {
            "min_q16": min_q16,
            "max_q16": max_q16,
            "saturated_count": saturated_count,
        },
        "produced_at_iso": produced_at_iso,
        "produced_by": "mind-nerve quantize",
    }


# ---------------------------------------------------------------------------
# Resolver helpers
# ---------------------------------------------------------------------------


def resolve_default_output_dir() -> Path:
    """Return the default ``--output`` directory.

    Resolution order, first hit wins:

      1. ``$MIND_NERVE_RUNTIME_DIR`` if set.
      2. ``~/.cache/mind-nerve/q16/``.

    The returned directory is created at mode 0700 if absent. The 0700
    mode matches the rest of mind-nerve's per-user runtime hardening.
    """
    env_dir = os.environ.get("MIND_NERVE_RUNTIME_DIR")
    if env_dir:
        out = Path(env_dir).expanduser()
    else:
        out = Path.home() / ".cache" / "mind-nerve" / "q16"
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(out, 0o700)
    except OSError:
        # Best-effort; not fatal on filesystems that don't honour mode bits.
        pass
    return out


# ---------------------------------------------------------------------------
# Main quantize entry
# ---------------------------------------------------------------------------


def quantize_catalog(
    catalog_path: Path,
    output_dir: Path,
    *,
    checkpoint_path: Path | None = None,
    hidden_dim: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Quantize an ``f32`` catalog ``.npy`` into ``route_table.q16.bin``.

    Returns the meta-JSON dict (also written to ``route_table.q16.meta.json``
    unless ``dry_run`` is True).
    """
    if not catalog_path.is_file():
        raise FileNotFoundError(f"catalog not found: {catalog_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog = np.load(str(catalog_path))
    if catalog.ndim != 2:
        raise ValueError(f"catalog must be 2-D (N_rows, hidden_dim); got shape {catalog.shape}")
    n_rows, cat_hidden = catalog.shape
    if hidden_dim is not None and cat_hidden != hidden_dim:
        raise ValueError(
            f"catalog hidden_dim {cat_hidden} does not match --hidden-dim {hidden_dim}"
        )
    catalog_dtype = str(catalog.dtype)
    # Always promote to float32 for the quantize call; the spec pins f64
    # internally but the input contract is f32 ``.npy``. Promotion via
    # astype with copy=False keeps the array identity when already f32.
    f32_view = catalog.astype(np.float32, copy=False)

    catalog_sha256 = _sha256_file(catalog_path)
    checkpoint_hash = _checkpoint_hash(checkpoint_path)

    q16_int64, saturated_count = quantize_array(f32_view)
    # Force C-contiguous + native byte order so .tobytes() is well-defined.
    q16_int64 = np.ascontiguousarray(q16_int64.reshape(n_rows, cat_hidden), dtype="<i8")

    min_q16 = int(q16_int64.min()) if q16_int64.size else 0
    max_q16 = int(q16_int64.max()) if q16_int64.size else 0

    bin_bytes = q16_int64.tobytes(order="C")
    byte_size = len(bin_bytes)
    expected = n_rows * cat_hidden * 8
    if byte_size != expected:
        raise RuntimeError(
            f"internal: bin size {byte_size} != expected {expected}; "
            "likely a non-contiguous reshape"
        )
    bin_sha256 = _sha256_bytes(bin_bytes)

    produced_at_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta = _ordered_meta(
        n_rows=n_rows,
        hidden_dim=cat_hidden,
        byte_size=byte_size,
        bin_sha256=bin_sha256,
        catalog_path=catalog_path,
        catalog_sha256=catalog_sha256,
        catalog_dtype=catalog_dtype,
        checkpoint_path=checkpoint_path,
        checkpoint_hash=checkpoint_hash,
        min_q16=min_q16,
        max_q16=max_q16,
        saturated_count=saturated_count,
        produced_at_iso=produced_at_iso,
    )

    if dry_run:
        return meta

    bin_path = output_dir / "route_table.q16.bin"
    meta_path = output_dir / "route_table.q16.meta.json"

    # Atomic-ish write: temp + rename. Same dir → rename is atomic on POSIX.
    tmp_bin = bin_path.with_suffix(".bin.tmp")
    tmp_bin.write_bytes(bin_bytes)
    os.replace(tmp_bin, bin_path)

    # Round-trip the meta dict through json.dumps with sort_keys=False so
    # the spec-mandated key order survives. Pretty-print for human diff.
    meta_text = json.dumps(meta, indent=2, sort_keys=False) + "\n"
    tmp_meta = meta_path.with_suffix(".json.tmp")
    tmp_meta.write_text(meta_text, encoding="utf-8")
    os.replace(tmp_meta, meta_path)

    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="quantize_phase1_to_q16",
        description="Quantize an FP32 catalog .npy into a Q16.16 route_table.q16.bin blob.",
    )
    ap.add_argument(
        "--catalog",
        required=True,
        help="Path to route_table.npy (float32, shape (N_rows, hidden_dim))",
    )
    ap.add_argument(
        "--input",
        default=None,
        help=(
            "Optional: path to the PyTorch checkpoint directory (or single file). "
            "Hashed into the meta JSON for reproducibility; not read for weights "
            "in Phase 1. Pass ``:none:`` or omit to skip."
        ),
    )
    ap.add_argument(
        "--output",
        default=None,
        help=("Output directory. Default: $MIND_NERVE_RUNTIME_DIR or ~/.cache/mind-nerve/q16/"),
    )
    ap.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Expected hidden dimension (default: catalog's column count).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the meta JSON to stdout without writing any file.",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    catalog_path = Path(args.catalog).expanduser()
    output_dir = Path(args.output).expanduser() if args.output else resolve_default_output_dir()
    checkpoint_path: Path | None
    if args.input in (None, ":none:", ""):
        checkpoint_path = None
    else:
        checkpoint_path = Path(args.input).expanduser()

    try:
        meta = quantize_catalog(
            catalog_path=catalog_path,
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            hidden_dim=args.hidden_dim,
            dry_run=args.dry_run,
        )
    except FileNotFoundError as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1
    except ValueError as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 2

    print(json.dumps(meta, indent=2, sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
