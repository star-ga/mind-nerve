"""Offline Phase 6.x quantizer — FP32 encoder checkpoint → ``encoder_weights.q16.bin``.

Reads the Phase-1 fine-tuned SentenceTransformer checkpoint
(``model.safetensors``, a 12-layer BGE-small-en-v1.5 / BERT encoder) and
emits the single contiguous Q16.16 weight blob consumed by the native
MIND encoder kernel ``mn_encoder_encode`` (``mind/exports/c_abi.mind`` →
``mind/kernels/encode.mind``).

  * ``encoder_weights.q16.bin`` — flat Q16.16 blob, ``int64`` little-
    endian per element, **no header**. The byte layout is dictated
    entirely by the *compiled* kernel's fixed pointer arithmetic in
    ``mind/kernels/encode.mind`` (``emb_block_bytes``, ``layer_stride_bytes``
    and the per-layer/embedding offset functions). It is NOT derived
    from the true tensor shapes.
  * ``encoder_weights.q16.meta.json`` — deterministic reproducibility
    metadata including the SHA-256 of the binary blob.

Quantization scheme (authoritative spec: ``spec/quantization.md``):

  * scale = ``2^16 = 65536``
  * rounding = round-half-to-even (``numpy.round``)
  * saturation = clamp to ``[INT32_MIN, INT32_MAX]``
  * on-disk encoding = ``int64`` little-endian (Q16.16 widened for the
    MIND heap ABI's i64-only loads)

The quantize core (``quantize_array``) is imported verbatim from
``quantize_phase1_to_q16.py`` — there is exactly one Q16.16 implementation
in the tree and both blobs share it.

Layout (must match ``mind/kernels/encode.mind`` byte-for-byte)
--------------------------------------------------------------

Embedding block (``emb_block_bytes`` = 95_380_480 bytes from ``wb + 0``)::

    word_table   wb + 0           30522 rows (true vocab) + 3968 pad slots
    pos_table    wb + 93_795_328  rows = 512
    type_table   wb + 95_368_192  rows = 2
    emb_ln_g     wb + 95_374_336  384 elems
    emb_ln_b     wb + 95_377_408  384 elems

The kernel's ``pos_table_ptr`` is the hard-coded byte offset 93_795_328
(= 11_724_416 i64 slots). The true word table is 30522 * 384 =
11_720_448 i64 slots, so there are exactly 3_968 zero i64 slots between
the word table and the position table. That pad is NOT a whole number
of 384-wide rows; the kernel never indexes past word row 30521 (token
IDs are vocabulary-bounded), so the trailing slots stay zero and are
never read. We MUST land pos_table at exactly that byte offset.

Per-layer block (``layer_stride_bytes`` = 14_195_712 bytes, 12 layers,
first layer at ``wb + emb_block_bytes``). Internal offsets are the
``off_*`` functions in ``encode.mind``::

    Wq   off 0           (384,384)  in-major  (PyTorch (out,in) transposed)
    Wk   off 1_179_648   (384,384)  in-major
    Wv   off 2_359_296   (384,384)  in-major
    Wo   off 3_538_944   (384,384)  in-major
    Wf1  off 4_718_592   (384,1536) in-major  (PyTorch (1536,384) transposed)
    Wf2  off 9_437_184   (1536,384) in-major  (PyTorch (384,1536) transposed)
    bq   off 14_155_776  (384,)
    bk   off 14_158_848  (384,)
    bv   off 14_161_920  (384,)
    bo   off 14_164_992  (384,)
    bf1  off 14_168_064  (1536,)
    bf2  off 14_180_352  (384,)
    ln1g off 14_183_424  (384,)   attention.output.LayerNorm.gamma
    ln1b off 14_186_496  (384,)   attention.output.LayerNorm.beta
    ln2g off 14_189_568  (384,)   output.LayerNorm.gamma
    ln2b off 14_192_640  (384,)   output.LayerNorm.beta

The MIND matmul kernel computes ``c[i,j] = sum_k a[i,k] * b[k,j]`` with
``b`` row-major ``(K, N)``. A PyTorch ``nn.Linear`` stores its weight as
``(out_features, in_features)`` and computes ``x @ W.T``. So every 2-D
weight matrix is **transposed** to ``(in_features, out_features)`` here.

Total blob size = 95_380_480 + 12 * 14_195_712 = 265_729_024 bytes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Reuse the single Q16.16 quantize core from the catalog quantizer.
# There is exactly one implementation of the scheme in the tree.
# ---------------------------------------------------------------------------
_TOOLS_DIR = Path(__file__).resolve().parent
_CATALOG_TOOL = _TOOLS_DIR / "quantize_phase1_to_q16.py"


def _load_catalog_tool() -> Any:
    spec = importlib.util.spec_from_file_location("quantize_phase1_to_q16", _CATALOG_TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load catalog quantizer from {_CATALOG_TOOL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CATALOG = _load_catalog_tool()
quantize_array = _CATALOG.quantize_array  # reused verbatim
SCALE = _CATALOG.SCALE
INT32_MAX = _CATALOG.INT32_MAX
INT32_MIN = _CATALOG.INT32_MIN
_sha256_bytes = _CATALOG._sha256_bytes
_sha256_checkpoint_dir = _CATALOG._sha256_checkpoint_dir

QUANTIZER_VERSION = "1.0"
SCHEMA_VERSION = 1
KIND = "mind_nerve.quantize.encoder_weights"

# ---------------------------------------------------------------------------
# Layout constants — these mirror mind/kernels/encode.mind EXACTLY.
# Changing any of these without rebuilding the .so is a correctness bug.
# ---------------------------------------------------------------------------
HIDDEN = 384
FFN = 1536
N_LAYERS = 12
N_HEADS = 12

# Embedding block (bytes). The kernel's pos_table_ptr is a hard-coded
# byte offset (= wb + 93_795_328 = 11_724_416 i64 slots). The true word
# table is 30522 * 384 = 11_720_448 i64 slots, so there are exactly
# WORD_TABLE_PAD_SLOTS (3_968) zero i64 slots between the word table and
# the position table. The pad is NOT a whole number of 384-wide rows;
# the kernel never indexes past row 30521 (token IDs are vocab-bounded)
# so the trailing slots stay zero and are never read.
WORD_TABLE_ROWS = 30522  # true vocabulary; the kernel reads at most this many
WORD_TABLE_PAD_SLOTS = 3_968  # OFF_POS_TABLE/8 - 30522*384
POS_TABLE_ROWS = 512
TYPE_TABLE_ROWS = 2

OFF_POS_TABLE = 93_795_328
OFF_TYPE_TABLE = 95_368_192
OFF_EMB_LNG = 95_374_336
OFF_EMB_LNB = 95_377_408
EMB_BLOCK_BYTES = 95_380_480

LAYER_STRIDE_BYTES = 14_195_712

# Per-layer internal byte offsets (mirror encode.mind off_* functions).
OFF_WQ = 0
OFF_WK = 1_179_648
OFF_WV = 2_359_296
OFF_WO = 3_538_944
OFF_WF1 = 4_718_592
OFF_WF2 = 9_437_184
OFF_BQ = 14_155_776
OFF_BK = 14_158_848
OFF_BV = 14_161_920
OFF_BO = 14_164_992
OFF_BF1 = 14_168_064
OFF_BF2 = 14_180_352
OFF_LN1G = 14_183_424
OFF_LN1B = 14_186_496
OFF_LN2G = 14_189_568
OFF_LN2B = 14_192_640

TOTAL_BYTES = EMB_BLOCK_BYTES + N_LAYERS * LAYER_STRIDE_BYTES  # 265_729_024


# ---------------------------------------------------------------------------
# Safetensors loading
# ---------------------------------------------------------------------------


def _checkpoint_safetensors(checkpoint_dir: Path) -> Path:
    st = checkpoint_dir / "model.safetensors"
    if not st.is_file():
        raise FileNotFoundError(
            f"encoder checkpoint not found: {st} "
            f"(expected a safetensors-format SentenceTransformer checkpoint)"
        )
    return st


def _load_state_dict(st_path: Path) -> dict[str, np.ndarray]:
    """Load every tensor from a safetensors file as a numpy float64 ndarray.

    ``safetensors`` is a transitive dependency of ``sentence-transformers``
    (already a runtime dep); it is imported here only on the offline tools
    path, never on the inference hot path.
    """
    from safetensors import safe_open  # transitive dep; offline path only

    out: dict[str, np.ndarray] = {}
    with safe_open(str(st_path), framework="np") as f:
        for key in f.keys():
            out[key] = np.asarray(f.get_tensor(key)).astype(np.float64, copy=False)
    return out


# ---------------------------------------------------------------------------
# Blob assembly
# ---------------------------------------------------------------------------


def _q16(arr: np.ndarray) -> tuple[np.ndarray, int]:
    """Quantize an ndarray to Q16.16 int64, flattened C-order.

    ``quantize_array`` preserves the input shape; transposed views are
    materialised C-contiguous *before* quantizing so the flatten matches
    the kernel's row-major ``(in, out)`` expectation.
    """
    q, sat = quantize_array(np.ascontiguousarray(arr))
    return np.ascontiguousarray(q).reshape(-1).astype(np.int64, copy=False), sat


def _place(blob: np.ndarray, byte_off: int, values: np.ndarray) -> None:
    """Write a flat int64 ndarray into ``blob`` starting at ``byte_off``."""
    start = byte_off // 8
    end = start + values.size
    if end > blob.size:
        raise RuntimeError(
            f"internal: write [{start},{end}) exceeds blob of {blob.size} i64 slots "
            f"(byte_off={byte_off}, n={values.size})"
        )
    blob[start:end] = values


def build_encoder_blob(state: dict[str, np.ndarray]) -> tuple[np.ndarray, int]:
    """Assemble the full Q16.16 encoder-weights blob.

    Returns ``(blob_int64, saturated_count)``. The blob is a C-contiguous
    ``int64`` ndarray of exactly ``TOTAL_BYTES / 8`` elements. Unused
    word-table padding rows stay zero.
    """
    blob = np.zeros(TOTAL_BYTES // 8, dtype=np.int64)
    saturated = 0

    def need(name: str, shape: tuple[int, ...]) -> np.ndarray:
        if name not in state:
            raise KeyError(f"checkpoint missing tensor: {name}")
        t = state[name]
        if tuple(t.shape) != shape:
            raise ValueError(f"{name}: expected shape {shape}, got {tuple(t.shape)}")
        return t

    # --- Embedding block ---
    word = need("embeddings.word_embeddings.weight", (30522, HIDDEN))
    pos = need("embeddings.position_embeddings.weight", (POS_TABLE_ROWS, HIDDEN))
    typ = need("embeddings.token_type_embeddings.weight", (TYPE_TABLE_ROWS, HIDDEN))
    emb_lng = need("embeddings.LayerNorm.gamma", (HIDDEN,))
    emb_lnb = need("embeddings.LayerNorm.beta", (HIDDEN,))

    # The true 30522 word rows go at offset 0; the WORD_TABLE_PAD_SLOTS
    # i64 slots before OFF_POS_TABLE stay zero (never indexed by the kernel).
    q_word, s = _q16(word)
    saturated += s
    _place(blob, 0, q_word)  # rows 0..30521 at byte 0; pad rows already zero

    q_pos, s = _q16(pos)
    saturated += s
    _place(blob, OFF_POS_TABLE, q_pos)

    q_typ, s = _q16(typ)
    saturated += s
    _place(blob, OFF_TYPE_TABLE, q_typ)

    q, s = _q16(emb_lng)
    saturated += s
    _place(blob, OFF_EMB_LNG, q)
    q, s = _q16(emb_lnb)
    saturated += s
    _place(blob, OFF_EMB_LNB, q)

    # --- Per-layer blocks ---
    for layer in range(N_LAYERS):
        base = EMB_BLOCK_BYTES + layer * LAYER_STRIDE_BYTES
        pfx = f"encoder.layer.{layer}."

        # PyTorch nn.Linear weight is (out, in); the MIND matmul wants
        # b laid out (in, out) row-major → transpose then C-contiguous.
        wq = need(pfx + "attention.self.query.weight", (HIDDEN, HIDDEN)).T
        wk = need(pfx + "attention.self.key.weight", (HIDDEN, HIDDEN)).T
        wv = need(pfx + "attention.self.value.weight", (HIDDEN, HIDDEN)).T
        wo = need(pfx + "attention.output.dense.weight", (HIDDEN, HIDDEN)).T
        # intermediate.dense: (FFN, HIDDEN) → (HIDDEN, FFN)
        wf1 = need(pfx + "intermediate.dense.weight", (FFN, HIDDEN)).T
        # output.dense: (HIDDEN, FFN) → (FFN, HIDDEN)
        wf2 = need(pfx + "output.dense.weight", (HIDDEN, FFN)).T

        bq = need(pfx + "attention.self.query.bias", (HIDDEN,))
        bk = need(pfx + "attention.self.key.bias", (HIDDEN,))
        bv = need(pfx + "attention.self.value.bias", (HIDDEN,))
        bo = need(pfx + "attention.output.dense.bias", (HIDDEN,))
        bf1 = need(pfx + "intermediate.dense.bias", (FFN,))
        bf2 = need(pfx + "output.dense.bias", (HIDDEN,))

        ln1g = need(pfx + "attention.output.LayerNorm.gamma", (HIDDEN,))
        ln1b = need(pfx + "attention.output.LayerNorm.beta", (HIDDEN,))
        ln2g = need(pfx + "output.LayerNorm.gamma", (HIDDEN,))
        ln2b = need(pfx + "output.LayerNorm.beta", (HIDDEN,))

        for off, mat in (
            (OFF_WQ, wq),
            (OFF_WK, wk),
            (OFF_WV, wv),
            (OFF_WO, wo),
            (OFF_WF1, wf1),
            (OFF_WF2, wf2),
            (OFF_BQ, bq),
            (OFF_BK, bk),
            (OFF_BV, bv),
            (OFF_BO, bo),
            (OFF_BF1, bf1),
            (OFF_BF2, bf2),
            (OFF_LN1G, ln1g),
            (OFF_LN1B, ln1b),
            (OFF_LN2G, ln2g),
            (OFF_LN2B, ln2b),
        ):
            q, s = _q16(mat)
            saturated += s
            _place(blob, base + off, q)

    return np.ascontiguousarray(blob, dtype="<i8"), saturated


# ---------------------------------------------------------------------------
# Meta JSON
# ---------------------------------------------------------------------------


def _ordered_meta(
    *,
    byte_size: int,
    bin_sha256: str,
    checkpoint_path: Path,
    checkpoint_hash: str,
    min_q16: int,
    max_q16: int,
    saturated_count: int,
    produced_at_iso: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "quantizer_version": QUANTIZER_VERSION,
        "hidden_dim": HIDDEN,
        "ffn_dim": FFN,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "word_table_rows": WORD_TABLE_ROWS,
        "word_table_pad_slots": WORD_TABLE_PAD_SLOTS,
        "scale": SCALE,
        "rounding": "half_to_even",
        "saturation": "int32",
        "dtype_disk": "int64_le",
        "dtype_value": "int32_q16_16",
        "emb_block_bytes": EMB_BLOCK_BYTES,
        "layer_stride_bytes": LAYER_STRIDE_BYTES,
        "byte_size": byte_size,
        "sha256": bin_sha256,
        "source": {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_hash": checkpoint_hash,
        },
        "stats": {
            "min_q16": min_q16,
            "max_q16": max_q16,
            "saturated_count": saturated_count,
        },
        "produced_at_iso": produced_at_iso,
        "produced_by": "mind-nerve quantize-encoder",
    }


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def quantize_encoder(
    checkpoint_dir: Path,
    output_dir: Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Quantize a safetensors encoder checkpoint into ``encoder_weights.q16.bin``.

    Returns the meta-JSON dict (also written unless ``dry_run`` is True).
    """
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint dir not found: {checkpoint_dir}")
    st_path = _checkpoint_safetensors(checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = _load_state_dict(st_path)
    blob, saturated_count = build_encoder_blob(state)

    bin_bytes = blob.tobytes(order="C")
    byte_size = len(bin_bytes)
    if byte_size != TOTAL_BYTES:
        raise RuntimeError(f"internal: blob size {byte_size} != expected {TOTAL_BYTES}")
    bin_sha256 = _sha256_bytes(bin_bytes)
    checkpoint_hash = _sha256_checkpoint_dir(checkpoint_dir)

    min_q16 = int(blob.min())
    max_q16 = int(blob.max())
    produced_at_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    meta = _ordered_meta(
        byte_size=byte_size,
        bin_sha256=bin_sha256,
        checkpoint_path=checkpoint_dir,
        checkpoint_hash=checkpoint_hash,
        min_q16=min_q16,
        max_q16=max_q16,
        saturated_count=saturated_count,
        produced_at_iso=produced_at_iso,
    )

    if dry_run:
        return meta

    bin_path = output_dir / "encoder_weights.q16.bin"
    meta_path = output_dir / "encoder_weights.q16.meta.json"

    tmp_bin = bin_path.with_suffix(".bin.tmp")
    tmp_bin.write_bytes(bin_bytes)
    os.replace(tmp_bin, bin_path)

    meta_text = json.dumps(meta, indent=2, sort_keys=False) + "\n"
    tmp_meta = meta_path.with_suffix(".json.tmp")
    tmp_meta.write_text(meta_text, encoding="utf-8")
    os.replace(tmp_meta, meta_path)

    return meta


def resolve_default_output_dir() -> Path:
    """Default ``--output`` dir: ``$MIND_NERVE_RUNTIME_DIR`` or the user runtime dir.

    The native encoder runtime looks for ``encoder_weights.q16.bin``
    alongside ``route_table.npy`` in the runtime dir, so the default
    output mirrors that location.
    """
    env_dir = os.environ.get("MIND_NERVE_RUNTIME_DIR")
    if env_dir:
        out = Path(env_dir).expanduser()
    else:
        out = Path.home() / ".local" / "share" / "mind-nerve" / "runtime"
    out.mkdir(parents=True, exist_ok=True)
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="quantize_encoder_to_q16",
        description=(
            "Quantize a safetensors encoder checkpoint into the Q16.16 "
            "encoder_weights.q16.bin blob consumed by mn_encoder_encode."
        ),
    )
    ap.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the checkpoint directory containing model.safetensors.",
    )
    ap.add_argument(
        "--output",
        default=None,
        help=(
            "Output directory. Default: $MIND_NERVE_RUNTIME_DIR or the user "
            "runtime dir (~/.local/share/mind-nerve/runtime)."
        ),
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

    checkpoint_dir = Path(args.checkpoint).expanduser()
    output_dir = Path(args.output).expanduser() if args.output else resolve_default_output_dir()

    try:
        meta = quantize_encoder(
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            dry_run=args.dry_run,
        )
    except FileNotFoundError as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1
    except (ValueError, KeyError) as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 2

    print(json.dumps(meta, indent=2, sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
