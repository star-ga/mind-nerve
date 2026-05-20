"""Phase 6.x encoder-weights quantizer tests — layout, bit-identity, CLI.

Verifies that ``tools/quantize_encoder_to_q16.py`` produces the contract
the native ``mn_encoder_encode`` kernel consumes:

  * The flat blob is exactly ``emb_block_bytes + 12 * layer_stride_bytes``
    bytes and the internal offsets mirror ``mind/kernels/encode.mind``.
  * Same checkpoint → byte-identical ``encoder_weights.q16.bin``.
  * Meta JSON carries a valid SHA-256 that matches the .bin bytes.
  * The Q16.16 scheme is the *same core* as the catalog quantizer.
  * CLI surface (``mind-nerve quantize-encoder``) wires through.

The real Phase-1 checkpoint is large and not in CI; tests that need a
checkpoint use a deterministic synthetic safetensors file built inline,
and self-skip cleanly if ``safetensors`` is unavailable (mirrors the
skip-guard discipline from commit ecb196a).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "quantize_encoder_to_q16.py"

safetensors = pytest.importorskip(
    "safetensors", reason="safetensors not installed (offline-tools dep)"
)


def _load_tool_module() -> Any:
    spec = importlib.util.spec_from_file_location("quantize_encoder_to_q16", TOOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module from {TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def emod() -> Any:
    return _load_tool_module()


SCALE = 65536
HIDDEN = 384
FFN = 1536
N_LAYERS = 12
EMB_BLOCK_BYTES = 95_380_480
LAYER_STRIDE_BYTES = 14_195_712
TOTAL_BYTES = EMB_BLOCK_BYTES + N_LAYERS * LAYER_STRIDE_BYTES  # 265_729_024


# ---------------------------------------------------------------------------
# Synthetic checkpoint
# ---------------------------------------------------------------------------


def _make_synthetic_checkpoint(ckpt_dir: Path, seed: int = 1337) -> None:
    """Write a deterministic synthetic BERT-small safetensors checkpoint.

    Shapes match the real BGE-small-en-v1.5 head so the layout assertions
    exercise the exact transpose / offset code paths.
    """
    from safetensors.numpy import save_file

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    tensors: dict[str, np.ndarray] = {}

    def w(shape: tuple[int, ...]) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)

    tensors["embeddings.word_embeddings.weight"] = w((30522, HIDDEN))
    tensors["embeddings.position_embeddings.weight"] = w((512, HIDDEN))
    tensors["embeddings.token_type_embeddings.weight"] = w((2, HIDDEN))
    tensors["embeddings.LayerNorm.gamma"] = w((HIDDEN,))
    tensors["embeddings.LayerNorm.beta"] = w((HIDDEN,))
    for layer in range(N_LAYERS):
        p = f"encoder.layer.{layer}."
        tensors[p + "attention.self.query.weight"] = w((HIDDEN, HIDDEN))
        tensors[p + "attention.self.query.bias"] = w((HIDDEN,))
        tensors[p + "attention.self.key.weight"] = w((HIDDEN, HIDDEN))
        tensors[p + "attention.self.key.bias"] = w((HIDDEN,))
        tensors[p + "attention.self.value.weight"] = w((HIDDEN, HIDDEN))
        tensors[p + "attention.self.value.bias"] = w((HIDDEN,))
        tensors[p + "attention.output.dense.weight"] = w((HIDDEN, HIDDEN))
        tensors[p + "attention.output.dense.bias"] = w((HIDDEN,))
        tensors[p + "attention.output.LayerNorm.gamma"] = w((HIDDEN,))
        tensors[p + "attention.output.LayerNorm.beta"] = w((HIDDEN,))
        tensors[p + "intermediate.dense.weight"] = w((FFN, HIDDEN))
        tensors[p + "intermediate.dense.bias"] = w((FFN,))
        tensors[p + "output.dense.weight"] = w((HIDDEN, FFN))
        tensors[p + "output.dense.bias"] = w((HIDDEN,))
        tensors[p + "output.LayerNorm.gamma"] = w((HIDDEN,))
        tensors[p + "output.LayerNorm.beta"] = w((HIDDEN,))

    save_file(tensors, str(ckpt_dir / "model.safetensors"))


# ---------------------------------------------------------------------------
# Quantize-core reuse
# ---------------------------------------------------------------------------


def test_reuses_catalog_quantize_core(emod: Any) -> None:
    """The encoder quantizer imports the *same* quantize_array as the catalog."""
    import importlib.util as ilu

    cat_spec = ilu.spec_from_file_location(
        "quantize_phase1_to_q16", REPO_ROOT / "tools" / "quantize_phase1_to_q16.py"
    )
    assert cat_spec is not None and cat_spec.loader is not None
    cat = ilu.module_from_spec(cat_spec)
    cat_spec.loader.exec_module(cat)
    rng = np.random.default_rng(0)
    arr = rng.uniform(-1.0, 1.0, size=257).astype(np.float32)
    a, _ = emod.quantize_array(arr)
    b, _ = cat.quantize_array(arr)
    assert np.array_equal(a, b)
    assert emod.SCALE == cat.SCALE == SCALE


# ---------------------------------------------------------------------------
# Layout constants must mirror encode.mind exactly
# ---------------------------------------------------------------------------


def test_layout_constants_match_kernel(emod: Any) -> None:
    """Blob layout constants equal the fixed pointer arithmetic in encode.mind."""
    assert emod.EMB_BLOCK_BYTES == EMB_BLOCK_BYTES
    assert emod.LAYER_STRIDE_BYTES == LAYER_STRIDE_BYTES
    assert emod.TOTAL_BYTES == TOTAL_BYTES
    # pos_table_ptr is the hard-coded byte offset 93_795_328 = 11_724_416
    # i64 slots; the true word table is 30522*384 slots, leaving exactly
    # WORD_TABLE_PAD_SLOTS (3_968) zero slots before the position table.
    assert emod.WORD_TABLE_ROWS == 30522
    assert emod.OFF_POS_TABLE == 93_795_328
    assert emod.OFF_POS_TABLE // 8 - 30522 * HIDDEN == emod.WORD_TABLE_PAD_SLOTS
    assert emod.WORD_TABLE_PAD_SLOTS == 3_968
    # Per-layer internal offsets (encode.mind off_* functions).
    assert emod.OFF_WK == 1_179_648
    assert emod.OFF_WF1 == 4_718_592
    assert emod.OFF_WF2 == 9_437_184
    assert emod.OFF_BQ == 14_155_776
    assert emod.OFF_LN2B == 14_192_640


# ---------------------------------------------------------------------------
# End-to-end blob assembly + bit-identity
# ---------------------------------------------------------------------------


def test_blob_size_and_meta(emod: Any, tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    _make_synthetic_checkpoint(ckpt, seed=1)
    out = tmp_path / "out"
    meta = emod.quantize_encoder(checkpoint_dir=ckpt, output_dir=out)

    bin_path = out / "encoder_weights.q16.bin"
    meta_path = out / "encoder_weights.q16.meta.json"
    assert bin_path.is_file()
    assert meta_path.is_file()
    assert meta["byte_size"] == TOTAL_BYTES
    assert bin_path.stat().st_size == TOTAL_BYTES
    assert meta["scale"] == SCALE
    assert meta["rounding"] == "half_to_even"
    assert meta["dtype_disk"] == "int64_le"
    assert meta["n_layers"] == N_LAYERS
    assert meta["word_table_rows"] == 30522
    assert meta["word_table_pad_slots"] == 3_968

    file_hash = hashlib.sha256(bin_path.read_bytes()).hexdigest()
    assert meta["sha256"] == file_hash

    blob = np.fromfile(str(bin_path), dtype="<i8")
    assert blob.shape == (TOTAL_BYTES // 8,)
    assert int(blob.min()) >= -2_147_483_648
    assert int(blob.max()) <= 2_147_483_647


def test_word_table_padding_slots_are_zero(emod: Any, tmp_path: Path) -> None:
    """The 3968 i64 slots between word table and pos table are zero padding."""
    ckpt = tmp_path / "ckpt"
    _make_synthetic_checkpoint(ckpt, seed=2)
    out = tmp_path / "out"
    emod.quantize_encoder(checkpoint_dir=ckpt, output_dir=out)
    blob = np.fromfile(str(out / "encoder_weights.q16.bin"), dtype="<i8")
    pad = blob[30522 * HIDDEN : 93_795_328 // 8]
    assert pad.size == 3_968
    assert np.all(pad == 0)
    # pos_table starts immediately after the pad and is non-trivial.
    assert np.any(blob[93_795_328 // 8 : 93_795_328 // 8 + HIDDEN] != 0)


def test_weight_layout_orientation(emod: Any, tmp_path: Path) -> None:
    """Wq is stored (in, out) — the transpose of the PyTorch (out, in)
    weight. Track A's `__mind_nerve_blas_matmul_q16_i64` consumes B
    as (K, N) row-major. (v0.3.0b7's thesis-pure (N, K) layout was
    yanked in v0.3.0b8 — see CHANGELOG for the dot_q16_v stride bug
    that motivated the revert.)"""
    from safetensors.numpy import save_file

    ckpt = tmp_path / "ckpt"
    _make_synthetic_checkpoint(ckpt, seed=3)
    # Re-stamp layer-0 query weight with a known asymmetric pattern.
    from safetensors import safe_open

    state: dict[str, np.ndarray] = {}
    with safe_open(str(ckpt / "model.safetensors"), framework="np") as f:
        for k in f.keys():
            state[k] = f.get_tensor(k)
    wq = np.arange(HIDDEN * HIDDEN, dtype=np.float32).reshape(HIDDEN, HIDDEN) * 1e-4
    state["encoder.layer.0.attention.self.query.weight"] = wq
    save_file(state, str(ckpt / "model.safetensors"))

    out = tmp_path / "out"
    emod.quantize_encoder(checkpoint_dir=ckpt, output_dir=out)
    blob = np.fromfile(str(out / "encoder_weights.q16.bin"), dtype="<i8")

    base = EMB_BLOCK_BYTES // 8  # layer 0, OFF_WQ == 0
    stored = blob[base : base + HIDDEN * HIDDEN].reshape(HIDDEN, HIDDEN)
    # Track A layout: stored == quantize(wq.T).
    expected, _ = emod.quantize_array(np.ascontiguousarray(wq.T))
    assert np.array_equal(stored, expected.reshape(HIDDEN, HIDDEN))


def test_bit_identity_same_checkpoint(emod: Any, tmp_path: Path) -> None:
    """**Reproducibility claim**: same checkpoint → byte-identical .bin."""
    ckpt = tmp_path / "ckpt"
    _make_synthetic_checkpoint(ckpt, seed=7)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    meta_a = emod.quantize_encoder(checkpoint_dir=ckpt, output_dir=out_a)
    meta_b = emod.quantize_encoder(checkpoint_dir=ckpt, output_dir=out_b)
    bin_a = (out_a / "encoder_weights.q16.bin").read_bytes()
    bin_b = (out_b / "encoder_weights.q16.bin").read_bytes()
    assert bin_a == bin_b, "two quantizer runs on the same checkpoint diverged"
    assert meta_a["sha256"] == meta_b["sha256"]
    assert {k: v for k, v in meta_a.items() if k != "produced_at_iso"} == {
        k: v for k, v in meta_b.items() if k != "produced_at_iso"
    }


def test_dry_run_writes_nothing(emod: Any, tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    _make_synthetic_checkpoint(ckpt, seed=5)
    out = tmp_path / "out"
    meta = emod.quantize_encoder(checkpoint_dir=ckpt, output_dir=out, dry_run=True)
    assert meta["byte_size"] == TOTAL_BYTES
    assert not (out / "encoder_weights.q16.bin").exists()
    assert not (out / "encoder_weights.q16.meta.json").exists()


def test_rejects_missing_checkpoint(emod: Any, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        emod.quantize_encoder(checkpoint_dir=tmp_path / "nope", output_dir=tmp_path / "o")


def test_rejects_checkpoint_without_safetensors(emod: Any, tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}\n")
    with pytest.raises(FileNotFoundError):
        emod.quantize_encoder(checkpoint_dir=ckpt, output_dir=tmp_path / "o")


def test_meta_key_order(emod: Any, tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    _make_synthetic_checkpoint(ckpt, seed=6)
    out = tmp_path / "out"
    emod.quantize_encoder(checkpoint_dir=ckpt, output_dir=out)
    loaded = json.loads((out / "encoder_weights.q16.meta.json").read_text())
    assert list(loaded.keys()) == [
        "schema_version",
        "kind",
        "quantizer_version",
        "hidden_dim",
        "ffn_dim",
        "n_layers",
        "n_heads",
        "word_table_rows",
        "word_table_pad_slots",
        "scale",
        "rounding",
        "saturation",
        "dtype_disk",
        "dtype_value",
        "emb_block_bytes",
        "layer_stride_bytes",
        "byte_size",
        "sha256",
        "source",
        "stats",
        "produced_at_iso",
        "produced_by",
    ]


def test_default_output_dir_honors_env(
    emod: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "rtdir"
    monkeypatch.setenv("MIND_NERVE_RUNTIME_DIR", str(target))
    resolved = emod.resolve_default_output_dir()
    assert resolved == target
    assert resolved.is_dir()


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_subcommand_quantize_encoder(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    _make_synthetic_checkpoint(ckpt, seed=12)
    out = tmp_path / "out"
    env = dict(os.environ)
    env.pop("MIND_NERVE_RUNTIME_DIR", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mind_nerve.cli",
            "quantize-encoder",
            "--checkpoint",
            str(ckpt),
            "--output",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    payload = json.loads(result.stdout)
    assert payload["byte_size"] == TOTAL_BYTES
    assert (out / "encoder_weights.q16.bin").is_file()
