"""Phase 6.2 quantizer tests — round-trip, bit-identity, CLI smoke.

Verifies that the offline FP32 → Q16.16 quantizer (``tools/quantize_phase1_to_q16.py``)
produces the contract defined in ``spec/quantization.md``:

  * Round-trip error stays within ``2 * 2^-16`` (≈ 3.05e-5).
  * Same input → byte-identical ``route_table.q16.bin``.
  * Meta JSON carries a valid SHA-256 that matches the .bin bytes.
  * Saturation clamps out-of-range inputs to ``[INT32_MIN, INT32_MAX]``.
  * CLI surface (``mind-nerve quantize``) wires through to the tool.
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
TOOL_PATH = REPO_ROOT / "tools" / "quantize_phase1_to_q16.py"


def _load_tool_module() -> Any:
    """Import ``quantize_phase1_to_q16`` from ``tools/`` (not on sys.path)."""
    spec = importlib.util.spec_from_file_location("quantize_phase1_to_q16", TOOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module from {TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def qmod() -> Any:
    return _load_tool_module()


# ---------------------------------------------------------------------------
# Constants — duplicated locally so a regression in the spec is visible here.
# ---------------------------------------------------------------------------

SCALE = 65536
INT32_MAX = 2_147_483_647
INT32_MIN = -2_147_483_648
HALF_LSB = 2.0 / SCALE  # max absolute round-trip error allowed


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_q16_one_round_trips_exactly(qmod: Any) -> None:
    """The single integer 65536 ↔ 1.0 round trip is exact."""
    assert qmod.f32_to_q16(1.0) == SCALE
    assert qmod.q16_to_f32(SCALE) == 1.0


def test_q16_zero(qmod: Any) -> None:
    assert qmod.f32_to_q16(0.0) == 0
    assert qmod.q16_to_f32(0) == 0.0


def test_q16_negative_one(qmod: Any) -> None:
    assert qmod.f32_to_q16(-1.0) == -SCALE
    assert qmod.q16_to_f32(-SCALE) == -1.0


def test_round_trip_1000_random_floats(qmod: Any) -> None:
    """1000 random ``f32`` values in [-1, 1] round-trip within half-LSB.

    This is the spec-mandated regression: max abs error MUST be strictly
    less than ``2 * 2^-16``.
    """
    rng = np.random.default_rng(seed=1337)
    raw = rng.uniform(-1.0, 1.0, size=1000).astype(np.float32)
    errors = []
    for f in raw.tolist():
        q = qmod.f32_to_q16(f)
        back = qmod.q16_to_f32(q)
        errors.append(abs(back - f))
    max_err = max(errors)
    assert max_err < HALF_LSB, f"max abs round-trip error {max_err} >= {HALF_LSB}"


def test_round_trip_idempotent_on_integers(qmod: Any) -> None:
    """Q16.16 integers round-trip through float exactly (no rounding loss)."""
    samples = [0, 1, -1, SCALE, -SCALE, INT32_MAX, INT32_MIN, 12345, -98765]
    for q in samples:
        back = qmod.f32_to_q16(qmod.q16_to_f32(q))
        assert back == q, f"q={q} → f={qmod.q16_to_f32(q)} → back={back}"


def test_saturation_positive(qmod: Any) -> None:
    """Inputs above the representable max saturate to INT32_MAX."""
    # 2.0e6 * 65536 ≈ 1.3e11 — well above INT32_MAX.
    assert qmod.f32_to_q16(2.0e6) == INT32_MAX
    assert qmod.f32_to_q16(1.0e9) == INT32_MAX


def test_saturation_negative(qmod: Any) -> None:
    """Inputs below the representable min saturate to INT32_MIN."""
    assert qmod.f32_to_q16(-2.0e6) == INT32_MIN
    assert qmod.f32_to_q16(-1.0e9) == INT32_MIN


def test_round_half_to_even(qmod: Any) -> None:
    """``np.round`` is round-half-to-even; verify on the half-LSB boundary.

    ``0.5 / 65536`` quantizes to ``0`` (ties to even). ``1.5 / 65536`` to ``2``.
    ``2.5 / 65536`` to ``2`` (ties to even). ``3.5 / 65536`` to ``4``.
    """
    assert qmod.f32_to_q16(0.5 / SCALE) == 0
    assert qmod.f32_to_q16(1.5 / SCALE) == 2
    assert qmod.f32_to_q16(2.5 / SCALE) == 2
    assert qmod.f32_to_q16(3.5 / SCALE) == 4


def test_non_finite_rejected(qmod: Any) -> None:
    with pytest.raises(ValueError):
        qmod.f32_to_q16(float("nan"))
    with pytest.raises(ValueError):
        qmod.f32_to_q16(float("inf"))
    with pytest.raises(ValueError):
        qmod.f32_to_q16(float("-inf"))


# ---------------------------------------------------------------------------
# Vectorized quantize_array
# ---------------------------------------------------------------------------


def test_quantize_array_matches_scalar(qmod: Any) -> None:
    """``quantize_array`` and the scalar ``f32_to_q16`` agree element-wise."""
    rng = np.random.default_rng(seed=42)
    arr = rng.uniform(-1.0, 1.0, size=512).astype(np.float32)
    vec, sat = qmod.quantize_array(arr)
    assert sat == 0
    expected = np.array([qmod.f32_to_q16(float(x)) for x in arr], dtype=np.int64)
    assert np.array_equal(vec, expected)


def test_quantize_array_int64_dtype(qmod: Any) -> None:
    """Vectorized output is ``int64`` (i.e. widened-Q16.16 for the i64 ABI)."""
    arr = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    vec, _ = qmod.quantize_array(arr)
    assert vec.dtype == np.int64


def test_quantize_array_saturation_count(qmod: Any) -> None:
    """Out-of-range inputs increment ``saturated_count``."""
    arr = np.array([1e9, -1e9, 0.0, 1.0], dtype=np.float32)
    vec, sat = qmod.quantize_array(arr)
    assert sat == 2
    assert int(vec[0]) == INT32_MAX
    assert int(vec[1]) == INT32_MIN


def test_quantize_array_empty(qmod: Any) -> None:
    vec, sat = qmod.quantize_array(np.empty((0,), dtype=np.float32))
    assert vec.shape == (0,)
    assert sat == 0


# ---------------------------------------------------------------------------
# End-to-end catalog quantization + bit-identity
# ---------------------------------------------------------------------------


def _make_fixture_catalog(path: Path, seed: int, n_rows: int = 100, hd: int = 384) -> None:
    """Write a deterministic synthetic L2-normalized catalog to *path*."""
    rng = np.random.default_rng(seed=seed)
    raw = rng.standard_normal(size=(n_rows, hd)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True) + 1e-12
    np.save(str(path), (raw / norms).astype(np.float32))


def test_quantize_catalog_writes_bin_and_meta(qmod: Any, tmp_path: Path) -> None:
    """End-to-end: produces .bin + meta.json with the right shape + size."""
    cat = tmp_path / "route_table.npy"
    _make_fixture_catalog(cat, seed=1, n_rows=100, hd=384)
    out_dir = tmp_path / "out"
    meta = qmod.quantize_catalog(catalog_path=cat, output_dir=out_dir)

    bin_path = out_dir / "route_table.q16.bin"
    meta_path = out_dir / "route_table.q16.meta.json"
    assert bin_path.is_file()
    assert meta_path.is_file()
    assert meta["n_rows"] == 100
    assert meta["hidden_dim"] == 384
    assert meta["scale"] == SCALE
    assert meta["rounding"] == "half_to_even"
    assert meta["saturation"] == "int32"
    assert meta["dtype_disk"] == "int64_le"
    assert meta["byte_size"] == 100 * 384 * 8
    assert bin_path.stat().st_size == meta["byte_size"]

    # SHA-256 in meta matches the actual file bytes.
    file_hash = hashlib.sha256(bin_path.read_bytes()).hexdigest()
    assert meta["sha256"] == file_hash

    # On-disk dtype: int64 LE, divisible into hidden_dim chunks per row.
    loaded = np.fromfile(str(bin_path), dtype="<i8")
    assert loaded.shape == (100 * 384,)
    assert int(loaded.min()) >= INT32_MIN
    assert int(loaded.max()) <= INT32_MAX


def test_bit_identity_same_input_same_output(qmod: Any, tmp_path: Path) -> None:
    """**Phase 6.2 reproducibility claim**: same input → byte-identical .bin."""
    cat = tmp_path / "route_table.npy"
    _make_fixture_catalog(cat, seed=7, n_rows=50, hd=384)

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    meta_a = qmod.quantize_catalog(catalog_path=cat, output_dir=out_a)
    meta_b = qmod.quantize_catalog(catalog_path=cat, output_dir=out_b)

    bin_a = (out_a / "route_table.q16.bin").read_bytes()
    bin_b = (out_b / "route_table.q16.bin").read_bytes()
    assert bin_a == bin_b, "two runs of the quantizer on the same input diverged"
    assert meta_a["sha256"] == meta_b["sha256"]
    # The on-disk binary bytes are equal; meta JSON differs only on produced_at_iso.
    assert {k: v for k, v in meta_a.items() if k != "produced_at_iso"} == {
        k: v for k, v in meta_b.items() if k != "produced_at_iso"
    }


def test_bit_identity_round_trip_via_inference_helpers(qmod: Any, tmp_path: Path) -> None:
    """Cross-check with ``mind_nerve._native._f32_to_q16``.

    The runtime catalog-quantization helper in ``python/mind_nerve/_native.py``
    uses the same scheme as this offline tool. The two MUST agree on every
    element so the inference path consumes a value-equivalent blob.
    """
    from mind_nerve._native import _f32_to_q16

    cat = tmp_path / "route_table.npy"
    _make_fixture_catalog(cat, seed=11, n_rows=10, hd=384)
    out_dir = tmp_path / "out"
    qmod.quantize_catalog(catalog_path=cat, output_dir=out_dir)

    loaded_offline = np.fromfile(str(out_dir / "route_table.q16.bin"), dtype="<i8")
    raw = np.load(str(cat))
    loaded_inline = _f32_to_q16(raw).reshape(-1)

    assert loaded_offline.shape == loaded_inline.shape
    assert np.array_equal(loaded_offline, loaded_inline), (
        "offline tool diverges from inference._native._f32_to_q16 — "
        "the inference path would observe a different catalog"
    )


def test_meta_includes_checkpoint_hash_when_present(qmod: Any, tmp_path: Path) -> None:
    """``--input <ckpt_dir>`` populates ``source.checkpoint_hash``."""
    cat = tmp_path / "route_table.npy"
    _make_fixture_catalog(cat, seed=2, n_rows=10, hd=384)

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text('{"foo": 1}\n')
    (ckpt / "model.safetensors").write_bytes(b"\x00" * 64)

    out_dir = tmp_path / "out"
    meta = qmod.quantize_catalog(catalog_path=cat, output_dir=out_dir, checkpoint_path=ckpt)
    assert meta["source"]["checkpoint_path"] == str(ckpt)
    assert meta["source"]["checkpoint_hash"] is not None
    assert len(meta["source"]["checkpoint_hash"]) == 64


def test_meta_checkpoint_hash_null_when_absent(qmod: Any, tmp_path: Path) -> None:
    """No ``--input`` ⇒ ``checkpoint_*`` fields are JSON null."""
    cat = tmp_path / "route_table.npy"
    _make_fixture_catalog(cat, seed=3, n_rows=10, hd=384)
    out_dir = tmp_path / "out"
    meta = qmod.quantize_catalog(catalog_path=cat, output_dir=out_dir)
    assert meta["source"]["checkpoint_path"] is None
    assert meta["source"]["checkpoint_hash"] is None


def test_rejects_non_2d_catalog(qmod: Any, tmp_path: Path) -> None:
    cat = tmp_path / "bad.npy"
    np.save(str(cat), np.zeros((3, 4, 5), dtype=np.float32))
    with pytest.raises(ValueError):
        qmod.quantize_catalog(catalog_path=cat, output_dir=tmp_path / "out")


def test_rejects_hidden_dim_mismatch(qmod: Any, tmp_path: Path) -> None:
    cat = tmp_path / "route_table.npy"
    _make_fixture_catalog(cat, seed=4, n_rows=5, hd=256)
    with pytest.raises(ValueError):
        qmod.quantize_catalog(catalog_path=cat, output_dir=tmp_path / "out", hidden_dim=384)


def test_dry_run_does_not_write(qmod: Any, tmp_path: Path) -> None:
    cat = tmp_path / "route_table.npy"
    _make_fixture_catalog(cat, seed=5, n_rows=10, hd=384)
    out_dir = tmp_path / "out"
    meta = qmod.quantize_catalog(catalog_path=cat, output_dir=out_dir, dry_run=True)
    assert meta["n_rows"] == 10
    # dry-run should not have created either output file
    assert not (out_dir / "route_table.q16.bin").exists()
    assert not (out_dir / "route_table.q16.meta.json").exists()


def test_meta_key_order(qmod: Any, tmp_path: Path) -> None:
    """Top-level meta keys are in the spec-mandated order."""
    cat = tmp_path / "route_table.npy"
    _make_fixture_catalog(cat, seed=6, n_rows=8, hd=384)
    out_dir = tmp_path / "out"
    qmod.quantize_catalog(catalog_path=cat, output_dir=out_dir)
    raw = (out_dir / "route_table.q16.meta.json").read_text()
    loaded: dict[str, Any] = json.loads(raw)
    expected_order = [
        "schema_version",
        "kind",
        "quantizer_version",
        "n_rows",
        "hidden_dim",
        "scale",
        "rounding",
        "saturation",
        "dtype_disk",
        "dtype_value",
        "byte_size",
        "sha256",
        "source",
        "stats",
        "produced_at_iso",
        "produced_by",
    ]
    assert list(loaded.keys()) == expected_order


# ---------------------------------------------------------------------------
# Default output dir
# ---------------------------------------------------------------------------


def test_default_output_dir_honors_env(
    qmod: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``$MIND_NERVE_RUNTIME_DIR`` overrides the default ~/.cache path."""
    target = tmp_path / "rtdir"
    monkeypatch.setenv("MIND_NERVE_RUNTIME_DIR", str(target))
    resolved = qmod.resolve_default_output_dir()
    assert resolved == target
    assert resolved.is_dir()


def test_default_output_dir_falls_back_to_cache(
    qmod: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absent env var ⇒ ~/.cache/mind-nerve/q16/."""
    monkeypatch.delenv("MIND_NERVE_RUNTIME_DIR", raising=False)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
    resolved = qmod.resolve_default_output_dir()
    assert resolved == fake_home / ".cache" / "mind-nerve" / "q16"
    assert resolved.is_dir()


# ---------------------------------------------------------------------------
# CLI smoke (mind-nerve quantize)
# ---------------------------------------------------------------------------


def test_cli_subcommand_quantize(tmp_path: Path) -> None:
    """``mind-nerve quantize`` invokes the tool and writes outputs."""
    cat = tmp_path / "route_table.npy"
    _make_fixture_catalog(cat, seed=12, n_rows=10, hd=384)
    out_dir = tmp_path / "out"

    env = dict(os.environ)
    # Make sure the test does not write into the user's real cache dir.
    env.pop("MIND_NERVE_RUNTIME_DIR", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mind_nerve.cli",
            "quantize",
            "--catalog",
            str(cat),
            "--output",
            str(out_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    payload = json.loads(result.stdout)
    assert payload["n_rows"] == 10
    assert payload["hidden_dim"] == 384
    assert (out_dir / "route_table.q16.bin").is_file()
    assert (out_dir / "route_table.q16.meta.json").is_file()


def test_cli_dry_run(tmp_path: Path) -> None:
    """``mind-nerve quantize --dry-run`` emits meta but writes nothing."""
    cat = tmp_path / "route_table.npy"
    _make_fixture_catalog(cat, seed=13, n_rows=4, hd=384)
    out_dir = tmp_path / "out"

    env = dict(os.environ)
    env.pop("MIND_NERVE_RUNTIME_DIR", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mind_nerve.cli",
            "quantize",
            "--catalog",
            str(cat),
            "--output",
            str(out_dir),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    payload = json.loads(result.stdout)
    assert payload["n_rows"] == 4
    assert not (out_dir / "route_table.q16.bin").exists()
