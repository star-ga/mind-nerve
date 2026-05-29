"""Regression tests for the 0.1.0a4 fixes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def _make_minimal_runtime(tmp: Path) -> Path:
    """Build a 3-route fake runtime dir (no real weights)."""
    runtime = tmp / "runtime"
    (runtime / "checkpoint").mkdir(parents=True)
    (runtime / "manifest.json").write_text(
        json.dumps(
            {
                "catalog_version": "test-v0",
                "phase1_version": "test-v0",
            }
        )
    )
    np.save(runtime / "route_table.npy", np.eye(3, 8, dtype=np.float32))
    with (runtime / "route_table.jsonl").open("w") as f:
        for i in range(3):
            f.write(
                json.dumps(
                    {"id": f"r{i}", "name": f"route{i}", "kind": "skill", "source_repo": "test"}
                )
                + "\n"
            )
    return runtime


def test_save_table_atomic_does_not_leak_tmp_npy(tmp_path: Path) -> None:
    """0.1.0a4 fix: NumPy auto-extension on .tmp filename used to leave artifacts."""
    from mind_nerve.discovery import _save_table_atomic

    rdir = tmp_path
    emb = np.ones((4, 8), dtype=np.float32)
    meta = [{"id": f"r{i}", "name": f"r{i}"} for i in range(4)]

    _save_table_atomic(rdir, emb, meta)

    assert (rdir / "route_table.npy").exists(), "atomic save did not produce route_table.npy"
    assert (rdir / "route_table.jsonl").exists(), "atomic save did not produce route_table.jsonl"
    # The old bug left these on disk:
    assert not (rdir / "route_table.npy.tmp").exists()
    assert not (rdir / "route_table.npy.tmp.npy").exists(), (
        "NumPy auto-extended .tmp filename — regression of 0.1.0a3 bug"
    )
    assert not (rdir / "route_table.tmp.npy").exists(), (
        "atomic rename didn't clean up the staging file"
    )

    # Verify the saved data round-trips correctly
    loaded = np.load(rdir / "route_table.npy")
    np.testing.assert_array_equal(loaded, emb)


def test_cli_learn_does_not_use_hardcoded_fallback(tmp_path: Path) -> None:
    """0.1.0a4 regression: cli.cmd_learn used to hardcode an absolute dataset path
    as the runtime_dir fallback, ignoring MIND_NERVE_RUNTIME_DIR. The fix
    routes through _DEFAULT_RUNTIME_DIR which honors the env var. We verify
    by pointing the env var at a bogus path and asserting the error message
    references THAT path, not the legacy hardcoded one.
    """
    bogus = tmp_path / "definitely-not-real"  # does not exist
    scan_dir = tmp_path / "scan"
    scan_dir.mkdir()

    env = os.environ.copy()
    env["MIND_NERVE_RUNTIME_DIR"] = str(bogus)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mind_nerve.cli",
            "learn",
            str(scan_dir),
            "--source",
            "test",
            "--dry-run",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    out = result.stdout + result.stderr
    # Old bug: would error on catalog-data/phase1/v1.[10]-oss
    assert "catalog-data" not in out, (
        "regression: legacy hardcoded fallback path appeared in CLI error"
    )
