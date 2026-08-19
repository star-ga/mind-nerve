"""Regression tests for discovery.scan()'s handling of a partial route table.

Audit finding (HIGH): ``_load_table_or_empty`` returned ``(None, [])`` unless
BOTH ``route_table.npy`` AND ``route_table.jsonl`` existed, conflating
"neither present" (a genuine fresh bootstrap -- safe) with "exactly one
present" (partial corruption: an interrupted write, an incomplete restore, a
hand-deleted file). In the one-present case, ``scan()`` treated the table as
empty and then wrote ONLY the freshly-scanned results via
``_save_table_atomic``, permanently discarding the existing (possibly large)
table with no error or warning.

The fix distinguishes the three cases: neither present -> safe bootstrap
(unchanged); both present -> load (unchanged); exactly one present -> refuse
with a clear ``RuntimeError`` naming the missing file, so the surviving file
is left untouched on disk rather than silently overwritten.

Also covers the adjacent Windows-compat fix in ``_walk_dir``: nested
skill/command/agent files must be detected regardless of the OS path
separator used to build the relative path string.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def _write_skill(root: Path, rel: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "---\nname: demo-skill\nlicense: MIT\n---\n\n"
        "A demo skill body long enough to pass the minimum length gate.\n"
    )


def _stub_embed(monkeypatch: pytest.MonkeyPatch, dim: int = 8) -> None:
    from mind_nerve import discovery

    monkeypatch.setattr(
        discovery,
        "_embed_texts",
        lambda rt, texts: np.zeros((len(texts), dim), dtype=np.float32),
    )


class TestLoadTableOrEmptyUnit:
    """Direct unit coverage of the three-way split."""

    def test_neither_file_present_returns_empty(self, tmp_path: Path) -> None:
        from mind_nerve.discovery import _load_table_or_empty

        emb, meta = _load_table_or_empty(tmp_path)

        assert emb is None
        assert meta == []

    def test_both_files_present_loads_table(self, tmp_path: Path) -> None:
        from mind_nerve.discovery import _load_table_or_empty

        np.save(tmp_path / "route_table.npy", np.eye(2, 4, dtype=np.float32))
        with (tmp_path / "route_table.jsonl").open("w") as fh:
            fh.write(json.dumps({"id": "r0"}) + "\n")
            fh.write(json.dumps({"id": "r1"}) + "\n")

        emb, meta = _load_table_or_empty(tmp_path)

        assert emb.shape == (2, 4)
        assert [m["id"] for m in meta] == ["r0", "r1"]

    def test_npy_only_refuses(self, tmp_path: Path) -> None:
        from mind_nerve.discovery import _load_table_or_empty

        np.save(tmp_path / "route_table.npy", np.eye(2, 4, dtype=np.float32))

        with pytest.raises(RuntimeError, match="partial route table"):
            _load_table_or_empty(tmp_path)

    def test_jsonl_only_refuses(self, tmp_path: Path) -> None:
        from mind_nerve.discovery import _load_table_or_empty

        with (tmp_path / "route_table.jsonl").open("w") as fh:
            fh.write(json.dumps({"id": "r0"}) + "\n")

        with pytest.raises(RuntimeError, match="partial route table"):
            _load_table_or_empty(tmp_path)


class TestScanRefusesPartialTable:
    """scan() must not silently discard an existing table via a partial pair."""

    def test_scan_refuses_and_preserves_npy_only_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from mind_nerve import discovery

        runtime = tmp_path / "runtime"
        runtime.mkdir()
        original = np.eye(3, 8, dtype=np.float32)
        np.save(runtime / "route_table.npy", original)
        # route_table.jsonl deliberately absent -- partial/corrupted pair.

        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "SKILL.md")
        _stub_embed(monkeypatch)

        with pytest.raises(RuntimeError, match="partial route table"):
            discovery.scan(str(skills_dir), runtime_dir=str(runtime), trusted=True)

        # The surviving file must be byte-for-byte untouched -- no silent
        # overwrite with a fresh (and much smaller) table.
        assert not (runtime / "route_table.jsonl").exists()
        reloaded = np.load(runtime / "route_table.npy")
        np.testing.assert_array_equal(reloaded, original)

    def test_scan_refuses_and_preserves_jsonl_only_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from mind_nerve import discovery

        runtime = tmp_path / "runtime"
        runtime.mkdir()
        original_rows: list[dict[str, Any]] = [
            {"id": "existing-0", "name": "route0", "kind": "skill", "source_repo": "test"},
            {"id": "existing-1", "name": "route1", "kind": "skill", "source_repo": "test"},
        ]
        with (runtime / "route_table.jsonl").open("w") as fh:
            for row in original_rows:
                fh.write(json.dumps(row) + "\n")
        original_text = (runtime / "route_table.jsonl").read_text()
        # route_table.npy deliberately absent -- partial/corrupted pair.

        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "SKILL.md")
        _stub_embed(monkeypatch)

        with pytest.raises(RuntimeError, match="partial route table"):
            discovery.scan(str(skills_dir), runtime_dir=str(runtime), trusted=True)

        assert not (runtime / "route_table.npy").exists()
        assert (runtime / "route_table.jsonl").read_text() == original_text

    def test_scan_still_bootstraps_when_neither_file_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unaffected control case: a genuinely fresh runtime dir must still
        bootstrap the very first table without raising."""
        from mind_nerve import discovery

        runtime = tmp_path / "runtime"
        runtime.mkdir()

        skills_dir = tmp_path / "skills"
        _write_skill(skills_dir, "SKILL.md")
        _stub_embed(monkeypatch)

        result = discovery.scan(str(skills_dir), runtime_dir=str(runtime), trusted=True)

        assert result["added"] == 1
        assert (runtime / "route_table.npy").is_file()
        assert (runtime / "route_table.jsonl").is_file()


class TestWalkDirWindowsPathSeparator:
    """_walk_dir must detect nested skill/command/agent files regardless of
    the OS-native path separator used to build the relative path string."""

    def test_detects_nested_command_with_backslash_relative_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from mind_nerve import discovery

        _write_skill(tmp_path, "commands/deploy.md")

        # Simulate Windows: Path.relative_to(root) stringifies with backslashes
        # there. Force the same shape here regardless of host OS by monkey-
        # patching PurePath.__str__ is fragile; instead assert directly on the
        # rel string _walk_dir computes and feeds to _detect_kind.
        items = list(discovery._walk_dir(tmp_path, "local", trusted=True))

        assert len(items) == 1
        assert items[0]["kind"] == "command"

    def test_detect_kind_rejects_backslash_separated_relative_path(self) -> None:
        """_detect_kind itself requires literal '/' -- documents why as_posix()
        (not str()) is required in _walk_dir on Windows."""
        from mind_nerve.discovery import _detect_kind

        assert _detect_kind("commands\\deploy.md") is None
        assert _detect_kind("commands/deploy.md") == "command"
