"""Regression tests for discovery.scan()'s duplicate-on-edit defect.

``scan()`` deduplicated purely on CONTENT hash: ``seen_ids`` was the set of
``sha256`` values already in the table, and anything not in it was APPENDED.
Editing an already-indexed file therefore produced a *new* sha256, which was
not "already indexed", so the edited row was appended while the row describing
the PRE-EDIT content survived. One file on disk, two rows in the table, both
competing for the same top-K slots -- and the stale row frequently outranked
the fresh one.

Observed in the wild: the governed hub table had grown to 2420 rows for 1379
distinct files (~1040 duplicated ``source_path``), roughly halving effective
top_k. Two earlier manual cleanups (``route_table.jsonl.pre-dedup-source-path-*``)
removed the duplicates without fixing the append, so the corpus re-duplicated.

The fix makes ``scan()`` UPSERT on ``source_path``: a scanned file whose path
already has a row REPLACES that row (in place, preserving row order and hence
npy/jsonl alignment) instead of appending a second one.

Note ``add_route()`` is deliberately NOT changed: its ``source_path`` defaults
to ``""`` for programmatic callers, so many legitimate rows share that empty
value and it is not a usable identity key there.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def _write_skill(root: Path, rel: str, body: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"---\nname: demo-skill\nlicense: MIT\n---\n\n{body}\n")
    return p


def _stub_runtime(monkeypatch: pytest.MonkeyPatch, dim: int = 8) -> None:
    """Stub both the encoder and the runtime load.

    A second scan over an existing table reaches ``load_default_runtime``,
    which needs a real manifest + encoder weights. Routing quality is not what
    these tests assert -- row identity is -- so a stub keeps them hermetic.
    """
    from mind_nerve import discovery

    monkeypatch.setattr(
        discovery,
        "_embed_texts",
        lambda rt, texts: np.zeros((len(texts), dim), dtype=np.float32),
    )

    class _FakeRT:
        def __init__(self, routes: list[dict[str, Any]]) -> None:
            self.routes = routes

    def _fake_load(runtime_dir: Any = None) -> _FakeRT:
        rdir = Path(str(runtime_dir))
        meta_path = rdir / "route_table.jsonl"
        rows: list[dict[str, Any]] = []
        if meta_path.is_file():
            rows = [
                json.loads(line)
                for line in meta_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        return _FakeRT(rows)

    _fake_load.cache_clear = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(discovery, "load_default_runtime", _fake_load)


def _rows(runtime: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (runtime / "route_table.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class TestScanUpsertsOnEdit:
    def test_editing_a_skill_replaces_its_row_rather_than_appending(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from mind_nerve import discovery

        runtime = tmp_path / "runtime"
        runtime.mkdir()
        skills = tmp_path / "skills"
        skill = _write_skill(skills, "demo/SKILL.md", "Body version one, long enough to pass.")
        _stub_runtime(monkeypatch)

        first = discovery.scan(str(skills), runtime_dir=str(runtime), trusted=True)
        assert first["added"] == 1
        before = _rows(runtime)
        assert len(before) == 1
        old_sha = before[0]["sha256"]

        # Edit in place: same path, new content -> new sha256.
        skill.write_text(
            "---\nname: demo-skill\nlicense: MIT\n---\n\n"
            "Body version two, edited and still long enough to pass.\n"
        )

        second = discovery.scan(str(skills), runtime_dir=str(runtime), trusted=True)

        after = _rows(runtime)
        assert len(after) == 1, f"edited skill duplicated its row: {after}"
        assert after[0]["source_path"] == str(skill)
        assert after[0]["sha256"] != old_sha, "row still describes the pre-edit content"
        assert second["updated"] == 1
        assert second["added"] == 0
        assert second["total_routes_after"] == 1

        # npy and jsonl must stay row-aligned after an in-place replacement.
        emb = np.load(runtime / "route_table.npy")
        assert emb.shape[0] == len(after)

    def test_unchanged_skill_is_still_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Control: rescanning an untouched tree must be a no-op, as before."""
        from mind_nerve import discovery

        runtime = tmp_path / "runtime"
        runtime.mkdir()
        skills = tmp_path / "skills"
        _write_skill(skills, "demo/SKILL.md", "Body version one, long enough to pass.")
        _stub_runtime(monkeypatch)

        discovery.scan(str(skills), runtime_dir=str(runtime), trusted=True)
        second = discovery.scan(str(skills), runtime_dir=str(runtime), trusted=True)

        assert second["added"] == 0
        assert second.get("updated", 0) == 0
        assert second["skipped"]["already_indexed"] == 1
        assert len(_rows(runtime)) == 1

    def test_new_skill_still_appends(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Control: a genuinely new file must still be added, not swallowed."""
        from mind_nerve import discovery

        runtime = tmp_path / "runtime"
        runtime.mkdir()
        skills = tmp_path / "skills"
        _write_skill(skills, "demo/SKILL.md", "Body version one, long enough to pass.")
        _stub_runtime(monkeypatch)

        discovery.scan(str(skills), runtime_dir=str(runtime), trusted=True)
        _write_skill(skills, "other/SKILL.md", "A second distinct skill body, long enough.")
        second = discovery.scan(str(skills), runtime_dir=str(runtime), trusted=True)

        assert second["added"] == 1
        rows = _rows(runtime)
        assert len(rows) == 2
        assert len({r["source_path"] for r in rows}) == 2

    def test_edit_preserves_row_position(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An upsert must replace in place, not delete-and-append.

        Row order is the npy row index. Moving a row on edit would silently
        re-key every row after it if the two files were ever written by
        different code paths; replacing in place keeps the mapping stable.
        """
        from mind_nerve import discovery

        runtime = tmp_path / "runtime"
        runtime.mkdir()
        skills = tmp_path / "skills"
        _write_skill(skills, "aaa/SKILL.md", "First skill body, long enough to pass.")
        _write_skill(skills, "zzz/SKILL.md", "Second skill body, long enough to pass.")
        _stub_runtime(monkeypatch)

        discovery.scan(str(skills), runtime_dir=str(runtime), trusted=True)
        before = [r["source_path"] for r in _rows(runtime)]
        assert len(before) == 2

        target = before[0]
        Path(target).write_text(
            "---\nname: demo-skill\nlicense: MIT\n---\n\n"
            "First skill body, now edited and long enough to pass.\n"
        )
        discovery.scan(str(skills), runtime_dir=str(runtime), trusted=True)

        after = [r["source_path"] for r in _rows(runtime)]
        assert after == before, "upsert reordered rows instead of replacing in place"
