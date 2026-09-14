"""Regression controls for the standalone hook's projected generations."""

from __future__ import annotations

import json
import os
import runpy
import time
from pathlib import Path

import pytest

HOOK = Path(__file__).parents[2] / "integrations" / "hook" / "mind-nerve-hook"
SESSION_ENV_KEYS = (
    "MIND_NERVE_SESSION_ID",
    "CODEX_SESSION_ID",
    "CLAUDE_SESSION_ID",
    "GEMINI_SESSION_ID",
    "CURSOR_SESSION_ID",
)


def _load_hook(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    session: str | None,
    session_env: str = "MIND_NERVE_SESSION_ID",
    max_session_generations: int = 64,
):
    for key in SESSION_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    if session is not None:
        monkeypatch.setenv(session_env, session)
    monkeypatch.setenv("MIND_NERVE_PROJECTED_DIR", str(tmp_path / "skills"))
    monkeypatch.setenv("MIND_NERVE_SOURCE_DIR", str(tmp_path / "source"))
    monkeypatch.setenv("MIND_NERVE_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MIND_NERVE_ENV_FILE", str(tmp_path / "no-env"))
    monkeypatch.setenv("MIND_NERVE_KEEP_GENERATIONS", "3")
    monkeypatch.setenv(
        "MIND_NERVE_MAX_SESSION_GENERATIONS", str(max_session_generations)
    )
    monkeypatch.setenv("MIND_NERVE_LOG", "")
    monkeypatch.setenv("MIND_NERVE_TELEMETRY", "")
    return runpy.run_path(str(HOOK), run_name="mind_nerve_projection_test")


def _publish(ns: dict[str, object], tmp_path: Path, count: int) -> list[Path]:
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    write_projection = ns["write_projection"]
    projected_dir = ns["PROJECTED_DIR"]
    targets: list[Path] = []
    for index in range(count):
        skill = source / f"skill-{index}"
        skill.mkdir()
        (skill / "SKILL.md").write_text(f"# skill {index}\n", encoding="utf-8")
        assert callable(write_projection)
        write_projection([(f"skill-{index}", skill)])
        assert isinstance(projected_dir, Path)
        targets.append(Path(os.readlink(projected_dir)))
    return targets


def test_active_session_lease_keeps_old_skill_path(monkeypatch, tmp_path):
    ns = _load_hook(monkeypatch, tmp_path, "codex-session-1", "CODEX_SESSION_ID")
    targets = _publish(ns, tmp_path, 5)

    assert (targets[0] / "skill-0" / "SKILL.md").is_file()
    lease_dir = targets[0].parent / ".leases"
    lease_files = list(lease_dir.glob("*.json"))
    assert len(lease_files) == 1
    payload = json.loads(lease_files[0].read_text(encoding="utf-8"))
    assert targets[0].name in payload["generations"]
    assert "codex-session-1" not in lease_files[0].name


def test_unknown_session_keeps_existing_bounded_gc(monkeypatch, tmp_path):
    ns = _load_hook(monkeypatch, tmp_path, None)
    targets = _publish(ns, tmp_path, 5)
    generation_dirs = [
        entry
        for entry in targets[-1].parent.iterdir()
        if entry.is_dir() and not entry.is_symlink() and entry.name != ".leases"
    ]

    assert not targets[0].exists()
    assert len(generation_dirs) == ns["KEEP_GENERATIONS"]


def test_session_lease_expires_and_gc_reclaims_generation(monkeypatch, tmp_path):
    monkeypatch.setenv("MIND_NERVE_GENERATION_LEASE_HOURS", "0")
    ns = _load_hook(monkeypatch, tmp_path, "short-lived")
    targets = _publish(ns, tmp_path, 6)
    current = targets[-1].name
    lease_file = next((targets[-1].parent / ".leases").glob("*.json"))
    old_time = time.time() - 61
    os.utime(lease_file, (old_time, old_time))
    ns["_gc_generations"](current)

    assert not targets[0].exists()
    assert targets[-1].exists()


def test_session_lease_has_a_generation_bound(monkeypatch, tmp_path):
    ns = _load_hook(monkeypatch, tmp_path, "bounded-session", max_session_generations=3)
    targets = _publish(ns, tmp_path, 6)
    generation_dirs = [
        entry
        for entry in targets[-1].parent.iterdir()
        if entry.is_dir() and not entry.is_symlink() and entry.name != ".leases"
    ]

    assert not targets[0].exists()
    lease_file = next((targets[-1].parent / ".leases").glob("*.json"))
    assert len(json.loads(lease_file.read_text(encoding="utf-8"))["generations"]) == 3
    # The normal rolling retention and the bounded session lease may overlap
    # by the current generation, so the total is bounded by both policies.
    assert len(generation_dirs) <= ns["KEEP_GENERATIONS"] + 2
