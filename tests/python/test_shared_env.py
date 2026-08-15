"""Tests for mind_nerve.shared_env — the ~/.mind-nerve/env fallback pins.

Covers the three consumers' contract: explicit env / CLI-config pins ALWAYS
win, the file only fills unset vars, malformed lines are ignored, and a
missing file is fine. The hook's inline copy of the parser is checked for
byte-compatible behaviour by exec'ing its AST-extracted function.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

from mind_nerve import ensure
from mind_nerve.shared_env import apply_shared_env, load_shared_env, shared_env_path

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK_PATH = REPO_ROOT / "integrations" / "hook" / "mind-nerve-hook"

ENV_FIXTURE = """\
# shared pins for every CLI
MIND_NERVE_RUNTIME_DIR=/srv/mind-nerve/runtime

MIND_NERVE_BACKEND=sentencetransformer
not-a-key-line
=missing-key
1BAD_KEY=value
MIND_NERVE_SPACED = spaced value
"""


def test_missing_file_is_fine(tmp_path):
    assert load_shared_env(tmp_path / "does-not-exist") == {}
    env = {"ALREADY": "1"}
    applied = apply_shared_env(env, path=tmp_path / "does-not-exist")
    assert applied == {}
    assert env == {"ALREADY": "1"}


def test_parse_and_malformed_lines_ignored(tmp_path):
    cfg = tmp_path / "env"
    cfg.write_text(ENV_FIXTURE, encoding="utf-8")
    out = load_shared_env(cfg)
    assert out == {
        "MIND_NERVE_RUNTIME_DIR": "/srv/mind-nerve/runtime",
        "MIND_NERVE_BACKEND": "sentencetransformer",
        "MIND_NERVE_SPACED": "spaced value",
    }


def test_overlong_line_ignored(tmp_path):
    cfg = tmp_path / "env"
    cfg.write_text(f"VALID=1\nHUGE={'x' * 5000}\n", encoding="utf-8")
    assert load_shared_env(cfg) == {"VALID": "1"}


def test_explicit_env_always_wins(tmp_path):
    cfg = tmp_path / "env"
    cfg.write_text(ENV_FIXTURE, encoding="utf-8")
    env = {"MIND_NERVE_BACKEND": "explicit-export"}
    applied = apply_shared_env(env, path=cfg)
    assert env["MIND_NERVE_BACKEND"] == "explicit-export"  # untouched
    assert env["MIND_NERVE_RUNTIME_DIR"] == "/srv/mind-nerve/runtime"  # filled
    assert applied == {
        "MIND_NERVE_RUNTIME_DIR": "/srv/mind-nerve/runtime",
        "MIND_NERVE_SPACED": "spaced value",
    }


def test_env_file_override_path(tmp_path, monkeypatch):
    cfg = tmp_path / "custom-env"
    cfg.write_text("MIND_NERVE_DEVICE=cpu\n", encoding="utf-8")
    monkeypatch.setenv("MIND_NERVE_ENV_FILE", str(cfg))
    assert shared_env_path() == cfg
    assert load_shared_env() == {"MIND_NERVE_DEVICE": "cpu"}


def test_spawn_daemon_env_merges_shared_under_environ(tmp_path, monkeypatch):
    """ensure._spawn_daemon: file fills unset pins, os.environ wins."""
    cfg = tmp_path / "env"
    cfg.write_text(
        "MIND_NERVE_RUNTIME_DIR=/from-file\nMIND_NERVE_BACKEND=file-backend\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MIND_NERVE_ENV_FILE", str(cfg))
    monkeypatch.delenv("MIND_NERVE_RUNTIME_DIR", raising=False)
    monkeypatch.setenv("MIND_NERVE_BACKEND", "explicit-backend")

    captured: dict = {}

    class _FakePopen:
        def __init__(self, argv, **kwargs):
            captured["argv"] = argv
            captured.update(kwargs)

    monkeypatch.setattr(ensure.subprocess, "Popen", _FakePopen)
    ensure._spawn_daemon("mind-nerve-routed", tmp_path / "log" / "daemon.log")
    env = captured["env"]
    assert env["MIND_NERVE_RUNTIME_DIR"] == "/from-file"  # filled from file
    assert env["MIND_NERVE_BACKEND"] == "explicit-backend"  # explicit wins
    assert (tmp_path / "log" / "daemon.log").is_file()


def _hook_load_shared_env():
    """Extract the hook's inline _load_shared_env (standalone script, no import)."""
    tree = ast.parse(HOOK_PATH.read_text(encoding="utf-8"))
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_load_shared_env"
    )
    ns: dict = {"os": os, "re": __import__("re"), "_HOME": "/nonexistent-home"}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<hook>", "exec"), ns)  # noqa: S102
    return ns["_load_shared_env"]


def test_hook_inline_parser_byte_compatible(tmp_path, monkeypatch):
    """The hook's inline reader behaves identically to the package parser."""
    cfg = tmp_path / "env"
    cfg.write_text(ENV_FIXTURE, encoding="utf-8")
    monkeypatch.setenv("MIND_NERVE_ENV_FILE", str(cfg))
    for key in ("MIND_NERVE_RUNTIME_DIR", "MIND_NERVE_BACKEND", "MIND_NERVE_SPACED"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MIND_NERVE_BACKEND", "explicit-backend")

    _hook_load_shared_env()()
    assert os.environ["MIND_NERVE_RUNTIME_DIR"] == "/srv/mind-nerve/runtime"
    assert os.environ["MIND_NERVE_BACKEND"] == "explicit-backend"  # explicit wins
    assert os.environ["MIND_NERVE_SPACED"] == "spaced value"


def test_hook_inline_parser_missing_file_never_raises(monkeypatch):
    monkeypatch.setenv("MIND_NERVE_ENV_FILE", "/nonexistent/dir/env")
    _hook_load_shared_env()()  # must not raise
