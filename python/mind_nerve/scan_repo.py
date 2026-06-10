"""Repo-scan → capability bundle.

Given a target repository, extract a deterministic set of capability
signals (languages, package manifests, framework markers) and route each
one over the *governed* mind-nerve table, then merge the per-signal route
lists into a single deduped bundle of skills/agents the agent is likely to
need for that repo.

Design constraints (wedge guardrails):

* Signal extraction is a pure function of the repo's on-disk bytes — no
  network, no clock, no environment. The same repo yields the same signals
  on every host.
* Routing goes only through :func:`mind_nerve.inference.route`, i.e. the
  local governed route table. There is **no** dependency on any public
  capability catalog; the bundle is reproducible.
* Bundle ordering reuses the existing cross-arch tie-break contract
  (score desc, then SHA-256(route_id) asc) so the bundle is bit-stable.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any

from .inference import route

# Files whose mere presence implies a capability signal. Mapped to the
# phrase(s) we route on. Kept small and obvious — extend deliberately.
_MANIFEST_SIGNALS: dict[str, tuple[str, ...]] = {
    "package.json": ("javascript node npm project",),
    "pnpm-lock.yaml": ("pnpm monorepo javascript",),
    "yarn.lock": ("yarn javascript dependencies",),
    "tsconfig.json": ("typescript project configuration",),
    "pyproject.toml": ("python packaging project",),
    "setup.py": ("python package setup",),
    "requirements.txt": ("python pip dependencies",),
    "Cargo.toml": ("rust cargo crate",),
    "go.mod": ("go module dependencies",),
    "pom.xml": ("java maven build",),
    "build.gradle": ("gradle jvm build",),
    "Gemfile": ("ruby bundler gems",),
    "composer.json": ("php composer dependencies",),
    "Dockerfile": ("docker container image build",),
    "docker-compose.yml": ("docker compose multi-service",),
    "docker-compose.yaml": ("docker compose multi-service",),
    ".github": ("github actions ci workflow",),
    "Makefile": ("make build automation",),
    "kustomization.yaml": ("kubernetes manifests deployment",),
    "Chart.yaml": ("helm kubernetes chart",),
    "Mind.toml": ("mind language project",),
}

# Source-file extensions → language phrase. First hit per language wins.
_EXT_SIGNALS: dict[str, str] = {
    ".py": "python source code",
    ".ts": "typescript source code",
    ".tsx": "react typescript frontend",
    ".js": "javascript source code",
    ".jsx": "react javascript frontend",
    ".rs": "rust source code",
    ".go": "go source code",
    ".java": "java source code",
    ".rb": "ruby source code",
    ".php": "php source code",
    ".c": "c source code",
    ".cpp": "c++ source code",
    ".cc": "c++ source code",
    ".cs": "c# dotnet source code",
    ".swift": "swift source code",
    ".kt": "kotlin source code",
    ".sql": "sql database query",
    ".mind": "mind language source",
    ".sh": "shell scripting",
    ".tf": "terraform infrastructure as code",
}

# Directories never worth descending into for signal extraction.
_SKIP_DIRS = frozenset(
    {
        ".git",
        "node_modules",
        "target",
        "dist",
        "build",
        "__pycache__",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        "vendor",
    }
)


@dataclass(frozen=True)
class RepoSignals:
    """Deterministic capability signals extracted from a repo."""

    root: str
    phrases: list[str]
    files_seen: int


def extract_signals(repo_path: str, *, max_files: int = 20000) -> RepoSignals:
    """Walk ``repo_path`` and return a sorted, deduped list of capability phrases.

    Pure function of the repo bytes (modulo ``max_files`` truncation, which
    is itself deterministic because :func:`os.walk` entries are sorted).
    """
    root = os.path.abspath(repo_path)
    if not os.path.isdir(root):
        raise NotADirectoryError(f"repo_path is not a directory: {repo_path!r}")
    phrases: set[str] = set()
    langs_seen: set[str] = set()
    files_seen = 0

    for dirpath, dirnames, filenames in os.walk(root):
        # Deterministic traversal + prune skip dirs in place.
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        filenames = sorted(filenames)

        # Directory-name manifest signals (e.g. ".github").
        base = os.path.basename(dirpath)
        if base in _MANIFEST_SIGNALS:
            phrases.update(_MANIFEST_SIGNALS[base])

        for fn in filenames:
            if files_seen >= max_files:
                break
            files_seen += 1

            if fn in _MANIFEST_SIGNALS:
                phrases.update(_MANIFEST_SIGNALS[fn])

            ext = os.path.splitext(fn)[1].lower()
            if ext in _EXT_SIGNALS and ext not in langs_seen:
                langs_seen.add(ext)
                phrases.add(_EXT_SIGNALS[ext])

        if files_seen >= max_files:
            break

    return RepoSignals(root=root, phrases=sorted(phrases), files_seen=files_seen)


def _tie_key(route_id: str) -> bytes:
    return hashlib.sha256(route_id.encode("utf-8")).digest()


def scan_repo(
    repo_path: str,
    *,
    per_signal_k: int = 5,
    bundle_size: int = 15,
    runtime_dir: str | None = None,
    max_files: int = 20000,
) -> dict[str, Any]:
    """Scan ``repo_path`` and return a recommended capability bundle.

    For each extracted signal phrase, route the top ``per_signal_k`` over the
    governed table, merge by route id (keeping the max score across signals),
    and return the top ``bundle_size`` with deterministic ordering.
    """
    if not 1 <= per_signal_k <= 64:
        raise ValueError(f"per_signal_k must be in [1, 64]; got {per_signal_k}")
    if bundle_size < 1:
        raise ValueError(f"bundle_size must be >= 1; got {bundle_size}")

    sig = extract_signals(repo_path, max_files=max_files)

    # route_id -> (best_score, route_dict, set_of_matching_signals)
    merged: dict[str, tuple[float, dict[str, Any], set[str]]] = {}
    for phrase in sig.phrases:
        # Pass ``runtime_dir`` straight through (``None`` when unset) so
        # route()'s own resolution order applies: explicit arg → env var →
        # default. Pre-binding the default proxy here would shadow the
        # MIND_NERVE_RUNTIME_DIR pin the daemon relies on.
        result = route(phrase, top_k=per_signal_k, runtime_dir=runtime_dir)
        for r in result.routes:
            prev = merged.get(r.id)
            if prev is None or r.score > prev[0]:
                # Copy the accumulated signal set rather than alias it, so the
                # replaced tuple's set can never be mutated through this entry.
                signals = set(prev[2]) if prev else set()
                signals.add(phrase)
                merged[r.id] = (r.score, r.as_dict(), signals)
            else:
                prev[2].add(phrase)

    ranked = sorted(
        merged.items(),
        key=lambda kv: (-kv[1][0], _tie_key(kv[0])),
    )

    bundle = []
    for _route_id, (_score, rdict, signals) in ranked[:bundle_size]:
        entry = dict(rdict)
        entry["matched_signals"] = sorted(signals)
        bundle.append(entry)

    return {
        "repo": sig.root,
        "files_seen": sig.files_seen,
        "signals": sig.phrases,
        "bundle_size": len(bundle),
        "bundle": bundle,
    }
