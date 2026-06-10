"""Tests for the repo-scan → capability bundle feature.

`scan_repo` walks a target repository, extracts deterministic capability
signals, routes each over the governed table, and merges them into a
ranked bundle. Two properties matter:

  * signal extraction is a pure function of the repo's on-disk bytes
    (same repo → same signals on every host), and
  * the merged bundle is bit-stable: ordered by score desc, then by
    ascending SHA-256(route_id), with per-route signal provenance.

These tests stub `route()` so no runtime/encoder is required.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from mind_nerve.types import Route, RouteResult


def _mkfile(root: Path, rel: str, body: str = "x") -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)


def _route_result(query: str, routes: list[Route]) -> RouteResult:
    return RouteResult(
        query=query,
        top_k=len(routes),
        routes=routes,
        encode_ms=0.0,
        rank_ms=0.0,
        catalog_size=len(routes),
        catalog_version="test",
        model_version="test",
    )


# ---------------------------------------------------------------------------
# extract_signals: pure function of repo bytes
# ---------------------------------------------------------------------------


class TestExtractSignals:
    def test_manifest_and_extension_signals(self, tmp_path: Path) -> None:
        from mind_nerve.scan_repo import extract_signals

        _mkfile(tmp_path, "pyproject.toml")
        _mkfile(tmp_path, "src/app.py")
        _mkfile(tmp_path, "go.mod")
        _mkfile(tmp_path, "cmd/main.go")

        sig = extract_signals(str(tmp_path))

        assert "python packaging project" in sig.phrases
        assert "python source code" in sig.phrases
        assert "go module dependencies" in sig.phrases
        assert "go source code" in sig.phrases

    def test_phrases_sorted_and_deduped(self, tmp_path: Path) -> None:
        from mind_nerve.scan_repo import extract_signals

        # Two python files must yield exactly one language phrase.
        _mkfile(tmp_path, "a.py")
        _mkfile(tmp_path, "b.py")

        sig = extract_signals(str(tmp_path))

        assert sig.phrases == sorted(sig.phrases)
        assert sig.phrases.count("python source code") == 1

    def test_skip_dirs_pruned(self, tmp_path: Path) -> None:
        from mind_nerve.scan_repo import extract_signals

        # A rust file buried only inside node_modules must not register.
        _mkfile(tmp_path, "node_modules/dep/index.rs")
        _mkfile(tmp_path, "main.py")

        sig = extract_signals(str(tmp_path))

        assert "rust source code" not in sig.phrases
        assert "python source code" in sig.phrases

    def test_deterministic_across_calls(self, tmp_path: Path) -> None:
        from mind_nerve.scan_repo import extract_signals

        _mkfile(tmp_path, "package.json")
        _mkfile(tmp_path, "tsconfig.json")
        _mkfile(tmp_path, "src/index.ts")
        _mkfile(tmp_path, "Dockerfile")

        first = extract_signals(str(tmp_path))
        second = extract_signals(str(tmp_path))

        assert first.phrases == second.phrases

    def test_github_dir_signal(self, tmp_path: Path) -> None:
        from mind_nerve.scan_repo import extract_signals

        _mkfile(tmp_path, ".github/workflows/ci.yml")

        sig = extract_signals(str(tmp_path))

        assert "github actions ci workflow" in sig.phrases

    def test_max_files_truncation_is_deterministic(self, tmp_path: Path) -> None:
        from mind_nerve.scan_repo import extract_signals

        for i in range(50):
            _mkfile(tmp_path, f"f{i:03d}.py")

        a = extract_signals(str(tmp_path), max_files=5)
        b = extract_signals(str(tmp_path), max_files=5)

        assert a.files_seen == b.files_seen
        assert a.phrases == b.phrases

    def test_max_files_cap_is_strict(self, tmp_path: Path) -> None:
        from mind_nerve.scan_repo import extract_signals

        for i in range(50):
            _mkfile(tmp_path, f"f{i:03d}.py")

        sig = extract_signals(str(tmp_path), max_files=5)
        assert sig.files_seen <= 5

    def test_terraform_extension_signal(self, tmp_path: Path) -> None:
        from mind_nerve.scan_repo import extract_signals

        _mkfile(tmp_path, "main.tf")

        sig = extract_signals(str(tmp_path))
        assert "terraform infrastructure as code" in sig.phrases

    def test_empty_repo(self, tmp_path: Path) -> None:
        from mind_nerve.scan_repo import extract_signals

        sig = extract_signals(str(tmp_path))
        assert sig.phrases == []
        assert sig.files_seen == 0

    def test_nonexistent_path_raises(self, tmp_path: Path) -> None:
        from mind_nerve.scan_repo import extract_signals

        with pytest.raises(NotADirectoryError):
            extract_signals(str(tmp_path / "does-not-exist"))


# ---------------------------------------------------------------------------
# scan_repo: bundle merge + ordering + provenance
# ---------------------------------------------------------------------------


class TestScanRepo:
    def test_bundle_merges_by_route_id_keeping_max_score(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mind_nerve.scan_repo as sr

        _mkfile(tmp_path, "app.py")
        _mkfile(tmp_path, "main.go")

        # "shared" route appears for both signals with different scores;
        # the bundle must keep the max (0.9) and record both signals.
        def fake_route(query: str, top_k: int, runtime_dir: object = None) -> RouteResult:
            if "python" in query:
                routes = [
                    Route("shared", "Shared", "skill", 0.5, "repo"),
                    Route("py-only", "PyOnly", "skill", 0.7, "repo"),
                ]
            else:
                routes = [
                    Route("shared", "Shared", "skill", 0.9, "repo"),
                    Route("go-only", "GoOnly", "agent", 0.6, "repo"),
                ]
            return _route_result(query, routes)

        monkeypatch.setattr(sr, "route", fake_route)

        out = sr.scan_repo(str(tmp_path), per_signal_k=2, bundle_size=10)
        by_id = {e["id"]: e for e in out["bundle"]}

        assert by_id["shared"]["score"] == pytest.approx(0.9)
        assert len(by_id["shared"]["matched_signals"]) >= 2
        assert "py-only" in by_id
        assert "go-only" in by_id

    def test_bundle_ordered_score_desc_then_sha256(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mind_nerve.scan_repo as sr

        _mkfile(tmp_path, "app.py")

        def fake_route(query: str, top_k: int, runtime_dir: object = None) -> RouteResult:
            routes = [
                Route("zzz", "Z", "skill", 0.8, "repo"),
                Route("aaa", "A", "skill", 0.8, "repo"),  # tie with zzz
                Route("low", "L", "skill", 0.2, "repo"),
            ]
            return _route_result(query, routes)

        monkeypatch.setattr(sr, "route", fake_route)

        out = sr.scan_repo(str(tmp_path), per_signal_k=3, bundle_size=10)
        ids = [e["id"] for e in out["bundle"]]

        # 0.2 route last; the two 0.8 routes ordered by ascending SHA-256(id).
        assert ids[-1] == "low"
        tie = ids[:2]
        expected_tie = sorted(["zzz", "aaa"], key=lambda i: hashlib.sha256(i.encode()).digest())
        assert tie == expected_tie

    def test_bundle_size_caps_output(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import mind_nerve.scan_repo as sr

        _mkfile(tmp_path, "app.py")

        def fake_route(query: str, top_k: int, runtime_dir: object = None) -> RouteResult:
            routes = [Route(f"r{i}", f"R{i}", "skill", 0.9 - i * 0.01, "repo") for i in range(20)]
            return _route_result(query, routes)

        monkeypatch.setattr(sr, "route", fake_route)

        out = sr.scan_repo(str(tmp_path), per_signal_k=20, bundle_size=5)
        assert out["bundle_size"] == 5
        assert len(out["bundle"]) == 5

    def test_full_scan_is_deterministic(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mind_nerve.scan_repo as sr

        _mkfile(tmp_path, "pyproject.toml")
        _mkfile(tmp_path, "app.py")
        _mkfile(tmp_path, "Cargo.toml")
        _mkfile(tmp_path, "lib.rs")

        def fake_route(query: str, top_k: int, runtime_dir: object = None) -> RouteResult:
            digest = int(hashlib.sha256(query.encode()).hexdigest(), 16)
            routes = [
                Route(f"r{(digest + i) % 7}", "N", "skill", 0.5 + i * 0.1, "repo") for i in range(3)
            ]
            return _route_result(query, routes)

        monkeypatch.setattr(sr, "route", fake_route)

        a = sr.scan_repo(str(tmp_path), per_signal_k=3, bundle_size=10)
        b = sr.scan_repo(str(tmp_path), per_signal_k=3, bundle_size=10)
        assert a == b

    def test_empty_repo_yields_empty_bundle(self, tmp_path: Path) -> None:
        from mind_nerve.scan_repo import scan_repo

        out = scan_repo(str(tmp_path))
        assert out["bundle_size"] == 0
        assert out["bundle"] == []
        assert out["signals"] == []

    def test_scan_respects_max_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import mind_nerve.scan_repo as sr

        for i in range(50):
            _mkfile(tmp_path, f"f{i:03d}.py")

        def fake_route(query: str, top_k: int, runtime_dir: object = None) -> RouteResult:
            return _route_result(query, [Route("r0", "R", "skill", 0.5, "repo")])

        monkeypatch.setattr(sr, "route", fake_route)

        a = sr.scan_repo(str(tmp_path), max_files=5)
        b = sr.scan_repo(str(tmp_path), max_files=5)
        assert a == b
        assert a["files_seen"] <= 5

    def test_invalid_params_rejected(self, tmp_path: Path) -> None:
        from mind_nerve.scan_repo import scan_repo

        with pytest.raises(ValueError):
            scan_repo(str(tmp_path), per_signal_k=0)
        with pytest.raises(ValueError):
            scan_repo(str(tmp_path), per_signal_k=999)
        with pytest.raises(ValueError):
            scan_repo(str(tmp_path), bundle_size=0)
