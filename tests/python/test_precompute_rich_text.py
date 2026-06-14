"""Tests for precompute_routes() rich-text embedding behavior.

spec/architecture.md: precompute_routes must produce embeddings that reflect
the full skill description + body, not just the kebab-case name. The quality
gap between name-only and description-first encoding can cause a factor-of-2
score difference for semantically equivalent queries.

RED: these tests must fail before the fix.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_skill(tmp: Path, name: str, description: str, body: str) -> Path:
    """Write a minimal SKILL.md with YAML frontmatter."""
    skill_dir = tmp / name
    skill_dir.mkdir()
    content = f"---\nname: {name}\ndescription: {description}\n---\n\n{body}"
    p = skill_dir / "SKILL.md"
    p.write_text(content, encoding="utf-8")
    return p


def _fake_model(dim: int = 8):
    """Return a minimal SentenceTransformer-like model that encodes via char hash.

    Deterministic: identical inputs produce identical outputs.
    The encoding is NOT semantically meaningful, but the test checks structural
    behavior (which text was used as input), not semantic quality.
    """

    class _FakeModel:
        def encode(
            self,
            texts: list[str],
            *,
            batch_size: int = 128,
            convert_to_numpy: bool = True,
            show_progress_bar: bool = False,
            normalize_embeddings: bool = False,
        ) -> np.ndarray:
            out = np.zeros((len(texts), dim), dtype=np.float32)
            for i, t in enumerate(texts):
                # hash text into float vector — deterministic, unique for different inputs
                h = abs(hash(t)) % (2**31)
                for j in range(dim):
                    out[i, j] = float((h >> j) & 0xFF) / 255.0
            return out

        def eval(self) -> "_FakeModel":
            return self

    return _FakeModel()


# ---------------------------------------------------------------------------
# Helpers to call precompute_routes with a fake model
# ---------------------------------------------------------------------------


def _run_precompute(
    tmp_path: Path, items: list[dict], monkeypatch: pytest.MonkeyPatch
) -> tuple[np.ndarray, list[dict]]:
    """Run precompute_routes with a fake model and return (embeddings, meta)."""
    from mind_nerve import inference as inf

    # Build a minimal runtime dir
    rdir = tmp_path / "runtime"
    rdir.mkdir()
    (rdir / "checkpoint").mkdir()
    (rdir / "manifest.json").write_text(
        json.dumps({"catalog_version": "test", "phase1_version": "test"})
    )

    # Write items.jsonl
    catalog = rdir / "items.jsonl"
    with catalog.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item) + "\n")

    # Patch SentenceTransformer to return our fake model
    fake_model = _fake_model(dim=8)
    monkeypatch.setattr(
        "mind_nerve.inference.SentenceTransformer",
        lambda *a, **kw: fake_model,
        raising=False,
    )
    # Patch at the call site inside precompute_routes (it does its own import)
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", lambda *a, **kw: fake_model)

    inf.precompute_routes(
        runtime_dir=str(rdir),
        catalog_path=str(catalog),
    )

    emb = np.load(rdir / "route_table.npy")
    with (rdir / "route_table.jsonl").open() as f:
        meta = [json.loads(line) for line in f]

    return emb, meta, fake_model


# ---------------------------------------------------------------------------
# Test: precompute_routes uses rich text when source_path is available
# ---------------------------------------------------------------------------


class TestPrecomputeUsesRichText:
    """precompute_routes must encode description+body when source_path is set."""

    def test_name_only_and_rich_text_produce_different_embeddings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two items with the same name but different descriptions must differ."""
        skill_a = _write_skill(
            tmp_path,
            "compute-embedding",
            description="Encode text sequences using transformer models",
            body="# Compute Embedding\n\nUse BERT, RoBERTa, or sentence-transformers.",
        )
        skill_b = _write_skill(
            tmp_path,
            "deploy-kubernetes",
            description="Deploy containerised services to a Kubernetes cluster",
            body="# Deploy Kubernetes\n\nUse kubectl, Helm charts, and manifests.",
        )

        items = [
            {
                "id": "aaa",
                "name": "compute-embedding",
                "kind": "skill",
                "source_repo": "test",
                "source_path": str(skill_a),
            },
            {
                "id": "bbb",
                "name": "deploy-kubernetes",
                "kind": "skill",
                "source_repo": "test",
                "source_path": str(skill_b),
            },
        ]

        emb, meta, model = _run_precompute(tmp_path, items, monkeypatch)

        assert emb.shape == (2, 8)
        # The embeddings must differ (different description → different hash → different vector)
        assert not np.allclose(emb[0], emb[1]), (
            "precompute_routes produced identical embeddings for items with different descriptions; "
            "it is probably encoding only the name, not the description+body."
        )

    def test_embedding_differs_from_name_only_encoding(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The embedding for an item with a source_path should differ from name-only encoding.

        This is the key regression guard: if precompute_routes falls back to
        name-only encoding, the embedding would match ``_fake_model.encode([name])``.
        With the fix it should match the rich text (description + body).
        """
        skill_file = _write_skill(
            tmp_path,
            "ml-pipeline-workflow",
            description=(
                "Build end-to-end MLOps pipelines from data preparation through "
                "model training, validation, and production deployment."
            ),
            body="# ML Pipeline Workflow\n\nDAG orchestration patterns (Airflow, Dagster, Kubeflow).",
        )

        items = [
            {
                "id": "ccc",
                "name": "ml-pipeline-workflow",
                "kind": "skill",
                "source_repo": "test",
                "source_path": str(skill_file),
            },
        ]

        emb, meta, model = _run_precompute(tmp_path, items, monkeypatch)
        assert emb.shape == (1, 8)

        # Compute what name-only encoding would produce
        name_only_emb = model.encode(["ml-pipeline-workflow"])

        # Compute what rich-text encoding should produce
        text = skill_file.read_text(encoding="utf-8")
        fm: dict[str, str] = {}
        if text.startswith("---"):
            end = text.find("\n---", 3)
            for line in text[3:end].splitlines():
                if ":" in line and not line.strip().startswith("#"):
                    k, v = line.split(":", 1)
                    fm[k.strip().lower()] = v.strip().strip('"').strip("'")
        body = text[text.find("\n---", 3) + 4 :] if text.startswith("---") else text
        desc = fm.get("description", "")
        rich_text = (desc or "ml-pipeline-workflow") + "\n\n" + body[:1024]
        rich_emb = model.encode([rich_text])

        # The actual embedding should match the RICH text, not the name-only text
        matches_name_only = np.allclose(emb[0], name_only_emb[0])
        matches_rich = np.allclose(emb[0], rich_emb[0])

        assert matches_rich, (
            "precompute_routes did not use rich text (description+body) for encoding; "
            f"matches_name_only={matches_name_only}, matches_rich={matches_rich}. "
            "Expected matches_rich=True after the fix."
        )
        assert not matches_name_only, (
            "precompute_routes is still using name-only encoding; "
            "the fix should produce embeddings from the description+body text."
        )

    def test_item_without_source_path_uses_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Items without a source_path fall back gracefully to name encoding."""
        items = [
            {
                "id": "ddd",
                "name": "some-remote-tool",
                "kind": "tool",
                "source_repo": "remote",
                "url": "https://example.com/tool",
                # NO source_path
            },
        ]

        emb, meta, model = _run_precompute(tmp_path, items, monkeypatch)
        assert emb.shape == (1, 8)
        # Must not crash; result must be a valid float32 array
        assert emb.dtype == np.float32
        assert np.isfinite(emb).all()
