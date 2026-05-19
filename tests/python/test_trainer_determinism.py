"""Integration regression for trainer determinism + ``run.json`` schema.

Two trainer invocations with the same ``TrainConfig`` and the same dataset
bytes must produce manifest/eval payloads whose metric values match within
floating-point tolerance and a ``run.json`` whose reproducibility fields
agree on identity (git SHA, dataset hash, hyper-parameters, seed).

The test is marked ``@pytest.mark.slow`` and skips automatically when the
PyTorch / sentence-transformers dependencies are not importable so it
does not regress the unit tier on a torch-less runner.
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Skip preconditions
# ---------------------------------------------------------------------------

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
_ST_AVAILABLE = importlib.util.find_spec("sentence_transformers") is not None

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (_TORCH_AVAILABLE and _ST_AVAILABLE),
        reason="trainer determinism requires torch + sentence-transformers",
    ),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _has_network() -> bool:
    """Best-effort: a fresh base model has to be downloaded from HF."""
    try:
        socket.create_connection(("huggingface.co", 443), timeout=2).close()
        return True
    except OSError:
        return False


def _write_tiny_corpus(path: Path) -> None:
    """Twenty synthetic routes, five queries each — 100 rows."""
    rows: list[str] = []
    base_routes = [
        ("deploy-build", "skill", "ship a new build to the staging environment"),
        ("rollback-release", "skill", "revert the production deploy to the previous tag"),
        ("query-logs", "skill", "search application logs for an error pattern"),
        ("rotate-keys", "skill", "rotate the production API signing keys"),
        ("scale-cluster", "skill", "scale the kubernetes cluster up by two nodes"),
        ("backup-db", "skill", "take a point-in-time backup of the primary database"),
        ("restore-db", "skill", "restore the database from yesterday's snapshot"),
        ("create-user", "skill", "create a new application user with read-only access"),
        ("delete-user", "skill", "delete an application user and revoke all tokens"),
        ("send-email", "skill", "send a transactional email to a customer address"),
        ("crawl-site", "skill", "crawl a website and extract the article text"),
        ("upload-file", "skill", "upload a binary file to object storage with checksum"),
        ("download-file", "skill", "download a binary file from object storage by sha"),
        ("generate-report", "skill", "generate a weekly status report PDF from metrics"),
        ("run-benchmark", "skill", "execute the latency benchmark suite on the box"),
        ("compile-code", "skill", "compile the rust workspace with the release profile"),
        ("lint-code", "skill", "lint the python tree with ruff and report issues"),
        ("format-code", "skill", "format the typescript tree with prettier in place"),
        ("publish-package", "skill", "publish the python wheel to the package registry"),
        ("tag-release", "skill", "create an annotated git tag for the release commit"),
    ]
    for name, kind, body in base_routes:
        # Five variants per route — the trainer wants ≥16-char bodies.
        rows.append(f"{name}\t{kind}\t{body}")
        rows.append(f"{name}\t{kind}\t{body} for the production environment")
        rows.append(f"{name}\t{kind}\t{body} so the on-call engineer is unblocked")
        rows.append(f"{name}\t{kind}\t{body} (automation request)")
        rows.append(f"{name}\t{kind}\t{body} -- requested by the team lead")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


@pytest.fixture()
def synthetic_catalog(tmp_path: Path) -> Path:
    catalog = tmp_path / "catalog.tsv"
    _write_tiny_corpus(catalog)
    return catalog


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_once(catalog: Path, out_dir: Path, seed: int) -> dict:
    from mind_nerve.mind_train import TrainConfig, train

    cfg = TrainConfig(
        catalog_path=catalog,
        output_dir=out_dir,
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        epochs=1,
        batch_size=8,
        lr=2e-5,
        max_len=64,
        seed=seed,
        eval_frac=0.2,
        smoke_test=True,
        backend="python",
        deterministic=True,
    )
    result = train(cfg)
    return {
        "manifest": _read_json(result.manifest_path),
        "run": _read_json(out_dir / "run.json"),
        "eval": _read_json(out_dir / "eval.json"),
        "result_metrics": result.metrics,
    }


def test_two_runs_same_seed_produce_identical_metrics(
    tmp_path: Path,
    synthetic_catalog: Path,
) -> None:
    if not _has_network() and not os.environ.get("MIND_NERVE_TEST_OFFLINE_OK"):
        pytest.skip(
            "base model download requires network; set MIND_NERVE_TEST_OFFLINE_OK to override"
        )

    run_a = _run_once(synthetic_catalog, tmp_path / "run-a", seed=4242)
    run_b = _run_once(synthetic_catalog, tmp_path / "run-b", seed=4242)

    metrics_a = run_a["manifest"]["final_metrics"]
    metrics_b = run_b["manifest"]["final_metrics"]

    # Required eval keys are all present.
    for key in ("top1", "top5", "top10", "mrr", "ndcg@1", "ndcg@5", "ndcg@10", "ece"):
        assert key in metrics_a, f"manifest missing {key!r}"
        assert key in metrics_b, f"manifest missing {key!r}"

    # Identical-input determinism: every numeric metric agrees within float
    # tolerance. ``candidate_pool`` is an integer count.
    for key, value_a in metrics_a.items():
        value_b = metrics_b[key]
        if isinstance(value_a, (int, float)):
            assert value_a == pytest.approx(value_b, abs=1e-9), (
                f"metric {key!r} drifted across runs: {value_a} vs {value_b}"
            )
        else:
            assert value_a == value_b, f"metric {key!r} drifted: {value_a} vs {value_b}"


def test_run_json_schema_fields_present(
    tmp_path: Path,
    synthetic_catalog: Path,
) -> None:
    if not _has_network() and not os.environ.get("MIND_NERVE_TEST_OFFLINE_OK"):
        pytest.skip(
            "base model download requires network; set MIND_NERVE_TEST_OFFLINE_OK to override"
        )

    run = _run_once(synthetic_catalog, tmp_path / "run", seed=99)
    payload = run["run"]
    required = {
        "schema_version",
        "kind",
        "git_sha",
        "requirements_lock_sha256",
        "dataset_manifest_sha256",
        "hf_revision",
        "seed",
        "epochs",
        "batch_size",
        "lr",
        "max_length",
        "eval_fraction",
        "deterministic",
        "backend",
        "base_model",
        "hostname",
        "platform",
        "python_version",
        "torch_version",
        "cuda_available",
        "cuda_version",
        "cpu_info",
        "started_at",
        "finished_at",
        "manifest",
    }
    missing = required - set(payload)
    assert not missing, f"run.json is missing required fields: {sorted(missing)}"
    assert payload["seed"] == 99
    assert payload["deterministic"] is True
    assert payload["backend"] == "python"
    # The dataset hash is deterministic for identical bytes — same catalog
    # produces the same value across runs.
    second = _run_once(synthetic_catalog, tmp_path / "run-2", seed=99)
    assert second["run"]["dataset_manifest_sha256"] == payload["dataset_manifest_sha256"]
