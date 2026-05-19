# Reproducing a mind-nerve training run

This document is the contract for the public training surface
(`mind_nerve.mind_train.train`) and the `run.json` manifest the trainer
emits alongside every checkpoint. It exists so an external reviewer can
take a published artifact, re-run the recipe, and verify the metrics
without reading any source code.

## What the trainer writes

Every successful `train(config)` produces, inside `config.output_dir`:

```
output_dir/
├── checkpoint/                # sentence-transformers model directory
├── manifest.json              # legacy training manifest (existing schema)
├── eval.json                  # retrieval + calibration metrics (see below)
└── run.json                   # full reproducibility manifest (new)
```

`manifest.json` is preserved verbatim for backward compatibility with
older consumers. The new `run.json` is the single source of truth for
reproducibility.

## `run.json` schema (v1)

```jsonc
{
  "schema_version": 1,
  "kind": "mind_nerve.train.run",

  // Source identity
  "git_sha": "b9b6401...",                // null if not in a git checkout
  "requirements_lock_sha256": "…",        // null if no requirements.lock found
  "dataset_manifest_sha256": "…",         // SHA-256 of the catalog TSV bytes
  "hf_revision": "main",                  // env MIND_NERVE_HF_REVISION or null

  // Hyper-parameters (one-to-one with TrainConfig)
  "seed": 1337,
  "epochs": 3,
  "batch_size": 32,
  "lr": 2e-5,
  "max_length": 256,
  "eval_fraction": 0.1,
  "deterministic": true,
  "backend": "python",
  "base_model": "BAAI/bge-small-en-v1.5",

  // Host facts captured at runtime
  "hostname": "host.example",
  "platform": "Linux-6.17.0-23-generic-x86_64-with-glibc2.39",
  "python_version": "3.12.3",
  "torch_version": "2.3.1",
  "cuda_available": false,
  "cuda_version": null,
  "cpu_info": "Intel(R) Xeon(R) CPU E5-2670 v3 @ 2.30GHz",

  // Time bounds (ISO-8601 UTC)
  "started_at": "2026-05-18T19:14:02Z",
  "finished_at": "2026-05-18T19:18:51Z",

  // Augmentation — the legacy manifest, embedded verbatim
  "manifest": { … }
}
```

### Field reference

| Field                       | Type      | Description                                                                                       |
| --------------------------- | --------- | ------------------------------------------------------------------------------------------------- |
| `git_sha`                   | string?   | Output of `git rev-parse HEAD` inside the current process working directory.                       |
| `requirements_lock_sha256`  | string?   | SHA-256 of the nearest `requirements.lock` (walking upward from the package). Null if absent.     |
| `dataset_manifest_sha256`   | string?   | SHA-256 of `TrainConfig.catalog_path` bytes. Stable for identical TSV content.                    |
| `hf_revision`               | string?   | Value of `MIND_NERVE_HF_REVISION` (set this env var to pin the base-model revision).               |
| `seed`                      | int       | Master RNG seed; controls Python / NumPy / Torch / CUDA seeding.                                  |
| `deterministic`             | bool      | `True` when deterministic flags were applied (see below).                                         |
| `backend`                   | string    | `"python"` (default) or `"native"` (Q16.16 path; not yet trainable).                              |
| `cpu_info`                  | string    | First `model name` line from `/proc/cpuinfo` if available, else `platform.processor()`.            |
| `started_at`, `finished_at` | string    | ISO-8601 UTC timestamps captured by the trainer.                                                  |
| `manifest`                  | object    | The complete `manifest.json` payload — no fields are removed when `run.json` lands.                |

## Deterministic mode

`TrainConfig(deterministic=True)` is the default and applies the
following before any model is constructed:

```python
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
torch.cuda.manual_seed_all(cfg.seed)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
```

### What this guarantees

* Two runs with the same `TrainConfig` and the same catalog bytes on the
  same hardware produce eval metrics that agree within float tolerance
  `1e-9`.
* `dataset_manifest_sha256` is byte-stable for identical TSV input.
* Seeds for Python / NumPy / Torch / CUDA / cuDNN are pinned.

### What this does **not** guarantee

* **Cross-architecture bit-identity.** FP16/BF16 accumulators on
  different GPUs (Ampere vs Hopper, etc.) produce slightly different
  losses; treat the determinism contract as "same machine, same
  hardware" only.
* **Cross-Torch version stability.** A new PyTorch release can change
  the kernel selection table; pin `torch_version` via
  `requirements.lock` to lock this down.
* **Cross-OS reproducibility for the base model.** `sentence-transformers`
  may download fresh weights if HF cache is empty; set
  `MIND_NERVE_HF_REVISION` to pin the artifact revision and re-export
  the produced cache directory for archival reproduction.
* **Throughput.** `torch.use_deterministic_algorithms(True, warn_only=True)`
  plus cuDNN deterministic mode typically slows GPU training by 10-25%
  on the BGE-small recipe. The CPU path is unaffected.

### Opting out

```python
from mind_nerve.mind_train import TrainConfig, train

result = train(TrainConfig(
    catalog_path=Path("catalog.tsv"),
    output_dir=Path("./run"),
    deterministic=False,            # accept run-to-run drift for speed
))
```

`deterministic=False` skips the deterministic-algorithms flag and cuDNN
pinning. Seed-based RNG seeding is still applied so the data split and
weight initialization remain stable.

## Eval metrics

`eval.json` and `manifest.json["final_metrics"]` now contain, in
addition to the historical fields:

| Key             | Definition                                                                                            |
| --------------- | ----------------------------------------------------------------------------------------------------- |
| `top1`/`top5`/`top10` | Hit-rate at cut-off `k`. Existing fields, preserved verbatim.                                   |
| `mrr`           | Mean Reciprocal Rank over the eval pool. `0` when the truth is never retrieved.                       |
| `ndcg@1`/`ndcg@5`/`ndcg@10` | Mean nDCG@k with binary relevance: `1/log2(1 + rank)` if the truth is in the top-k, else `0`. |
| `ece`           | Expected Calibration Error using the top-1 softmax confidence vs whether the top-1 was correct.        |
| `candidate_pool` | Number of positives in the retrieval pool.                                                            |

All metrics are computed by the pure-Python reference implementations
in `mind_nerve.eval_metrics`; they are deterministic across NumPy
versions to within IEEE-754 rounding.

## Reproducing a published run

Assume someone has published a training output containing `run.json`,
`manifest.json`, `eval.json`, and a `checkpoint/` directory.

```bash
# 1. Inspect the published run.json
cat run.json | jq '{git_sha, dataset_manifest_sha256, hf_revision, seed, deterministic, base_model, torch_version}'

# 2. Recreate the recorded environment.
git checkout "$(jq -r .git_sha run.json)"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock         # SHA-locked deps; same hash as run.json
export MIND_NERVE_HF_REVISION="$(jq -r .hf_revision run.json)"

# 3. Re-run the trainer with the exact same TrainConfig.
python - <<'PY'
import json
from pathlib import Path
from mind_nerve.mind_train import TrainConfig, train

run = json.loads(Path("run.json").read_text())
cfg = TrainConfig(
    catalog_path=Path("catalog.tsv"),
    output_dir=Path("./reproduce"),
    base_model=run["base_model"],
    epochs=run["epochs"],
    batch_size=run["batch_size"],
    lr=run["lr"],
    max_len=run["max_length"],
    seed=run["seed"],
    eval_frac=run["eval_fraction"],
    deterministic=run["deterministic"],
    backend=run["backend"],
)
result = train(cfg)
print(result.metrics)
PY

# 4. Diff the new run.json against the published one.
diff <(jq -S 'del(.started_at, .finished_at, .hostname, .platform, .manifest.elapsed_seconds, .manifest.trained_at_iso, .manifest.started_at, .manifest.finished_at)' run.json) \
     <(jq -S 'del(.started_at, .finished_at, .hostname, .platform, .manifest.elapsed_seconds, .manifest.trained_at_iso, .manifest.started_at, .manifest.finished_at)' ./reproduce/run.json)
```

The fields excluded above (`started_at`, `finished_at`, hostname,
platform, wall-clock elapsed) are documented as **non-reproducible** —
everything else, including all metrics, the model hash, and the dataset
hash, must match.

## Caveats and known non-guarantees

* The trainer does **not** record CPU thread count or `OMP_NUM_THREADS`.
  Some torch kernels reduce non-deterministically across threads when
  `warn_only=True`; pin these in the wrapper environment if you need
  full bit-identity.
* The native MIND backend (`backend="native"`) raises
  `NotImplementedError` in this release. Native bit-identity is gated on
  the Q16.16 kernel landing.
* Calibration (`ece`) depends on the softmax over similarities being a
  meaningful probability — it is most useful for relative comparison
  between checkpoints, not as an absolute miscalibration claim.
