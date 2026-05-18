"""mind-nerve inference — PyTorch Phase 1 path.

Loads the fine-tuned sentence-transformers checkpoint + the
precomputed catalog embeddings, encodes one query, returns top-K.

The Phase 1 inference path is intentionally simple. Phase 2 swaps the
PyTorch encoder for the native MIND inference binary; the public
API in `__init__.py` stays unchanged.

Runtime directory resolution
----------------------------
The runtime dir holds `manifest.json`, `checkpoint/`, `route_table.npy`,
and `route_table.jsonl`. Resolution order, first hit wins:

  1. Explicit ``runtime_dir`` argument to ``route()`` / ``load_default_runtime()``
  2. ``MIND_NERVE_RUNTIME_DIR`` env var
  3. ``~/.local/share/mind-nerve/runtime/`` (auto-seeded from
     ``star-ga/mind-nerve-phase1`` on Hugging Face on first use)
"""

from __future__ import annotations

import functools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .types import Route, RouteResult

_HF_REPO_ID = "star-ga/mind-nerve-phase1"
_USER_RUNTIME_DIR = Path.home() / ".local" / "share" / "mind-nerve" / "runtime"


def _seed_from_hf(target: Path) -> None:
    """Snapshot-download the Phase-1 weights from Hugging Face into *target*.

    Idempotent: skips files that already exist. Prints a one-line progress
    notice to stderr on first download (sub-second on cache-hot machines,
    ~150 MB cold).
    """
    from huggingface_hub import snapshot_download

    print(
        f"mind-nerve: downloading Phase-1 weights ({_HF_REPO_ID}, ~150 MB) to {target}",
        file=sys.stderr,
    )
    cached = Path(snapshot_download(repo_id=_HF_REPO_ID, repo_type="model"))
    target.mkdir(parents=True, exist_ok=True)
    import shutil

    for item in cached.iterdir():
        if item.name.startswith("."):
            continue
        dst = target / item.name
        if dst.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, dst, symlinks=False)
        else:
            shutil.copy2(item, dst)


def _resolve_runtime_dir(runtime_dir: str | None = None) -> Path:
    """Return a Path to a valid mind-nerve runtime directory.

    Auto-seeds ``~/.local/share/mind-nerve/runtime/`` from Hugging Face when
    no explicit runtime is provided.
    """
    if runtime_dir:
        p = Path(runtime_dir).expanduser()
        if not p.is_dir():
            raise FileNotFoundError(f"runtime dir {p} does not exist")
        return p
    env_dir = os.environ.get("MIND_NERVE_RUNTIME_DIR")
    if env_dir:
        p = Path(env_dir).expanduser()
        if not p.is_dir():
            raise FileNotFoundError(f"MIND_NERVE_RUNTIME_DIR={env_dir} does not exist")
        return p
    if not (_USER_RUNTIME_DIR / "manifest.json").exists():
        _seed_from_hf(_USER_RUNTIME_DIR)
    return _USER_RUNTIME_DIR


# Compatibility shim: discovery.py and the CLI used to import this constant.
# It now lazy-evaluates on first attribute access so the HF download isn't
# triggered at import time.
class _DefaultRuntimeDirProxy(str):  # type: ignore[misc]
    """str-compatible proxy that resolves to the runtime dir on str-cast."""

    def __new__(cls):
        return super().__new__(cls, "<lazy:mind-nerve-runtime>")

    def __str__(self) -> str:
        return str(_resolve_runtime_dir())

    def __fspath__(self) -> str:
        return str(_resolve_runtime_dir())


_DEFAULT_RUNTIME_DIR = _DefaultRuntimeDirProxy()


class _Runtime:
    """Loaded model + precomputed catalog embeddings."""

    def __init__(self, runtime_dir: Path):
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self.dir = runtime_dir
        self.manifest = json.loads((runtime_dir / "manifest.json").read_text())

        # Device selection. `MIND_NERVE_DEVICE=cpu` forces CPU even if a GPU
        # is visible — useful when sharing the GPU with another resident
        # model (e.g. a local LLM). Otherwise we attempt the default
        # sentence-transformers selection (CUDA → MPS → CPU) and fall back
        # to CPU on OOM rather than crashing the user's first prompt.
        forced = os.environ.get("MIND_NERVE_DEVICE")
        if forced:
            self.model = SentenceTransformer(str(runtime_dir / "checkpoint"), device=forced)
        else:
            try:
                self.model = SentenceTransformer(str(runtime_dir / "checkpoint"))
            except Exception as exc:  # noqa: BLE001  fall through to CPU on GPU failure
                msg = str(exc).lower()
                if (
                    "out of memory" in msg
                    or "cuda" in msg
                    or "cudaerror" in msg
                    or "no cuda" in msg
                ):
                    print(
                        f"mind-nerve: GPU init failed ({exc.__class__.__name__}), "
                        f"falling back to CPU",
                        file=sys.stderr,
                    )
                    self.model = SentenceTransformer(str(runtime_dir / "checkpoint"), device="cpu")
                else:
                    raise
        self.model.eval()

        emb_path = runtime_dir / "route_table.npy"
        meta_path = runtime_dir / "route_table.jsonl"
        if not emb_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Precomputed catalog embeddings not found at {emb_path}. "
                f"Run mind_nerve.installer.precompute_routes() first."
            )
        self.embeddings: "np.ndarray" = np.load(emb_path)
        self.routes: list[dict] = [json.loads(ln) for ln in meta_path.open("r")]
        if self.embeddings.shape[0] != len(self.routes):
            raise RuntimeError("Route table embeddings/meta length mismatch")

        # L2-normalize once so query-time is a single matmul.
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-12
        self.embeddings = (self.embeddings / norms).astype(np.float32)

        # Catalog v2: optional per-route log-prior column. When present, it
        # is added to the dot-product score before top-k selection (Bayesian
        # combination of likelihood + frequency prior). Loaded from
        # `route_table_prior.npy` if it exists; absent file means v1 catalog
        # and the runtime falls through to the plain dot-product path.
        prior_path = runtime_dir / "route_table_prior.npy"
        if prior_path.exists():
            log_prior = np.load(prior_path).astype(np.float32)
            if log_prior.shape != (self.embeddings.shape[0],):
                raise RuntimeError(
                    f"Route prior shape mismatch: expected ({self.embeddings.shape[0]},), "
                    f"got {log_prior.shape}"
                )
            self.log_prior: "np.ndarray | None" = log_prior
        else:
            self.log_prior = None

    @property
    def catalog_size(self) -> int:
        return len(self.routes)

    @property
    def catalog_version(self) -> str:
        return str(self.manifest.get("catalog_version", "unknown"))

    @property
    def model_version(self) -> str:
        return str(self.manifest.get("phase1_version", "unknown"))


@functools.lru_cache(maxsize=4)
def _load_cached(runtime_dir_str: str) -> _Runtime:
    return _Runtime(Path(runtime_dir_str))


def load_default_runtime(runtime_dir: str | None = None) -> _Runtime:
    """Cached runtime loader — call once per process.

    Auto-downloads the Phase-1 weights from Hugging Face the first time
    it's called without an explicit ``runtime_dir`` or ``MIND_NERVE_RUNTIME_DIR``.
    """
    p = _resolve_runtime_dir(runtime_dir)
    return _load_cached(str(p))


def route(query: str, top_k: int = 5, *, runtime_dir: str | None = None) -> RouteResult:
    """Return the top-K routing candidates for a query.

    Side-effect-free. Thread-safe given the LRU-cached runtime.
    """
    import numpy as np

    rt = load_default_runtime(runtime_dir)

    t0 = time.perf_counter()
    qv = rt.model.encode(
        [query], convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True
    ).astype(np.float32)[0]
    t_encode = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    scores = rt.embeddings @ qv  # (N,)
    # Catalog v2: combine the dot-product likelihood with the per-route
    # log-prior, when present. log-space addition is equivalent to
    # P(route|query) ∝ P(query|route) · P(route).
    if rt.log_prior is not None:
        scores = scores + rt.log_prior
    k = min(top_k, scores.shape[0])
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]  # exact sort over the k
    t_rank = (time.perf_counter() - t0) * 1000.0

    out: list[Route] = []
    for i in top:
        meta = rt.routes[int(i)]
        out.append(
            Route(
                id=str(meta.get("id", "")),
                name=str(meta.get("name", "")),
                kind=str(meta.get("kind", "")),
                score=float(scores[int(i)]),
                source_repo=str(meta.get("source_repo", "")),
                url=meta.get("url"),
            )
        )

    return RouteResult(
        query=query,
        top_k=top_k,
        routes=out,
        encode_ms=t_encode,
        rank_ms=t_rank,
        catalog_size=rt.catalog_size,
        catalog_version=rt.catalog_version,
        model_version=rt.model_version,
    )


def precompute_routes(
    runtime_dir: str | None = None,
    catalog_path: str | None = None,
    cooccurrence_path: str | None = None,
    emit_prior: bool = False,
) -> dict[str, Any]:
    """Encode every catalog item and write route_table.npy + .jsonl.

    Run once after training. The result lives inside runtime_dir so the
    runtime loader can pick it up at startup.

    Catalog-v2 (SOTA-track #1): when ``emit_prior=True`` or
    ``cooccurrence_path`` is provided, also emit ``route_table_prior.npy``
    with one ``float32`` log-prior per route. The runtime adds this column
    to the dot-product score before top-k selection. With no
    co-occurrence stats the priors default to ``log(2) ≈ 0.693`` per route
    (uniform Laplace prior), making the file behaviorally a no-op until
    real frequency data is available.
    """
    import math

    import numpy as np
    from sentence_transformers import SentenceTransformer

    rdir = _resolve_runtime_dir(runtime_dir)
    if not (rdir / "checkpoint").is_dir():
        raise FileNotFoundError(f"no trained checkpoint at {rdir / 'checkpoint'}")
    if catalog_path is None:
        catalog_path = str(rdir / "items.jsonl")
        if not Path(catalog_path).exists():
            raise FileNotFoundError(
                f"no catalog_path provided and no items.jsonl found at {catalog_path}"
            )

    model = SentenceTransformer(str(rdir / "checkpoint"))
    items: list[dict] = []
    texts: list[str] = []
    with open(catalog_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("name", "")
            # Tool entries have url; for them include url in the text.
            if obj.get("kind") == "tool" and obj.get("url"):
                text = f"{text} — {obj['url']}"
            items.append(obj)
            texts.append(text)

    emb = model.encode(
        texts,
        batch_size=128,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,
    )
    emb = np.asarray(emb, dtype=np.float32)

    np.save(rdir / "route_table.npy", emb)
    with (rdir / "route_table.jsonl").open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, separators=(",", ":")) + "\n")

    result: dict[str, Any] = {
        "count": len(items),
        "dim": int(emb.shape[1]),
        "bytes_npy": (rdir / "route_table.npy").stat().st_size,
        "bytes_jsonl": (rdir / "route_table.jsonl").stat().st_size,
    }

    # Catalog-v2: optional log-prior column. Drop the file even when no
    # co-occurrence stats are provided so installers can ship a v2 runtime
    # by default; the uniform prior is behaviorally identical to v1
    # scoring until real frequency data lands.
    if emit_prior or cooccurrence_path is not None:
        counts: dict[str, int] = {}
        if cooccurrence_path is not None:
            with open(cooccurrence_path, "r", encoding="utf-8") as cf:
                for line in cf:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    rid = obj.get("route_id")
                    if rid is None:
                        continue
                    counts[rid] = counts.get(rid, 0) + int(obj.get("count", 1))
        # Laplace smoothing: freq_r = raw_count + 1, log_prior = log(1+freq_r).
        log_prior = np.empty(len(items), dtype=np.float32)
        for i, item in enumerate(items):
            raw = counts.get(item.get("name", ""), 0)
            log_prior[i] = float(math.log(1.0 + (raw + 1)))
        prior_path = rdir / "route_table_prior.npy"
        np.save(prior_path, log_prior)
        result["bytes_prior"] = prior_path.stat().st_size
        result["prior_uniform"] = cooccurrence_path is None

    return result
