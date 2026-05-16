"""mind-nerve inference — PyTorch Phase 1 path.

Loads the fine-tuned sentence-transformers checkpoint + the
precomputed catalog embeddings, encodes one query, returns top-K.

The Phase 1 inference path is intentionally simple. Phase 2 swaps the
PyTorch encoder for the native MIND inference binary; the public
API in `__init__.py` stays unchanged.
"""

from __future__ import annotations

import functools
import json
import os
import time
from pathlib import Path
from typing import Any

from .types import Route, RouteResult

_DEFAULT_RUNTIME_DIR = os.environ.get(
    "MIND_NERVE_RUNTIME_DIR",
    "/data/datasets/mind-nerve-catalog/phase1/v1.1-oss",
)


class _Runtime:
    """Loaded model + precomputed catalog embeddings."""

    def __init__(self, runtime_dir: Path):
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self.dir = runtime_dir
        self.manifest = json.loads((runtime_dir / "manifest.json").read_text())
        self.model = SentenceTransformer(str(runtime_dir / "checkpoint"))
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
def load_default_runtime(runtime_dir: str = _DEFAULT_RUNTIME_DIR) -> _Runtime:
    """Cached runtime loader — call once per process."""
    p = Path(runtime_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"runtime dir {p} does not exist")
    return _Runtime(p)


def route(query: str, top_k: int = 5, *, runtime_dir: str | None = None) -> RouteResult:
    """Return the top-K routing candidates for a query.

    Side-effect-free. Thread-safe given the LRU-cached runtime.
    """
    import numpy as np

    rt = load_default_runtime(runtime_dir or _DEFAULT_RUNTIME_DIR)

    t0 = time.perf_counter()
    qv = rt.model.encode([query], convert_to_numpy=True, show_progress_bar=False,
                         normalize_embeddings=True).astype(np.float32)[0]
    t_encode = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    scores = rt.embeddings @ qv                           # (N,)
    k = min(top_k, scores.shape[0])
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]                   # exact sort over the k
    t_rank = (time.perf_counter() - t0) * 1000.0

    out: list[Route] = []
    for i in top:
        meta = rt.routes[int(i)]
        out.append(Route(
            id=str(meta.get("id", "")),
            name=str(meta.get("name", "")),
            kind=str(meta.get("kind", "")),
            score=float(scores[int(i)]),
            source_repo=str(meta.get("source_repo", "")),
            url=meta.get("url"),
        ))

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


def precompute_routes(runtime_dir: str = _DEFAULT_RUNTIME_DIR,
                      catalog_path: str = "/data/datasets/mind-nerve-catalog/freeze/v1.0/items.jsonl",
                      ) -> dict[str, Any]:
    """Encode every catalog item and write route_table.npy + .jsonl.

    Run once after training. The result lives inside runtime_dir so the
    runtime loader can pick it up at startup.
    """
    import numpy as np
    from sentence_transformers import SentenceTransformer

    rdir = Path(runtime_dir)
    if not (rdir / "checkpoint").is_dir():
        raise FileNotFoundError(f"no trained checkpoint at {rdir/'checkpoint'}")

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

    emb = model.encode(texts, batch_size=128, convert_to_numpy=True,
                       show_progress_bar=True, normalize_embeddings=False)
    emb = np.asarray(emb, dtype=np.float32)

    np.save(rdir / "route_table.npy", emb)
    with (rdir / "route_table.jsonl").open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, separators=(",", ":")) + "\n")

    return {
        "count": len(items),
        "dim": int(emb.shape[1]),
        "bytes_npy": (rdir / "route_table.npy").stat().st_size,
        "bytes_jsonl": (rdir / "route_table.jsonl").stat().st_size,
    }
