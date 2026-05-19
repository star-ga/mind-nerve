"""mind-nerve inference — backend-selectable encoder path.

Loads the fine-tuned sentence-transformers checkpoint + the
precomputed catalog embeddings, encodes one query, returns top-K.

Backend selection (MIND_NERVE_BACKEND env var):
    native  — (default in v0.4.0) ctypes binding to libmind_nerve_encoder.so
              compiled from mind/exports/c_abi.mind. No torch dependency.
    pytorch — sentence-transformers path (Phase 1). Requires torch.

The public API in ``__init__.py`` stays unchanged regardless of backend.

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

import numpy as np

from .types import Route, RouteResult

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

_BACKEND_ENV_VAR = "MIND_NERVE_BACKEND"
_BACKEND_NATIVE = "native"
_BACKEND_PYTORCH = "pytorch"


def _active_backend() -> str:
    """Return the active backend name, lower-cased and validated."""
    raw = os.environ.get(_BACKEND_ENV_VAR, _BACKEND_NATIVE).strip().lower()
    if raw not in (_BACKEND_NATIVE, _BACKEND_PYTORCH):
        raise ValueError(
            f"Unknown MIND_NERVE_BACKEND={raw!r}. "
            f"Valid values: {_BACKEND_NATIVE!r}, {_BACKEND_PYTORCH!r}."
        )
    return raw


def _load_runtime() -> "_Runtime | _NativeEncoderRuntime":
    """Return the appropriate runtime based on MIND_NERVE_BACKEND."""
    backend = _active_backend()
    rdir = _resolve_runtime_dir()
    if backend == _BACKEND_NATIVE:
        return _NativeEncoderRuntime(rdir)
    return _Runtime(rdir)


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
    """Loaded model + precomputed catalog embeddings (pytorch backend)."""

    def __init__(self, runtime_dir: Path):
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

        # Catalog v2 (SOTA-track #4): optional per-route frequency-adaptive
        # scale column. Multiplies each L2-normalized embedding row in
        # place at load — zero runtime cost. Rare routes get higher scale,
        # common routes get lower scale (floor 0.5), addressing the long-
        # tail drown-out problem. Absent file = unchanged v1 behavior.
        freq_path = runtime_dir / "route_table_freq_scale.npy"
        if freq_path.exists():
            freq_scale = np.load(freq_path).astype(np.float32)
            if freq_scale.shape != (self.embeddings.shape[0],):
                raise RuntimeError(
                    f"Route freq_scale shape mismatch: expected ({self.embeddings.shape[0]},), "
                    f"got {freq_scale.shape}"
                )
            self.embeddings = (self.embeddings * freq_scale[:, None]).astype(np.float32)
            self.freq_scale: "np.ndarray | None" = freq_scale
        else:
            self.freq_scale = None

        # Catalog v2 (SOTA-track #3): optional entropy → stride threshold
        # table. Consumed by the native-MIND windowed encoder once mindc
        # 0.3.0 cdylib lands; in the Phase 1 sentence-transformers path
        # it's load-only metadata for forward compatibility.
        stride_path = runtime_dir / "stride_thresholds.json"
        if stride_path.exists():
            self.stride_thresholds: "dict | None" = json.loads(stride_path.read_text())
        else:
            self.stride_thresholds = None

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
def _load_cached_pytorch(runtime_dir_str: str) -> "_Runtime":
    return _Runtime(Path(runtime_dir_str))


# ---------------------------------------------------------------------------
# Native encoder runtime (MIND_NERVE_BACKEND=native)
# ---------------------------------------------------------------------------


class _NativeEncoderRuntime:
    """Native Q16.16 encoder runtime backed by libmind_nerve_encoder.so.

    Provides the same catalog/metadata surface as _Runtime but routes
    encode calls through the ctypes binding in _native.py instead of
    sentence-transformers.

    The WordPiece tokenizer is still Python-side; token_ids are produced
    by the same HuggingFace tokenizer used in the pytorch path.
    """

    def __init__(self, runtime_dir: Path) -> None:
        self.dir = runtime_dir
        self.manifest = json.loads((runtime_dir / "manifest.json").read_text())

        # Load the native encoder binding. If the .so is not present the
        # import will raise FileNotFoundError with a build instruction.
        from ._native import _f32_to_q16, _NativeRuntime, _q16_to_f32

        self._native = _NativeRuntime()
        self._f32_to_q16 = _f32_to_q16
        self._q16_to_f32 = _q16_to_f32

        # Load HF tokenizer for WordPiece tokenization (stays Python-side).
        self._tokenizer = self._load_tokenizer(runtime_dir)

        # Weight blob: loaded from route_table.q16.bin when present.
        # For A1.3 we initialise the handle with a placeholder zero-length
        # blob so that mn_encoder_init allocates scratch buffers; the weight
        # blob is null until the offline quantizer ships in Phase 6.2.
        self._handle: int = 0
        q16_blob_path = runtime_dir / "route_table.q16.bin"
        if q16_blob_path.exists():
            self._weights = np.fromfile(str(q16_blob_path), dtype=np.int64)
            self._weights_pinned = np.ascontiguousarray(self._weights)
            blob_addr = int(
                self._weights_pinned.ctypes.data_as(
                    __import__("ctypes").POINTER(__import__("ctypes").c_int64)
                ).__int__()
            )
            self._handle = self._native.init(blob_addr, self._weights.nbytes)
        else:
            # Placeholder handle: no valid weights yet.
            self._handle = self._native.init(0, 0)

        # Catalog embeddings (float32 from .npy, Q16.16 quantised at load).
        emb_path = runtime_dir / "route_table.npy"
        meta_path = runtime_dir / "route_table.jsonl"
        if not emb_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Precomputed catalog not found at {emb_path}. "
                f"Run mind_nerve.installer.precompute_routes() first."
            )
        embeddings_f32 = np.load(emb_path).astype(np.float32)
        norms = np.linalg.norm(embeddings_f32, axis=1, keepdims=True) + 1e-12
        embeddings_f32 = (embeddings_f32 / norms).astype(np.float32)

        # Freq-adaptive scale (catalog v2).
        freq_path = runtime_dir / "route_table_freq_scale.npy"
        if freq_path.exists():
            freq_scale = np.load(freq_path).astype(np.float32)
            embeddings_f32 = (embeddings_f32 * freq_scale[:, None]).astype(np.float32)

        # Store as Q16.16 int64 for native scoring path.
        self._catalog_q16: np.ndarray = np.ascontiguousarray(self._f32_to_q16(embeddings_f32))

        self.routes: list[dict] = [json.loads(ln) for ln in meta_path.open("r")]
        if self._catalog_q16.shape[0] != len(self.routes):
            raise RuntimeError("Native catalog embeddings/meta length mismatch")

        # Log-prior (catalog v2, optional).
        prior_path = runtime_dir / "route_table_prior.npy"
        if prior_path.exists():
            lp = np.load(prior_path).astype(np.float32)
            self._log_prior_q16: np.ndarray | None = np.ascontiguousarray(self._f32_to_q16(lp))
        else:
            self._log_prior_q16 = None

    def _load_tokenizer(self, runtime_dir: Path) -> Any:
        """Load the HuggingFace fast tokenizer from the checkpoint directory."""
        try:
            from transformers import AutoTokenizer  # type: ignore[import]

            return AutoTokenizer.from_pretrained(str(runtime_dir / "checkpoint"), use_fast=True)
        except ImportError:
            # transformers not installed; tokenizer unavailable.
            # route() will raise a clear error if encode is called.
            return None

    def _tokenize(self, text: str) -> np.ndarray:
        """Return int32 token IDs for *text* (max 512 tokens)."""
        if self._tokenizer is None:
            raise RuntimeError(
                "transformers is not installed; cannot tokenize text for the "
                "native backend. Install transformers or set "
                "MIND_NERVE_BACKEND=pytorch."
            )
        enc = self._tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors="np",
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return enc["input_ids"][0].astype(np.int32)

    def encode_query(self, text: str) -> np.ndarray:
        """Tokenize and encode a query string; returns Q16.16 int64 vector (384,)."""
        token_ids = self._tokenize(text)
        return self._native.encode(self._handle, token_ids)

    @property
    def catalog_size(self) -> int:
        return len(self.routes)

    @property
    def catalog_version(self) -> str:
        return str(self.manifest.get("catalog_version", "unknown"))

    @property
    def model_version(self) -> str:
        return str(self.manifest.get("phase1_version", "native"))

    def __del__(self) -> None:
        if self._handle != 0:
            try:
                self._native.free(self._handle)
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Backend-aware cached loader
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=4)
def _load_cached(runtime_dir_str: str, backend: str) -> "_Runtime | _NativeEncoderRuntime":
    rdir = Path(runtime_dir_str)
    if backend == _BACKEND_NATIVE:
        return _NativeEncoderRuntime(rdir)
    return _Runtime(rdir)


def load_default_runtime(
    runtime_dir: str | None = None,
) -> "_Runtime | _NativeEncoderRuntime":
    """Cached runtime loader — call once per process.

    Auto-downloads the Phase-1 weights from Hugging Face the first time
    it's called without an explicit ``runtime_dir`` or ``MIND_NERVE_RUNTIME_DIR``.
    The backend is selected by ``MIND_NERVE_BACKEND`` (default: ``native``).
    """
    p = _resolve_runtime_dir(runtime_dir)
    return _load_cached(str(p), _active_backend())


def route(query: str, top_k: int = 5, *, runtime_dir: str | None = None) -> RouteResult:
    """Return the top-K routing candidates for a query.

    Side-effect-free. Thread-safe given the LRU-cached runtime.
    Dispatches to the native Q16.16 encoder path (MIND_NERVE_BACKEND=native,
    default) or the pytorch sentence-transformers path
    (MIND_NERVE_BACKEND=pytorch).
    """
    rt = load_default_runtime(runtime_dir)
    backend = _active_backend()

    if backend == _BACKEND_NATIVE:
        return _route_native(query, top_k, rt)  # type: ignore[arg-type]
    return _route_pytorch(query, top_k, rt)  # type: ignore[arg-type]


def _route_native(
    query: str,
    top_k: int,
    rt: "_NativeEncoderRuntime",
) -> RouteResult:
    """route() implementation for MIND_NERVE_BACKEND=native."""
    t0 = time.perf_counter()
    qv_q16 = rt.encode_query(query)  # int64 ndarray (384,) in Q16.16
    t_encode = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    scores_q16 = rt._native.score(rt._handle, qv_q16, rt._catalog_q16)

    # Catalog v2: add log-prior in Q16.16 (integer add, same as float add
    # after both are in Q16.16 space).
    if rt._log_prior_q16 is not None:
        scores_q16 = scores_q16 + rt._log_prior_q16

    k = min(top_k, scores_q16.shape[0])
    indices_q16, top_scores_q16 = rt._native.topk(scores_q16, k)
    t_rank = (time.perf_counter() - t0) * 1000.0

    out: list[Route] = []
    for pos in range(k):
        i = int(indices_q16[pos])
        meta = rt.routes[i]
        out.append(
            Route(
                id=str(meta.get("id", "")),
                name=str(meta.get("name", "")),
                kind=str(meta.get("kind", "")),
                score=float(top_scores_q16[pos]) / 65536.0,
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


def _route_pytorch(
    query: str,
    top_k: int,
    rt: "_Runtime",
) -> RouteResult:
    """route() implementation for MIND_NERVE_BACKEND=pytorch."""
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
    emit_freq_scale: bool = False,
    emit_stride_thresholds: bool = False,
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

    Catalog-v2 (SOTA-track #4): when ``emit_freq_scale=True`` or a
    ``cooccurrence_path`` is provided, also emit
    ``route_table_freq_scale.npy`` with one ``float32`` scalar per route
    equal to ``max(1/sqrt(freq), 0.5)`` (Laplace-smoothed). The runtime
    multiplies each embedding row by this scale at load time. With no
    co-occurrence stats every scale defaults to ``1.0`` (raw_count=0 →
    freq=1 → 1/sqrt(1)=1), which is behaviorally identical to v1.

    Catalog-v2 (SOTA-track #3): when ``emit_stride_thresholds=True`` also
    emit ``stride_thresholds.json`` with a calibrated entropy → stride map
    consumed by the native-MIND windowed encoder once mindc 0.3.0 lands.
    The Phase-1 sentence-transformers path ignores this file; emit is
    forward-compatible bookkeeping.
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

    # Catalog-v2: load co-occurrence counts once; reused by prior +
    # freq_scale emit paths. Empty dict when no log provided.
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

    # Catalog-v2: optional log-prior column. Drop the file even when no
    # co-occurrence stats are provided so installers can ship a v2 runtime
    # by default; the uniform prior is behaviorally identical to v1
    # scoring until real frequency data lands.
    if emit_prior or cooccurrence_path is not None:
        # Laplace smoothing: freq_r = raw_count + 1, log_prior = log(1+freq_r).
        log_prior = np.empty(len(items), dtype=np.float32)
        for i, item in enumerate(items):
            raw = counts.get(item.get("name", ""), 0)
            log_prior[i] = float(math.log(1.0 + (raw + 1)))
        prior_path = rdir / "route_table_prior.npy"
        np.save(prior_path, log_prior)
        result["bytes_prior"] = prior_path.stat().st_size
        result["prior_uniform"] = cooccurrence_path is None

    # Catalog-v2 (SOTA-track #4): per-route freq-adaptive scale column.
    # scale = max(1/sqrt(freq), 0.5) with freq = raw_count + 1 (Laplace).
    # Floor at 0.5 caps the de-emphasis of very common routes.
    if emit_freq_scale or cooccurrence_path is not None:
        freq_scale = np.empty(len(items), dtype=np.float32)
        for i, item in enumerate(items):
            raw = counts.get(item.get("name", ""), 0)
            freq = raw + 1
            freq_scale[i] = float(max(1.0 / math.sqrt(freq), 0.5))
        freq_path = rdir / "route_table_freq_scale.npy"
        np.save(freq_path, freq_scale)
        result["bytes_freq_scale"] = freq_path.stat().st_size
        result["freq_scale_uniform"] = cooccurrence_path is None

    # Catalog-v2 (SOTA-track #3): entropy → stride threshold table.
    # Defaults chosen so widest stride covers the common low-entropy
    # CLI commands; tightest stride reserved for multi-clause queries.
    if emit_stride_thresholds:
        stride_table = {
            "schema_version": 1,
            "feature": "token_entropy_first16",
            "breakpoints": [
                {"max_entropy": 0.4, "stride": 256},
                {"max_entropy": 0.7, "stride": 192},
                {"max_entropy": None, "stride": 96},
            ],
            "default_stride": 192,
            "calibration": "default-uncalibrated",
        }
        stride_path = rdir / "stride_thresholds.json"
        stride_path.write_text(json.dumps(stride_table, indent=2))
        result["bytes_stride_thresholds"] = stride_path.stat().st_size

    return result
