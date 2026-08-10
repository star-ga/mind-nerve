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
     ``star-ga/mind-nerve`` on Hugging Face on first use)
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .types import Route, RouteResult

# ---------------------------------------------------------------------------
# Shared skill-text helpers (used by precompute_routes and discovery.scan)
# ---------------------------------------------------------------------------


def _parse_skill_frontmatter(text: str) -> dict[str, str]:
    """Parse YAML frontmatter from a skill file; returns key→value dict."""
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    out: dict[str, str] = {}
    for line in text[3:end].splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z_][\w-]*)\s*:\s*(.*)$", line)
        if m:
            key = m.group(1).strip().lower()
            val = m.group(2).strip().strip('"').strip("'")
            out[key] = val
    return out


# ---------------------------------------------------------------------------
# Ingest normalisation — bounds the retrieval-poisoning surface
# ---------------------------------------------------------------------------
#
# Descriptions became the ranking substrate when the catalog was re-embedded on
# name + description + tags instead of the kebab-case name alone. That is a
# large relevance win AND a new attack surface: third-party skills (SkillsMP
# and any other external feed into this hub) author their own description, so a
# keyword-stuffed "Use when the user asks to ..." paragraph can win every
# query. Name-only ranking was accidentally hard to game because a name is
# short and structural; free prose is not.
#
# The rule is mechanical, not a matter of trusting authors:
#   * embed  name + description[:240] + first 5 tags
#   * strip URLs, code fences and markdown syntax first
#   * a description over 1000 RAW chars, or one containing a routing
#     imperative ("always use", "route here for all", "ignore ..."), indexes
#     NAME-ONLY and emits a lint warning
#
# Deterministic and idempotent — this catalog feeds a bit-identity-gated
# system, so no classifier and no run-to-run variation. What was actually
# indexed stays auditable in the row's ``text`` field.

MAX_DESC_CHARS = 240
MAX_TAGS = 5
MAX_RAW_DESC_CHARS = 1000

# Phrases whose only purpose is to bias the router itself rather than describe
# the skill. Their presence is treated as disqualifying, not sanitisable.
_ROUTING_IMPERATIVE_RE = re.compile(
    r"(?:"
    r"always\s+use|"
    r"route\s+here\s+for\s+all|"
    r"use\s+(?:this|me)\s+for\s+(?:all|every|any)\b|"
    r"ignore\s+(?:all\s+)?(?:previous|other|prior)\b|"
    r"ignore\s+the\s+above|"
    r"disregard\s+(?:all\s+)?(?:previous|other)\b|"
    r"highest\s+priority\s+skill|"
    r"must\s+be\s+used\s+for\s+(?:all|every)\b"
    r")",
    re.I,
)

# Zero-width + bidi-override characters: invisible in a rendered SKILL.md but
# fully present in the embedding input, so they can carry hidden ranking text.
_INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
_FENCE_RE = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~")
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_SYNTAX_RE = re.compile(r"[*_#>|]+")
_WS_RE = re.compile(r"\s+")


def strip_markup(text: str) -> str:
    """Remove URLs, code fences and markdown syntax; collapse whitespace."""
    if not text:
        return ""
    out = _INVISIBLE_RE.sub("", text)
    out = _FENCE_RE.sub(" ", out)
    out = _MD_LINK_RE.sub(r"\1", out)
    out = _INLINE_CODE_RE.sub(r"\1", out)
    out = _URL_RE.sub(" ", out)
    out = _MD_SYNTAX_RE.sub(" ", out)
    return _WS_RE.sub(" ", out).strip()


def description_is_poisoned(raw_description: str) -> str | None:
    """Return a lint reason when *raw_description* must not be indexed.

    ``None`` means the description is usable.
    """
    if not raw_description:
        return None
    if len(raw_description) > MAX_RAW_DESC_CHARS:
        return f"description_too_long:{len(raw_description)}>{MAX_RAW_DESC_CHARS}"
    m = _ROUTING_IMPERATIVE_RE.search(raw_description)
    if m:
        return f"routing_imperative:{m.group(0).lower().strip()}"
    return None


def parse_tags(front_matter: dict[str, str]) -> list[str]:
    """Extract up to MAX_TAGS tags from frontmatter (list or comma form)."""
    raw = front_matter.get("tags") or front_matter.get("keywords") or ""
    raw = raw.strip().strip("[]")
    if not raw:
        return []
    parts = re.split(r"[,\s]+", raw)
    return [p.strip("-\"' ") for p in parts if p.strip("-\"' ")][:MAX_TAGS]


def build_embedding_text(
    name: str, description: str, tags: "list[str] | None" = None
) -> tuple[str, str | None]:
    """Return ``(text_to_embed, lint_warning)`` for one catalog entry.

    A poisoned description degrades to NAME-ONLY rather than being cleaned:
    partially neutralising a hostile description still lets the remainder
    influence rank, and the whole point is that the surface is bounded.
    """
    name = (name or "").strip()
    warning = description_is_poisoned(description or "")
    if warning:
        return name, warning
    desc = strip_markup(description or "")[:MAX_DESC_CHARS]
    tag_part = " ".join(f"- {t}" for t in (tags or [])[:MAX_TAGS])
    return " ".join(p for p in (name, desc, tag_part) if p), None


def _skill_embedding_text(item: dict[str, Any]) -> str:
    """Return the canonical text to encode for a catalog item.

    Priority:
      1. When ``source_path`` is set and the file is readable: the normalised
         ``name + description[:240] + first 5 tags`` (see
         ``build_embedding_text``). A description failing the ingest lint
         degrades to name-only.
      2. Tool items with a ``url``: ``name — url``.
      3. Fallback: ``name`` only.

    This function is the single source of truth for what text goes into the
    embedding — both ``precompute_routes`` (batch rebuild) and
    ``discovery.scan`` (incremental) must produce the same text for the
    same source file.
    """
    name: str = item.get("name", "")
    source_path = item.get("source_path", "")
    if source_path:
        p = Path(source_path)
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            if 32 <= len(text) <= 256 * 1024:
                fm = _parse_skill_frontmatter(text)
                desc = fm.get("description", "")
                embed_text, warning = build_embedding_text(name, desc, parse_tags(fm))
                if warning:
                    # Loud but non-fatal: the entry is still indexed, by NAME
                    # ONLY, and the reason is on the record.
                    sys.stderr.write(
                        f"[mind-nerve] ingest lint: {source_path}: {warning} "
                        f"-- indexing name-only\n"
                    )
                return embed_text
        except OSError:
            pass
    if item.get("kind") == "tool" and item.get("url"):
        return f"{name} — {item['url']}"
    return name


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


_HF_REPO_ID = "star-ga/mind-nerve"


def _default_user_runtime_dir() -> Path:
    """Resolve the default runtime dir, preferring a populated STARGA-local
    table over the HF auto-seed location.

    Two runtime dirs coexist on STARGA hosts:

    * ``~/.local/share/mind-nerve-runtime``  (dash) — the curated STARGA route
      table (first-party trust root, ``trusted_paths.json`` present); this is
      what the route daemon's systemd drop-in pins via ``MIND_NERVE_RUNTIME_DIR``.
    * ``~/.local/share/mind-nerve/runtime``  (slash) — the HF auto-seed target,
      which may default to the larger OSS catalog.

    When ``MIND_NERVE_RUNTIME_DIR`` is not exported (e.g. an interactive
    ``mind-nerve learn`` with no systemd env), resolution previously fell to the
    slash path, so ``learn`` appended to the wrong table and trust-root
    detection silently failed. Preferring the dash path when it has a real
    ``manifest.json`` makes every entry point (CLI, daemon, MCP) land on the
    curated table by default — the env/drop-in becomes a pin, not a requirement.
    """
    dash = Path.home() / ".local" / "share" / "mind-nerve-runtime"
    if (dash / "manifest.json").exists():
        return dash
    return Path.home() / ".local" / "share" / "mind-nerve" / "runtime"


_USER_RUNTIME_DIR = _default_user_runtime_dir()


def _seed_from_hf(target: Path) -> None:
    """Snapshot-download the Phase-1 weights from Hugging Face into *target*.

    Idempotent: skips files that already exist. Prints a one-line progress
    notice to stderr on first download (sub-second on cache-hot machines,
    ~150 MB cold).
    """
    from huggingface_hub import snapshot_download

    revision = os.environ.get("MIND_NERVE_HF_REVISION", "71221fd435f119cc50c92df4786352ac594efa17")
    print(
        f"mind-nerve: downloading Phase-1 weights ({_HF_REPO_ID}@{revision[:12]}, ~150 MB) to {target}",
        file=sys.stderr,
    )
    cached = Path(
        snapshot_download(
            repo_id=_HF_REPO_ID,
            repo_type="model",
            revision=revision,
            allow_patterns=[
                "manifest.json",
                "checkpoint/*",
                "route_table*.npy",
                "route_table.jsonl",
                "stride_thresholds.json",
            ],
        )
    )
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
    # ``_DEFAULT_RUNTIME_DIR`` is a lazy ``str`` proxy whose raw value is the
    # sentinel ``"<lazy:mind-nerve-runtime>"``.  When it is threaded through a
    # CLI default (e.g. ``mind-nerve learn`` with no ``--runtime-dir``) it can
    # arrive here verbatim instead of resolving; treat it as "unset" so we fall
    # back to env / HF resolution rather than ``Path("<lazy:...>")``.
    if runtime_dir is not None and runtime_dir == "<lazy:mind-nerve-runtime>":
        runtime_dir = None
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
class _DefaultRuntimeDirProxy(str):
    """str-compatible proxy that resolves to the runtime dir on str-cast."""

    def __new__(cls) -> "_DefaultRuntimeDirProxy":
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
        with meta_path.open("r") as _f:
            self.routes: list[dict[str, Any]] = [json.loads(ln) for ln in _f]
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
            self.stride_thresholds: "dict[str, Any] | None" = json.loads(stride_path.read_text())
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

        # Encoder-weights blob: the Q16.16 weight tables consumed by
        # mn_encoder_encode (NOT the catalog — that is route_table.q16.bin).
        # Produced offline by tools/quantize_encoder_to_q16.py. Resolution:
        #   1. $MIND_NERVE_ENCODER_WEIGHTS (explicit override)
        #   2. <runtime_dir>/encoder_weights.q16.bin (default)
        # When absent, the handle is initialised with a zero-length blob so
        # mn_encoder_init still allocates scratch buffers; encode then yields
        # an all-zero embedding (caller can detect via the missing blob).
        self._handle: int = 0
        env_blob = os.environ.get("MIND_NERVE_ENCODER_WEIGHTS")
        if env_blob:
            q16_blob_path = Path(env_blob).expanduser()
        else:
            q16_blob_path = runtime_dir / "encoder_weights.q16.bin"
        self._encoder_weights_path = q16_blob_path
        self._encoder_weights_loaded = q16_blob_path.exists()
        if q16_blob_path.exists():
            import ctypes as _ct

            # Pin the blob for the lifetime of the handle; the native side
            # stores the raw address and reads it on every encode call.
            self._weights = np.fromfile(str(q16_blob_path), dtype=np.int64)
            self._weights_pinned = np.ascontiguousarray(self._weights, dtype=np.int64)
            blob_addr = (
                _ct.cast(
                    self._weights_pinned.ctypes.data_as(_ct.POINTER(_ct.c_int64)),
                    _ct.c_void_p,
                ).value
                or 0
            )
            self._handle = self._native.init(blob_addr, self._weights_pinned.nbytes)
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

        with meta_path.open("r") as _f:
            self.routes: list[dict[str, Any]] = [json.loads(ln) for ln in _f]
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
            from transformers import AutoTokenizer

            # nosec B615 — loads from a LOCAL checkpoint directory, not a remote
            # Hub repo; the runtime is already revision-pinned at seed time by
            # _seed_from_hf, so a `revision=` argument is not applicable here.
            return AutoTokenizer.from_pretrained(  # nosec B615
                str(runtime_dir / "checkpoint"), use_fast=True
            )
        except ImportError:
            # transformers not installed; tokenizer unavailable.
            # route() will raise a clear error if encode is called.
            return None

    def _tokenize(self, text: str) -> np.ndarray:
        """Return int32 token IDs for *text*, truncated to the model's
        ``max_seq_length`` (256).

        #228: this MUST match the reference SentenceTransformer, which
        truncates to ``sentence_bert_config.json`` ``max_seq_length`` (256)
        then CLS-pools the single window. Previously this used
        ``max_length=512``, so any input >256 tokens reached the native
        encoder's sliding-window ("later-window-wins") path and silently
        produced a *different* embedding than pytorch — the A1.5 gate never
        caught it because its harness tokenizes at 256. Truncating at 256
        here makes native route()/encode pytorch-SentenceTransformer-
        identical for all inputs; the sliding-window kernel stays available
        for explicit long-document use but is never silently on this path.
        """
        if self._tokenizer is None:
            raise RuntimeError(
                "transformers is not installed; cannot tokenize text for the "
                "native backend. Install transformers or set "
                "MIND_NERVE_BACKEND=pytorch."
            )
        enc = self._tokenizer(
            text,
            truncation=True,
            max_length=256,  # = model max_seq_length; pytorch-ST-equivalent (#228)
            return_tensors="np",
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return np.asarray(enc["input_ids"][0], dtype=np.int32)

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
        # __init__ may raise before _handle/_native are bound (e.g. the native
        # .so is absent). getattr guards keep __del__ from masking the real
        # construction error with a spurious AttributeError.
        handle = getattr(self, "_handle", 0)
        native = getattr(self, "_native", None)
        if handle != 0 and native is not None:
            try:
                native.free(handle)
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Backend-aware cached loader
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=4)
def _load_cached(runtime_dir_str: str, backend: str) -> "_Runtime | _NativeEncoderRuntime":
    rdir = Path(runtime_dir_str)
    if backend == _BACKEND_NATIVE:
        try:
            return _NativeEncoderRuntime(rdir)
        except (FileNotFoundError, ImportError, OSError) as exc:
            # The native Q16.16 encoder shared library is absent or unloadable
            # — the common case on Windows/macOS and for a plain pip install
            # where the Linux .so is not shipped. Degrade to the pure-Python
            # (numpy + sentence-transformers) backend so route() keeps working,
            # just slower. One-line notice; never crash on the default backend.
            # (FileNotFoundError: missing .so; ImportError: ctypes/_native gap;
            #  OSError: present-but-unloadable .so, e.g. wrong arch/ABI.)
            print(
                f"mind-nerve: native encoder unavailable ({exc}); "
                f"falling back to the pure-Python backend.",
                file=sys.stderr,
            )
            return _Runtime(rdir)
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


# The actual LRU cache lives on ``_load_cached``; expose its ``cache_clear`` on
# the public wrapper so callers (e.g. ``discovery.scan`` after a route-table
# rebuild) can invalidate the in-memory runtime without an AttributeError.
load_default_runtime.cache_clear = _load_cached.cache_clear  # type: ignore[attr-defined]


def _count_bpe_tokens(query: str, rt: "_Runtime | _NativeEncoderRuntime") -> int:
    """Return the BPE token count for *query* using the runtime's tokenizer.

    For the pytorch backend we call SentenceTransformer.tokenize() directly,
    which uses the same WordPiece vocabulary as the encoder. For the native
    backend the tokenizer is already loaded as rt._tokenizer.
    """
    try:
        if isinstance(rt, _NativeEncoderRuntime):
            if rt._tokenizer is None:
                return 0
            # Native backend: uses HF AutoTokenizer.
            enc = rt._tokenizer(
                query,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            return len(enc["input_ids"])
        # Pytorch backend: SentenceTransformer.tokenize().
        tokens = rt.model.tokenize([query])
        return int(tokens["input_ids"].shape[1])
    except Exception:  # noqa: BLE001
        # If tokenization is unavailable, skip the check rather than crashing.
        return 0


def route(query: str, top_k: int = 5, *, runtime_dir: str | None = None) -> RouteResult:
    """Return the top-K routing candidates for a query.

    Side-effect-free. Thread-safe given the LRU-cached runtime.
    Dispatches to the native Q16.16 encoder path (MIND_NERVE_BACKEND=native,
    default) or the pytorch sentence-transformers path
    (MIND_NERVE_BACKEND=pytorch).

    Raises:
        ValueError: if top_k is outside [1, 64] or if the query exceeds
            1024 BPE tokens (``RequestTooLong``).
    """
    if not 1 <= top_k <= 64:
        raise ValueError(f"top_k must be in [1, 64]; got {top_k}")

    rt = load_default_runtime(runtime_dir)

    token_count = _count_bpe_tokens(query, rt)
    if token_count > 1024:
        raise ValueError(f"RequestTooLong: query exceeds 1024 tokens (got {token_count})")

    # Dispatch on the resolved runtime *instance*, not the env var: when the
    # native backend was requested but the .so was unavailable, _load_cached
    # transparently falls back to _Runtime, and the result must be ranked via
    # the pytorch path. Selecting on type keeps that fallback correct.
    if isinstance(rt, _NativeEncoderRuntime):
        return _route_native(query, top_k, rt)
    return _route_pytorch(query, top_k, rt)


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
                source_path=meta.get("source_path") or None,
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


# ---------------------------------------------------------------------------
# Deterministic top-K helpers
# ---------------------------------------------------------------------------


def _tie_key(route_id: str) -> bytes:
    """Return SHA-256 digest of route_id for stable tie-breaking.

    The spec mandates that equal-score routes are ordered ascending by
    SHA-256(route_id) so that the same input produces the same ranking on
    every architecture (x86, ARM, CUDA). This is the load-bearing contract
    for cross-arch Q16.16 bit-identity.
    """
    return hashlib.sha256(route_id.encode("utf-8")).digest()


def _deterministic_topk(
    scores: "np.ndarray",
    route_ids: list[str],
    k: int,
) -> "np.ndarray":
    """Return indices of the top-k routes with stable SHA-256 tie-breaking.

    Primary sort: descending score.
    Tie-break: ascending SHA-256(route_id) digest (bytes comparison).
    """
    cand = np.argpartition(-scores, k - 1)[:k]
    ordered = sorted(
        cand.tolist(),
        key=lambda i: (-float(scores[i]), _tie_key(route_ids[i])),
    )
    return np.asarray(ordered, dtype=np.int64)


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
    route_ids = [str(r.get("id", "")) for r in rt.routes]
    top = _deterministic_topk(scores, route_ids, k)
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
                source_path=meta.get("source_path") or None,
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
    consumed by the native-MIND windowed encoder once mind-nerve's wheel-side
    cdylib integration lands (the mindc 0.3.0 compiler-side shipped 2026-05-18).
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
    items: list[dict[str, Any]] = []
    texts: list[str] = []
    with open(catalog_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            # Use rich body text when available (description + body[:1024]), matching
            # the discovery.scan path so batch rebuilds produce equivalent embeddings.
            texts.append(_skill_embedding_text(obj))
            items.append(obj)

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
