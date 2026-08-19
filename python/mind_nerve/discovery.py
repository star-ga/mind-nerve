"""mind-nerve auto-discovery — pick up new skills as users add them.

Encoder is frozen at training time; new entries are *embedded* into the
existing route table on the fly. No retraining required for routing
quality on the new entries (BGE-small was pretrained on broad English
and generalises to unseen skill descriptions well enough for top-K
retrieval; periodic full retrains happen on a separate cadence).

Three discovery modes:

1. **scan**: one-shot walk of a directory, extract new SKILL.md /
   skills/*.md / agents/*.md / commands/*.md / .cursorrules entries,
   embed + append.
2. **watch**: inotify-style daemon that monitors directories and
   triggers `scan` on every file change.
3. **add**: programmatic one-call API — `add_route({...})` for hosts
   that already parse their own skill catalog.

License gate: by default the discovery layer refuses to ingest any
skill it can't classify as OSS-compatible
(license: MIT / Apache-2.0 / BSD / CC0 / ISC / etc.). Override with
`--include-unknown` (CLI) or `include_unknown=True` (Python).

First-party trust: the license gate exists to vet *external* content;
trust in the operator's own skills is an origin question, not a license
question. Directories listed in `$MIND_NERVE_TRUSTED_PATHS`
(os.pathsep-separated) or in `<runtime_dir>/trusted_paths.json` (a JSON
list of paths) are trust roots: anything scanned under them classifies
as `first_party_ok` — never license-scanned, never refused. A scan can
also be forced trusted with `--trusted` (CLI) / `trusted=True` (Python).

The route table is updated atomically: write to `route_table.npy.tmp`
+ `route_table.jsonl.tmp` then `os.replace()` so a concurrent reader
never sees a partial state.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from .inference import (
    _DEFAULT_RUNTIME_DIR,
    _NativeEncoderRuntime,
    _skill_embedding_text,
    load_default_runtime,
)
from .security_scan import _SKIP_DIRS as _SCANNER_SKIP_DIRS


def _embed_texts(rt: Any, texts: list[str]) -> "Any":
    """Embed *texts* into a float32 ``(N, 384)`` matrix for the route table.

    The on-disk route table (``route_table.npy``) is float32 produced by the
    reference sentence-transformers model. New rows MUST be embedded by that
    same reference model so they are directly comparable: the native Q16.16
    encoder is a *quantized scoring approximation* (built for fast routing of an
    already-built table), not a table-builder.

    The previous code called ``rt.model.encode(...)`` unconditionally. Under the
    default ``MIND_NERVE_BACKEND=native`` the runtime is a
    ``_NativeEncoderRuntime`` with no ``.model`` attribute, so ``learn`` crashed
    with ``AttributeError: '_NativeEncoderRuntime' object has no attribute
    'model'``. Two further hazards ruled out routing the native encoder into the
    table builder when no ``encoder_weights.q16.bin`` blob is present:

      * ``encode_query`` then produces all-zero (unrankable) embeddings — a
        silent corruption of the table; and
      * calling ``encode_query`` against the zero-length-blob handle SEGFAULTs
        the process.

    So when routing on the native backend, we obtain the reference ``_Runtime``
    explicitly just for embedding. The route daemon keeps scoring with the
    native Q16.16 path; only table *construction* uses pytorch.

    deferred: a native table-builder (consistent Q16.16 round-trip + a shipped
    ``encoder_weights.q16.bin``) would let ``learn`` run fully on the native
    backend — stubbed here because the weights blob may be absent and the bare
    handle segfaults. Upgrade path: ship/point ``MIND_NERVE_ENCODER_WEIGHTS`` at
    a real blob, prove encode_query round-trips f32 within tolerance, then
    dispatch on isinstance like inference._route_native does.
    """
    import numpy as np

    model_rt = rt
    if isinstance(rt, _NativeEncoderRuntime) or not hasattr(rt, "model"):
        # Build the table with the reference pytorch model regardless of the
        # routing backend, so we never re-enter the native path here.
        from .inference import _resolve_runtime_dir, _Runtime

        model_rt = _Runtime(_resolve_runtime_dir(None))

    return model_rt.model.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    ).astype(np.float32)


PUBLIC_LICENSES = {
    "apache-2.0",
    "apache 2.0",
    "apache2",
    "mit",
    "bsd-3-clause",
    "bsd-2-clause",
    "isc",
    "cc0",
    "cc0-1.0",
    "unlicense",
    "cc-by-4.0",
}

COMMERCIAL_MARKERS = re.compile(
    r"\b(starga[\s-]*commercial|proprietary|confidential|"
    r"all[\s-]*rights[\s-]*reserved|do[\s-]*not[\s-]*distribute|"
    r"closed[\s-]*source)\b",
    re.IGNORECASE,
)


def _trusted_roots(runtime_dir: str | Path | None = None) -> list[Path]:
    """First-party trust roots: $MIND_NERVE_TRUSTED_PATHS plus
    <runtime_dir>/trusted_paths.json (a JSON list of directories)."""
    roots: list[Path] = []
    for part in os.environ.get("MIND_NERVE_TRUSTED_PATHS", "").split(os.pathsep):
        part = part.strip()
        if part:
            roots.append(Path(part).expanduser())
    # Resolve the runtime dir through the canonical resolver so the lazy
    # ``_DEFAULT_RUNTIME_DIR`` proxy ("<lazy:mind-nerve-runtime>") — which the
    # CLI threads in when ``--runtime-dir`` is omitted — is unwrapped to the
    # real path BEFORE we look for ``trusted_paths.json``. Without this,
    # ``Path("<lazy:...>")/trusted_paths.json`` never exists, no trust roots are
    # found, and a scan of the first-party hub is wrongly license-gated.
    try:
        from .inference import _resolve_runtime_dir

        resolved_rdir: Path | None = _resolve_runtime_dir(
            str(runtime_dir) if runtime_dir is not None else None
        )
    except Exception:  # noqa: BLE001 — never let trust resolution crash a scan
        resolved_rdir = Path(str(runtime_dir)) if runtime_dir else None
    if resolved_rdir is not None:
        cfg = resolved_rdir / "trusted_paths.json"
        try:
            if cfg.is_file():
                entries = json.loads(cfg.read_text(encoding="utf-8"))
                if isinstance(entries, list):
                    roots.extend(Path(str(e)).expanduser() for e in entries)
        except (OSError, ValueError):
            pass
    resolved = []
    for r in roots:
        try:
            resolved.append(r.resolve())
        except OSError:
            continue
    return resolved


def _is_trusted_dir(directory: str | Path, runtime_dir: str | Path | None = None) -> bool:
    try:
        d = Path(directory).expanduser().resolve()
    except OSError:
        return False
    return any(d == root or root in d.parents for root in _trusted_roots(runtime_dir))


SKILL_PATTERNS = [
    (re.compile(r"(^|/)SKILL\.md$", re.I), "skill"),
    (re.compile(r"(^|/)skills?/[^/]+\.md$", re.I), "skill"),
    (re.compile(r"(\.claude|^claude)/(skills|commands|agents)/.*\.md$", re.I), "skill"),
    (re.compile(r"(^|/)commands?/[^/]+\.md$", re.I), "command"),
    (re.compile(r"(^|/)agents?/[^/]+\.md$", re.I), "agent"),
    (re.compile(r"(^|/)subagents?/[^/]+\.md$", re.I), "agent"),
    (re.compile(r"\.cursorrules$", re.I), "rule"),
    (re.compile(r"\.mdc$", re.I), "rule"),
]
# `.system` holds the hub's OWN internals, not routable skills. It is excluded
# from the INDEX (not merely from projection) because four of its entries
# shadow real hub skills by name -- plugin-creator, openai-docs, skill-creator
# and imagegen each exist as BOTH `.system/<name>/SKILL.md` and
# `<name>/SKILL.md`. Projection symlinks are keyed on name, so whichever row
# the router returned would silently overwrite the other. Excluding at ingest
# means the ambiguity cannot reach any consumer.
# Canonical source: security_scan._SKIP_DIRS (single set; fable#5 drift).
SKIP_DIRS = _SCANNER_SKIP_DIRS


# ---------------------------------------------------------------------------
# Frontmatter + license classification
# ---------------------------------------------------------------------------


def _parse_frontmatter(text: str) -> dict[str, str]:
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
            key, val = m.group(1).strip().lower(), m.group(2).strip().strip('"').strip("'")
            out[key] = val
    return out


def _classify(text: str, fm: dict[str, str], *, trusted: bool = False) -> tuple[str, list[str]]:
    """Return (bucket, reasons)."""
    # First-party content from a trust root is exempt from the license gate
    # entirely: the gate vets external content, and a trusted origin is the
    # operator's own. Marker words in the body ("proprietary", "internal")
    # are subject matter here, not a license declaration.
    if trusted:
        return "first_party_ok", ["trusted source path"]

    lic = (fm.get("license") or "").lower().strip()

    # An explicit private/internal visibility flag always wins.
    if fm.get("visibility", "").lower() in {"private", "confidential", "internal"}:
        return "commercial_risk", [f"visibility={fm['visibility']}"]
    # A declared public license in the frontmatter is authoritative and must
    # take precedence over body-text markers: security/forensics skills legitimately
    # *discuss* words like "proprietary"/"confidential"/"do not distribute" as subject
    # matter, which would otherwise be mis-bucketed as commercial_risk.
    if lic in PUBLIC_LICENSES:
        return "public_ok", [f"license={lic}"]
    if lic and "commercial" in lic:
        return "commercial_risk", [f"license={lic}"]
    # No authoritative license: fall back to scanning the body for commercial markers
    # (catches skills that declare a commercial/internal license only in prose).
    if COMMERCIAL_MARKERS.search(text):
        return "commercial_risk", ["matched commercial marker"]

    return "unknown", ["no license declared"]


def _detect_kind(rel: str) -> str | None:
    for pat, kind in SKILL_PATTERNS:
        if pat.search(rel):
            return kind
    return None


# ---------------------------------------------------------------------------
# Route table I/O
# ---------------------------------------------------------------------------


def _load_table(runtime_dir: Path) -> tuple[Any, list[dict[str, Any]]]:
    import numpy as np

    emb = np.load(runtime_dir / "route_table.npy")
    with (runtime_dir / "route_table.jsonl").open(encoding="utf-8") as _f:
        meta = [json.loads(line) for line in _f]
    return emb, meta


def _load_table_or_empty(runtime_dir: Path) -> tuple[Any, list[dict[str, Any]]]:
    """Load the route table, or return ``(None, [])`` when NEITHER half exists.

    A fresh runtime dir (own/seeded checkpoint but no precomputed table yet) has
    no ``route_table.npy``/``.jsonl``. ``scan``/``learn`` then BOOTSTRAPS the
    table directly from the scanned SKILL.md files rather than crashing on the
    missing file — the chicken-and-egg fix: ``precompute-routes`` needs an
    ``items.jsonl`` a fresh install lacks, but ``scan`` reads skills straight
    off disk and can seed the very first table.

    When EXACTLY ONE of the pair is present, this is NOT a fresh bootstrap —
    it is a partial/corrupted table (interrupted write, incomplete restore, a
    hand-deleted file). Treating that the same as "neither present" would let
    the caller's next atomic write silently overwrite the surviving file with
    an otherwise-empty table, permanently discarding a possibly large existing
    route table with no error. Refuse instead, after warning which file is
    missing.
    """
    emb_path = runtime_dir / "route_table.npy"
    meta_path = runtime_dir / "route_table.jsonl"
    emb_exists = emb_path.is_file()
    meta_exists = meta_path.is_file()
    if emb_exists and meta_exists:
        return _load_table(runtime_dir)
    if not emb_exists and not meta_exists:
        return None, []
    present = emb_path.name if emb_exists else meta_path.name
    missing = meta_path.name if emb_exists else emb_path.name
    print(
        f"mind-nerve: WARNING — partial route table in {runtime_dir}: "
        f"{present} is present but {missing} is missing (interrupted write, "
        "partial restore, or manual deletion). Refusing to treat this as an "
        f"empty table -- silently continuing would discard the existing "
        f"{present} data on the next write.",
        file=sys.stderr,
    )
    raise RuntimeError(
        f"partial route table in {runtime_dir}: {present} present, {missing} missing "
        "-- restore the missing file, or remove both to bootstrap fresh, before scanning"
    )


def _save_table_atomic(runtime_dir: Path, emb: Any, meta: list[dict[str, Any]]) -> None:
    import numpy as np

    # Invariant: the embeddings matrix and the metadata list are row-aligned.
    # Refuse to WRITE a drifted table — a mismatch here only surfaces later as a
    # load-time "Route table embeddings/meta length mismatch" RuntimeError, far
    # from the code that caused it. Callers repair an EXISTING drift via
    # reconcile_table().
    n_emb = int(np.asarray(emb).shape[0]) if emb is not None else 0
    if n_emb != len(meta):
        raise ValueError(
            f"refusing to write a drifted route table: {n_emb} embeddings vs "
            f"{len(meta)} meta rows (npy and jsonl must be row-aligned)"
        )
    tmp_npy = runtime_dir / "route_table.tmp.npy"
    tmp_jsonl = runtime_dir / "route_table.jsonl.tmp"
    with tmp_npy.open("wb") as f:
        np.save(f, emb)
    with tmp_jsonl.open("w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, separators=(",", ":")) + "\n")
    os.replace(tmp_npy, runtime_dir / "route_table.npy")
    os.replace(tmp_jsonl, runtime_dir / "route_table.jsonl")


def reconcile_table(runtime_dir: str | Path = _DEFAULT_RUNTIME_DIR) -> dict[str, Any]:
    """Realign ``route_table.npy`` and ``route_table.jsonl`` to identical length.

    The two files are written together, but a crash between the two
    ``os.replace`` calls — or a hand-edit — can leave them different lengths,
    after which the runtime raises ``Route table embeddings/meta length
    mismatch`` on every load and routing is dead. This truncates BOTH to their
    common prefix (the rows present in both, on identical indices ``[0, n)``),
    rewrites them atomically, and reports how many rows were dropped. A no-op
    when they already match or when the table is absent. Deterministic.
    """
    import numpy as np

    from .inference import _resolve_runtime_dir

    rdir = _resolve_runtime_dir(str(runtime_dir) if runtime_dir is not None else None)
    emb_path = rdir / "route_table.npy"
    meta_path = rdir / "route_table.jsonl"
    if not (emb_path.is_file() and meta_path.is_file()):
        return {"reconciled": False, "reason": "no route table", "runtime_dir": str(rdir)}
    emb, meta = _load_table(rdir)
    n_emb = int(np.asarray(emb).shape[0])
    n_meta = len(meta)
    if n_emb == n_meta:
        return {"reconciled": False, "aligned": n_emb, "runtime_dir": str(rdir)}
    n = min(n_emb, n_meta)
    print(
        f"mind-nerve: WARNING — route table drift in {rdir}: npy={n_emb} rows, "
        f"jsonl={n_meta} rows; truncating both to {n} "
        f"(dropping {n_emb - n} embeddings, {n_meta - n} meta rows)",
        file=sys.stderr,
    )
    _save_table_atomic(rdir, emb[:n], [meta[i] for i in range(n)])
    load_default_runtime.cache_clear()  # type: ignore[attr-defined]
    return {
        "reconciled": True,
        "aligned": n,
        "dropped_embeddings": n_emb - n,
        "dropped_meta": n_meta - n,
        "runtime_dir": str(rdir),
    }


# ---------------------------------------------------------------------------
# Item extraction from a single file
# ---------------------------------------------------------------------------


def _first_h1(body: str) -> str | None:
    for line in body.splitlines()[:50]:
        s = line.strip()
        if s.startswith("# "):
            return s[2:].strip()
    return None


def _item_from_file(
    path: Path, source_repo: str, kind: str, trusted: bool = False
) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if len(text) < 32 or len(text) > 256 * 1024:
        return None
    fm = _parse_frontmatter(text)
    bucket, reasons = _classify(text, fm, trusted=trusted)
    name = fm.get("name") or _first_h1(text) or path.stem
    sha = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
    item: dict[str, Any] = {
        "id": sha[:16],
        "source_repo": source_repo,
        "source_path": str(path),
        "kind": kind,
        "name": name,
        "size_bytes": len(text),
        "sha256": sha,
        "tokens_est": max(1, len(text) // 4),
        "_license_bucket": bucket,
        "_license_reasons": reasons,
    }
    # Use the shared helper so discovery and precompute_routes produce identical text.
    item["_embedded_text"] = _skill_embedding_text(item)
    return item


def _walk_dir(root: Path, source_repo: str, trusted: bool = False) -> Iterable[dict[str, Any]]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        parts = set(p.relative_to(root).parts)
        if parts & SKIP_DIRS:
            continue
        # SKILL_PATTERNS require literal '/' separators; on Windows
        # str(p.relative_to(root)) is backslash-separated, which would make
        # _detect_kind silently return None for every nested file (skills/,
        # commands/, agents/ entries never match). as_posix() normalizes to
        # forward slashes on every platform.
        rel = p.relative_to(root).as_posix()
        kind = _detect_kind(rel)
        if not kind and p.name == "SKILL.md":
            kind = "skill"
        if not kind:
            continue
        item = _item_from_file(p, source_repo, kind, trusted=trusted)
        if item is not None:
            yield item


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan(
    directory: str | Path,
    source_repo: str = "local",
    include_unknown: bool = False,
    runtime_dir: str = _DEFAULT_RUNTIME_DIR,
    dry_run: bool = False,
    trusted: bool | None = None,
) -> dict[str, Any]:
    """One-shot scan: discover new skills under `directory`, embed, persist.

    `trusted=None` (default) auto-detects: the scan is trusted when
    `directory` falls under a configured trust root (see module docstring).
    Pass `trusted=True`/`False` to force.

    Returns a summary dict.
    """
    import numpy as np

    # Resolve through the canonical resolver so the lazy _DEFAULT_RUNTIME_DIR
    # proxy ("<lazy:mind-nerve-runtime>") is unwrapped before table I/O —
    # route_table.npy lives under the REAL dir, not the literal proxy string.
    from .inference import _resolve_runtime_dir

    rdir = _resolve_runtime_dir(str(runtime_dir) if runtime_dir is not None else None)

    # Bootstrap-friendly load: a fresh runtime dir has no route_table.npy yet, so
    # load_default_runtime() would raise FileNotFoundError before we ever reach
    # table I/O. Only load the full runtime when the table exists; otherwise
    # start from an empty index and let _embed_texts(None, ...) pull the
    # embedding model from the default resolver (chicken-and-egg fix).
    table_exists = (rdir / "route_table.npy").is_file() and (rdir / "route_table.jsonl").is_file()
    rt: Any = load_default_runtime(runtime_dir) if table_exists else None
    seen_ids = {m["sha256"] for m in rt.routes} if rt is not None else set()
    if trusted is None:
        trusted = _is_trusted_dir(directory, rdir)

    new_items: list[dict[str, Any]] = []
    skipped: dict[str, int] = {"already_indexed": 0, "license_excluded": 0, "unknown_excluded": 0}

    for item in _walk_dir(Path(directory), source_repo, trusted=trusted):
        if item["sha256"] in seen_ids:
            skipped["already_indexed"] += 1
            continue
        bucket = item["_license_bucket"]
        if bucket == "commercial_risk":
            skipped["license_excluded"] += 1
            continue
        if bucket == "unknown" and not include_unknown:
            skipped["unknown_excluded"] += 1
            continue
        new_items.append(item)

    if not new_items:
        return {
            "added": 0,
            "skipped": skipped,
            "total_routes_after": len(rt.routes) if rt is not None else 0,
            "trusted": trusted,
        }

    if dry_run:
        return {
            "would_add": len(new_items),
            "names_preview": [i["name"] for i in new_items[:10]],
            "skipped": skipped,
            "dry_run": True,
        }

    # Embed the new items using the same model the route table was built with.
    # (_embed_texts tolerates rt=None — it resolves the reference model itself.)
    texts = [i["_embedded_text"] for i in new_items]
    new_emb = _embed_texts(rt, texts)

    # Load raw table (rt.embeddings is the *normalised* in-memory copy). On a
    # fresh runtime dir there is no table yet → bootstrap from empty.
    raw_emb, meta = _load_table_or_empty(rdir)
    # Repair any pre-existing npy/jsonl drift BEFORE appending, so a mismatch
    # already on disk is healed here rather than surfacing later as a load-time
    # RuntimeError — and so _save_table_atomic's row-alignment assert can never
    # trip on inherited drift.
    if raw_emb is not None and int(raw_emb.shape[0]) != len(meta):
        n = min(int(raw_emb.shape[0]), len(meta))
        print(
            f"mind-nerve: WARNING — healing route table drift before append "
            f"(npy={raw_emb.shape[0]}, jsonl={len(meta)} → {n})",
            file=sys.stderr,
        )
        raw_emb, meta = raw_emb[:n], [meta[i] for i in range(n)]
    new_meta_rows = [{k: v for k, v in i.items() if not k.startswith("_")} for i in new_items]
    if raw_emb is None:
        combined_emb = new_emb
    else:
        combined_emb = np.concatenate([raw_emb, new_emb], axis=0)
    combined_meta = list(meta) + new_meta_rows
    _save_table_atomic(rdir, combined_emb, combined_meta)

    # Invalidate the in-memory cache so the next route() call reloads
    load_default_runtime.cache_clear()  # type: ignore[attr-defined]

    return {
        "added": len(new_items),
        "skipped": skipped,
        "total_routes_after": len(combined_meta),
        "names_added": [i["name"] for i in new_items[:20]],
        "trusted": trusted,
    }


def add_route(
    item: dict[str, Any],
    runtime_dir: str = _DEFAULT_RUNTIME_DIR,
    include_unknown: bool = False,
    trusted: bool = False,
) -> dict[str, Any]:
    """Programmatic single-route registration.

    `item` must contain at minimum {name, description, kind} and may
    include {license, source_repo, url}. `trusted=True` marks the item
    first-party and bypasses the license gate.
    """
    import numpy as np

    name = item.get("name", "").strip()
    desc = item.get("description", "").strip()
    if not name or not desc:
        raise ValueError("item must include non-empty 'name' and 'description'")

    text = f"---\nname: {name}\ndescription: {desc}\n"
    if item.get("license"):
        text += f"license: {item['license']}\n"
    text += "---\n\n" + desc

    fm = _parse_frontmatter(text)
    bucket, reasons = _classify(text, fm, trusted=trusted)
    if bucket == "commercial_risk":
        raise PermissionError(f"refusing commercial-risk item: {reasons}")
    if bucket == "unknown" and not include_unknown:
        raise PermissionError(
            f"refusing unknown-license item: {reasons}; pass include_unknown=True to override"
        )

    sha = hashlib.sha256(text.encode()).hexdigest()
    rt = load_default_runtime(runtime_dir)
    # Resolve through the canonical resolver so the lazy _DEFAULT_RUNTIME_DIR
    # proxy ("<lazy:mind-nerve-runtime>") is unwrapped before table I/O —
    # route_table.npy lives under the REAL dir, not the literal proxy string.
    from .inference import _resolve_runtime_dir

    rdir = _resolve_runtime_dir(str(runtime_dir) if runtime_dir is not None else None)

    if any(m["sha256"] == sha for m in rt.routes):
        return {"added": 0, "reason": "already indexed", "sha256": sha}

    emb = _embed_texts(rt, [f"{name}\n\n{desc}"])
    raw_emb, meta = _load_table(rdir)
    combined_emb = np.concatenate([raw_emb, emb], axis=0)
    new_meta = {
        "id": sha[:16],
        "source_repo": item.get("source_repo", "programmatic"),
        "source_path": item.get("source_path", ""),
        "kind": item.get("kind", "skill"),
        "name": name,
        "size_bytes": len(text),
        "sha256": sha,
        "tokens_est": max(1, len(text) // 4),
    }
    if item.get("url"):
        new_meta["url"] = item["url"]
    meta.append(new_meta)
    _save_table_atomic(rdir, combined_emb, meta)
    load_default_runtime.cache_clear()  # type: ignore[attr-defined]
    return {"added": 1, "sha256": sha, "total_routes_after": len(meta)}


# ---------------------------------------------------------------------------
# Polling watcher (no inotify dependency required)
# ---------------------------------------------------------------------------


class Watcher:
    """Polling watcher that rescans directories at `interval` seconds.

    Polling beats inotify here because we want to work on macOS,
    Windows, and containers without a Linux-specific dependency. The
    rescan is cheap: it walks files, hashes them, and skips anything
    already in `seen_ids`.
    """

    def __init__(
        self,
        directories: list[tuple[str | Path, str]],
        *,
        interval: float = 5.0,
        include_unknown: bool = False,
        runtime_dir: str = _DEFAULT_RUNTIME_DIR,
    ):
        self.dirs = [(Path(d), src) for d, src in directories]
        self.interval = interval
        self.include_unknown = include_unknown
        self.runtime_dir = runtime_dir
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_summary: dict[str, Any] = {}

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                for d, src in self.dirs:
                    if d.is_dir():
                        out = scan(
                            d,
                            source_repo=src,
                            include_unknown=self.include_unknown,
                            runtime_dir=self.runtime_dir,
                        )
                        self._last_summary = {"dir": str(d), "summary": out, "ts": time.time()}
            except Exception as exc:  # noqa: BLE001
                self._last_summary = {"error": str(exc), "ts": time.time()}
            self._stop.wait(self.interval)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="mind-nerve-watcher")
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    @property
    def last(self) -> dict[str, Any]:
        return dict(self._last_summary)
