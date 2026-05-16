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

The route table is updated atomically: write to `route_table.npy.tmp`
+ `route_table.jsonl.tmp` then `os.replace()` so a concurrent reader
never sees a partial state.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from .inference import _DEFAULT_RUNTIME_DIR, load_default_runtime

PUBLIC_LICENSES = {
    "apache-2.0", "apache 2.0", "apache2", "mit", "bsd-3-clause",
    "bsd-2-clause", "isc", "cc0", "cc0-1.0", "unlicense", "cc-by-4.0",
}

COMMERCIAL_MARKERS = re.compile(
    r"\b(starga[\s-]*commercial|proprietary|confidential|"
    r"all[\s-]*rights[\s-]*reserved|do[\s-]*not[\s-]*distribute|"
    r"closed[\s-]*source|naestro-defense|mind-internal)\b",
    re.IGNORECASE,
)

SKILL_PATTERNS = [
    (re.compile(r"(^|/)SKILL\.md$",                       re.I), "skill"),
    (re.compile(r"(^|/)skills?/[^/]+\.md$",               re.I), "skill"),
    (re.compile(r"(\.claude|^claude)/(skills|commands|agents)/.*\.md$", re.I), "skill"),
    (re.compile(r"(^|/)commands?/[^/]+\.md$",             re.I), "command"),
    (re.compile(r"(^|/)agents?/[^/]+\.md$",               re.I), "agent"),
    (re.compile(r"(^|/)subagents?/[^/]+\.md$",            re.I), "agent"),
    (re.compile(r"\.cursorrules$",                        re.I), "rule"),
    (re.compile(r"\.mdc$",                                re.I), "rule"),
]
SKIP_DIRS = {".git", "node_modules", "dist", "build", "target", "__pycache__"}


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


def _classify(text: str, fm: dict[str, str]) -> tuple[str, list[str]]:
    """Return (bucket, reasons)."""
    lic = (fm.get("license") or "").lower().strip()

    if COMMERCIAL_MARKERS.search(text):
        return "commercial_risk", ["matched commercial marker"]
    if fm.get("visibility", "").lower() in {"private", "confidential", "internal"}:
        return "commercial_risk", [f"visibility={fm['visibility']}"]
    if lic in PUBLIC_LICENSES:
        return "public_ok", [f"license={lic}"]
    if lic and "commercial" in lic:
        return "commercial_risk", [f"license={lic}"]

    return "unknown", ["no license declared"]


def _detect_kind(rel: str) -> str | None:
    for pat, kind in SKILL_PATTERNS:
        if pat.search(rel):
            return kind
    return None


# ---------------------------------------------------------------------------
# Route table I/O
# ---------------------------------------------------------------------------


def _load_table(runtime_dir: Path) -> tuple[Any, list[dict]]:
    import numpy as np
    emb = np.load(runtime_dir / "route_table.npy")
    meta = [json.loads(line) for line in (runtime_dir / "route_table.jsonl").open()]
    return emb, meta


def _save_table_atomic(runtime_dir: Path, emb, meta: list[dict]) -> None:
    import numpy as np
    tmp_npy = runtime_dir / "route_table.npy.tmp"
    tmp_jsonl = runtime_dir / "route_table.jsonl.tmp"
    np.save(tmp_npy, emb)
    with tmp_jsonl.open("w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, separators=(",", ":")) + "\n")
    os.replace(tmp_npy, runtime_dir / "route_table.npy")
    os.replace(tmp_jsonl, runtime_dir / "route_table.jsonl")


# ---------------------------------------------------------------------------
# Item extraction from a single file
# ---------------------------------------------------------------------------


def _first_h1(body: str) -> str | None:
    for line in body.splitlines()[:50]:
        s = line.strip()
        if s.startswith("# "):
            return s[2:].strip()
    return None


def _item_from_file(path: Path, source_repo: str, kind: str) -> dict | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if len(text) < 32 or len(text) > 256 * 1024:
        return None
    fm = _parse_frontmatter(text)
    bucket, reasons = _classify(text, fm)
    name = fm.get("name") or _first_h1(text) or path.stem
    body = text[text.find("\n---", 3) + 4:] if text.startswith("---") else text
    sha = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
    return {
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
        "_embedded_text": (fm.get("description") or name) + "\n\n" + body[:1024],
    }


def _walk_dir(root: Path, source_repo: str) -> Iterable[dict]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        parts = set(p.relative_to(root).parts)
        if parts & SKIP_DIRS:
            continue
        rel = str(p.relative_to(root))
        kind = _detect_kind(rel)
        if not kind and p.name == "SKILL.md":
            kind = "skill"
        if not kind:
            continue
        item = _item_from_file(p, source_repo, kind)
        if item is not None:
            yield item


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan(directory: str | Path, source_repo: str = "local",
         include_unknown: bool = False,
         runtime_dir: str = _DEFAULT_RUNTIME_DIR,
         dry_run: bool = False) -> dict:
    """One-shot scan: discover new skills under `directory`, embed, persist.

    Returns a summary dict.
    """
    import numpy as np

    rt = load_default_runtime(runtime_dir)
    rdir = Path(runtime_dir)
    seen_ids = {m["sha256"] for m in rt.routes}

    new_items: list[dict] = []
    skipped: dict[str, int] = {"already_indexed": 0, "license_excluded": 0,
                                "unknown_excluded": 0}

    for item in _walk_dir(Path(directory), source_repo):
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
        return {"added": 0, "skipped": skipped,
                "total_routes_after": len(rt.routes)}

    if dry_run:
        return {
            "would_add": len(new_items),
            "names_preview": [i["name"] for i in new_items[:10]],
            "skipped": skipped,
            "dry_run": True,
        }

    # Embed the new items using the same model the route table was built with
    texts = [i["_embedded_text"] for i in new_items]
    new_emb = rt.model.encode(texts, batch_size=64, convert_to_numpy=True,
                              show_progress_bar=False,
                              normalize_embeddings=False).astype(np.float32)

    # Load raw table (rt.embeddings is the *normalised* in-memory copy)
    raw_emb, meta = _load_table(rdir)
    combined_emb = np.concatenate([raw_emb, new_emb], axis=0)
    combined_meta = list(meta) + [{k: v for k, v in i.items()
                                    if not k.startswith("_")}
                                   for i in new_items]
    _save_table_atomic(rdir, combined_emb, combined_meta)

    # Invalidate the in-memory cache so the next route() call reloads
    load_default_runtime.cache_clear()

    return {
        "added": len(new_items),
        "skipped": skipped,
        "total_routes_after": len(combined_meta),
        "names_added": [i["name"] for i in new_items[:20]],
    }


def add_route(item: dict, runtime_dir: str = _DEFAULT_RUNTIME_DIR,
              include_unknown: bool = False) -> dict:
    """Programmatic single-route registration.

    `item` must contain at minimum {name, description, kind} and may
    include {license, source_repo, url}.
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
    bucket, reasons = _classify(text, fm)
    if bucket == "commercial_risk":
        raise PermissionError(f"refusing commercial-risk item: {reasons}")
    if bucket == "unknown" and not include_unknown:
        raise PermissionError(f"refusing unknown-license item: {reasons}; "
                              "pass include_unknown=True to override")

    sha = hashlib.sha256(text.encode()).hexdigest()
    rt = load_default_runtime(runtime_dir)
    rdir = Path(runtime_dir)

    if any(m["sha256"] == sha for m in rt.routes):
        return {"added": 0, "reason": "already indexed", "sha256": sha}

    emb = rt.model.encode([f"{name}\n\n{desc}"], convert_to_numpy=True,
                          show_progress_bar=False,
                          normalize_embeddings=False).astype(np.float32)
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
    load_default_runtime.cache_clear()
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

    def __init__(self, directories: list[tuple[str | Path, str]], *,
                 interval: float = 5.0, include_unknown: bool = False,
                 runtime_dir: str = _DEFAULT_RUNTIME_DIR):
        self.dirs = [(Path(d), src) for d, src in directories]
        self.interval = interval
        self.include_unknown = include_unknown
        self.runtime_dir = runtime_dir
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_summary: dict[str, Any] = {}

    def _loop(self):
        while not self._stop.is_set():
            try:
                for d, src in self.dirs:
                    if d.is_dir():
                        out = scan(d, source_repo=src,
                                   include_unknown=self.include_unknown,
                                   runtime_dir=self.runtime_dir)
                        self._last_summary = {"dir": str(d), "summary": out,
                                              "ts": time.time()}
            except Exception as exc:                 # noqa: BLE001
                self._last_summary = {"error": str(exc), "ts": time.time()}
            self._stop.wait(self.interval)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                         name="mind-nerve-watcher")
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    @property
    def last(self) -> dict[str, Any]:
        return dict(self._last_summary)
