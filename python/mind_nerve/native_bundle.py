"""python/mind_nerve/native_bundle.py — U4b native (MIND-loader) bundle writer.

Producer for the native binary bundle ``src/mcp.mind``'s ``tools/call`` reads
via ``src/loader.mind``: ``route_table.cat`` (MNC1) + ``encoder_weights.mnw``
(MNW1). Written ALONGSIDE the existing Python-path ``route_table.jsonl`` /
``encoder_weights.q16.bin`` — never replacing them. No flag-day: both wire
formats coexist in the same runtime dir; the Python routing path keeps
reading the ``.jsonl``/``.q16.bin`` pair exactly as before, the native path
reads the new ``.cat``/``.mnw`` pair.

The wire-format encoders live in ``catalog-builder/format/cat_v2.py`` (MNC1)
and ``tools/quantize_encoder_to_q16.py`` (MNW1, placeholder mode); this
module is the seed-time GLUE that turns an already-built Python route table
(``route_table.jsonl``) into the native catalog, and always emits the same
deterministic placeholder MNW1 weights bundle.

deferred: the route embeddings written into the produced MNC1 catalog are
zero-vector placeholders (``ROUTE_EMBEDDING_DIM=256``, ``src/lib.mind``) —
the Python route table is 384-dim (BGE-small); there is no trained
256-dim/2-layer native encoder checkpoint to source real embeddings/weights
from yet. This mirrors the precedent already set by
``catalog-builder/build_index.py``'s MNC1 emit (see its docstring). Upgrade
path: once a real 256-dim checkpoint exists, source real embeddings/weights
here instead of the placeholders — the container/header code does not need
to change.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
from pathlib import Path
from typing import Any

CATALOG_FILENAME = "route_table.cat"
WEIGHTS_FILENAME = "encoder_weights.mnw"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CAT_MODULE_PATH = _REPO_ROOT / "catalog-builder" / "format" / "cat_v2.py"
_QUANTIZE_ENCODER_MODULE_PATH = _REPO_ROOT / "tools" / "quantize_encoder_to_q16.py"


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cat_module() -> Any:
    return _load_module(_CAT_MODULE_PATH, "mind_nerve._native_bundle_cat")


def _quantize_encoder_module() -> Any:
    return _load_module(_QUANTIZE_ENCODER_MODULE_PATH, "mind_nerve._native_bundle_quantize_encoder")


def route_ids_and_external_ids(
    meta: list[dict[str, Any]],
) -> tuple[list[bytes], list[bytes]]:
    """Derive (route_id, external_id) pairs from Python route_table.jsonl rows.

    Mirrors ``catalog-builder/build_index.py``'s derivation: route_id is the
    SHA-256 of a stable per-row identifier string (prefers ``sha256``, falls
    back to ``id``, then a synthetic ``route_{i:06d}`` when a row carries
    neither); external_id is that same string, UTF-8 encoded, written to the
    MNC1 trailer.
    """
    route_ids: list[bytes] = []
    external_ids: list[bytes] = []
    for i, row in enumerate(meta):
        text = str(row.get("sha256") or row.get("id") or f"route_{i:06d}")
        route_ids.append(hashlib.sha256(text.encode("utf-8")).digest())
        external_ids.append(text.encode("utf-8"))
    return route_ids, external_ids


def build_route_table_cat_bytes(meta: list[dict[str, Any]]) -> bytes:
    """Build an MNC1 ``route_table.cat`` blob from Python route metadata.

    See the module docstring's deferred-work note: embeddings are
    zero-vector placeholders (dimension ``EMBEDDING_DIM``, currently 256).
    """
    cat = _cat_module()
    route_ids, external_ids = route_ids_and_external_ids(meta)
    embeddings = [[0] * cat.EMBEDDING_DIM for _ in meta]
    return bytes(cat.encode_mnc1(route_ids, embeddings, external_ids))


def build_encoder_weights_mnw_bytes() -> bytes:
    """Build the (placeholder) MNW1 ``encoder_weights.mnw`` blob.

    See ``tools/quantize_encoder_to_q16.py``'s MNW1 section docstring — a
    deterministic placeholder until a trained 2-layer/256-hidden checkpoint
    exists.
    """
    qe = _quantize_encoder_module()
    return bytes(qe.build_mnw1_placeholder_bundle())


def _validate_mnw1_header(data: bytes) -> None:
    """Cheap structural self-check mirroring ``loader_parse_weights``'s
    header gates (magic / version / reserved / shape) — catches a producer
    bug before it ships, without re-walking the whole ~32 MiB blob."""
    if len(data) < 80:
        raise ValueError("MNW1 blob shorter than the fixed 80-byte header")
    if data[:4] != b"MNW1":
        raise ValueError(f"expected MNW1 magic, got {data[:4]!r}")
    (version,) = struct.unpack_from("<H", data, 4)
    if version not in (1, 2):
        raise ValueError(f"unsupported MNW1 version {version}")
    (reserved,) = struct.unpack_from("<H", data, 6)
    if reserved != 0:
        raise ValueError(f"MNW1 reserved field must be zero, got {reserved}")
    (layers,) = struct.unpack_from("<I", data, 72)
    (hidden,) = struct.unpack_from("<I", data, 76)
    if layers != 2 or hidden != 256:
        raise ValueError(f"unexpected MNW1 shape: layers={layers} hidden={hidden} (want 2/256)")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def write_native_bundle(runtime_dir: str | Path, meta: list[dict[str, Any]]) -> dict[str, Any]:
    """Write ``route_table.cat`` (MNC1) + ``encoder_weights.mnw`` (MNW1) into
    *runtime_dir*, ALONGSIDE the existing Python-path
    ``route_table.jsonl``/``encoder_weights.q16.bin`` — never replacing them.

    Self-verifies each blob with a pure-Python reference decoder/structural
    check BEFORE writing — a producer must never ship bytes the native
    loader would reject (``decode_mnc1`` replays every
    ``loader_parse_catalog`` gate; ``_validate_mnw1_header`` replays the
    ``loader_parse_weights`` header gates).
    """
    cat = _cat_module()
    runtime_dir = Path(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    cat_bytes = build_route_table_cat_bytes(meta)
    cat.decode_mnc1(cat_bytes)  # raises ValueError if the loader would reject this

    weights_bytes = build_encoder_weights_mnw_bytes()
    _validate_mnw1_header(weights_bytes)

    cat_path = runtime_dir / CATALOG_FILENAME
    weights_path = runtime_dir / WEIGHTS_FILENAME
    _atomic_write_bytes(cat_path, cat_bytes)
    _atomic_write_bytes(weights_path, weights_bytes)

    return {
        "route_table_cat": str(cat_path),
        "route_table_cat_bytes": len(cat_bytes),
        "encoder_weights_mnw": str(weights_path),
        "encoder_weights_mnw_bytes": len(weights_bytes),
        "route_count": len(meta),
    }


def write_native_bundle_from_route_table(runtime_dir: str | Path) -> dict[str, Any] | None:
    """Read the existing ``route_table.jsonl`` in *runtime_dir* (if present)
    and write the matching native bundle alongside it.

    Returns ``None`` (no-op) when ``route_table.jsonl`` is absent — there is
    nothing to derive route ids from yet, and this must never fabricate a
    catalog the Python route table does not agree with.
    """
    runtime_dir = Path(runtime_dir)
    meta_path = runtime_dir / "route_table.jsonl"
    if not meta_path.is_file():
        return None
    meta: list[dict[str, Any]] = []
    with meta_path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                meta.append(json.loads(stripped))
    return write_native_bundle(runtime_dir, meta)
