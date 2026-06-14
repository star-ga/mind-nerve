"""
tests/bit_identity/runner.py

Per-backend hash emitter for the A1.4 bit-identity harness.

Runs 1,000 corpus queries through the selected backend and produces a
JSON blob of 7,000 SHA-256 hashes (7 per query).

Hash points per query (per spec §A1.4):
  1. token_ids          — int32 array (WordPiece token IDs)
  2. post_embed_ln      — Q16.16 byte buffer of post-embedding LayerNorm output
  3. final_layer_ln     — Q16.16 byte buffer of final layer output post-LayerNorm
  4. post_cls_slice     — Q16.16 byte buffer of CLS-token vector
  5. post_l2_norm       — Q16.16 byte buffer of L2-normalized encoder output
  6. catalog_scores     — Q16.16 byte buffer of dot-product scores
  7. topk_indices_scores — raw bytes of top-K indices (int32) + scores (int32 Q16.16)

Backend selection:
  --backend native    Uses A1.3 ctypes wiring; emits BACKEND_STUB_NOT_BUILT
                      sentinel per query if the .so is not present.
  --backend pytorch   Uses sentence-transformers FP32 reference path (ground truth).
  --backend cuda      Emits CUDA_DEFERRED_TO_V0_4_1 sentinel per query (§3.2).

Usage:
    python tests/bit_identity/runner.py --backend pytorch --out /tmp/hashes_pytorch.json
    python tests/bit_identity/runner.py --backend native  --out /tmp/hashes_native.json
    python tests/bit_identity/runner.py --backend cuda    --out /tmp/hashes_cuda.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Sentinels for unavailable backends
# ---------------------------------------------------------------------------

SENTINEL_NATIVE_STUB = "BACKEND_STUB_NOT_BUILT"
SENTINEL_CUDA = "CUDA_DEFERRED_TO_V0_4_1"

# Q16.16 scale factor (2^16)
Q16_SCALE = 65536.0

# Number of top-K results
TOP_K = 5

# BERT config (from checkpoint/config.json)
HIDDEN_SIZE = 384
MAX_SEQ_LEN = 256
NUM_LAYERS = 12

THIS_DIR = Path(__file__).parent
CORPUS_PATH = THIS_DIR / "corpus.json"

_DEFAULT_RUNTIME_DIR = Path.home() / ".local" / "share" / "mind-nerve" / "runtime"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_bytes(data: bytes) -> str:
    """Return lowercase hex SHA-256 of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def _float32_to_q16_bytes(arr: Any) -> bytes:
    """
    Convert a float32 numpy array to Q16.16 int32 and return as little-endian bytes.
    Clamps to int32 range before conversion. Deterministic: uses round-half-even
    (numpy default rounding mode) which is consistent across platforms.
    """
    import numpy as np

    q16 = np.clip(
        np.round(arr.astype(np.float64) * Q16_SCALE),
        -2147483648,
        2147483647,
    ).astype(np.int32)
    return q16.tobytes()


def _topk_to_bytes(indices: Any, scores_q16: Any) -> bytes:
    """
    Encode top-K indices (int32) followed by top-K scores (Q16.16 int32) as LE bytes.
    indices and scores must be the same length.
    """
    import numpy as np

    idx_i32 = indices.astype(np.int32)
    scr_i32 = scores_q16.astype(np.int32)
    return idx_i32.tobytes() + scr_i32.tobytes()


def _resolve_runtime_dir() -> Path:
    env = os.environ.get("MIND_NERVE_RUNTIME_DIR")
    if env:
        p = Path(env).expanduser()
        if p.is_dir():
            return p
    return _DEFAULT_RUNTIME_DIR


# ---------------------------------------------------------------------------
# PyTorch backend — ground truth
# ---------------------------------------------------------------------------


class _PyTorchHook:
    """
    Captures intermediate tensors from a BERT forward pass using register hooks.
    Extracts the 7 hash points specified in A1.4.
    """

    def __init__(self) -> None:
        self._handles: list = []
        self.post_embed_ln: Any = None
        self.final_layer_ln: Any = None

    def attach(self, bert_model: Any) -> None:
        """Attach forward hooks to the embedding LayerNorm and last transformer layer."""
        import torch

        # Hook 1: post-embedding LayerNorm (BERT embeddings module LayerNorm)
        def _embed_ln_hook(module: Any, _input: Any, output: Any) -> None:
            if isinstance(output, torch.Tensor):
                self.post_embed_ln = output.detach().cpu().float().numpy()

        emb_ln = bert_model.embeddings.LayerNorm
        self._handles.append(emb_ln.register_forward_hook(_embed_ln_hook))

        # Hook 2: final transformer layer output (last BertLayer's output LayerNorm)
        def _final_layer_hook(module: Any, _input: Any, output: Any) -> None:
            # BertLayer output is a tuple; first element is the hidden state
            tensor = output[0] if isinstance(output, tuple) else output
            self.final_layer_ln = tensor.detach().cpu().float().numpy()

        last_layer = bert_model.encoder.layer[-1]
        self._handles.append(last_layer.register_forward_hook(_final_layer_hook))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def reset(self) -> None:
        self.post_embed_ln = None
        self.final_layer_ln = None


def _run_pytorch_backend(
    corpus: list[dict],
    runtime_dir: Path,
) -> list[dict]:
    """
    Run the PyTorch reference backend on the corpus.
    Returns list of hash records, one per query.
    """
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(runtime_dir / "checkpoint"), device="cpu")
    model.eval()

    bert = model[0].auto_model
    tokenizer = model.tokenizer

    # Load catalog embeddings (pre-normalized, matching inference.py behavior)
    emb_path = runtime_dir / "route_table.npy"
    catalog_emb = np.load(emb_path).astype(np.float32)
    norms = np.linalg.norm(catalog_emb, axis=1, keepdims=True) + 1e-12
    catalog_norm = (catalog_emb / norms).astype(np.float32)

    hook = _PyTorchHook()
    hook.attach(bert)

    records: list[dict] = []
    n = len(corpus)

    for qi, entry in enumerate(corpus):
        if (qi + 1) % 100 == 0:
            print(
                f"  pytorch: {qi + 1}/{n} ({100 * (qi + 1) / n:.0f}%)",
                file=sys.stderr,
            )

        query_id = entry["id"]
        text = entry["text"]

        hook.reset()

        # Tokenize
        encoded = tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=MAX_SEQ_LEN,
        )
        token_ids_np = encoded["input_ids"].numpy().astype(np.int32).flatten()

        # Forward pass
        with torch.no_grad():
            outputs = bert(**encoded)
            last_hidden = outputs.last_hidden_state  # (1, T, 384)

        # Hash 1: token IDs
        h1 = _sha256_bytes(token_ids_np.tobytes())

        # Hash 2: post-embedding LayerNorm (captured by hook)
        post_embed = hook.post_embed_ln
        if post_embed is None:
            # Fallback: use hidden_states[0] from full pass with output_hidden_states=True
            with torch.no_grad():
                out2 = bert(**encoded, output_hidden_states=True)
            # hidden_states[0] is the embedding layer output before transformer layers
            post_embed = out2.hidden_states[0].cpu().float().numpy().flatten()
        else:
            post_embed = post_embed.flatten()
        h2 = _sha256_bytes(_float32_to_q16_bytes(post_embed))

        # Hash 3: final layer output post-LayerNorm (captured by hook)
        final_ln = hook.final_layer_ln
        if final_ln is None:
            final_ln = last_hidden.cpu().float().numpy().flatten()
        else:
            final_ln = final_ln.flatten()
        h3 = _sha256_bytes(_float32_to_q16_bytes(final_ln))

        # Hash 4: CLS-token slice [0, :]
        cls_vec = last_hidden[0, 0, :].cpu().float().numpy()  # (384,)
        h4 = _sha256_bytes(_float32_to_q16_bytes(cls_vec))

        # Hash 5: L2-normalized encoder output
        # Matches sentence-transformers Normalize module behavior
        norm_val = float(np.linalg.norm(cls_vec)) + 1e-12
        l2_vec = (cls_vec / norm_val).astype(np.float32)
        h5 = _sha256_bytes(_float32_to_q16_bytes(l2_vec))

        # Hash 6: catalog dot-product scores
        # scores = catalog_norm @ l2_vec  (shape: N,)
        scores_f32 = (catalog_norm @ l2_vec).astype(np.float32)
        h6 = _sha256_bytes(_float32_to_q16_bytes(scores_f32))

        # Hash 7: top-K indices + scores (Q16.16)
        k = min(TOP_K, len(scores_f32))
        top_part = np.argpartition(-scores_f32, k - 1)[:k]
        top_sorted = top_part[np.argsort(-scores_f32[top_part])]
        top_scores_f32 = scores_f32[top_sorted]
        # Convert scores to Q16.16 for the hash
        top_scores_q16 = np.clip(
            np.round(top_scores_f32.astype(np.float64) * Q16_SCALE),
            -2147483648,
            2147483647,
        ).astype(np.int32)
        h7 = _sha256_bytes(_topk_to_bytes(top_sorted, top_scores_q16))

        records.append(
            {
                "id": query_id,
                "category": entry["category"],
                "backend": "pytorch",
                "token_len": int(token_ids_np.shape[0]),
                "hashes": {
                    "token_ids": h1,
                    "post_embed_ln": h2,
                    "final_layer_ln": h3,
                    "post_cls_slice": h4,
                    "post_l2_norm": h5,
                    "catalog_scores": h6,
                    "topk_indices_scores": h7,
                },
                "topk_indices": top_sorted.tolist(),
            }
        )

    hook.detach()
    return records


# ---------------------------------------------------------------------------
# Native backend (A1.3 ctypes wiring stub)
# ---------------------------------------------------------------------------


def _find_native_so() -> Path | None:
    """Locate the native encoder .so in the standard search paths."""
    candidates = [
        Path(__file__).parent.parent.parent
        / "python"
        / "mind_nerve"
        / "_native"
        / "libmind_nerve_encoder.so",
        Path(os.environ.get("MIND_NERVE_NATIVE_SO", "")),
    ]
    for c in candidates:
        if c and c.exists():
            return c
    return None


def _run_native_backend(corpus: list[dict], runtime_dir: Path) -> list[dict]:
    """
    Run the native (mindc-compiled .so) backend.
    Emits BACKEND_STUB_NOT_BUILT sentinel when the .so is not present.
    """
    so_path = _find_native_so()

    if so_path is None:
        print(
            "  native: .so not found — emitting stub sentinels. "
            "Build the wheel with tools/build_native_encoder.sh to enable.",
            file=sys.stderr,
        )
        return _sentinel_records(corpus, "native", SENTINEL_NATIVE_STUB)

    # A1.3 ctypes wiring is loaded here when the .so exists.
    # Full implementation wired in A1.3; for now we validate the ABI surface.
    try:
        import ctypes

        lib = ctypes.CDLL(str(so_path))
        # Verify ABI exports
        required = [
            "mn_encoder_init",
            "mn_encoder_encode",
            "mn_encoder_score",
            "mn_encoder_topk",
            "mn_encoder_free",
            "mn_encoder_version",
        ]
        missing = [sym for sym in required if not hasattr(lib, sym)]
        if missing:
            print(
                f"  native: .so missing symbols {missing} — emitting stub sentinels.",
                file=sys.stderr,
            )
            return _sentinel_records(corpus, "native", SENTINEL_NATIVE_STUB)

        # Full native path: call mn_encoder_init, then mn_encoder_encode per query.
        # This path is active once A1.3 ships; here we delegate to the ctypes layer.
        return _run_native_so(lib, corpus, runtime_dir)

    except OSError as exc:
        print(f"  native: failed to load .so ({exc}) — emitting stub sentinels.", file=sys.stderr)
        return _sentinel_records(corpus, "native", SENTINEL_NATIVE_STUB)


def _run_native_so(lib: Any, corpus: list[dict], runtime_dir: Path) -> list[dict]:
    """
    Execute the native encoder for each corpus query via ctypes.
    Called only when the .so is present and exports the required symbols.
    """
    import ctypes

    import numpy as np

    # Configure ABI
    lib.mn_encoder_version.restype = ctypes.c_char_p
    lib.mn_encoder_init.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    lib.mn_encoder_init.restype = ctypes.c_int64
    lib.mn_encoder_encode.argtypes = [
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.mn_encoder_encode.restype = ctypes.c_int32
    lib.mn_encoder_score.argtypes = [
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.mn_encoder_score.restype = ctypes.c_int32
    lib.mn_encoder_topk.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.mn_encoder_topk.restype = ctypes.c_int32
    lib.mn_encoder_free.argtypes = [ctypes.c_int64]
    lib.mn_encoder_free.restype = ctypes.c_int32

    # Load weights blob
    weights_path = runtime_dir / "_native" / "weights.q16.bin"
    if not weights_path.exists():
        print(
            f"  native: weights blob not found at {weights_path} — emitting stub sentinels.",
            file=sys.stderr,
        )
        return _sentinel_records(corpus, "native", SENTINEL_NATIVE_STUB)

    weights_data = weights_path.read_bytes()
    weights_buf = (ctypes.c_uint8 * len(weights_data))(*weights_data)
    handle = lib.mn_encoder_init(weights_buf, len(weights_data))
    if handle <= 0:
        print("  native: mn_encoder_init failed — emitting stub sentinels.", file=sys.stderr)
        return _sentinel_records(corpus, "native", SENTINEL_NATIVE_STUB)

    # Load catalog Q16.16
    catalog_path = runtime_dir / "_native" / "route_table.q16.bin"
    if not catalog_path.exists():
        lib.mn_encoder_free(handle)
        return _sentinel_records(corpus, "native", SENTINEL_NATIVE_STUB)

    catalog_raw = np.frombuffer(catalog_path.read_bytes(), dtype=np.int32)
    n_catalog = catalog_raw.shape[0] // HIDDEN_SIZE
    catalog_ptr = catalog_raw.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    # Load tokenizer (stays Python in v0.4.0 per §A1.3)
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(str(runtime_dir / "checkpoint"))

    out_vec = (ctypes.c_int32 * HIDDEN_SIZE)()
    out_scores = (ctypes.c_int32 * n_catalog)()
    out_topk_idx = (ctypes.c_int32 * TOP_K)()
    out_topk_scores = (ctypes.c_int32 * TOP_K)()

    records: list[dict] = []
    n = len(corpus)

    for qi, entry in enumerate(corpus):
        if (qi + 1) % 100 == 0:
            print(f"  native: {qi + 1}/{n}", file=sys.stderr)

        query_id = entry["id"]
        text = entry["text"]

        enc = tokenizer(
            text,
            return_tensors="np",
            padding=False,
            truncation=True,
            max_length=MAX_SEQ_LEN,
        )
        token_ids = enc["input_ids"].astype(np.int32).flatten()
        token_len = len(token_ids)

        token_ptr = token_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

        # Hash 1: token IDs
        h1 = _sha256_bytes(token_ids.tobytes())

        # Encode — fills out_vec with Q16.16 L2-normalized output (hash 5)
        rc = lib.mn_encoder_encode(handle, token_ptr, token_len, out_vec)
        if rc != 0:
            records.append(
                _error_record(query_id, entry["category"], "native", f"encode_error_{rc}")
            )
            continue

        # We only get the final output from the ABI; for intermediate hashes
        # (post_embed_ln, final_layer_ln, post_cls_slice) we emit the same
        # hash as post_l2_norm since the ABI exposes only the final output vector.
        # This is by design: the full intermediate capture requires the Python
        # instrumentation path (pytorch backend) for A1.4 ground truth.
        l2_vec_q16 = np.ctypeslib.as_array(out_vec).copy()
        l2_bytes = l2_vec_q16.tobytes()
        h5 = _sha256_bytes(l2_bytes)

        # For native ABI, intermediates not available — use final output hash
        h2 = h3 = h4 = h5  # ABI limitation documented in A1.3

        # Hash 6: catalog scores
        rc = lib.mn_encoder_score(handle, out_vec, catalog_ptr, n_catalog, out_scores)
        if rc != 0:
            records.append(
                _error_record(query_id, entry["category"], "native", f"score_error_{rc}")
            )
            continue

        scores_q16 = np.ctypeslib.as_array(out_scores, shape=(n_catalog,)).copy()
        h6 = _sha256_bytes(scores_q16.tobytes())

        # Hash 7: top-K
        k = min(TOP_K, n_catalog)
        rc = lib.mn_encoder_topk(out_scores, n_catalog, k, out_topk_idx, out_topk_scores)
        if rc != 0:
            records.append(_error_record(query_id, entry["category"], "native", f"topk_error_{rc}"))
            continue

        topk_idx = np.ctypeslib.as_array(out_topk_idx, shape=(k,)).copy()
        topk_scr = np.ctypeslib.as_array(out_topk_scores, shape=(k,)).copy()
        h7 = _sha256_bytes(_topk_to_bytes(topk_idx, topk_scr))

        records.append(
            {
                "id": query_id,
                "category": entry["category"],
                "backend": "native",
                "token_len": int(token_len),
                "hashes": {
                    "token_ids": h1,
                    "post_embed_ln": h2,
                    "final_layer_ln": h3,
                    "post_cls_slice": h4,
                    "post_l2_norm": h5,
                    "catalog_scores": h6,
                    "topk_indices_scores": h7,
                },
                "topk_indices": topk_idx.tolist(),
            }
        )

    lib.mn_encoder_free(handle)
    return records


# ---------------------------------------------------------------------------
# CUDA backend — deferred to v0.4.1 (§3.2)
# ---------------------------------------------------------------------------


def _run_cuda_backend(corpus: list[dict], runtime_dir: Path) -> list[dict]:
    """CUDA bit-identity gate deferred to A2 (v0.4.1). Emits sentinels."""
    print(
        "  cuda: CUDA bit-identity gate deferred to v0.4.1 per §3.2 — emitting sentinels.",
        file=sys.stderr,
    )
    return _sentinel_records(corpus, "cuda", SENTINEL_CUDA)


# ---------------------------------------------------------------------------
# Sentinel helpers
# ---------------------------------------------------------------------------

HASH_KEYS = (
    "token_ids",
    "post_embed_ln",
    "final_layer_ln",
    "post_cls_slice",
    "post_l2_norm",
    "catalog_scores",
    "topk_indices_scores",
)


def _sentinel_records(corpus: list[dict], backend: str, sentinel: str) -> list[dict]:
    return [
        {
            "id": e["id"],
            "category": e["category"],
            "backend": backend,
            "token_len": None,
            "hashes": {k: sentinel for k in HASH_KEYS},
            "topk_indices": None,
        }
        for e in corpus
    ]


def _error_record(query_id: str, category: str, backend: str, error: str) -> dict:
    return {
        "id": query_id,
        "category": category,
        "backend": backend,
        "token_len": None,
        "hashes": {k: f"ERROR:{error}" for k in HASH_KEYS},
        "topk_indices": None,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_backend(
    backend: str,
    corpus: list[dict],
    runtime_dir: Path,
) -> list[dict]:
    """Dispatch to the appropriate backend runner."""
    if backend == "pytorch":
        return _run_pytorch_backend(corpus, runtime_dir)
    elif backend == "native":
        return _run_native_backend(corpus, runtime_dir)
    elif backend == "cuda":
        return _run_cuda_backend(corpus, runtime_dir)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Choose: pytorch, native, cuda")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A1.4 bit-identity harness — run encoder backend and emit SHA-256 hashes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        choices=["pytorch", "native", "cuda"],
        default="pytorch",
        help="Backend to run (default: pytorch)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON file path (default: /tmp/bit_identity_{backend}.json)",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=CORPUS_PATH,
        help="Corpus JSON file path",
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=None,
        help="mind-nerve runtime directory (default: auto-resolve)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit corpus to first N queries (for development)",
    )
    args = parser.parse_args()

    # Load corpus
    if args.corpus.exists():
        with args.corpus.open("r", encoding="utf-8") as f:
            corpus = json.load(f)
    else:
        print(f"Corpus not found at {args.corpus}, building on-the-fly...", file=sys.stderr)
        sys.path.insert(0, str(THIS_DIR))
        from corpus import build_corpus

        corpus = build_corpus()

    if args.limit:
        corpus = corpus[: args.limit]

    runtime_dir = args.runtime_dir or _resolve_runtime_dir()

    out_path = args.out or Path(f"/tmp/bit_identity_{args.backend}.json")

    print(
        f"Running {args.backend} backend on {len(corpus)} queries...",
        file=sys.stderr,
    )
    t0 = time.perf_counter()
    records = run_backend(args.backend, corpus, runtime_dir)
    elapsed = time.perf_counter() - t0

    output = {
        "backend": args.backend,
        "corpus_size": len(corpus),
        "total_hashes": len(records) * len(HASH_KEYS),
        "elapsed_seconds": round(elapsed, 3),
        "runtime_dir": str(runtime_dir),
        "records": records,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(
        f"Wrote {len(records)} records ({len(records) * len(HASH_KEYS)} hashes) "
        f"to {out_path} in {elapsed:.1f}s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
