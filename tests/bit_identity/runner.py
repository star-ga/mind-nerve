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
  --backend native    Loads through production's _NativeEncoderRuntime
                      (mind_nerve.inference) and hashes the REAL .so output —
                      the same load path the routing daemon uses, so the
                      harness can never drift from production. Emits
                      BACKEND_STUB_NOT_BUILT only when the encoder-weights
                      blob (encoder_weights.q16.bin) is genuinely absent.
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
    """Resolve the runtime dir EXACTLY as production inference does.

    Delegates to ``mind_nerve.inference._resolve_runtime_dir`` so the harness can
    never drift from the routing daemon: it respects ``MIND_NERVE_RUNTIME_DIR``
    and otherwise lands on the curated STARGA dash-form runtime dir
    (``~/.local/share/mind-nerve-runtime``, production's ``_USER_RUNTIME_DIR``) —
    NOT the OSS-leftover slash-form dir this harness used to default to. The old
    default is what silently pointed the native path at a dir with no encoder
    weights, part of the permanently fake-green gate.
    """
    try:
        from mind_nerve.inference import _resolve_runtime_dir as _prod_resolve
    except ImportError:
        _prod_resolve = None
    if _prod_resolve is not None:
        return _prod_resolve(None)
    # Fallback only when mind_nerve is unimportable: mirror production's
    # dash-preferred preference by hand rather than falling to the OSS dir.
    env = os.environ.get("MIND_NERVE_RUNTIME_DIR")
    if env:
        p = Path(env).expanduser()
        if p.is_dir():
            return p
    dash = Path.home() / ".local" / "share" / "mind-nerve-runtime"
    if (dash / "manifest.json").exists():
        return dash
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
# Native backend — loads through production's _NativeEncoderRuntime
# ---------------------------------------------------------------------------


def _native_weights_path(runtime_dir: Path) -> Path:
    """Resolve the encoder-weights blob path the SAME way _NativeEncoderRuntime
    does: ``$MIND_NERVE_ENCODER_WEIGHTS`` override, else
    ``<runtime_dir>/encoder_weights.q16.bin``.
    """
    env_blob = os.environ.get("MIND_NERVE_ENCODER_WEIGHTS")
    if env_blob:
        return Path(env_blob).expanduser()
    return runtime_dir / "encoder_weights.q16.bin"


def _run_native_backend(corpus: list[dict], runtime_dir: Path) -> list[dict]:
    """Run the native (mindc-compiled .so) backend through the SAME loader the
    routing daemon uses: ``mind_nerve.inference._NativeEncoderRuntime``.

    Loading through production is deliberate and load-bearing. The previous
    implementation re-implemented the native load path here (a hand-rolled
    ctypes ABI + hard-coded ``_native/weights.q16.bin`` / ``route_table.q16.bin``
    paths that production never writes), so it ALWAYS fell through to
    ``BACKEND_STUB_NOT_BUILT`` and the gate was permanently fake-green. Delegating
    to ``_NativeEncoderRuntime`` means this harness can never again drift from
    the daemon: the runtime reads the encoder weights from
    ``encoder_weights.q16.bin`` and DERIVES the Q16 catalog on load from
    ``route_table.npy``.

    Sentinel policy (de-fake-green):
      * encoder-weights blob ABSENT  -> ``BACKEND_STUB_NOT_BUILT`` (the ONE
        legitimate stub: the .so needs quantized weights to produce a real
        embedding).
      * weights PRESENT but the runtime fails to load/encode -> RAISE. A
        broken-but-present backend must fail loudly, never degrade to a
        green stub.
    """
    weights_path = _native_weights_path(runtime_dir)
    if not weights_path.exists():
        print(
            f"  native: encoder-weights blob absent at {weights_path} — emitting "
            f"stub sentinels (quantize/build the encoder to enable).",
            file=sys.stderr,
        )
        return _sentinel_records(corpus, "native", SENTINEL_NATIVE_STUB)

    # Weights present: load through production. Any failure past this point is a
    # real regression and must surface, not silently stub.
    from mind_nerve.inference import _NativeEncoderRuntime

    rt = _NativeEncoderRuntime(runtime_dir)
    if not getattr(rt, "_encoder_weights_loaded", False) or rt._handle == 0:
        raise RuntimeError(
            f"native backend: encoder weights present at {weights_path} but the "
            f"runtime did not load them (handle={getattr(rt, '_handle', 0)}). "
            f"Refusing to emit a fake-green stub."
        )

    records: list[dict] = []
    n = len(corpus)

    for qi, entry in enumerate(corpus):
        if (qi + 1) % 100 == 0:
            print(f"  native: {qi + 1}/{n}", file=sys.stderr)

        query_id = entry["id"]
        text = entry["text"]

        # Token IDs — int32, tokenized exactly as the daemon does (max_len 256,
        # pytorch-SentenceTransformer-equivalent, #228).
        token_ids = rt._tokenize(text)
        token_len = int(token_ids.shape[0])
        h1 = _sha256_bytes(token_ids.tobytes())

        # Final Q16.16 L2-normalized embedding straight from the .so
        # (int64 (384,)). Re-quantize via _float32_to_q16_bytes so the record
        # format is byte-for-byte the same int32 Q16.16 LE as the pytorch path.
        qv_q16 = rt._native.encode(rt._handle, token_ids)
        l2_f32 = rt._q16_to_f32(qv_q16)
        h5 = _sha256_bytes(_float32_to_q16_bytes(l2_f32))

        # Intermediate LayerNorm hashes are NOT observable through the C ABI
        # (mn_encoder_encode returns only the final vector). Set them equal to
        # the final-output hash — the honest handling documented in A1.3.
        h2 = h3 = h4 = h5

        # Catalog scores — mirror _route_native EXACTLY (incl. the optional
        # log-prior add) so the pinned reference equals the daemon's real
        # top-K output.
        scores_q16 = rt._native.score(rt._handle, qv_q16, rt._catalog_q16)
        if rt._log_prior_q16 is not None:
            scores_q16 = scores_q16 + rt._log_prior_q16
        scores_f32 = rt._q16_to_f32(scores_q16)
        h6 = _sha256_bytes(_float32_to_q16_bytes(scores_f32))

        # Top-K indices + scores (Q16.16) via the native selector.
        k = min(TOP_K, int(scores_q16.shape[0]))
        topk_idx, topk_scr = rt._native.topk(scores_q16, k)
        h7 = _sha256_bytes(_topk_to_bytes(topk_idx, topk_scr))

        records.append(
            {
                "id": query_id,
                "category": entry["category"],
                "backend": "native",
                "token_len": token_len,
                "hashes": {
                    "token_ids": h1,
                    "post_embed_ln": h2,
                    "final_layer_ln": h3,
                    "post_cls_slice": h4,
                    "post_l2_norm": h5,
                    "catalog_scores": h6,
                    "topk_indices_scores": h7,
                },
                "topk_indices": [int(i) for i in topk_idx.tolist()],
            }
        )

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
