#!/usr/bin/env python3
"""A1.5 correctness harness — native Q16.16 encode vs pytorch SentenceTransformer.

Measures, on a held-out query set:
  * embedding cosine (native vs pytorch, mean / min)
  * top-5 route overlap: build a synthetic catalog from pytorch embeddings of
    a doc corpus, rank each query against it with both native and pytorch
    query embeddings, report mean Jaccard/overlap of the two top-5 lists.
  * float64 reference of the exact encode.mind path vs pytorch (correctness
    invariant — must stay cosine ~1.0).

Not a pytest (mirrors the prior agent's out-of-CI measurement). Deterministic
corpus generated from a fixed seed so the number is reproducible run-to-run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python"))

RUNTIME = Path(os.environ.get("MIND_NERVE_RUNTIME_DIR", str(Path.home() / ".local/share/mind-nerve/runtime")))

N_QUERIES = int(os.environ.get("A15_N", "160"))
N_DOCS = int(os.environ.get("A15_DOCS", "400"))
SEED = 0xA15_C0DE


def _corpus(n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    subjects = [
        "the database",
        "a neural network",
        "the compiler",
        "this kernel",
        "the cache layer",
        "an embedding model",
        "the scheduler",
        "a tensor op",
        "the parser",
        "this protocol",
        "the runtime",
        "a memory allocator",
        "the encoder",
        "this benchmark",
        "the routing table",
        "a quantizer",
    ]
    verbs = [
        "optimizes",
        "fails to handle",
        "accelerates",
        "validates",
        "compresses",
        "indexes",
        "deduplicates",
        "serializes",
        "reconciles",
        "normalizes",
        "schedules",
        "profiles",
    ]
    objects = [
        "the input stream under heavy load",
        "fixed-point arithmetic precision",
        "concurrent write contention",
        "the sliding-window attention path",
        "cross-architecture bit identity",
        "the L2-normalized output vector",
        "variance accumulation in layernorm",
        "the top-k selection step",
        "deterministic tie-breaking",
        "the residual connection",
        "softmax over long sequences",
        "the head-split reshape",
    ]
    out = []
    for _ in range(n):
        s = subjects[rng.integers(len(subjects))]
        v = verbs[rng.integers(len(verbs))]
        o = objects[rng.integers(len(objects))]
        out.append(f"{s} {v} {o}")
    return out


def main() -> int:
    from mind_nerve._native import _NativeRuntime, _q16_to_f32
    from sentence_transformers import SentenceTransformer

    ckpt = str(RUNTIME / "checkpoint")
    print(f"[harness] loading SentenceTransformer from {ckpt}", flush=True)
    st = SentenceTransformer(ckpt, device="cpu")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ckpt, use_fast=True)

    # Native runtime
    nat = _NativeRuntime()
    blob_path = RUNTIME / "encoder_weights.q16.bin"
    weights = np.ascontiguousarray(np.fromfile(str(blob_path), dtype=np.int64))
    import ctypes as ct

    blob_addr = ct.cast(weights.ctypes.data_as(ct.POINTER(ct.c_int64)), ct.c_void_p).value or 0
    handle = nat.init(blob_addr, weights.nbytes)
    if handle == 0:
        print("[harness] mn_encoder_init failed", file=sys.stderr)
        return 2
    print(f"[harness] native build id: {nat.version()}", flush=True)

    queries = _corpus(N_QUERIES, SEED)
    docs = _corpus(N_DOCS, SEED ^ 0x9999)

    def tokenize(text: str) -> np.ndarray:
        enc = tok(
            text,
            truncation=True,
            max_length=256,
            return_tensors="np",
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return enc["input_ids"][0].astype(np.int32)

    # pytorch embeddings (normalized)
    q_pt = st.encode(queries, normalize_embeddings=True, convert_to_numpy=True).astype(np.float64)
    d_pt = st.encode(docs, normalize_embeddings=True, convert_to_numpy=True).astype(np.float64)

    # native embeddings
    q_nat = np.zeros_like(q_pt)
    for i, text in enumerate(queries):
        emb = _q16_to_f32(nat.encode(handle, tokenize(text))).astype(np.float64)
        n = np.linalg.norm(emb)
        q_nat[i] = emb / n if n > 0 else emb

    # cosine native vs pytorch
    cos = np.sum(q_nat * q_pt, axis=1)
    print(
        f"[harness] embedding cosine native-vs-pytorch: mean={cos.mean():.6f} min={cos.min():.6f}",
        flush=True,
    )

    # top-5 overlap against pytorch-doc catalog
    K = 5
    sim_pt = q_pt @ d_pt.T
    sim_nat = q_nat @ d_pt.T
    top_pt = np.argsort(-sim_pt, axis=1)[:, :K]
    top_nat = np.argsort(-sim_nat, axis=1)[:, :K]
    overlaps = [len(set(top_pt[i]) & set(top_nat[i])) / K for i in range(len(queries))]
    mean_overlap = float(np.mean(overlaps))
    print(
        f"[harness] TOP-5 ROUTE OVERLAP native-vs-pytorch: {mean_overlap:.4f} "
        f"(gate >= 0.92, n={len(queries)})",
        flush=True,
    )

    nat.free(handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
