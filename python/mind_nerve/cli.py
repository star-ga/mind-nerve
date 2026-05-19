"""mind-nerve CLI — the single binary every host (claude-code, codex,
vibe, gemini, cursor, etc.) ultimately calls.

Two modes:

  Subprocess mode (default — used by claude-code, codex):
    $ echo "git status" | mind-nerve route --top-k 5
    {"routes":[{"id":"...","name":"...","kind":"skill","score":0.93,...}, ...]}

  Diagnostic mode:
    $ mind-nerve info
    $ mind-nerve precompute-routes
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .discovery import Watcher
from .discovery import scan as discovery_scan
from .inference import _DEFAULT_RUNTIME_DIR, load_default_runtime, precompute_routes, route


def cmd_route(args) -> int:
    if args.query:
        query = args.query
    else:
        query = sys.stdin.read().strip()
    if not query:
        print(json.dumps({"error": "empty query"}), file=sys.stderr)
        return 2

    kwargs = {}
    if args.runtime_dir:
        kwargs["runtime_dir"] = args.runtime_dir
    result = route(query, top_k=args.top_k, **kwargs)
    out = result.as_dict() if args.json else None
    if args.json:
        print(json.dumps(out, separators=(",", ":")))
    elif args.ids_only:
        for r in result.routes:
            print(r.id)
    else:
        for r in result.routes:
            url = f"  {r.url}" if r.url else ""
            print(f"{r.score:.4f}  {r.kind:<10}  {r.name}{url}")
    return 0


def cmd_info(args) -> int:
    rt = load_default_runtime(args.runtime_dir) if args.runtime_dir else load_default_runtime()
    out = {
        "version": __version__,
        "runtime_dir": str(rt.dir),
        "catalog_size": rt.catalog_size,
        "catalog_version": rt.catalog_version,
        "model_version": rt.model_version,
        "model_manifest_keys": sorted(rt.manifest.keys()),
    }
    print(json.dumps(out, indent=2))
    return 0


def cmd_precompute(args) -> int:
    kwargs = {}
    if args.runtime_dir:
        kwargs["runtime_dir"] = args.runtime_dir
    if args.cooccurrence:
        kwargs["cooccurrence_path"] = args.cooccurrence
    if args.emit_prior:
        kwargs["emit_prior"] = True
    if args.emit_freq_scale:
        kwargs["emit_freq_scale"] = True
    if args.emit_stride_thresholds:
        kwargs["emit_stride_thresholds"] = True
    out = precompute_routes(**kwargs)
    print(json.dumps(out, indent=2))
    return 0


def cmd_learn(args) -> int:
    out = discovery_scan(
        args.dir,
        source_repo=args.source or "local",
        include_unknown=args.include_unknown,
        runtime_dir=args.runtime_dir or _DEFAULT_RUNTIME_DIR,
        dry_run=args.dry_run,
    )
    print(json.dumps(out, indent=2))
    return 0


def cmd_train(args) -> int:
    from pathlib import Path

    from .mind_train import TrainConfig, config_to_dict, train

    config = TrainConfig(
        catalog_path=Path(args.catalog),
        output_dir=Path(args.out),
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_len=args.max_len,
        seed=args.seed,
        eval_frac=args.eval_frac,
        smoke_test=args.smoke_test,
        backend=args.backend,
    )
    try:
        result = train(config)
    except NotImplementedError as e:
        print(json.dumps({"error": str(e), "backend": args.backend}), file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1

    out = {
        "config": config_to_dict(config),
        "checkpoint_dir": str(result.checkpoint_dir),
        "manifest_path": str(result.manifest_path),
        "model_hash": result.model_hash,
        "epochs_completed": result.epochs_completed,
        "train_pairs": result.train_pairs,
        "eval_pairs": result.eval_pairs,
        "metrics": result.metrics,
        "baseline_metrics": result.baseline_metrics,
        "elapsed_seconds": result.elapsed_seconds,
        "backend_used": result.backend_used,
        "extras": result.extras,
    }
    print(json.dumps(out, indent=2))
    return 0


def cmd_quantize(args) -> int:
    """Phase 6.2 offline Q16.16 quantizer — wraps ``tools/quantize_phase1_to_q16.py``.

    Imports the tool module lazily so the rest of the CLI surface stays
    free of the dependency. The tool's ``main`` accepts the same argv
    surface, so this handler simply forwards.
    """
    import importlib.util as _ilu
    from pathlib import Path as _Path

    tool_path = _Path(__file__).resolve().parents[2] / "tools" / "quantize_phase1_to_q16.py"
    spec = _ilu.spec_from_file_location("quantize_phase1_to_q16", tool_path)
    if spec is None or spec.loader is None:
        print(
            json.dumps({"error": f"quantizer tool not found at {tool_path}"}),
            file=sys.stderr,
        )
        return 1
    module = _ilu.module_from_spec(spec)
    spec.loader.exec_module(module)

    argv = ["--catalog", args.catalog]
    if args.input is not None:
        argv.extend(["--input", args.input])
    if args.output is not None:
        argv.extend(["--output", args.output])
    if args.hidden_dim is not None:
        argv.extend(["--hidden-dim", str(args.hidden_dim)])
    if args.dry_run:
        argv.append("--dry-run")
    return module.main(argv)


def cmd_quantize_encoder(args) -> int:
    """Phase 6.x offline encoder-weights quantizer.

    Wraps ``tools/quantize_encoder_to_q16.py``: loads a safetensors
    checkpoint and emits ``encoder_weights.q16.bin`` for ``mn_encoder_encode``.
    """
    import importlib.util as _ilu
    from pathlib import Path as _Path

    tool_path = _Path(__file__).resolve().parents[2] / "tools" / "quantize_encoder_to_q16.py"
    spec = _ilu.spec_from_file_location("quantize_encoder_to_q16", tool_path)
    if spec is None or spec.loader is None:
        print(
            json.dumps({"error": f"encoder quantizer tool not found at {tool_path}"}),
            file=sys.stderr,
        )
        return 1
    module = _ilu.module_from_spec(spec)
    spec.loader.exec_module(module)

    argv = ["--checkpoint", args.checkpoint]
    if args.output is not None:
        argv.extend(["--output", args.output])
    if args.dry_run:
        argv.append("--dry-run")
    return module.main(argv)


def cmd_attest_sign(args) -> int:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    from .attestation import binding_message, serialize_binding_record, sign_binding

    try:
        mn = bytes.fromhex(args.mind_nerve_hash)
        ml = bytes.fromhex(args.mindllm_hash)
        nonce = bytes.fromhex(args.nonce)
        sk = bytes.fromhex(args.private_key_hex)
    except ValueError as e:
        print(json.dumps({"error": f"hex decode: {e}"}), file=sys.stderr)
        return 2

    if len(sk) != 32:
        print(json.dumps({"error": "private key must be 32 bytes hex"}), file=sys.stderr)
        return 2

    private_key = Ed25519PrivateKey.from_private_bytes(sk)
    pub_bytes = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    msg = binding_message(mn, ml, nonce)
    sig = sign_binding(private_key, msg)
    record = serialize_binding_record(mn, ml, nonce, sig, pub_bytes)
    print(
        json.dumps(
            {
                "record_hex": record.hex(),
                "record_bytes": len(record),
                "signer_pubkey_hex": pub_bytes.hex(),
                "binding_msg_hex": msg.hex(),
            },
            indent=2 if args.indent else None,
            separators=None if args.indent else (",", ":"),
        )
    )
    return 0


def cmd_attest_verify(args) -> int:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    from .attestation import application_verify_binding, deserialize_binding_record

    try:
        wire = bytes.fromhex(args.record_hex)
    except ValueError as e:
        print(json.dumps({"error": f"hex decode: {e}"}), file=sys.stderr)
        return 2

    try:
        rec = deserialize_binding_record(wire)
    except ValueError as e:
        print(json.dumps({"result": "ParseError", "error": str(e)}, separators=(",", ":")))
        return 1

    trust_anchor = args.pubkey_hex
    if trust_anchor:
        try:
            anchor_bytes = bytes.fromhex(trust_anchor)
        except ValueError as e:
            print(json.dumps({"error": f"hex decode: {e}"}), file=sys.stderr)
            return 2
        if anchor_bytes != rec.signer_pubkey:
            print(
                json.dumps(
                    {"result": "UntrustedSigner", "record_pubkey": rec.signer_pubkey.hex()},
                    separators=(",", ":"),
                )
            )
            return 1
        pubkey_bytes = anchor_bytes
    else:
        pubkey_bytes = rec.signer_pubkey

    public_key = Ed25519PublicKey.from_public_bytes(pubkey_bytes)
    result = application_verify_binding(
        rec.mind_nerve_hash, rec.mindllm_hash, rec.nonce, rec.signature, public_key
    )
    print(
        json.dumps(
            {
                "result": result,
                "mind_nerve_hash": rec.mind_nerve_hash.hex(),
                "mindllm_hash": rec.mindllm_hash.hex(),
                "nonce": rec.nonce.hex(),
                "signer_pubkey": rec.signer_pubkey.hex(),
                "version": rec.version,
            },
            indent=2 if args.indent else None,
            separators=None if args.indent else (",", ":"),
        )
    )
    return 0 if result == "ok" else 1


def cmd_rollback(args) -> int:
    """Restore a target CLI's config files from their last ``.bak`` snapshots.

    Thin wrapper over :func:`mind_nerve.installer.rollback_last` so users
    can call either ``mind-nerve rollback --target claude`` or
    ``mind-nerve-install rollback --target claude``.
    """
    from .installer import rollback_last

    result = rollback_last(args.target)
    print(json.dumps(result, indent=2))
    return 1 if result.get("errors") else 0


def cmd_watch(args) -> int:
    dirs = [(d, args.source or "local") for d in args.dirs]
    w = Watcher(
        dirs,
        interval=args.interval,
        include_unknown=args.include_unknown,
        runtime_dir=args.runtime_dir or _DEFAULT_RUNTIME_DIR,
    )
    w.start()
    print(
        f"[mind-nerve watch] watching {len(dirs)} dirs every {args.interval}s; ctrl-c to stop",
        file=sys.stderr,
    )
    import time

    try:
        while True:
            time.sleep(args.interval)
            last = w.last
            if last:
                print(json.dumps(last, separators=(",", ":")), flush=True)
    except KeyboardInterrupt:
        w.stop()
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mind-nerve")
    ap.add_argument("--version", action="version", version=f"mind-nerve {__version__}")
    ap.add_argument(
        "--runtime-dir",
        default=None,
        help="Override the runtime directory (default: $MIND_NERVE_RUNTIME_DIR, "
        "or ~/.local/share/mind-nerve/runtime/ which is auto-seeded from "
        "Hugging Face on first use).",
    )

    sub = ap.add_subparsers(dest="cmd", required=True)

    p_route = sub.add_parser("route", help="Return top-K routes for a query")
    p_route.add_argument("query", nargs="?", help="Query text (or read from stdin if omitted)")
    p_route.add_argument("--top-k", type=int, default=5)
    p_route.add_argument("--json", action="store_true", default=True, help="Emit JSON (default).")
    p_route.add_argument(
        "--ids-only", action="store_true", help="One route id per line (overrides --json)"
    )
    p_route.set_defaults(func=cmd_route)

    p_info = sub.add_parser("info", help="Print runtime info as JSON")
    p_info.set_defaults(func=cmd_info)

    p_pre = sub.add_parser(
        "precompute-routes", help="(One-time) encode the catalog into route_table.npy"
    )
    p_pre.add_argument(
        "--cooccurrence",
        default=None,
        help="Path to a JSONL co-occurrence log; enables catalog-v2 prior emit "
        "(`route_table_prior.npy`).",
    )
    p_pre.add_argument(
        "--emit-prior",
        action="store_true",
        help="Emit `route_table_prior.npy` even without a co-occurrence log "
        "(uniform Laplace prior; behaviorally identical to v1 scoring).",
    )
    p_pre.add_argument(
        "--emit-freq-scale",
        action="store_true",
        help="Emit `route_table_freq_scale.npy` (SOTA-track #4 freq-adaptive "
        "scale). With no co-occurrence log, every scale defaults to 1.0.",
    )
    p_pre.add_argument(
        "--emit-stride-thresholds",
        action="store_true",
        help="Emit `stride_thresholds.json` (SOTA-track #3 entropy → stride "
        "map). Consumed by the native-MIND encoder; ignored on Phase 1.",
    )
    p_pre.set_defaults(func=cmd_precompute)

    p_learn = sub.add_parser(
        "learn", help="Scan a directory for new skills and add them to the route table"
    )
    p_learn.add_argument("dir", help="Directory to scan (e.g. ~/.agents/skills)")
    p_learn.add_argument("--source", default=None, help="Source-repo label (default: 'local')")
    p_learn.add_argument(
        "--include-unknown",
        action="store_true",
        help="Include items with no declared license. OFF by default.",
    )
    p_learn.add_argument("--dry-run", action="store_true")
    p_learn.set_defaults(func=cmd_learn)

    p_train = sub.add_parser(
        "train",
        help="Train a mind-nerve encoder checkpoint (v0.3.0-beta.1 bring-up: PyTorch)",
    )
    p_train.add_argument(
        "--catalog",
        required=True,
        help="Path to corpus.tsv (tab-separated: name\\tkind\\tbody)",
    )
    p_train.add_argument(
        "--out", required=True, help="Output directory (checkpoint + manifest written here)"
    )
    p_train.add_argument(
        "--backend",
        choices=["python", "native"],
        default="python",
        help="'python' = PyTorch bring-up (available now); 'native' = MIND cdylib "
        "(NotImplementedError until mindc 0.3.0).",
    )
    p_train.add_argument("--base-model", default="BAAI/bge-small-en-v1.5")
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=2e-5)
    p_train.add_argument("--max-len", type=int, default=256)
    p_train.add_argument("--seed", type=int, default=1337)
    p_train.add_argument("--eval-frac", type=float, default=0.1)
    p_train.add_argument(
        "--smoke-test",
        action="store_true",
        help="500 pairs, 1 epoch — ~1 min run to validate the pipeline.",
    )
    p_train.set_defaults(func=cmd_train)

    p_quant = sub.add_parser(
        "quantize",
        help="Phase 6.2 offline FP32 → Q16.16 quantizer (produces route_table.q16.bin)",
    )
    p_quant.add_argument(
        "--catalog",
        required=True,
        help="Path to route_table.npy (float32, shape (N_rows, hidden_dim))",
    )
    p_quant.add_argument(
        "--input",
        default=None,
        help=(
            "Optional PyTorch checkpoint dir or file; hashed into the meta "
            "JSON only. Pass ``:none:`` or omit to skip."
        ),
    )
    p_quant.add_argument(
        "--output",
        default=None,
        help=("Output directory. Default: $MIND_NERVE_RUNTIME_DIR or ~/.cache/mind-nerve/q16/"),
    )
    p_quant.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Expected hidden dimension (default: catalog's column count).",
    )
    p_quant.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the meta JSON without writing any file.",
    )
    p_quant.set_defaults(func=cmd_quantize)

    p_quant_enc = sub.add_parser(
        "quantize-encoder",
        help="Offline FP32 → Q16.16 encoder-weights quantizer (encoder_weights.q16.bin)",
    )
    p_quant_enc.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the checkpoint directory containing model.safetensors.",
    )
    p_quant_enc.add_argument(
        "--output",
        default=None,
        help=(
            "Output directory. Default: $MIND_NERVE_RUNTIME_DIR or the user "
            "runtime dir (~/.local/share/mind-nerve/runtime)."
        ),
    )
    p_quant_enc.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the meta JSON without writing any file.",
    )
    p_quant_enc.set_defaults(func=cmd_quantize_encoder)

    p_attest = sub.add_parser(
        "attest", help="MindLLM cross-binding handshake (sign/verify BindingRecords)"
    )
    attest_sub = p_attest.add_subparsers(dest="attest_cmd", required=True)

    p_sign = attest_sub.add_parser("sign", help="Produce a 200-byte BindingRecord (hex)")
    p_sign.add_argument("--mind-nerve-hash", required=True, help="SHA-256 manifest aggregate (hex)")
    p_sign.add_argument("--mindllm-hash", required=True, help="MindLLM manifest aggregate (hex)")
    p_sign.add_argument("--nonce", required=True, help="Caller-supplied 32-byte nonce (hex)")
    p_sign.add_argument(
        "--private-key-hex",
        required=True,
        help="Ed25519 32-byte private key seed (hex). Treat as secret.",
    )
    p_sign.add_argument("--indent", action="store_true", help="Pretty-print JSON output")
    p_sign.set_defaults(func=cmd_attest_sign)

    p_verify = attest_sub.add_parser(
        "verify", help="Verify a 200-byte BindingRecord. Exit 0 if 'ok', else non-zero."
    )
    p_verify.add_argument("--record-hex", required=True, help="200-byte BindingRecord (hex)")
    p_verify.add_argument(
        "--pubkey-hex",
        default=None,
        help="Optional trust-anchor Ed25519 pubkey (hex). If supplied, the "
        "embedded signer_pubkey MUST match it or the result is UntrustedSigner.",
    )
    p_verify.add_argument("--indent", action="store_true", help="Pretty-print JSON output")
    p_verify.set_defaults(func=cmd_attest_verify)

    p_rb = sub.add_parser(
        "rollback",
        help="Restore a target CLI's config files from their .bak snapshots "
        "written by mind-nerve-install.",
    )
    p_rb.add_argument(
        "--target",
        required=True,
        help="Target name: claude, claude-code, claude-code-hook, claude-desktop, "
        "cursor, codex, gemini, vibe",
    )
    p_rb.set_defaults(func=cmd_rollback)

    p_watch = sub.add_parser(
        "watch", help="Daemon: poll one or more dirs for new skills (no inotify dep)"
    )
    p_watch.add_argument("dirs", nargs="+", help="Directories to watch")
    p_watch.add_argument("--source", default=None)
    p_watch.add_argument("--interval", type=float, default=5.0, help="Poll interval (sec)")
    p_watch.add_argument("--include-unknown", action="store_true")
    p_watch.set_defaults(func=cmd_watch)

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
