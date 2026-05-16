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
from .inference import route, load_default_runtime, precompute_routes


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
    out = precompute_routes(**kwargs)
    print(json.dumps(out, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mind-nerve")
    ap.add_argument("--version", action="version", version=f"mind-nerve {__version__}")
    ap.add_argument("--runtime-dir", default=None,
                    help="Override the runtime directory (default: $MIND_NERVE_RUNTIME_DIR or "
                         "/data/datasets/mind-nerve-catalog/phase1/v1.0)")

    sub = ap.add_subparsers(dest="cmd", required=True)

    p_route = sub.add_parser("route", help="Return top-K routes for a query")
    p_route.add_argument("query", nargs="?", help="Query text (or read from stdin if omitted)")
    p_route.add_argument("--top-k", type=int, default=5)
    p_route.add_argument("--json", action="store_true", default=True,
                         help="Emit JSON (default).")
    p_route.add_argument("--ids-only", action="store_true",
                         help="One route id per line (overrides --json)")
    p_route.set_defaults(func=cmd_route)

    p_info = sub.add_parser("info", help="Print runtime info as JSON")
    p_info.set_defaults(func=cmd_info)

    p_pre = sub.add_parser("precompute-routes",
                           help="(One-time) encode the catalog into route_table.npy")
    p_pre.set_defaults(func=cmd_precompute)

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
