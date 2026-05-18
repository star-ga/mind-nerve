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
        help="Override the runtime directory (default: $MIND_NERVE_RUNTIME_DIR or "
        "/data/datasets/mind-nerve-catalog/phase1/v1.1-oss)",
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
