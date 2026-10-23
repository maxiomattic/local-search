"""Command-line entry point: `localsearch index|search images|docs ...`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import documents, images

KINDS = {"images": images, "docs": documents}


def _progress(done: int, total: int) -> None:
    print(f"\r{done}/{total}", end="", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="localsearch", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    idx = sub.add_parser("index", help="index a directory (incremental)")
    idx.add_argument("kind", choices=KINDS)
    idx.add_argument("directory", type=Path)
    idx.add_argument("--force", action="store_true", help="re-embed everything")

    q = sub.add_parser("search", help="search with a plain-English query")
    q.add_argument("kind", choices=KINDS)
    q.add_argument("query")
    q.add_argument("-n", "--limit", type=int, default=5)

    args = parser.parse_args(argv)
    module = KINDS[args.kind]

    if args.command == "index":
        if not args.directory.is_dir():
            parser.error(f"not a directory: {args.directory}")
        count = module.index(args.directory, force=args.force, progress=_progress)
        print(f"\nindexed {count} {'images' if args.kind == 'images' else 'chunks'}", file=sys.stderr)
        return 0

    for hit in module.search(args.query, args.limit):
        p = hit["payload"]
        where = f" (chunk {p['chunk']}/{p['chunks']})" if "chunk" in p else ""
        print(f"{hit['score']:.3f}  {p['path']}{where}")
        if "text" in p:
            print(f"       {p['text'][:300]}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
