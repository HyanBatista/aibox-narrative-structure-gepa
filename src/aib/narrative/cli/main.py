"""CLI entry point for aib-narrative."""

from __future__ import annotations

import argparse
import sys

from .commands import classify, datasets, evaluate, optimize
from .commands import run as run_cmd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aib-narrative",
        description="Rhetorical narrative structure classification and GEPA prompt optimization.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    classify.add_parser(subparsers)
    evaluate.add_parser(subparsers)
    optimize.add_parser(subparsers)
    run_cmd.add_parser(subparsers)
    datasets.add_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "datasets":
        return datasets.handle(args)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.error(f"unknown command: {args.command}")
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
