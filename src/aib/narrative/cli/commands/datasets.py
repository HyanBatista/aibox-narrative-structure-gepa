"""Dataset registry CLI commands."""

from __future__ import annotations

import argparse
from typing import Any

from ...data.registry import list_datasets


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("datasets", help="Dataset registry commands")
    dataset_subparsers = parser.add_subparsers(dest="datasets_command", required=True)
    list_parser = dataset_subparsers.add_parser("list", help="List available datasets")
    list_parser.set_defaults(handler=handle_list)


def handle_list(_args: argparse.Namespace) -> int:
    print(f"{'NAME':<18}{'TRAIN':<7}{'VAL':<6}CATEGORIES")
    for dataset in list_datasets():
        category_names = ", ".join(category.name for category in dataset.categories)
        print(f"{dataset.name:<18}{len(dataset.train):<7}{len(dataset.val):<6}{category_names}")
    return 0


def handle(args: argparse.Namespace) -> int:
    handler = getattr(args, "handler", None)
    if handler is None:
        return 2
    return handler(args)
