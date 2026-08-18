"""Classify narrative text into rhetorical categories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ...data.registry import get_categories
from ...tasks import RhetoricalCategoryTask
from .._model import load_model


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "classify", help="Assign rhetorical categories to narrative text"
    )
    parser.add_argument("text", nargs="?", help="Narrative text to classify")
    parser.add_argument("--file", help="Read narrative text from a file")
    _add_shared_flags(parser)
    parser.add_argument(
        "--categories",
        default="freytag-sample",
        help="Category set from the dataset registry (default: freytag-sample)",
    )
    parser.set_defaults(handler=handle)


def _add_shared_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="HF model id")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--prompt-file", help="Prompt file to use")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)


def handle(args: argparse.Namespace) -> int:
    text = args.text
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8").strip()
    if not text:
        print("error: provide narrative text as an argument or via --file", file=sys.stderr)
        return 2

    prompt = None
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()

    categories = get_categories(args.categories)
    task = RhetoricalCategoryTask(categories, prompt=prompt)
    model = load_model(args.model, args.device, args.temperature)
    prediction = task.predict(model, text, prompt)

    print(
        json.dumps(
            {
                "labels": list(prediction.labels),
                "scores": dict(prediction.scores),
                "evidence": list(prediction.evidence),
            },
            indent=2,
        )
    )
    return 0
