"""Evaluate rhetorical classification on a labeled dataset."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from ...pipelines.experiment import evaluate_dataset
from .._model import load_model


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "evaluate", help="Benchmark rhetorical classification on a labeled dataset"
    )
    parser.add_argument("--dataset", required=True, help="Dataset name from registry")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--run-dir", default="runs/evaluate")
    _add_shared_flags(parser)
    parser.set_defaults(handler=handle)


def _add_shared_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="HF model id")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--prompt-file", help="Prompt file to use")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)


def handle(args: argparse.Namespace) -> int:
    model = load_model(args.model, args.device, args.temperature)
    try:
        result = evaluate_dataset(
            model,
            args.dataset,
            split=args.split,
            prompt_file=args.prompt_file,
            run_dir=args.run_dir,
            seed=args.seed,
            model_id=args.model,
            temperature=args.temperature,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"F1: {result.score:.4f}")
    print(f"Run ID: {result.run_id}")
    print(f"Artifacts: {result.run_path}")
    return 0
