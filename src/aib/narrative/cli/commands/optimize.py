"""Optimize rhetorical classification prompts with GEPA."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from ...pipelines.experiment import optimize_prompt
from .._model import load_model


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "optimize", help="Evolve the rhetorical classification prompt with GEPA"
    )
    parser.add_argument("--dataset", required=True, help="Dataset name from registry")
    parser.add_argument("--run-dir", default="runs/optimize")
    parser.add_argument("--max-metric-calls", type=int, default=25)
    parser.add_argument("--reflection-minibatch-size", type=int, default=2)
    _add_shared_flags(parser)
    parser.set_defaults(handler=handle)


def _add_shared_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="HF model id")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--prompt-file", help="Seed prompt file")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)


def handle(args: argparse.Namespace) -> int:
    model = load_model(args.model, args.device, args.temperature)
    try:
        result = optimize_prompt(
            model,
            model,
            args.dataset,
            prompt_file=args.prompt_file,
            run_dir=args.run_dir,
            max_metric_calls=args.max_metric_calls,
            reflection_minibatch_size=args.reflection_minibatch_size,
            seed=args.seed,
            temperature=args.temperature,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Best validation F1: {result.best_score:.4f}")
    print(f"Best prompt written to: {args.run_dir}/best_prompt.txt")
    return 0
