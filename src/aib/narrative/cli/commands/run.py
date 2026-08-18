"""Run the full rhetorical classification experiment pipeline."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from ...pipelines.experiment import run_full_experiment
from .._model import load_model


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "run", help="Run baseline, GEPA optimization, and validation in one pipeline"
    )
    parser.add_argument("--dataset", required=True, help="Dataset name from registry")
    parser.add_argument("--run-dir", default="runs/experiment")
    parser.add_argument("--max-metric-calls", type=int, default=25)
    parser.add_argument("--reflection-minibatch-size", type=int, default=2)
    parser.add_argument("--skip-baseline", action="store_true")
    _add_shared_flags(parser)
    parser.set_defaults(handler=handle)


def _add_shared_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct", help="HF model id")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--prompt-file", help="Seed prompt file for optimization")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)


def handle(args: argparse.Namespace) -> int:
    model = load_model(args.model, args.device, args.temperature)
    try:
        run_full_experiment(
            model,
            model,
            args.dataset,
            prompt_file=args.prompt_file,
            run_dir=args.run_dir,
            max_metric_calls=args.max_metric_calls,
            reflection_minibatch_size=args.reflection_minibatch_size,
            seed=args.seed,
            model_id=args.model,
            temperature=args.temperature,
            skip_baseline=args.skip_baseline,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
