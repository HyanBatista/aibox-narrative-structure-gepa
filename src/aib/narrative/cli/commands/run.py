"""Run the full rhetorical classification experiment pipeline."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from ...data.registry import load_dataset
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
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)


def handle(args: argparse.Namespace) -> int:
    model = load_model(args.model, args.device, args.temperature)
    try:
        result = run_full_experiment(
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

    dataset = load_dataset(args.dataset)
    print("=== Rhetorical Classification Experiment ===")
    train_count = len(dataset.train)
    val_count = len(dataset.val)
    print(f"Dataset:      {result.dataset_name} ({train_count} train / {val_count} val)")
    if result.baseline_f1 is not None:
        print(f"Baseline F1:  {result.baseline_f1:.2f}  (default prompt)")
    print(f"Optimized F1: {result.optimized_f1:.2f}  (GEPA-evolved prompt)")
    if result.baseline_f1 is not None:
        print(f"Delta:        {result.optimized_f1 - result.baseline_f1:+.2f}")
    print()
    print("Best prompt:")
    print(f"  {result.best_prompt[:120]}{'...' if len(result.best_prompt) > 120 else ''}")
    print()
    print("Artifacts:")
    if result.baseline_run_id is not None:
        print(f"  {args.run_dir}/baseline/baseline/{result.baseline_run_id}/")
    print(f"  {result.optimize_run_dir}/")
    print(f"  {args.run_dir}/optimized/optimized/{result.optimized_run_id}/")
    return 0
