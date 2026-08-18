"""Console and artifact logging for experiment pipelines."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def log_section(title: str) -> None:
    print(f"\n--- {title} ---")


def log_prompt(prompt: str) -> None:
    print("Prompt:")
    lines = prompt.splitlines() or [prompt]
    for line in lines:
        print(f"  {line}")


def log_metrics(score: float, metrics: Mapping[str, float] | None = None) -> None:
    print(f"F1: {score:.4f}")
    if metrics:
        for name, value in sorted(metrics.items()):
            if name != "f1":
                print(f"  {name}: {value:.4f}")


def log_run_location(run_path: Path) -> None:
    print(f"Artifacts: {run_path}")


def write_experiment_summary(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
