"""Pipelines for rhetorical classification experiments."""

from .experiment import (
    EvaluateResult,
    ExperimentPipelineResult,
    evaluate_dataset,
    optimize_prompt,
    run_full_experiment,
)

__all__ = [
    "EvaluateResult",
    "ExperimentPipelineResult",
    "evaluate_dataset",
    "optimize_prompt",
    "run_full_experiment",
]
