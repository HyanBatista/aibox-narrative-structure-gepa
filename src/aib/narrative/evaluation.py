"""Evaluation protocols and common multilabel metrics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .models import NarrativeExample, Prediction


def _empty_metrics() -> dict[str, float]:
    return {}


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Aggregate and optional per-example evaluation information."""

    score: float
    metrics: dict[str, float] = field(default_factory=_empty_metrics)
    feedback: tuple[str, ...] = ()


class EvaluatorProtocol(Protocol):
    """Evaluates task predictions against examples."""

    def evaluate(
        self, examples: Sequence[NarrativeExample], predictions: Sequence[Prediction]
    ) -> EvaluationResult: ...


def multilabel_f1(expected: Iterable[str], predicted: Iterable[str]) -> float:
    """Return example-level F1 for two label collections."""

    expected_set = set(expected)
    predicted_set = set(predicted)
    if not expected_set and not predicted_set:
        return 1.0
    if not expected_set or not predicted_set:
        return 0.0
    true_positives = len(expected_set & predicted_set)
    precision = true_positives / len(predicted_set)
    recall = true_positives / len(expected_set)
    return 2 * precision * recall / (precision + recall)


class F1Evaluator:
    """Macro-average example-level multilabel F1."""

    def evaluate(
        self, examples: Sequence[NarrativeExample], predictions: Sequence[Prediction]
    ) -> EvaluationResult:
        if len(examples) != len(predictions):
            raise ValueError("Examples and predictions must have the same length.")
        scores = [
            multilabel_f1(example.labels, prediction.labels)
            for example, prediction in zip(examples, predictions, strict=True)
        ]
        score = sum(scores) / len(scores) if scores else 0.0
        return EvaluationResult(score=score, metrics={"f1": score})
