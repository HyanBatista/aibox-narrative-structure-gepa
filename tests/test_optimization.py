from __future__ import annotations

from aib.narrative import (
    Category,
    DiagnosticF1Evaluator,
    GEPAOptimizer,
    NarrativeExample,
    OptimizationConfig,
    Prediction,
)
from aib.narrative.tasks import RhetoricalCategoryTask
from conftest import FakeLLM


def test_diagnostic_f1_evaluator_returns_label_diff_feedback() -> None:
    evaluator = DiagnosticF1Evaluator()
    examples = [NarrativeExample("text", labels=("climax",))]
    predictions = [Prediction(labels=("rising_action",))]

    result = evaluator.evaluate(examples, predictions)

    assert result.score == 0.0
    assert len(result.feedback) == 1
    assert "Expected labels: ['climax']" in result.feedback[0]
    assert "Predicted: ['rising_action']" in result.feedback[0]


def test_gepa_optimizer_returns_best_prompt() -> None:
    response = '{"labels": ["resolution"], "scores": {"resolution": 0.9}, "evidence": ["resolved"]}'
    model = FakeLLM(response)
    task = RhetoricalCategoryTask(
        [
            Category("resolution", "Conflict resolved."),
            Category("climax", "Peak tension."),
        ]
    )
    trainset = [
        NarrativeExample("Peace returned.", labels=("resolution",)),
        NarrativeExample("The battle peaked.", labels=("climax",)),
    ]
    valset = [NarrativeExample("They went home.", labels=("resolution",))]

    result = GEPAOptimizer(
        reflection_model=model,
        config=OptimizationConfig(max_metric_calls=3, reflection_minibatch_size=1, seed=0),
    ).optimize(
        task=task,
        model=model,
        trainset=trainset,
        valset=valset,
        evaluator=DiagnosticF1Evaluator(),
    )

    assert result.best_prompt
    assert isinstance(result.best_score, float)
