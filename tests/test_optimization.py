from __future__ import annotations

from aib.narrative import (
    Category,
    DiagnosticF1Evaluator,
    GEPAOptimizer,
    NarrativeExample,
    OptimizationConfig,
    Prediction,
)
from aib.narrative.optimization import _NarrativeAdapter, looks_like_classification_json
from aib.narrative.tasks import RhetoricalCategoryTask
from conftest import FakeLLM, RoutingFakeLLM


def test_diagnostic_f1_evaluator_returns_label_diff_feedback() -> None:
    evaluator = DiagnosticF1Evaluator()
    examples = [NarrativeExample("text", labels=("climax",))]
    predictions = [Prediction(labels=("rising_action",))]

    result = evaluator.evaluate(examples, predictions)

    assert result.score == 0.0
    assert len(result.feedback) == 1
    assert "Expected labels: ['climax']" in result.feedback[0]
    assert "Predicted: ['rising_action']" in result.feedback[0]


def test_looks_like_classification_json_detects_task_outputs() -> None:
    assert looks_like_classification_json('{"labels": [], "scores": {}, "evidence": []}')
    assert not looks_like_classification_json(
        "You classify narrative text into rhetorical categories. Respond with JSON only."
    )


def test_adapter_rejects_json_instruction_proposals() -> None:
    task = RhetoricalCategoryTask(
        [
            Category("resolution", "Conflict resolved."),
            Category("climax", "Peak tension."),
        ]
    )
    adapter = _NarrativeAdapter(
        task,
        FakeLLM('{"labels": ["resolution"], "scores": {}, "evidence": []}'),
        DiagnosticF1Evaluator(),
        reflection_model=RoutingFakeLLM(
            task_response='{"labels": ["resolution"], "scores": {}, "evidence": []}',
            reflection_response='{"labels": [], "scores": {}, "evidence": []}',
        ),
        reflection_template="Current:\n```\n<curr_param>\n```\nExamples:\n```\n<side_info>\n```",
        reflection_temperature=0.3,
    )
    reflective_dataset = {
        "prompt": [
            {
                "Inputs": {"narrative": "Peace returned.", "expected_categories": ["resolution"]},
                "Generated Outputs": {"classifier_json_prediction": {"labels": ["climax"]}},
                "Feedback": "wrong",
            }
        ]
    }

    new_texts = adapter._propose_new_texts({"prompt": task.prompt}, reflective_dataset, ["prompt"])

    assert new_texts == {}


def test_gepa_optimizer_returns_best_prompt() -> None:
    task_response = '{"labels": ["resolution"], "scores": {"resolution": 0.9}, "evidence": ["resolved"]}'
    reflection_response = (
        "```\n"
        "You classify narrative text into rhetorical story-structure categories. "
        "Read the narrative carefully, choose the best matching categories, and "
        "respond with a single JSON object containing labels, scores, and evidence.\n"
        "```"
    )
    model = RoutingFakeLLM(task_response, reflection_response)
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
    assert not looks_like_classification_json(result.best_prompt)
