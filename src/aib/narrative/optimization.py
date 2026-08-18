"""GEPA prompt optimization integration."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from .evaluation import EvaluatorProtocol
from .models import LLMProtocol, NarrativeExample
from .tasks import RhetoricalCategoryTask


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    max_metric_calls: int = 100
    reflection_minibatch_size: int = 3
    seed: int = 0
    run_dir: str | None = None


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    best_prompt: str
    best_score: float
    raw_result: Any = field(repr=False, default=None)


class GEPAOptimizer:
    """Optimize a rhetorical task prompt using a provider-neutral LLM."""

    def __init__(
        self, reflection_model: LLMProtocol, config: OptimizationConfig | None = None
    ) -> None:
        self.reflection_model = reflection_model
        self.config = config or OptimizationConfig()

    def optimize(
        self,
        task: RhetoricalCategoryTask,
        model: LLMProtocol,
        trainset: Sequence[NarrativeExample],
        evaluator: EvaluatorProtocol,
        valset: Sequence[NarrativeExample] | None = None,
    ) -> OptimizationResult:
        """Run GEPA against the task's prompt component."""

        adapter = _NarrativeAdapter(task, model, evaluator)
        gepa_api: Any = importlib.import_module("gepa.api")
        result = gepa_api.optimize(
            seed_candidate={"prompt": task.prompt},
            trainset=list(trainset),
            valset=list(valset) if valset is not None else None,
            adapter=cast(Any, adapter),
            reflection_lm=_ReflectionCallable(self.reflection_model),
            max_metric_calls=self.config.max_metric_calls,
            reflection_minibatch_size=self.config.reflection_minibatch_size,
            seed=self.config.seed,
            run_dir=self.config.run_dir,
        )
        best_candidate = cast(dict[str, str], result.best_candidate)
        best_prompt = best_candidate["prompt"]
        best_idx = int(result.best_idx)
        aggregate_scores = cast(Sequence[float], result.val_aggregate_scores)
        best_score = aggregate_scores[best_idx]
        return OptimizationResult(best_prompt=best_prompt, best_score=best_score, raw_result=result)


class _ReflectionCallable:
    def __init__(self, model: LLMProtocol) -> None:
        self.model = model

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [
                {"role": str(item["role"]), "content": str(item["content"])} for item in prompt
            ]
        return self.model.generate(messages).text


class _NarrativeAdapter:
    propose_new_texts = None

    def __init__(
        self, task: RhetoricalCategoryTask, model: LLMProtocol, evaluator: EvaluatorProtocol
    ) -> None:
        self.task = task
        self.model = model
        self.evaluator = evaluator

    def evaluate(
        self, batch: list[NarrativeExample], candidate: dict[str, str], capture_traces: bool = False
    ) -> Any:
        from gepa.core.adapter import EvaluationBatch

        outputs: list[dict[str, Any]] = []
        scores: list[float] = []
        trajectories: list[dict[str, Any]] | None = [] if capture_traces else None
        for example in batch:
            try:
                prediction = self.task.predict(self.model, example.text, candidate["prompt"])
                evaluation = self.evaluator.evaluate([example], [prediction])
                score = evaluation.score
                feedback = " ".join(evaluation.feedback)
                output = {"labels": list(prediction.labels), "raw_text": prediction.raw_text}
            except Exception as exc:
                score = 0.0
                feedback = f"Prediction failed: {type(exc).__name__}: {exc}"
                output = {"error": feedback}
            outputs.append(output)
            scores.append(score)
            if trajectories is not None:
                trajectories.append(
                    {"input": example.text, "output": output, "feedback": feedback, "score": score}
                )
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self, candidate: dict[str, str], eval_batch: Any, components_to_update: list[str]
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        trajectories = cast(list[dict[str, Any]] | None, eval_batch.trajectories) or []
        return {component: list(trajectories) for component in components_to_update}
