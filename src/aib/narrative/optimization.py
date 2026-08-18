"""GEPA prompt optimization integration."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from .evaluation import EvaluatorProtocol
from .models import Category, GenerationOptions, LLMProtocol, NarrativeExample
from .tasks import RhetoricalCategoryTask

_REFLECTION_RETRY_HINT = (
    "That response is a JSON classification result, not an instruction prompt. "
    "Write natural-language instructions that tell a classifier how to analyze "
    "narratives and choose rhetorical categories. Put the full instruction "
    "prompt inside ``` blocks only."
)


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    max_metric_calls: int = 100
    reflection_minibatch_size: int = 3
    seed: int = 0
    run_dir: str | None = None
    reflection_temperature: float = 0.3


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    best_prompt: str
    best_score: float
    raw_result: Any = field(repr=False, default=None)


def _build_reflection_template(categories: Sequence[Category]) -> str:
    category_lines = "\n".join(f"- {category.name}: {category.description}" for category in categories)
    return f"""You are revising the INSTRUCTION PROMPT for a narrative rhetorical-category classifier.

The current instruction prompt given to the classifier is:
```
<curr_param>
```

Below are examples of classifier runs on narrative texts. Each example includes:
- the narrative and gold-standard categories
- the classifier's JSON prediction (model OUTPUT — not the prompt to edit)
- feedback on prediction mistakes

```
<side_info>
```

Write a NEW instruction prompt for the narrative rhetorical-category classifier.

Requirements:
- Output natural-language instructions only. Do NOT output a JSON classification object.
- The instruction must explain how to read narratives and choose rhetorical categories.
- The instruction may require the classifier to answer with JSON containing labels, scores, and evidence, but the instruction itself must be prose.
- Include clear guidance for these categories:
{category_lines}
- Put the complete new instruction prompt inside ``` blocks and nowhere else."""


def looks_like_classification_json(text: str) -> bool:
    """Return True when text looks like a task JSON output, not an instruction prompt."""

    stripped = text.strip()
    if not stripped.startswith("{"):
        return False
    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError:
        return False
    return isinstance(loaded, dict) and ("labels" in loaded or "scores" in loaded)


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

        reflection_template = _build_reflection_template(task.categories)
        adapter = _NarrativeAdapter(
            task,
            model,
            evaluator,
            reflection_model=self.reflection_model,
            reflection_template=reflection_template,
            reflection_temperature=self.config.reflection_temperature,
        )
        gepa_api: Any = importlib.import_module("gepa.api")
        result = gepa_api.optimize(
            seed_candidate={"prompt": task.prompt},
            trainset=list(trainset),
            valset=list(valset) if valset is not None else None,
            adapter=cast(Any, adapter),
            reflection_lm=_ReflectionCallable(
                self.reflection_model,
                temperature=self.config.reflection_temperature,
            ),
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
    def __init__(self, model: LLMProtocol, *, temperature: float | None = None) -> None:
        self.model = model
        self.temperature = temperature

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [
                {"role": str(item["role"]), "content": str(item["content"])} for item in prompt
            ]
        options = (
            GenerationOptions(temperature=self.temperature)
            if self.temperature is not None
            else None
        )
        return self.model.generate(messages, options).text


class _NarrativeAdapter:
    propose_new_texts: Any = None

    def __init__(
        self,
        task: RhetoricalCategoryTask,
        model: LLMProtocol,
        evaluator: EvaluatorProtocol,
        *,
        reflection_model: LLMProtocol,
        reflection_template: str,
        reflection_temperature: float,
    ) -> None:
        self.task = task
        self.model = model
        self.evaluator = evaluator
        self._reflection_model = reflection_model
        self._reflection_template = reflection_template
        self._reflection_temperature = reflection_temperature
        self.propose_new_texts = self._propose_new_texts

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
                    {
                        "input": example.text,
                        "expected": list(example.labels),
                        "output": output,
                        "feedback": feedback,
                        "score": score,
                    }
                )
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self, candidate: dict[str, str], eval_batch: Any, components_to_update: list[str]
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        trajectories = cast(list[dict[str, Any]] | None, eval_batch.trajectories) or []
        records: list[Mapping[str, Any]] = []
        for traj in trajectories:
            records.append(
                {
                    "Inputs": {
                        "narrative": traj["input"],
                        "expected_categories": traj["expected"],
                    },
                    "Generated Outputs": {
                        "classifier_json_prediction": traj["output"],
                        "note": (
                            "This JSON is the classifier's answer for the narrative, "
                            "not the instruction prompt being optimized."
                        ),
                    },
                    "Feedback": traj["feedback"],
                }
            )
        return {component: records for component in components_to_update}

    def _propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        from gepa.strategies.instruction_proposal import InstructionProposalSignature

        reflection = _ReflectionCallable(
            self._reflection_model,
            temperature=self._reflection_temperature,
        )
        new_texts: dict[str, str] = {}
        for name in components_to_update:
            dataset = reflective_dataset.get(name)
            if not dataset:
                continue
            prompt = InstructionProposalSignature.prompt_renderer(
                {
                    "current_instruction_doc": candidate[name],
                    "dataset_with_feedback": dataset,
                    "prompt_template": self._reflection_template,
                }
            )
            messages: list[dict[str, str]] = (
                [{"role": "user", "content": prompt}]
                if isinstance(prompt, str)
                else [{"role": str(item["role"]), "content": str(item["content"])} for item in prompt]
            )
            for attempt in range(3):
                raw_output = reflection(messages)
                new_instruction = InstructionProposalSignature.output_extractor(raw_output.strip())[
                    "new_instruction"
                ]
                if (
                    not looks_like_classification_json(new_instruction)
                    and len(new_instruction.strip()) >= 40
                ):
                    new_texts[name] = new_instruction
                    break
                if attempt < 2:
                    messages = [
                        *messages,
                        {"role": "assistant", "content": raw_output},
                        {"role": "user", "content": _REFLECTION_RETRY_HINT},
                    ]
        return new_texts
