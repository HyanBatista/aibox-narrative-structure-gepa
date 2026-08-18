"""Task/system abstractions and the rhetorical-category task."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any, Protocol, cast

from .models import Category, GenerationOptions, LLMProtocol, NarrativeExample, Prediction


class TaskProtocol(Protocol):
    """Executable system contract used by the experiment runner."""

    name: str

    def fit(self, model: object, examples: Sequence[NarrativeExample]) -> None: ...

    def run(self, model: object, example: NarrativeExample) -> Prediction: ...


class RhetoricalCategoryTask:
    """Classify narrative text into a configurable set of categories."""

    name = "rhetorical-category"

    def __init__(
        self,
        categories: Sequence[Category],
        prompt: str | None = None,
        options: GenerationOptions | None = None,
    ) -> None:
        if not categories:
            raise ValueError("At least one category is required.")
        names = [category.name for category in categories]
        if len(set(names)) != len(names):
            raise ValueError("Category names must be unique.")
        self.categories = tuple(categories)
        self.prompt = prompt or self._default_prompt()
        self.options = options

    def fit(self, model: object, examples: Sequence[NarrativeExample]) -> None:
        """LLM-backed tasks do not require fitting."""

    def render_prompt(self, text: str, prompt: str | None = None) -> str:
        instruction = prompt or self.prompt
        category_text = "\n".join(
            f"- {category.name}: {category.description}" for category in self.categories
        )
        return f"{instruction}\n\nCategories:\n{category_text}\n\nNarrative:\n{text}"

    def run(self, model: object, example: NarrativeExample) -> Prediction:
        if not isinstance(model, LLMProtocol):
            raise TypeError("RhetoricalCategoryTask requires a model implementing LLMProtocol.")
        return self.predict(model, example.text)

    def predict(self, model: LLMProtocol, text: str, prompt: str | None = None) -> Prediction:
        response = model.generate(
            [{"role": "user", "content": self.render_prompt(text, prompt)}], self.options
        )
        return self.parse_prediction(response.text)

    def parse_prediction(self, raw_text: str) -> Prediction:
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip())
        try:
            loaded: Any = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM response is not valid JSON.") from exc
        if not isinstance(loaded, dict):
            raise ValueError("LLM response must be an object containing a labels list.")
        payload = cast(dict[str, Any], loaded)
        if not isinstance(payload.get("labels"), list):
            raise ValueError("LLM response must be an object containing a labels list.")
        known = {category.name for category in self.categories}
        raw_labels = cast(list[Any], payload["labels"])
        labels = tuple(label for label in raw_labels if isinstance(label, str))
        if len(labels) != len(raw_labels) or not all(label in known for label in labels):
            raise ValueError("LLM response contains an unknown or invalid category label.")
        raw_scores = payload.get("scores", {})
        if not isinstance(raw_scores, dict):
            raise ValueError("Prediction scores must be an object.")
        scores = {str(key): float(value) for key, value in cast(dict[Any, Any], raw_scores).items()}
        evidence_value = payload.get("evidence", [])
        if not isinstance(evidence_value, list) or not all(
            isinstance(item, str) for item in cast(list[Any], evidence_value)
        ):
            raise ValueError("Prediction evidence must be a list of strings.")
        evidence = tuple(cast(str, item) for item in cast(list[Any], evidence_value))
        return Prediction(labels=labels, scores=scores, evidence=evidence, raw_text=raw_text)

    def _default_prompt(self) -> str:
        return (
            "Classify the narrative into zero or more rhetorical categories. "
            "Return only JSON with labels, scores, and evidence: "
            '{"labels": ["category"], "scores": {"category": 0.0}, "evidence": ["..."]}'
        )


class EstimatorRhetoricalCategoryTask:
    """Adapt a generic estimator to the task protocol."""

    name = "rhetorical-category-estimator"

    def __init__(self, feature_extractor: Any) -> None:
        self.feature_extractor = feature_extractor

    def fit(self, model: object, examples: Sequence[NarrativeExample]) -> None:
        features = [self.feature_extractor(example) for example in examples]
        targets = [example.labels for example in examples]
        from .models import EstimatorProtocol

        if not isinstance(model, EstimatorProtocol):
            raise TypeError("EstimatorRhetoricalCategoryTask requires an EstimatorProtocol model.")
        model.fit(features, targets)

    def run(self, model: object, example: NarrativeExample) -> Prediction:
        features = [self.feature_extractor(example)]
        from .models import EstimatorProtocol

        if not isinstance(model, EstimatorProtocol):
            raise TypeError("EstimatorRhetoricalCategoryTask requires an EstimatorProtocol model.")
        raw_labels = model.predict(features)[0]
        if isinstance(raw_labels, str):
            labels = (raw_labels,)
        else:
            labels = tuple(str(label) for label in raw_labels)
        return Prediction(labels=labels)
