"""Typed data and model protocols used by narrative experiments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, Self, runtime_checkable


def _empty_metadata() -> dict[str, Any]:
    return {}


def _empty_usage() -> dict[str, int]:
    return {}


@dataclass(frozen=True, slots=True)
class Category:
    """A rhetorical category available to a task."""

    name: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Category name cannot be empty.")


@dataclass(frozen=True, slots=True)
class NarrativeExample:
    """One narrative input and its optional gold labels."""

    text: str
    labels: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Narrative text cannot be empty.")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("Narrative labels cannot contain duplicates.")


@dataclass(frozen=True, slots=True)
class Dataset:
    """A deterministic collection of narrative examples."""

    examples: tuple[NarrativeExample, ...]
    name: str = "dataset"

    @classmethod
    def from_examples(cls, examples: Sequence[NarrativeExample], name: str = "dataset") -> Self:
        if not examples:
            raise ValueError("Dataset must contain at least one example.")
        return cls(tuple(examples), name)

    @property
    def fingerprint(self) -> str:
        payload = [asdict(example) for example in self.examples]
        encoded = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    """Provider-neutral generation settings."""

    temperature: float | None = None
    max_tokens: int | None = None
    response_format: str | None = None


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Text returned by an LLM and optional provider metadata."""

    text: str
    usage: Mapping[str, int] = field(default_factory=_empty_usage)
    metadata: Mapping[str, Any] = field(default_factory=_empty_metadata)


@runtime_checkable
class LLMProtocol(Protocol):
    """Synchronous provider-neutral LLM contract."""

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        options: GenerationOptions | None = None,
    ) -> ModelResponse: ...


@runtime_checkable
class EstimatorProtocol(Protocol):
    """Minimal scikit-learn-like estimator contract."""

    def fit(self, features: Sequence[Any], targets: Sequence[Any]) -> Any: ...

    def predict(self, features: Sequence[Any]) -> Sequence[Any]: ...


@dataclass(frozen=True, slots=True)
class Prediction:
    """Validated rhetorical-category prediction."""

    labels: tuple[str, ...]
    scores: Mapping[str, float] = field(default_factory=lambda: {})
    evidence: tuple[str, ...] = ()
    raw_text: str | None = None

    def __post_init__(self) -> None:
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("Prediction labels cannot contain duplicates.")
        if any(not 0 <= value <= 1 for value in self.scores.values()):
            raise ValueError("Prediction scores must be between 0 and 1.")
