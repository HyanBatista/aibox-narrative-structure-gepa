from collections.abc import Mapping, Sequence

import pytest

from aib.narrative.models import GenerationOptions, ModelResponse


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[Sequence[Mapping[str, str]]] = []

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        options: GenerationOptions | None = None,
    ) -> ModelResponse:
        self.calls.append(messages)
        return ModelResponse(self.response)


class RoutingFakeLLM:
    """Return different canned responses for task vs reflection calls."""

    def __init__(self, task_response: str, reflection_response: str) -> None:
        self.task_response = task_response
        self.reflection_response = reflection_response
        self.calls: list[Sequence[Mapping[str, str]]] = []

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        options: GenerationOptions | None = None,
    ) -> ModelResponse:
        self.calls.append(messages)
        content = " ".join(str(message["content"]) for message in messages)
        if "instruction prompt" in content.lower():
            return ModelResponse(self.reflection_response)
        return ModelResponse(self.task_response)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM('{"labels": ["resolution"], "scores": {"resolution": 0.9}, "evidence": []}')
