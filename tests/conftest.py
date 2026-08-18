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


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM('{"labels": ["resolution"], "scores": {"resolution": 0.9}, "evidence": []}')
