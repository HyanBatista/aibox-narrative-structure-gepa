"""Lazy Hugging Face model loading for CLI commands."""

from __future__ import annotations

from ..llm.huggingface import HuggingFaceLLM
from ..models import LLMProtocol


def load_model(
    model_id: str,
    device: str,
    temperature: float,
) -> LLMProtocol:
    resolved_device = None if device == "auto" else device
    return HuggingFaceLLM(
        model_id=model_id,
        device=resolved_device,
        temperature=temperature,
    )
