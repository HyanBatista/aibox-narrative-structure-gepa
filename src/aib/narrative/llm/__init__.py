"""LLM backend implementations."""

from .huggingface import HuggingFaceLLM
from .pytorch import detect_device

__all__ = ["HuggingFaceLLM", "detect_device"]
