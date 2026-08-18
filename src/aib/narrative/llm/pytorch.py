# pyright: reportMissingImports=false, reportUnknownMemberType=false
"""Shared PyTorch utilities for local LLM backends."""

from __future__ import annotations


def detect_device(device: str | None = None) -> str:
    """Return cuda, mps, or cpu based on PyTorch availability."""

    if device is not None and device != "auto":
        return device
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is not installed. Run: uv sync --group huggingface"
        ) from exc
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
