# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportAttributeAccessIssue=false
"""Hugging Face transformers backend for LLMProtocol."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from ..models import GenerationOptions, ModelResponse
from .pytorch import detect_device


def _strip_markdown_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)


class HuggingFaceLLM:
    """Local Hugging Face causal LM implementing LLMProtocol."""

    def __init__(
        self,
        model_id: str,
        device: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "huggingface dependencies not installed. Run: uv sync --group huggingface"
            ) from exc

        resolved_device = detect_device(device)
        dtype = torch.float16 if resolved_device in {"cuda", "mps"} else torch.float32

        self.model_id = model_id
        self.device = resolved_device
        self.max_new_tokens = max_new_tokens
        self.default_temperature = temperature

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        self._model.to(resolved_device)
        self._model.eval()
        self._torch = torch

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        options: GenerationOptions | None = None,
    ) -> ModelResponse:
        temperature = self.default_temperature
        max_new_tokens = self.max_new_tokens
        if options is not None:
            if options.temperature is not None:
                temperature = options.temperature
            if options.max_tokens is not None:
                max_new_tokens = options.max_tokens

        chat_messages = [{"role": str(m["role"]), "content": str(m["content"])} for m in messages]
        prompt_text = self._tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self.device)

        generate_kwargs: dict[str, object] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature

        with self._torch.no_grad():
            output_ids = self._model.generate(**inputs, **generate_kwargs)

        new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return ModelResponse(text=_strip_markdown_fences(text))
