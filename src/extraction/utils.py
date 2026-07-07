"""Shared utilities for activation extraction scripts."""

from typing import Any

THINKING_MODE_SUFFIXES = ("_low", "_medium", "_high", "_reasoning")
THINKING_SEPARATOR = "assistantfinal"


def strip_thinking_suffix(model_name: str) -> str:
    """Strip thinking-mode suffix to get the base HF model id."""
    for suffix in THINKING_MODE_SUFFIXES:
        if model_name.endswith(suffix):
            return model_name[: -len(suffix)]
    return model_name


def generation_to_text(generation: Any) -> str:
    """Convert generated solution payloads (str/dict/list) to plain text."""
    if generation is None:
        return ""
    if isinstance(generation, str):
        return generation
    if isinstance(generation, dict):
        for key in ("text", "completion", "response", "output_text", "answer"):
            if key in generation:
                return generation_to_text(generation[key])
        if "content" in generation:
            content = generation["content"]
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [generation_to_text(item) for item in content]
                return "".join(part for part in parts if part)
            return generation_to_text(content)
        return str(generation)
    if isinstance(generation, list):
        parts = [generation_to_text(item) for item in generation]
        return "".join(part for part in parts if part)
    return str(generation)
