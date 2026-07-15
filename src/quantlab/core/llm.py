"""Thin, budget-aware wrapper around the OpenAI Chat Completions API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str


class LLM:
    """Minimal wrapper. Real implementation adds retries, JSON-mode, and cost tracking."""

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.2) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is not installed. `pip install openai`."
            ) from exc
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model
        self.temperature = temperature

    def complete(self, system: str, user: str, **kwargs: Any) -> LLMResponse:
        rsp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        choice = rsp.choices[0].message.content or ""
        usage = rsp.usage
        return LLMResponse(
            text=choice,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            model=self.model,
        )
