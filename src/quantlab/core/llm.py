"""Budget-aware wrapper around the OpenAI Chat Completions API.

Every agent that wants to call an LLM goes through this module. It keeps
three concerns separate from the agents themselves: whether a call is even
possible in the current environment, how much a call costs, and how JSON
output is parsed and validated. Agents are expected to treat any failure
here (missing key, missing package, budget exceeded, malformed JSON) as a
signal to fall back to their deterministic behaviour rather than crash the
pipeline.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
from dataclasses import dataclass
from typing import Any

from quantlab.core.state import ResearchState

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)

_PRICING_USD_PER_1K_TOKENS: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.005, 0.015),
    "gpt-4o-2024-08-06": (0.005, 0.015),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-3.5-turbo": (0.0005, 0.0015),
}
_DEFAULT_PRICING: tuple[float, float] = (0.005, 0.015)


class BudgetExceededError(RuntimeError):
    """Raised when a completion would push a run past its configured budget."""


class LLMUnavailableError(RuntimeError):
    """Raised when no LLM can be reached: missing key, missing package, or API error."""


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    cost_usd: float


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate the USD cost of a completion using a static local pricing table.

    The table is an approximation for local budget tracking only; it is not
    fetched from a billing API and may drift from the provider's current
    prices.
    """
    prompt_rate, completion_rate = _PRICING_USD_PER_1K_TOKENS.get(
        model, _DEFAULT_PRICING
    )
    return (prompt_tokens / 1000.0) * prompt_rate + (
        completion_tokens / 1000.0
    ) * completion_rate


def parse_json_response(text: str) -> dict[str, Any]:
    """Strip markdown code fences and parse an LLM completion as a JSON object.

    Raises ValueError if the cleaned text is not a valid JSON object. This
    is split out from complete_json so the parsing behaviour can be tested
    without making a network call.
    """
    cleaned = _JSON_FENCE_RE.sub("", text).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM did not return valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON response was not an object.")
    return parsed


class LLM:
    """Thin client around the OpenAI Chat Completions API.

    Construction fails fast with LLMUnavailableError if the openai package
    is not installed or OPENAI_API_KEY is not set, so callers can decide
    up front whether to attempt an LLM path at all.
    """

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.2) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMUnavailableError(
                "openai package is not installed. Install it with pip install openai."
            ) from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMUnavailableError("OPENAI_API_KEY is not set.")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    @staticmethod
    def available() -> bool:
        """Return True if an LLM call could plausibly succeed right now.

        This only checks for the openai package and an API key; it does not
        make a network call, so a returned True can still fail later on
        network or authentication errors.
        """
        if importlib.util.find_spec("openai") is None:
            return False
        return bool(os.environ.get("OPENAI_API_KEY"))

    def complete(self, system: str, user: str, **kwargs: Any) -> LLMResponse:
        try:
            rsp = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **kwargs,
            )
        except Exception as exc:
            raise LLMUnavailableError(
                f"OpenAI completion request failed: {exc}"
            ) from exc

        choice = rsp.choices[0].message.content or ""
        usage = rsp.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        return LLMResponse(
            text=choice,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=self.model,
            cost_usd=estimate_cost_usd(self.model, prompt_tokens, completion_tokens),
        )

    def complete_json(
        self, system: str, user: str, **kwargs: Any
    ) -> tuple[dict[str, Any], LLMResponse]:
        """Request a JSON object completion and parse it.

        Raises ValueError if the response is not valid JSON once fenced code
        blocks are stripped, or LLMUnavailableError if the request itself
        fails. Callers should treat both as a signal to fall back.
        """
        json_system = (
            f"{system}\n\nRespond with a single JSON object and nothing else: "
            "no prose, no markdown fences, no explanation."
        )
        response = self.complete(json_system, user, **kwargs)
        parsed = parse_json_response(response.text)
        return parsed, response


def within_budget(state: ResearchState, budget: dict[str, Any] | None) -> bool:
    """Return False if a run has already reached its configured token or USD cap.

    Agents should call this before attempting an LLM completion so that
    once a budget is exhausted, the rest of the run falls back to
    deterministic behaviour without making further doomed attempts.
    """
    if not budget:
        return True
    max_tokens = budget.get("max_tokens_per_run")
    max_usd = budget.get("max_usd_per_run")
    token_cap_hit = max_tokens is not None and state.get("tokens_used", 0) >= max_tokens
    usd_cap_hit = max_usd is not None and state.get("usd_spent", 0.0) >= max_usd
    return not (token_cap_hit or usd_cap_hit)


def apply_usage(
    state: ResearchState, response: LLMResponse, budget: dict[str, Any] | None = None
) -> None:
    """Add a completion's usage to the run's running totals.

    Raises BudgetExceededError after recording the usage if the configured
    max_tokens_per_run or max_usd_per_run has been exceeded, so callers can
    catch it and fall back to deterministic behaviour for the remainder of
    the run.
    """
    tokens = response.prompt_tokens + response.completion_tokens
    state["tokens_used"] = state.get("tokens_used", 0) + tokens
    state["usd_spent"] = state.get("usd_spent", 0.0) + response.cost_usd

    if not budget:
        return

    max_tokens = budget.get("max_tokens_per_run")
    max_usd = budget.get("max_usd_per_run")
    if max_tokens is not None and state["tokens_used"] > max_tokens:
        raise BudgetExceededError(
            f"Token budget exceeded: {state['tokens_used']} > {max_tokens}."
        )
    if max_usd is not None and state["usd_spent"] > max_usd:
        raise BudgetExceededError(
            f"USD budget exceeded: {state['usd_spent']:.4f} > {max_usd}."
        )
