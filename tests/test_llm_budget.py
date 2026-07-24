"""Unit tests for the LLM budget tracking and JSON parsing helpers.

None of these tests make a network call: OPENAI_API_KEY is unset by the
sandboxed_environment fixture in conftest.py, so LLM.available() is always
False here, and parse_json_response / apply_usage / within_budget are pure
functions tested directly.
"""

from __future__ import annotations

import pytest

from quantlab.core.llm import (
    LLM,
    BudgetExceededError,
    LLMResponse,
    apply_usage,
    estimate_cost_usd,
    parse_json_response,
    within_budget,
)


def test_available_is_false_without_api_key():
    assert LLM.available() is False


def test_available_is_false_when_openai_package_missing(monkeypatch):
    import importlib.util

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "openai":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-test")
    assert LLM.available() is False


def test_parse_json_response_strips_code_fences():
    text = '```json\n{"a": 1, "b": "two"}\n```'
    parsed = parse_json_response(text)
    assert parsed == {"a": 1, "b": "two"}


def test_parse_json_response_plain_object():
    assert parse_json_response('{"x": 1}') == {"x": 1}


def test_parse_json_response_rejects_non_json():
    with pytest.raises(ValueError):
        parse_json_response("not json at all")


def test_parse_json_response_rejects_json_array():
    with pytest.raises(ValueError):
        parse_json_response("[1, 2, 3]")


def test_estimate_cost_usd_scales_with_tokens():
    small = estimate_cost_usd("gpt-4o", 100, 100)
    large = estimate_cost_usd("gpt-4o", 1000, 1000)
    assert large > small
    assert small > 0.0


def test_apply_usage_accumulates_tokens_and_cost():
    state: dict = {}
    response = LLMResponse(
        text="hi", prompt_tokens=10, completion_tokens=5, model="gpt-4o", cost_usd=0.01
    )
    apply_usage(state, response, budget=None)
    assert state["tokens_used"] == 15
    assert state["usd_spent"] == pytest.approx(0.01)


def test_apply_usage_raises_when_token_budget_exceeded():
    state: dict = {}
    response = LLMResponse(
        text="hi",
        prompt_tokens=1000,
        completion_tokens=1000,
        model="gpt-4o",
        cost_usd=0.01,
    )
    with pytest.raises(BudgetExceededError):
        apply_usage(state, response, budget={"max_tokens_per_run": 500})


def test_apply_usage_raises_when_usd_budget_exceeded():
    state: dict = {}
    response = LLMResponse(
        text="hi", prompt_tokens=10, completion_tokens=10, model="gpt-4o", cost_usd=10.0
    )
    with pytest.raises(BudgetExceededError):
        apply_usage(state, response, budget={"max_usd_per_run": 1.0})


def test_within_budget_true_with_no_budget_configured():
    assert within_budget({"tokens_used": 999999}, None) is True


def test_within_budget_false_once_token_cap_reached():
    assert within_budget({"tokens_used": 500}, {"max_tokens_per_run": 500}) is False


def test_within_budget_true_below_cap():
    assert within_budget({"tokens_used": 10}, {"max_tokens_per_run": 500}) is True
