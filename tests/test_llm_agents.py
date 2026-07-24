"""Tests for the optional LLM path in planner, hypothesis, and literature agents.

Each agent is exercised twice: once with no model configured (must use the
deterministic fallback) and once with a fake LLM monkeypatched in that
returns a controlled payload (must validate and use it, or reject it and
fall back). No real network call or API key is used anywhere here.
"""

from __future__ import annotations

from quantlab.agents import hypothesis as hypothesis_agent
from quantlab.agents import planner as planner_agent
from quantlab.core.llm import LLMResponse
from quantlab.core.state import ResearchState


def _fake_llm_class(payload: dict):
    """Build a stand-in for quantlab.core.llm.LLM with no network access.

    available() reports True unconditionally and complete_json() always
    returns the given payload, so tests can control exactly what the
    "model" says without touching OPENAI_API_KEY or the network.
    """

    class _FakeLLM:
        def __init__(self, model: str) -> None:
            self.model = model

        @staticmethod
        def available() -> bool:
            return True

        def complete_json(self, system: str, user: str, **_: object):
            response = LLMResponse(
                text="",
                prompt_tokens=10,
                completion_tokens=10,
                model="fake",
                cost_usd=0.001,
            )
            return payload, response

    return _FakeLLM


def test_planner_uses_deterministic_dag_without_models_config():
    state: ResearchState = {"objective": "test objective", "run_config": {}}
    state = planner_agent.run(state)
    assert set(state["task_dag"].keys()) == {
        "literature",
        "hypothesis",
        "data",
        "model",
        "backtest",
        "evaluate",
        "report",
    }


def test_planner_uses_valid_llm_dag_when_available(monkeypatch):
    valid_dag = {
        stage: {"depends_on": [], "notes": "note"}
        for stage in (
            "literature",
            "hypothesis",
            "data",
            "model",
            "backtest",
            "evaluate",
            "report",
        )
    }
    monkeypatch.setattr(planner_agent, "LLM", _fake_llm_class(valid_dag))

    state: ResearchState = {
        "objective": "test",
        "run_config": {"models": {"planner": "gpt-4o"}},
    }
    state = planner_agent.run(state)
    assert state["task_dag"] == valid_dag


def test_planner_falls_back_when_llm_dag_is_malformed(monkeypatch):
    monkeypatch.setattr(planner_agent, "LLM", _fake_llm_class({"not": "a valid dag"}))

    state: ResearchState = {
        "objective": "test",
        "run_config": {"models": {"planner": "gpt-4o"}},
    }
    state = planner_agent.run(state)
    assert set(state["task_dag"].keys()) == {
        "literature",
        "hypothesis",
        "data",
        "model",
        "backtest",
        "evaluate",
        "report",
    }


def test_planner_skips_llm_path_when_budget_already_exhausted(monkeypatch):
    def _should_not_be_called(model: str):
        raise AssertionError(
            "LLM should not be constructed once the budget is exhausted"
        )

    fake_cls = _fake_llm_class({})
    monkeypatch.setattr(
        fake_cls, "__init__", lambda self, model: _should_not_be_called(model)
    )
    monkeypatch.setattr(planner_agent, "LLM", fake_cls)

    state: ResearchState = {
        "objective": "test",
        "run_config": {
            "models": {"planner": "gpt-4o"},
            "budget": {"max_tokens_per_run": 100},
        },
        "tokens_used": 500,
    }
    state = planner_agent.run(state)
    assert "task_dag" in state


def test_hypothesis_uses_deterministic_fields_without_models_config():
    state: ResearchState = {"objective": "test", "run_config": {}, "literature": []}
    state = hypothesis_agent.run(state)
    assert state["hypothesis"].expected_sign == "positive"
    assert state["hypothesis"].statement


def test_hypothesis_uses_valid_llm_fields_when_available(monkeypatch):
    payload = {
        "statement": "A custom statement.",
        "null": "A custom null.",
        "alternative": "A custom alternative.",
        "rationale": "A custom rationale.",
    }
    monkeypatch.setattr(hypothesis_agent, "LLM", _fake_llm_class(payload))

    state: ResearchState = {
        "objective": "test",
        "run_config": {"models": {"hypothesis": "gpt-4o"}},
        "literature": [],
    }
    state = hypothesis_agent.run(state)
    assert state["hypothesis"].statement == "A custom statement."
    assert state["hypothesis"].expected_sign == "positive"


def test_hypothesis_falls_back_when_llm_fields_are_missing_keys(monkeypatch):
    monkeypatch.setattr(
        hypothesis_agent, "LLM", _fake_llm_class({"statement": "only one field"})
    )

    state: ResearchState = {
        "objective": "test",
        "run_config": {"models": {"hypothesis": "gpt-4o"}},
        "literature": [],
    }
    state = hypothesis_agent.run(state)
    assert "positive risk-adjusted return" in state["hypothesis"].statement
