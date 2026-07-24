"""Research Planner Agent: decomposes the objective into a task DAG.

Uses an LLM when one is configured for this stage in run_config.models and
OPENAI_API_KEY is set. Always falls back to a deterministic, known-good DAG
if the LLM path is unavailable, exceeds the run's budget, or returns
something that does not validate against the expected schema.
"""

from __future__ import annotations

from typing import Any

from quantlab.core.llm import (
    LLM,
    BudgetExceededError,
    LLMUnavailableError,
    apply_usage,
    within_budget,
)
from quantlab.core.state import ResearchState

PLANNER_SYSTEM = """You are the research planner of QuantLab. Given a natural-language
research objective, output a JSON task DAG with keys: literature, hypothesis, data,
model, backtest, evaluate, report. Each entry has: depends_on (list) and notes.
Be concise and testable. Do not invent tools."""

_REQUIRED_STAGES = (
    "literature",
    "hypothesis",
    "data",
    "model",
    "backtest",
    "evaluate",
    "report",
)


def _deterministic_dag() -> dict[str, Any]:
    return {
        "literature": {
            "depends_on": [],
            "notes": "Retrieve 3 to 5 papers on the topic.",
        },
        "hypothesis": {
            "depends_on": ["literature"],
            "notes": "Formalise a testable hypothesis.",
        },
        "data": {
            "depends_on": ["hypothesis"],
            "notes": "Load prices and build features.",
        },
        "model": {"depends_on": ["data"], "notes": "Build ranked signal or ML model."},
        "backtest": {
            "depends_on": ["model"],
            "notes": "Vectorised backtest with costs.",
        },
        "evaluate": {
            "depends_on": ["backtest"],
            "notes": "Compute financial and quality metrics.",
        },
        "report": {"depends_on": ["evaluate"], "notes": "Compile markdown report."},
    }


def _validate_dag(candidate: Any) -> dict[str, Any] | None:
    if not isinstance(candidate, dict):
        return None
    if set(candidate.keys()) != set(_REQUIRED_STAGES):
        return None
    for stage in _REQUIRED_STAGES:
        entry = candidate[stage]
        if not isinstance(entry, dict):
            return None
        if not isinstance(entry.get("depends_on"), list):
            return None
        if not isinstance(entry.get("notes"), str):
            return None
    return candidate


def _llm_dag(
    objective: str, model_name: str, state: ResearchState
) -> dict[str, Any] | None:
    cfg = state.get("run_config", {})
    budget = cfg.get("budget")
    if not within_budget(state, budget):
        return None
    try:
        llm = LLM(model=model_name)
        parsed, response = llm.complete_json(PLANNER_SYSTEM, objective)
        apply_usage(state, response, budget)
    except (LLMUnavailableError, ValueError, BudgetExceededError):
        return None
    return _validate_dag(parsed)


def run(state: ResearchState) -> ResearchState:
    objective = state.get("objective", "")
    model_name = state.get("run_config", {}).get("models", {}).get("planner")

    dag = None
    if model_name and LLM.available():
        dag = _llm_dag(objective, model_name, state)

    state["task_dag"] = dag if dag is not None else _deterministic_dag()
    return state
