"""Hypothesis Generation Agent.

Uses an LLM when one is configured for this stage in run_config.models and
OPENAI_API_KEY is set, to draft the hypothesis statement, null, alternative,
and rationale text. expected_sign, horizon, and universe stay deterministic
since downstream agents consume them as typed values. Falls back to a
deterministic, known-good hypothesis if the LLM path is unavailable,
exceeds the run's budget, or returns something that does not validate.
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
from quantlab.core.state import Hypothesis, ResearchState

HYPOTHESIS_SYSTEM = """You are the hypothesis generation agent of QuantLab. Given a
research objective and a list of supporting paper titles, draft a single testable
finance research hypothesis. Output a JSON object with exactly these string keys:
statement, null, alternative, rationale. statement is the claim under test, null is
H0, alternative is H1, and rationale explains why the hypothesis is plausible given
the supporting literature. Keep each field to two sentences or fewer."""

_REQUIRED_FIELDS = ("statement", "null", "alternative", "rationale")


def _deterministic_fields(tc_label: str) -> dict[str, str]:
    return {
        "statement": (
            "A cross-sectional 12-1 month momentum portfolio on the NASDAQ 100, "
            "rebalanced monthly and long the top decile, delivers a positive "
            f"risk-adjusted return net of {tc_label} transaction costs."
        ),
        "null": "Sharpe ratio of the strategy net of costs is less than or equal to zero.",
        "alternative": "Sharpe ratio of the strategy net of costs is greater than zero.",
        "rationale": (
            "Cross-sectional momentum has been documented since Jegadeesh and Titman "
            "(1993) and has survived out of sample across markets and decades."
        ),
    }


def _validate_fields(candidate: Any) -> dict[str, str] | None:
    if not isinstance(candidate, dict):
        return None
    for field_name in _REQUIRED_FIELDS:
        value = candidate.get(field_name)
        if not isinstance(value, str) or not value.strip():
            return None
    return {field_name: candidate[field_name] for field_name in _REQUIRED_FIELDS}


def _llm_fields(
    objective: str, supporting: list[str], model_name: str, state: ResearchState
) -> dict[str, str] | None:
    cfg = state.get("run_config", {})
    budget = cfg.get("budget")
    if not within_budget(state, budget):
        return None
    user = (
        f"Objective: {objective}\n"
        f"Supporting papers: {', '.join(supporting) if supporting else 'none retrieved'}"
    )
    try:
        llm = LLM(model=model_name)
        parsed, response = llm.complete_json(HYPOTHESIS_SYSTEM, user)
        apply_usage(state, response, budget)
    except (LLMUnavailableError, ValueError, BudgetExceededError):
        return None
    return _validate_fields(parsed)


def run(state: ResearchState) -> ResearchState:
    supporting = [p.title for p in state.get("literature", [])]
    cfg = state.get("run_config", {})
    tc_bps = float(cfg.get("transaction_cost_bps", 5.0))
    tc_label = f"{tc_bps:g} bps"

    model_name = cfg.get("models", {}).get("hypothesis")
    fields = None
    if model_name and LLM.available():
        fields = _llm_fields(state.get("objective", ""), supporting, model_name, state)
    if fields is None:
        fields = _deterministic_fields(tc_label)

    state["hypothesis"] = Hypothesis(
        statement=fields["statement"],
        null=fields["null"],
        alternative=fields["alternative"],
        expected_sign="positive",
        horizon_days=21,
        universe="nasdaq_100",
        rationale=fields["rationale"],
        supporting_papers=supporting,
    )
    return state
