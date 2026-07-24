"""Tests for the Reflective Memory Agent's cross-run pattern detection."""

from __future__ import annotations

from quantlab.agents import reflection
from quantlab.core.state import ResearchState


def _run_literature_stage(run_id: str, n_papers: int) -> ResearchState:
    state: ResearchState = {
        "run_id": run_id,
        "reflections": [],
        "literature": [None] * n_papers,
    }
    return reflection.critique(state, "literature")


def test_single_run_with_few_papers_raises_a_warning():
    state = _run_literature_stage("run0", n_papers=1)
    assert any(r.severity == "warn" for r in state["reflections"])


def test_single_run_with_enough_papers_raises_nothing():
    state = _run_literature_stage("run0", n_papers=5)
    assert state["reflections"] == []


def test_recurring_pattern_surfaces_after_enough_prior_runs():
    final_state = None
    for i in range(5):
        final_state = _run_literature_stage(f"run{i}", n_papers=1)

    critiques = [r.critique for r in final_state["reflections"]]
    assert any("Recurring pattern" in c for c in critiques)


def test_recurring_pattern_does_not_fire_with_too_few_prior_runs():
    final_state = None
    for i in range(2):
        final_state = _run_literature_stage(f"run{i}", n_papers=1)

    critiques = [r.critique for r in final_state["reflections"]]
    assert not any("Recurring pattern" in c for c in critiques)


def test_recurring_pattern_does_not_fire_when_stage_is_consistently_healthy():
    final_state = None
    for i in range(5):
        final_state = _run_literature_stage(f"run{i}", n_papers=5)

    assert final_state["reflections"] == []
