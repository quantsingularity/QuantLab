"""Reflective Memory Agent.

Called after every node via the graph wrapper. Emits structured critiques,
persists them to the memory store, and recalls prior runs' critiques for
the same stage to surface recurring problems across runs. In the thesis
version an LLM also produces free-form critique text; here the sanity
checks are deterministic so the PoC runs offline, but the cross-run recall
itself is real and backed by quantlab.core.memory.
"""

from __future__ import annotations

from quantlab.core import memory
from quantlab.core.state import Reflection, ResearchState

_RECURRING_LOOKBACK = 20
_RECURRING_MIN_PRIOR_RUNS = 3
_RECURRING_RATE_THRESHOLD = 0.5


def _stage_reflections(state: ResearchState, stage: str) -> list[Reflection]:
    refs: list[Reflection] = []

    if stage == "literature" and len(state.get("literature", [])) < 3:
        refs.append(
            Reflection(
                agent="ReflectiveMemory",
                stage=stage,
                critique="Fewer than three papers retrieved; broaden the search.",
                severity="warn",
            )
        )

    if stage == "hypothesis" and "hypothesis" in state:
        h = state["hypothesis"]
        if not h.supporting_papers:
            refs.append(
                Reflection(
                    agent="ReflectiveMemory",
                    stage=stage,
                    critique="Hypothesis lacks explicit paper support.",
                    severity="warn",
                )
            )

    if stage == "evaluate" and "metrics" in state:
        m = state["metrics"]
        if m.sharpe < 0:
            refs.append(
                Reflection(
                    agent="ReflectiveMemory",
                    stage=stage,
                    critique=f"Negative Sharpe ({m.sharpe:.2f}). Reject H1.",
                    severity="info",
                )
            )
        if m.max_drawdown < -0.35:
            refs.append(
                Reflection(
                    agent="ReflectiveMemory",
                    stage=stage,
                    critique=f"Large drawdown ({m.max_drawdown:.2%}). Add volatility targeting.",
                    severity="warn",
                )
            )

    return refs


def _recurring_pattern_reflection(
    stage: str, current_has_issue: bool
) -> Reflection | None:
    """Look at prior runs' critiques for this stage and flag a recurring pattern.

    This is what makes memory genuinely cross-run: a stage that has raised
    warnings in most of its recent past occurrences gets an extra critique
    pointing that out, even if the current run's checks alone did not flag
    anything.
    """
    prior_records = [
        record
        for record in memory.recall(stage, k=_RECURRING_LOOKBACK)
        if record.get("stage") == stage
    ]
    if len(prior_records) < _RECURRING_MIN_PRIOR_RUNS:
        return None

    prior_issue_count = sum(
        1 for record in prior_records if record.get("n_reflections", 0) > 0
    )
    recurring_rate = prior_issue_count / len(prior_records)
    if recurring_rate < _RECURRING_RATE_THRESHOLD:
        return None
    if not current_has_issue and recurring_rate < _RECURRING_RATE_THRESHOLD * 1.5:
        return None

    return Reflection(
        agent="ReflectiveMemory",
        stage=stage,
        critique=(
            f"Recurring pattern: {prior_issue_count} of the last {len(prior_records)} "
            f"runs at the {stage} stage also raised a critique. Consider revisiting "
            f"the {stage} approach rather than the current run in isolation."
        ),
        severity="warn",
    )


def critique(state: ResearchState, stage: str) -> ResearchState:
    refs: list[Reflection] = list(state.get("reflections", []))

    stage_refs = _stage_reflections(state, stage)
    refs.extend(stage_refs)

    recurring = _recurring_pattern_reflection(stage, current_has_issue=bool(stage_refs))
    if recurring is not None:
        refs.append(recurring)

    state["reflections"] = refs
    memory.append(
        {
            "run_id": state.get("run_id"),
            "stage": stage,
            "n_reflections": len(stage_refs),
        }
    )
    return state
