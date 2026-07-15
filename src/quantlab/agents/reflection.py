"""Reflective Memory Agent.

Called after every node via the graph wrapper. Emits structured critiques and
persists them to the memory store. In the thesis version an LLM produces the
critique; here we ship deterministic sanity checks so the PoC runs offline.
"""

from __future__ import annotations

from quantlab.core import memory
from quantlab.core.state import Reflection, ResearchState


def critique(state: ResearchState, stage: str) -> ResearchState:
    refs: list[Reflection] = list(state.get("reflections", []))

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

    state["reflections"] = refs
    memory.append(
        {
            "run_id": state.get("run_id"),
            "stage": stage,
            "n_reflections": len(refs),
        }
    )
    return state
