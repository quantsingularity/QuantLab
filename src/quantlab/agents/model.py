"""Model Development Agent.

For the PoC the "model" is a deterministic ranking signal. In the thesis
version this agent will also fit XGBoost or linear models with cross-validated
hyperparameters.
"""

from __future__ import annotations

from quantlab.core.state import ModelArtifact, ResearchState


def run(state: ResearchState) -> ResearchState:
    state["model"] = ModelArtifact(
        kind="rank_signal",
        params={"lookback": 252, "skip": 21, "top_pct": 0.10, "long_only": True},
    )
    return state
