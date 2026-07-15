"""Hypothesis Generation Agent."""

from __future__ import annotations

from quantlab.core.state import Hypothesis, ResearchState


def run(state: ResearchState) -> ResearchState:
    supporting = [p.title for p in state.get("literature", [])]

    tc_bps = float(state.get("run_config", {}).get("transaction_cost_bps", 5.0))
    tc_label = f"{tc_bps:g} bps"

    state["hypothesis"] = Hypothesis(
        statement=(
            "A cross-sectional 12-1 month momentum portfolio on the NASDAQ 100, "
            "rebalanced monthly and long the top decile, delivers a positive "
            f"risk-adjusted return net of {tc_label} transaction costs."
        ),
        null="Sharpe ratio of the strategy net of costs is less than or equal to zero.",
        alternative="Sharpe ratio of the strategy net of costs is greater than zero.",
        expected_sign="positive",
        horizon_days=21,
        universe="nasdaq_100",
        rationale=(
            "Cross-sectional momentum has been documented since Jegadeesh and Titman "
            "(1993) and has survived out of sample across markets and decades."
        ),
        supporting_papers=supporting,
    )
    return state
