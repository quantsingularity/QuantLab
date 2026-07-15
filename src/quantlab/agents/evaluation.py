"""Evaluation Agent: computes financial and research-quality metrics."""

from __future__ import annotations

from quantlab.core.state import MetricsBundle, ResearchState
from quantlab.evaluation.metrics import compute_all


def run(state: ResearchState) -> ResearchState:
    daily = state.get("_daily_returns")
    if daily is None:
        raise RuntimeError("Backtest daily returns missing from state.")

    cfg = state.get("run_config", {})
    oos_start = str(cfg.get("oos_start", "2022-01-01"))
    daily_turnover = state.get("_daily_turnover")
    m = compute_all(daily, oos_start=oos_start, daily_turnover=daily_turnover)
    state["metrics"] = MetricsBundle(
        sharpe=m["sharpe"],
        cagr=m["cagr"],
        max_drawdown=m["max_drawdown"],
        win_rate=m["win_rate"],
        turnover=m.get("turnover", 0.0),
    )
    return state
