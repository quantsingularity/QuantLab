"""Backtesting Agent."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantlab.backtest.engine import run_backtest, save_result
from quantlab.core.state import BacktestResult, ResearchState
from quantlab.strategies.momentum import momentum_signal


def run(state: ResearchState) -> ResearchState:
    cfg = state.get("run_config", {})
    tc_bps = float(cfg.get("transaction_cost_bps", 5.0))

    prices = pd.read_parquet(state["features"][0].parquet_path)
    weights = momentum_signal(prices)
    out = run_backtest(prices, weights, transaction_cost_bps=tc_bps)

    out_dir = Path(state.get("output_dir", "./runs")) / state["run_id"]
    eq_path, trades_path = save_result(out, out_dir)

    state["backtest"] = BacktestResult(
        equity_curve_path=str(eq_path),
        trades_path=str(trades_path),
        start=out.equity_curve.index.min().to_pydatetime(),
        end=out.equity_curve.index.max().to_pydatetime(),
        transaction_cost_bps=tc_bps,
    )
    state["_daily_returns"] = out.daily_returns
    state["_daily_turnover"] = out.turnover
    return state
