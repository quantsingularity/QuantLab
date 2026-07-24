"""Backtesting Agent."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantlab.backtest.engine import run_backtest, save_result
from quantlab.core.seeding import DEFAULT_SEED
from quantlab.core.state import BacktestResult, ModelArtifact, ResearchState
from quantlab.strategies.ml_signal import ml_ranked_signal
from quantlab.strategies.momentum import momentum_signal


def _build_weights(
    prices: pd.DataFrame, model: ModelArtifact, seed: int
) -> pd.DataFrame:
    params = model.params

    if model.kind == "rank_signal":
        return momentum_signal(
            prices,
            lookback=int(params.get("lookback", 252)),
            skip=int(params.get("skip", 21)),
            top_pct=float(params.get("top_pct", 0.10)),
            long_only=bool(params.get("long_only", True)),
        )
    if model.kind in ("ridge", "xgboost"):
        return ml_ranked_signal(
            prices,
            lookback=int(params.get("lookback", 252)),
            skip=int(params.get("skip", 21)),
            top_pct=float(params.get("top_pct", 0.10)),
            long_only=bool(params.get("long_only", True)),
            kind=model.kind,
            min_train_periods=int(params.get("min_train_periods", 12)),
            seed=seed,
        )
    raise ValueError(f"Unknown model kind: {model.kind}")


def run(state: ResearchState) -> ResearchState:
    cfg = state.get("run_config", {})
    tc_bps = float(cfg.get("transaction_cost_bps", 5.0))
    seed = int(cfg.get("seed", DEFAULT_SEED))

    prices = pd.read_parquet(state["features"][0].parquet_path)
    weights = _build_weights(prices, state["model"], seed)
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
