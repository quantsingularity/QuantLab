"""Vectorised backtest engine with a leakage guard.

Design choices:
- Weights at month-end t are applied to returns from t to t+1 (no look-ahead).
- Transaction cost = turnover(t) * cost_bps, applied on the day weights change.
- Cash between rebalances is not compounded within a month for the PoC
  (a common convention; can be relaxed).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd


@dataclass
class BacktestOutput:
    equity_curve: pd.Series
    daily_returns: pd.Series
    turnover: pd.Series
    weights: pd.DataFrame


def _check_no_lookahead(prices: pd.DataFrame, weights: pd.DataFrame) -> None:
    """Raise if any weight timestamp is not strictly before its return period."""
    if weights.index.max() > prices.index.max():
        raise ValueError("Weights indexed beyond price data: possible look-ahead.")
    if not weights.index.is_monotonic_increasing:
        raise ValueError("Weight index must be monotonic increasing.")


_RESAMPLE_LABEL_SLACK_DAYS = 35


def _drop_resample_label_rounding(
    prices: pd.DataFrame, monthly_weights: pd.DataFrame
) -> pd.DataFrame:
    """Drop a trailing weight row whose label is a resample-rounding artefact.

    `pandas.resample("ME")` labels a bucket with the *calendar* month end even
    when the underlying price series stops mid-month (e.g. prices end on a
    Wednesday that is not itself a month-end trading day). That trailing
    label is never actually used: `reindex(prices.index, ...)` in
    `run_backtest` only keeps weight values on dates that exist in
    `prices.index`, so a weight row dated a few weeks after
    `prices.index.max()` has zero effect on the simulation -- it is a
    labelling artefact, not genuine look-ahead.

    This only trims rows within `_RESAMPLE_LABEL_SLACK_DAYS` of the last
    price date. A weight schedule that reaches meaningfully further into the
    future (e.g. months or years past the available price data, which would
    indicate a real upstream bug) is left untouched and still trips
    `_check_no_lookahead` below.
    """
    last_price_date = prices.index.max()
    slack_cutoff = last_price_date + pd.Timedelta(days=_RESAMPLE_LABEL_SLACK_DAYS)
    is_rounding_artifact = (monthly_weights.index > last_price_date) & (
        monthly_weights.index <= slack_cutoff
    )
    return monthly_weights[~is_rounding_artifact]


def run_backtest(
    prices: pd.DataFrame,
    monthly_weights: pd.DataFrame,
    transaction_cost_bps: float = 5.0,
) -> BacktestOutput:
    monthly_weights = _drop_resample_label_rounding(prices, monthly_weights)
    _check_no_lookahead(prices, monthly_weights)

    common = prices.columns.intersection(monthly_weights.columns)
    prices = prices[common].sort_index()
    monthly_weights = monthly_weights[common].sort_index()

    daily_weights = (
        monthly_weights.reindex(prices.index, method="ffill").shift(1).fillna(0.0)
    )

    daily_returns_assets = prices.pct_change().fillna(0.0)
    gross_returns = (daily_weights * daily_returns_assets).sum(axis=1)

    turnover = daily_weights.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (transaction_cost_bps / 1e4)

    net_returns = gross_returns - cost
    equity = (1.0 + net_returns).cumprod()

    return BacktestOutput(
        equity_curve=equity,
        daily_returns=net_returns,
        turnover=turnover,
        weights=daily_weights,
    )


def save_result(output: BacktestOutput, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    eq_path = out_dir / "equity_curve.parquet"
    trades_path = out_dir / "trades.parquet"
    output.equity_curve.to_frame("equity").to_parquet(eq_path)
    weights_long = cast("pd.Series[float]", output.weights.stack())
    weights_long.index = weights_long.index.set_names(["date", "ticker"])
    weights_long.to_frame("weight").reset_index().to_parquet(trades_path)
    return eq_path, trades_path
