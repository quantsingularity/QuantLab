"""Financial performance metrics.

All metrics accept a pd.Series of daily net returns indexed by date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sharpe(returns: pd.Series, rf_daily: float = 0.0) -> float:
    excess = returns - rf_daily
    std = excess.std(ddof=0)
    if not np.isfinite(std) or std < 1e-12:
        return 0.0
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / std)


def cagr(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    total = float(np.prod((1.0 + returns).to_numpy()))
    years = len(returns) / TRADING_DAYS
    if years <= 0 or total <= 0:
        return 0.0
    return float(total ** (1.0 / years) - 1.0)


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def win_rate(returns: pd.Series, freq: str = "ME") -> float:
    if returns.empty:
        return 0.0
    grouped = (1.0 + returns).resample(freq).prod() - 1.0
    if grouped.empty:
        return 0.0
    return float((grouped > 0).mean())


def compute_all(
    returns: pd.Series,
    oos_start: str | None = None,
    daily_turnover: pd.Series | None = None,
) -> dict[str, float]:
    if oos_start:
        returns = returns.loc[oos_start:]
        if daily_turnover is not None:
            daily_turnover = daily_turnover.loc[oos_start:]
    result = {
        "sharpe": sharpe(returns),
        "cagr": cagr(returns),
        "max_drawdown": max_drawdown(returns),
        "win_rate": win_rate(returns),
    }
    if daily_turnover is not None:
        result["turnover"] = (
            float(daily_turnover.mean() * TRADING_DAYS)
            if not daily_turnover.empty
            else 0.0
        )
    return result
