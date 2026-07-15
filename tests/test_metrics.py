"""Unit tests for the financial metrics module."""

import numpy as np
import pandas as pd

from quantlab.evaluation.metrics import cagr, max_drawdown, sharpe, win_rate


def _flat_returns(days: int = 252, r: float = 0.0004) -> pd.Series:
    idx = pd.bdate_range("2020-01-01", periods=days)
    return pd.Series(r, index=idx)


def test_sharpe_zero_volatility_returns_zero():
    r = _flat_returns()
    assert sharpe(r) == 0.0


def test_sharpe_positive_for_positive_mean_and_noise():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=1000)
    r = pd.Series(rng.normal(0.0005, 0.01, size=1000), index=idx)
    assert sharpe(r) > 0.0


def test_cagr_matches_expected():
    r = _flat_returns(252, 0.001)
    expected = (1.001) ** 252 - 1
    assert abs(cagr(r) - expected) < 1e-6


def test_max_drawdown_is_non_positive():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=500)
    r = pd.Series(rng.normal(0.0, 0.01, size=500), index=idx)
    assert max_drawdown(r) <= 0.0


def test_win_rate_bounds():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2020-01-01", periods=500)
    r = pd.Series(rng.normal(0.0, 0.01, size=500), index=idx)
    w = win_rate(r)
    assert 0.0 <= w <= 1.0
