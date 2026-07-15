"""Property-based sanity checks for the leakage guard."""

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.engine import run_backtest


def _random_prices(n_days: int = 500, n_assets: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    rets = rng.normal(0.0, 0.01, size=(n_days, n_assets))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=idx, columns=[f"A{i}" for i in range(n_assets)])


def test_weights_beyond_prices_raise():
    prices = _random_prices()
    monthly = pd.DataFrame(
        0.2,
        index=pd.date_range("2020-01-31", periods=30, freq="ME"),
        columns=prices.columns,
    )
    with pytest.raises(ValueError):
        run_backtest(prices, monthly)


def test_backtest_runs_and_produces_finite_output():
    prices = _random_prices()
    monthly = pd.DataFrame(
        0.2,
        index=prices.resample("ME").last().index[:-1],
        columns=prices.columns,
    )
    out = run_backtest(prices, monthly, transaction_cost_bps=5.0)
    assert out.equity_curve.notna().all()
    assert np.isfinite(out.equity_curve).all()


def test_month_end_resample_label_past_prices_does_not_false_positive():
    """Regression test: prices ending on a non-month-end date used to false-positive.

    `momentum_signal` builds its monthly weight index via `resample("ME")`,
    which labels the final bucket with the calendar month end even when the
    price series itself stops mid-month. That trailing label used to trip
    `_check_no_lookahead` even though it has zero effect on the simulation
    (see `docs/` review notes). This test pins the fix: a price series ending
    on an arbitrary weekday must not crash the backtest.
    """
    prices = _random_prices(n_days=400, seed=1)
    assert prices.index.max().day != prices.index.max().days_in_month

    monthly = pd.DataFrame(
        0.2,
        index=prices.resample("ME").last().index,
        columns=prices.columns,
    )
    assert monthly.index.max() > prices.index.max()

    out = run_backtest(prices, monthly, transaction_cost_bps=5.0)
    assert out.equity_curve.notna().all()
    assert np.isfinite(out.equity_curve).all()


def test_far_future_weights_still_raise_despite_slack_tolerance():
    """A weight schedule reaching genuinely far past the data must still raise."""
    prices = _random_prices(n_days=400, seed=2)
    far_future = pd.DataFrame(
        0.2,
        index=pd.date_range(
            prices.index.max() + pd.Timedelta(days=200), periods=6, freq="ME"
        ),
        columns=prices.columns,
    )
    with pytest.raises(ValueError):
        run_backtest(prices, far_future)
