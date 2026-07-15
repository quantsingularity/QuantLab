"""Cross-sectional 12-1 momentum signal.

`prices` is a wide DataFrame of daily close prices (index: date, columns: tickers).
The signal at date t is the return from t-252 to t-21 (skipping the last month
to avoid short-term reversal), ranked cross-sectionally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOOKBACK = 252
SKIP = 21


def momentum_signal(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a monthly signal with values in [-1, 1], long-short by cross-sectional rank."""
    r = prices.shift(SKIP) / prices.shift(LOOKBACK) - 1.0
    r = r.dropna(how="all")

    r_monthly = r.resample("ME").last()

    ranks = r_monthly.rank(axis=1, pct=True) * 2.0 - 1.0

    threshold = 0.8
    signal = (ranks >= threshold).astype(float)

    row_sum = signal.sum(axis=1).replace(0.0, np.nan)
    weights = signal.div(row_sum, axis=0).fillna(0.0)
    return weights
