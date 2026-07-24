"""Cross-sectional 12-1 momentum signal.

prices is a wide DataFrame of daily close prices, indexed by date with one
column per ticker. The signal at month end t ranks assets by their trailing
lookback-day return, skipping the last skip days to avoid short-term
reversal, and equal-weights the top top_pct of the cross-section. When
long_only is False, the bottom top_pct is equal-weighted short as well,
producing a dollar-neutral long-short portfolio.
"""

from __future__ import annotations

import pandas as pd

from quantlab.strategies.features import (
    monthly_momentum_feature,
    rank_weighted_portfolio,
)

DEFAULT_LOOKBACK = 252
DEFAULT_SKIP = 21
DEFAULT_TOP_PCT = 0.10


def momentum_signal(
    prices: pd.DataFrame,
    lookback: int = DEFAULT_LOOKBACK,
    skip: int = DEFAULT_SKIP,
    top_pct: float = DEFAULT_TOP_PCT,
    long_only: bool = True,
) -> pd.DataFrame:
    """Return a monthly weight DataFrame for the cross-sectional momentum strategy."""
    feature = monthly_momentum_feature(prices, lookback=lookback, skip=skip)
    return rank_weighted_portfolio(feature, top_pct=top_pct, long_only=long_only)
