"""Shared feature computation and portfolio construction for the momentum family.

Every strategy in this package starts from the same cross-sectional 12-1
momentum feature and turns cross-sectional scores into weights the same
way. Keeping both in one place means every strategy sees an identical
feature definition and an identical construction rule, which matters for
the leakage guard: the feature at date t only ever uses price history
strictly before t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def monthly_momentum_feature(
    prices: pd.DataFrame, lookback: int, skip: int
) -> pd.DataFrame:
    """Return the trailing lookback-day return, skipping the last skip days.

    The result is resampled to month end. Skipping the most recent skip
    trading days avoids short-term reversal contaminating the momentum
    signal, following Jegadeesh and Titman (1993).
    """
    trailing_return = prices.shift(skip) / prices.shift(lookback) - 1.0
    trailing_return = trailing_return.dropna(how="all")
    return trailing_return.resample("ME").last()


def rank_weighted_portfolio(
    scores: pd.DataFrame, top_pct: float, long_only: bool
) -> pd.DataFrame:
    """Turn a cross-sectional score matrix into equal-weighted monthly weights.

    The top top_pct of each row (by percentile rank) is equal-weighted long.
    When long_only is False, the bottom top_pct is equal-weighted short as
    well, producing a dollar-neutral long-short portfolio.
    """
    ranks = scores.rank(axis=1, pct=True)

    long_mask = ranks >= (1.0 - top_pct)
    long_weights = _equal_weight_from_mask(long_mask, sign=1.0)

    if long_only:
        return long_weights

    short_mask = ranks <= top_pct
    short_weights = _equal_weight_from_mask(short_mask, sign=-1.0)
    return long_weights.add(short_weights, fill_value=0.0)


def _equal_weight_from_mask(mask: pd.DataFrame, sign: float) -> pd.DataFrame:
    row_count = mask.sum(axis=1).replace(0.0, np.nan)
    return sign * mask.div(row_count, axis=0).fillna(0.0)
