"""Walk-forward ML signal: ridge regression or gradient-boosted trees.

At each monthly rebalance date t_i, a fresh model is fit only on
(feature, forward return) pairs whose forward return was already realised
strictly before t_i. The fitted model then scores the current cross-section
using the feature observed at t_i, and those scores are turned into weights
the same way the plain rank-based momentum signal is. This keeps the walk
forward loop leakage-safe by construction: no training pair ever uses
information from t_i or later, which is what quantlab.backtest.engine's
leakage guard also checks for the resulting weight schedule.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pandas as pd

from quantlab.core.seeding import DEFAULT_SEED
from quantlab.strategies.features import (
    monthly_momentum_feature,
    rank_weighted_portfolio,
)

MIN_TRAIN_SAMPLES = 10


def _resolve_kind(kind: str) -> str:
    """Fall back to ridge when xgboost was requested but is not installed."""
    if kind != "xgboost":
        return kind
    if importlib.util.find_spec("xgboost") is None:
        print(
            "[quantlab.strategies.ml_signal] xgboost is not installed, falling back to ridge."
        )
        return "ridge"
    return "xgboost"


def _new_model(kind: str, seed: int) -> Any:
    if kind == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=1.0, random_state=seed)
    if kind == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=seed,
            verbosity=0,
        )
    raise ValueError(f"Unknown ML model kind: {kind}")


def ml_ranked_signal(
    prices: pd.DataFrame,
    lookback: int,
    skip: int,
    top_pct: float,
    long_only: bool,
    kind: str,
    min_train_periods: int = 12,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Return a monthly weight DataFrame from a walk-forward fitted model."""
    resolved_kind = _resolve_kind(kind)

    feature = monthly_momentum_feature(prices, lookback=lookback, skip=skip)
    monthly_close = prices.resample("ME").last()
    common_index = feature.index.intersection(monthly_close.index).sort_values()
    feature = feature.loc[common_index]
    monthly_close = monthly_close.loc[common_index]

    forward_return = monthly_close.shift(-1) / monthly_close - 1.0

    scores = pd.DataFrame(index=feature.index, columns=feature.columns, dtype=float)

    for i in range(min_train_periods, len(feature.index)):
        train_x, train_y = _training_pairs(feature, forward_return, up_to=i)
        if train_x is None or len(train_x) < MIN_TRAIN_SAMPLES:
            continue

        current_feature = feature.iloc[i]
        valid_current = current_feature.notna()
        if not valid_current.any():
            continue

        model = _new_model(resolved_kind, seed)
        model.fit(train_x, train_y)
        current_x = current_feature[valid_current].to_numpy().reshape(-1, 1)
        predictions = model.predict(current_x)
        scores.loc[feature.index[i], current_feature.index[valid_current]] = predictions

    return rank_weighted_portfolio(scores, top_pct=top_pct, long_only=long_only)


def _training_pairs(
    feature: pd.DataFrame, forward_return: pd.DataFrame, up_to: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Collect all (feature, forward return) pairs realised strictly before index up_to."""
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for k in range(up_to):
        feature_row = feature.iloc[k]
        label_row = forward_return.iloc[k]
        valid = feature_row.notna() & label_row.notna()
        if not valid.any():
            continue
        x_parts.append(feature_row[valid].to_numpy().reshape(-1, 1))
        y_parts.append(label_row[valid].to_numpy())

    if not x_parts:
        return None, None
    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)
