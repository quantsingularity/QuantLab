"""Unit tests for the walk-forward ML signal."""

from __future__ import annotations

from quantlab.strategies.ml_signal import _resolve_kind, ml_ranked_signal


def test_resolve_kind_passes_through_ridge():
    assert _resolve_kind("ridge") == "ridge"


def test_resolve_kind_falls_back_when_xgboost_missing(monkeypatch):
    import importlib.util

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "xgboost":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    assert _resolve_kind("xgboost") == "ridge"


def test_ridge_signal_produces_valid_long_only_weights(price_frame):
    weights = ml_ranked_signal(
        price_frame,
        lookback=126,
        skip=10,
        top_pct=0.25,
        long_only=True,
        kind="ridge",
        min_train_periods=6,
        seed=1,
    )
    assert (weights >= 0.0).all().all()
    row_sums = weights.sum(axis=1)
    nonzero_rows = row_sums[row_sums.abs() > 1e-9]
    assert not nonzero_rows.empty
    assert (nonzero_rows.round(6) == 1.0).all()


def test_ridge_signal_is_deterministic_given_same_seed(price_frame):
    kwargs = {
        "lookback": 126,
        "skip": 10,
        "top_pct": 0.25,
        "long_only": True,
        "kind": "ridge",
        "min_train_periods": 6,
    }
    first = ml_ranked_signal(price_frame, seed=3, **kwargs)
    second = ml_ranked_signal(price_frame, seed=3, **kwargs)
    assert first.equals(second)


def test_ridge_signal_leaves_early_periods_empty_before_min_train(price_frame):
    weights = ml_ranked_signal(
        price_frame,
        lookback=126,
        skip=10,
        top_pct=0.25,
        long_only=True,
        kind="ridge",
        min_train_periods=6,
        seed=1,
    )
    early_rows = weights.iloc[:6]
    assert (early_rows.fillna(0.0) == 0.0).all().all()
