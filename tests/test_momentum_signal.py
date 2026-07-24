"""Unit tests for the cross-sectional momentum signal."""

from __future__ import annotations

from quantlab.strategies.momentum import momentum_signal


def test_long_only_weights_are_non_negative_and_sum_to_one(price_frame):
    weights = momentum_signal(price_frame, top_pct=0.25, long_only=True)
    assert (weights >= 0.0).all().all()
    row_sums = weights.sum(axis=1)
    nonzero_rows = row_sums[row_sums.abs() > 1e-9]
    assert not nonzero_rows.empty
    assert (nonzero_rows.round(6) == 1.0).all()


def test_long_only_selects_roughly_top_pct_of_universe(price_frame):
    top_pct = 0.25
    weights = momentum_signal(price_frame, top_pct=top_pct, long_only=True)
    n_assets = price_frame.shape[1]
    expected_selected = max(1, round(n_assets * top_pct))
    n_selected = (weights > 0).sum(axis=1)
    nonzero = n_selected[n_selected > 0]
    assert not nonzero.empty
    assert (nonzero - expected_selected).abs().max() <= 1


def test_long_short_is_approximately_dollar_neutral(price_frame):
    weights = momentum_signal(price_frame, top_pct=0.25, long_only=False)
    row_sums = weights.sum(axis=1)
    nonzero_rows = row_sums[row_sums.abs() > 1e-9]
    assert (nonzero_rows.abs() < 1e-6).all()


def test_smaller_top_pct_selects_fewer_names(price_frame):
    narrow = momentum_signal(price_frame, top_pct=0.10, long_only=True)
    wide = momentum_signal(price_frame, top_pct=0.40, long_only=True)
    narrow_count = (narrow > 0).sum(axis=1).mean()
    wide_count = (wide > 0).sum(axis=1).mean()
    assert narrow_count < wide_count


def test_shorter_lookback_changes_the_selection(price_frame):
    short_lb = momentum_signal(price_frame, lookback=63, skip=5, top_pct=0.25)
    long_lb = momentum_signal(price_frame, lookback=252, skip=21, top_pct=0.25)
    common_index = short_lb.index.intersection(long_lb.index)
    assert len(common_index) > 0
    assert not short_lb.loc[common_index].equals(long_lb.loc[common_index])
