"""Unit tests for the offline synthetic price fallback."""

from __future__ import annotations

import pandas as pd

from quantlab.data.loaders import NASDAQ_100_POC, load_prices


def test_load_prices_offline_returns_readable_parquet(tmp_path):
    path = load_prices(
        universe="nasdaq_100",
        start="2019-01-01",
        end="2019-06-01",
        cache_dir=tmp_path,
        seed=11,
    )
    assert path.exists()
    assert path.name.endswith("_synthetic.parquet")
    prices = pd.read_parquet(path)
    assert set(prices.columns) == set(NASDAQ_100_POC)
    assert (prices > 0.0).all().all()


def test_load_prices_offline_is_deterministic_given_same_seed(tmp_path):
    first = load_prices(
        "nasdaq_100", "2019-01-01", "2019-06-01", tmp_path / "a", seed=5
    )
    second = load_prices(
        "nasdaq_100", "2019-01-01", "2019-06-01", tmp_path / "b", seed=5
    )
    pd.testing.assert_frame_equal(pd.read_parquet(first), pd.read_parquet(second))


def test_load_prices_offline_differs_across_seeds(tmp_path):
    first = load_prices(
        "nasdaq_100", "2019-01-01", "2019-06-01", tmp_path / "a", seed=1
    )
    second = load_prices(
        "nasdaq_100", "2019-01-01", "2019-06-01", tmp_path / "b", seed=2
    )
    assert not pd.read_parquet(first).equals(pd.read_parquet(second))


def test_load_prices_caches_to_disk(tmp_path):
    first = load_prices("nasdaq_100", "2019-01-01", "2019-06-01", tmp_path, seed=1)
    mtime_before = first.stat().st_mtime
    second = load_prices("nasdaq_100", "2019-01-01", "2019-06-01", tmp_path, seed=1)
    assert first == second
    assert second.stat().st_mtime == mtime_before


def test_load_prices_single_ticker_universe(tmp_path):
    path = load_prices("AAPL", "2019-01-01", "2019-06-01", tmp_path, seed=1)
    prices = pd.read_parquet(path)
    assert list(prices.columns) == ["AAPL"]
