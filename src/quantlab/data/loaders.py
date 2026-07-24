"""Price loaders.

The PoC prefers yfinance for the NASDAQ 100. When yfinance is not installed,
the network is unreachable, or the download comes back empty, this module
falls back to a deterministic synthetic price series so the pipeline still
runs end to end offline. Set QUANTLAB_OFFLINE=1 to skip the network attempt
entirely and always use synthetic data, which is useful for tests and CI.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.core.seeding import DEFAULT_SEED

NASDAQ_100_POC = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
    "AVGO",
    "COST",
    "PEP",
    "ADBE",
    "CSCO",
    "NFLX",
    "AMD",
    "INTC",
    "QCOM",
    "TXN",
    "AMGN",
    "HON",
    "SBUX",
    "INTU",
    "BKNG",
    "GILD",
    "ISRG",
    "REGN",
    "VRTX",
    "MDLZ",
    "ADP",
    "LRCX",
    "ADI",
]


def _offline_forced() -> bool:
    return os.environ.get("QUANTLAB_OFFLINE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _tickers_for(universe: str) -> list[str]:
    return NASDAQ_100_POC if universe == "nasdaq_100" else [universe]


def _download_from_yfinance(
    tickers: list[str], start: str, end: str
) -> pd.DataFrame | None:
    try:
        import yfinance as yf
    except ImportError:
        return None

    try:
        raw = yf.download(
            tickers, start=start, end=end, auto_adjust=True, progress=False
        )
    except Exception:
        return None

    if raw is None or raw.empty:
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    close = close.dropna(how="all")
    return close if not close.empty else None


def _synthetic_prices(
    tickers: list[str], start: str, end: str, seed: int
) -> pd.DataFrame:
    """Generate deterministic geometric Brownian motion price paths.

    Used whenever real market data cannot be obtained. The paths are
    reproducible given the same tickers, date range, and seed, which keeps
    offline runs and tests deterministic.
    """
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start=start, end=end)
    n_days = len(index)
    n_assets = len(tickers)

    daily_drift = rng.uniform(0.00015, 0.00045, size=n_assets)
    daily_vol = rng.uniform(0.012, 0.028, size=n_assets)
    shocks = rng.normal(0.0, 1.0, size=(n_days, n_assets))
    log_returns = daily_drift + daily_vol * shocks
    log_prices = np.cumsum(log_returns, axis=0)

    start_price = rng.uniform(20.0, 400.0, size=n_assets)
    prices = start_price * np.exp(log_prices)

    return pd.DataFrame(prices, index=index, columns=tickers)


def load_prices(
    universe: str,
    start: str,
    end: str,
    cache_dir: Path,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Load daily close prices for universe and cache them to parquet.

    Returns the path to a wide-format parquet with a DatetimeIndex and one
    column per ticker. Prefers a real yfinance download; falls back to a
    deterministic synthetic series if that is unavailable or forced off via
    QUANTLAB_OFFLINE.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    tickers = _tickers_for(universe)

    if not _offline_forced():
        real_path = cache_dir / f"{universe}_{start}_{end}.parquet"
        if real_path.exists():
            return real_path
        downloaded = _download_from_yfinance(tickers, start, end)
        if downloaded is not None:
            downloaded.to_parquet(real_path)
            return real_path

    synthetic_path = cache_dir / f"{universe}_{start}_{end}_synthetic.parquet"
    if synthetic_path.exists():
        return synthetic_path
    synthetic = _synthetic_prices(tickers, start, end, seed)
    synthetic.to_parquet(synthetic_path)
    return synthetic_path
