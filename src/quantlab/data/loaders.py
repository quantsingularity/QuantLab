"""Price loaders. PoC uses yfinance for the NASDAQ 100."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Small, static NASDAQ 100 slice for the PoC. Replace with a point-in-time
# constituent list in the thesis version.
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


def load_prices(universe: str, start: str, end: str, cache_dir: Path) -> Path:
    """Download daily close prices and cache them to parquet.

    Returns the path to a wide-format parquet with a DatetimeIndex and one
    column per ticker.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{universe}_{start}_{end}.parquet"
    if out.exists():
        return out

    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "Install yfinance for the PoC: `pip install yfinance`."
        ) from e

    tickers = NASDAQ_100_POC if universe == "nasdaq_100" else [universe]
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    close = close.dropna(how="all")
    close.to_parquet(out)
    return out
