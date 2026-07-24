"""Shared pytest fixtures.

Every test runs fully offline against a throwaway memory file, regardless
of what a developer's shell environment happens to have set, so the suite
never touches a real Postgres instance or the developer's own
./runs/memory.jsonl.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _sandboxed_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUANTLAB_OFFLINE", "1")
    monkeypatch.setenv("QUANTLAB_MEMORY", str(tmp_path / "memory.jsonl"))
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def price_frame():
    """A small deterministic synthetic price panel for strategy-level tests."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(7)
    index = pd.bdate_range("2018-01-01", "2020-06-30")
    tickers = [f"T{i}" for i in range(12)]
    drift = rng.uniform(0.0002, 0.0005, size=len(tickers))
    vol = rng.uniform(0.01, 0.02, size=len(tickers))
    shocks = rng.normal(0.0, 1.0, size=(len(index), len(tickers)))
    log_returns = drift + vol * shocks
    log_prices = np.cumsum(log_returns, axis=0)
    start_price = rng.uniform(30.0, 150.0, size=len(tickers))
    prices = start_price * np.exp(log_prices)
    return pd.DataFrame(prices, index=index, columns=tickers)
