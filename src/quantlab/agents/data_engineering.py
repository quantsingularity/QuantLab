"""Data Engineering Agent: universe construction, price download, feature build."""

from __future__ import annotations

from pathlib import Path

from quantlab.core.state import FeatureSpec, ResearchState
from quantlab.data.loaders import load_prices

CACHE_DIR = Path("./runs/cache")


def run(state: ResearchState) -> ResearchState:
    cfg = state.get("run_config", {})
    universe = cfg.get("universe", state["hypothesis"].universe)
    start = cfg.get("start", "2010-01-01")
    end = cfg.get("end", "2025-01-01")
    prices_path = load_prices(
        universe=universe, start=start, end=end, cache_dir=CACHE_DIR
    )

    state["features"] = [
        FeatureSpec(
            name="mom_12_1",
            formula="close[t-21] / close[t-252] - 1",
            lookback_days=252,
            parquet_path=str(prices_path),
            lineage=["close"],
        )
    ]
    return state
