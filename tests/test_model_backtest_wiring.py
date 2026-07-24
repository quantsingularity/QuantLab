"""Regression test: model hyperparameters must actually reach the backtest.

The original repository computed a ModelArtifact with lookback/skip/top_pct
but the backtest agent silently ignored it and always used hardcoded
defaults. This test fails if that wiring ever breaks again.
"""

from __future__ import annotations

from pathlib import Path

from quantlab.agents import backtest as backtest_agent
from quantlab.agents import model as model_agent
from quantlab.core.state import FeatureSpec, Hypothesis, ResearchState
from quantlab.data.loaders import load_prices


def _base_state(tmp_path: Path, model_params: dict) -> ResearchState:
    prices_path = load_prices(
        "nasdaq_100", "2015-01-01", "2020-01-01", tmp_path, seed=9
    )
    hypothesis = Hypothesis(
        statement="test",
        null="test",
        alternative="test",
        expected_sign="positive",
        horizon_days=21,
        universe="nasdaq_100",
        rationale="test",
        supporting_papers=[],
    )
    return {
        "run_id": "wiring_test",
        "objective": "test",
        "run_config": {
            "model_kind": "rank_signal",
            "model_params": model_params,
            "transaction_cost_bps": 5.0,
            "seed": 1,
        },
        "output_dir": str(tmp_path / "out"),
        "hypothesis": hypothesis,
        "features": [
            FeatureSpec(
                name="mom_12_1",
                formula="close[t-21] / close[t-252] - 1",
                lookback_days=252,
                parquet_path=str(prices_path),
                lineage=["close"],
            )
        ],
    }


def test_model_params_are_read_from_config(tmp_path):
    state = _base_state(tmp_path, {"lookback": 126, "skip": 10, "top_pct": 0.2})
    state = model_agent.run(state)
    assert state["model"].params["lookback"] == 126
    assert state["model"].params["skip"] == 10
    assert state["model"].params["top_pct"] == 0.2


def test_changing_top_pct_changes_realised_turnover(tmp_path):
    narrow_state = model_agent.run(_base_state(tmp_path / "narrow", {"top_pct": 0.05}))
    narrow_state = backtest_agent.run(narrow_state)

    wide_state = model_agent.run(_base_state(tmp_path / "wide", {"top_pct": 0.5}))
    wide_state = backtest_agent.run(wide_state)

    narrow_weights_nonzero = (
        narrow_state["_daily_returns"] != wide_state["_daily_returns"]
    ).any()
    assert narrow_weights_nonzero


def test_backtest_dispatches_to_ml_signal_for_ridge_kind(tmp_path):
    state = _base_state(tmp_path, {"top_pct": 0.2, "min_train_periods": 6})
    state["run_config"]["model_kind"] = "ridge"
    state = model_agent.run(state)
    assert state["model"].kind == "ridge"
    state = backtest_agent.run(state)
    assert "backtest" in state
    assert state["_daily_returns"] is not None
