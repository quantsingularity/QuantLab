"""Smoke tests for the markdown report renderer."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from quantlab.core.state import (
    BacktestResult,
    FeatureSpec,
    Hypothesis,
    MetricsBundle,
    ModelArtifact,
    ResearchState,
)
from quantlab.data.loaders import load_prices
from quantlab.report.writer import render_markdown


def _state(tmp_path: Path) -> ResearchState:
    prices_path = load_prices(
        "nasdaq_100", "2015-01-01", "2020-01-01", tmp_path, seed=4
    )
    hypothesis = Hypothesis(
        statement="Momentum works.",
        null="Sharpe <= 0.",
        alternative="Sharpe > 0.",
        expected_sign="positive",
        horizon_days=21,
        universe="nasdaq_100",
        rationale="Because momentum.",
        supporting_papers=["Jegadeesh and Titman"],
    )
    metrics = MetricsBundle(
        sharpe=0.75, cagr=0.12, max_drawdown=-0.18, win_rate=0.55, turnover=6.0
    )
    backtest = BacktestResult(
        equity_curve_path=str(tmp_path / "equity.parquet"),
        trades_path=str(tmp_path / "trades.parquet"),
        start=datetime(2015, 1, 1, tzinfo=UTC),
        end=datetime(2020, 1, 1, tzinfo=UTC),
        transaction_cost_bps=5.0,
    )
    return {
        "run_id": "report_test",
        "objective": "Develop a momentum strategy for the NASDAQ 100",
        "run_config": {"oos_start": "2018-01-01"},
        "hypothesis": hypothesis,
        "metrics": metrics,
        "backtest": backtest,
        "model": ModelArtifact(
            kind="rank_signal", params={"lookback": 252, "top_pct": 0.1}
        ),
        "reflections": [],
        "literature": [],
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


def test_render_markdown_contains_all_numbered_sections(tmp_path):
    md = render_markdown(_state(tmp_path))
    for i in range(1, 10):
        assert f"## {i}." in md


def test_render_markdown_notes_synthetic_data_source(tmp_path):
    md = render_markdown(_state(tmp_path))
    assert "synthetic" in md.lower()


def test_render_markdown_includes_discussion_when_provided(tmp_path):
    md = render_markdown(
        _state(tmp_path),
        discussion="This result is broadly consistent with prior literature.",
    )
    assert "Discussion" in md
    assert "broadly consistent with prior literature" in md


def test_render_markdown_omits_discussion_when_not_provided(tmp_path):
    md = render_markdown(_state(tmp_path), discussion=None)
    assert "**Discussion:**" not in md


def test_render_markdown_reflects_model_kind(tmp_path):
    state = _state(tmp_path)
    state["model"] = ModelArtifact(
        kind="ridge", params={"lookback": 126, "top_pct": 0.2}
    )
    md = render_markdown(state)
    assert "`ridge`" in md
