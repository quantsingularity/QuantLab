"""Shared state schema for the QuantLab agent graph.

All agents read and write instances of ResearchState. The state is
serialisable to JSON and is persisted to PostgreSQL after every agent step
so that runs are fully reproducible and interruptible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, TypedDict


@dataclass
class PaperSummary:
    title: str
    authors: list[str]
    year: int
    url: str
    abstract: str
    key_findings: str
    relevance_score: float


@dataclass
class Hypothesis:
    """A single, testable research hypothesis.

    The statement is a free-text claim, for example "12-1 cross-sectional
    momentum yields positive alpha on NASDAQ 100". null is the H0 to be
    rejected, alternative is H1, expected_sign is either "positive" or
    "negative", and supporting_papers holds arXiv or SSRN identifiers.
    """

    statement: str
    null: str
    alternative: str
    expected_sign: str
    horizon_days: int
    universe: str
    rationale: str
    supporting_papers: list[str]


@dataclass
class FeatureSpec:
    """A single engineered feature.

    formula is a human-readable spec such as "close[t-21] / close[t-252] - 1",
    parquet_path is the local cache path for the underlying price data, and
    lineage lists the source columns consumed, checked by the leakage guard.
    """

    name: str
    formula: str
    lookback_days: int
    parquet_path: str
    lineage: list[str]


@dataclass
class ModelArtifact:
    """The fitted signal or model used by the strategy.

    kind is one of "rank_signal", "ridge", or "xgboost". params holds the
    hyperparameters consumed by the corresponding strategy implementation in
    quantlab.strategies. fitted_path is set when a persisted model artifact
    exists on disk; walk-forward signal kinds refit at every rebalance and
    leave it unset.
    """

    kind: str
    params: dict[str, Any]
    fitted_path: str | None = None


@dataclass
class BacktestResult:
    equity_curve_path: str
    trades_path: str
    start: datetime
    end: datetime
    transaction_cost_bps: float


@dataclass
class MetricsBundle:
    sharpe: float
    cagr: float
    max_drawdown: float
    win_rate: float
    turnover: float
    extra: dict[str, float] = field(default_factory=dict)


@dataclass
class Reflection:
    """A single critique emitted by the Reflective Memory Agent.

    severity is one of "info", "warn", or "error".
    """

    agent: str
    stage: str
    critique: str
    severity: str
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))


class ResearchState(TypedDict, total=False):
    """LangGraph passes this dict between nodes."""

    run_id: str
    objective: str
    run_config: dict[str, Any]
    output_dir: str
    task_dag: dict[str, Any]
    literature: list[PaperSummary]
    hypothesis: Hypothesis
    features: list[FeatureSpec]
    model: ModelArtifact
    backtest: BacktestResult
    metrics: MetricsBundle
    report_md: str
    reflections: list[Reflection]
    tokens_used: int
    usd_spent: float
    _daily_returns: Any
    _daily_turnover: Any
