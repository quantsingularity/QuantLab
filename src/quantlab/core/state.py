"""Shared state schema for the QuantLab agent graph.

All agents read and write instances of :class:`ResearchState`. The state is
serialisable to JSON and is persisted to PostgreSQL after every agent step
so that runs are fully reproducible and interruptible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, TypedDict


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
    statement: (
        str  # e.g. "12-1 cross-sectional momentum yields positive alpha on NASDAQ 100"
    )
    null: str  # H0
    alternative: str  # H1
    expected_sign: str  # "positive" | "negative"
    horizon_days: int
    universe: str
    rationale: str
    supporting_papers: list[str]  # arXiv or SSRN ids


@dataclass
class FeatureSpec:
    name: str
    formula: str  # human-readable spec, e.g. "close[t-21] / close[t-252] - 1"
    lookback_days: int
    parquet_path: str  # local cache path
    lineage: list[str]  # list of source columns, checked by the leakage guard


@dataclass
class ModelArtifact:
    kind: str  # "rank_signal" | "linear" | "xgboost" | ...
    params: dict[str, Any]
    fitted_path: Optional[str] = None


@dataclass
class BacktestResult:
    equity_curve_path: str  # parquet
    trades_path: str  # parquet
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
    agent: str
    stage: str
    critique: str
    severity: str  # "info" | "warn" | "error"
    ts: datetime = field(default_factory=datetime.utcnow)


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
