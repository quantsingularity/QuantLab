"""Model Development Agent.

Chooses the strategy's model kind and hyperparameters from the run config,
falling back to the deterministic rank-based momentum signal when the
config does not specify one. rank_signal ranks assets directly by trailing
momentum. ridge and xgboost instead walk-forward fit a regression model on
the same momentum feature at every rebalance date and rank by predicted
forward return; see quantlab.strategies.ml_signal for the fitting logic.
"""

from __future__ import annotations

from quantlab.core.state import ModelArtifact, ResearchState
from quantlab.strategies.momentum import DEFAULT_LOOKBACK, DEFAULT_SKIP, DEFAULT_TOP_PCT

DEFAULT_MIN_TRAIN_PERIODS = 12
_ML_KINDS = ("ridge", "xgboost")


def run(state: ResearchState) -> ResearchState:
    cfg = state.get("run_config", {})
    kind = cfg.get("model_kind", "rank_signal")
    overrides = cfg.get("model_params", {})

    params: dict[str, object] = {
        "lookback": int(overrides.get("lookback", DEFAULT_LOOKBACK)),
        "skip": int(overrides.get("skip", DEFAULT_SKIP)),
        "top_pct": float(overrides.get("top_pct", DEFAULT_TOP_PCT)),
        "long_only": bool(overrides.get("long_only", True)),
    }
    if kind in _ML_KINDS:
        params["min_train_periods"] = int(
            overrides.get("min_train_periods", DEFAULT_MIN_TRAIN_PERIODS)
        )

    state["model"] = ModelArtifact(kind=kind, params=params)
    return state
