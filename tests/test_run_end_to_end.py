"""End-to-end smoke test for the full agent graph, entirely offline."""

from __future__ import annotations

from pathlib import Path

from quantlab.core.graph import build_graph
from quantlab.core.seeding import apply_seed
from quantlab.core.state import ResearchState


def _initial_state(tmp_path: Path) -> ResearchState:
    return {
        "run_id": "e2e_test",
        "objective": "Develop a momentum strategy for the NASDAQ 100",
        "run_config": {
            "universe": "nasdaq_100",
            "start": "2016-01-01",
            "end": "2020-06-01",
            "oos_start": "2019-01-01",
            "transaction_cost_bps": 5.0,
            "seed": 3,
        },
        "output_dir": str(tmp_path),
        "reflections": [],
        "tokens_used": 0,
        "usd_spent": 0.0,
    }


def test_full_pipeline_runs_end_to_end_offline(tmp_path):
    apply_seed(3)
    graph = build_graph()
    final = graph.invoke(_initial_state(tmp_path))

    assert "metrics" in final
    assert final.get("report_md")
    assert "## 1. Research Hypothesis" in final["report_md"]

    out_dir = Path(final["backtest"].equity_curve_path).parent
    assert (out_dir / "report.md").exists()
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "equity_curve.parquet").exists()


def test_pipeline_without_reflection_skips_critique_layer(tmp_path):
    apply_seed(3)
    graph = build_graph(use_reflection=False)
    final = graph.invoke(_initial_state(tmp_path))
    assert final.get("reflections", []) == []
