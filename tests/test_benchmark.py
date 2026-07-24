"""Tests for the comparative benchmark harness."""

from __future__ import annotations

from pathlib import Path

import yaml

from quantlab.eval.run_benchmark import run_benchmark


def test_run_benchmark_produces_a_row_per_run_config(tmp_path):
    spec = {
        "output_dir": str(tmp_path / "benchmark"),
        "seed": 5,
        "runs": [
            {
                "name": "reflective",
                "objective": "Develop a momentum strategy for the NASDAQ 100",
                "use_reflection": True,
            },
            {
                "name": "non_reflective",
                "objective": "Develop a momentum strategy for the NASDAQ 100",
                "use_reflection": False,
            },
        ],
    }
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(yaml.dump(spec), encoding="utf-8")

    results = run_benchmark(config_path)

    assert len(results) == 2
    names = {r["config_name"] for r in results}
    assert names == {"reflective", "non_reflective"}
    for r in results:
        assert "sharpe" in r and "cagr" in r and "max_drawdown" in r

    out_dir = Path(spec["output_dir"])
    assert (out_dir / "benchmark_results.json").exists()
    assert (out_dir / "benchmark_summary.md").exists()
