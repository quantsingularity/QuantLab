"""Comparative benchmark harness.

Runs the QuantLab pipeline across the configurations listed in a YAML file
(typically `configs/benchmark.yaml`) and aggregates their financial and
system-efficiency metrics into a single CSV/markdown summary. This is the
concrete implementation of the "Comparative Evaluation" layer described in
`docs/03_Evaluation_Framework.md`.

Honest scope note: `agents/planner.py`, `agents/hypothesis.py`, and
`agents/literature.py` can optionally call an LLM when `run_config.models`
names one for that stage and `OPENAI_API_KEY` is set, but every one of them
still has a fully deterministic fallback and none is used by default in
`configs/benchmark.yaml`. This harness therefore cannot yet reproduce the
"single-agent GPT-4o baseline" or "human-assisted workflow baseline" from
the thesis proposal, since those describe a different system architecture,
not just an LLM call. What it *does* give you today is a genuine,
reproducible ablation of the reflection layer (`use_reflection: true` vs
`false`), which is the "non-reflective multi-agent baseline" row in the
same table, exercised end to end, not simulated.

Usage:
    python -m quantlab.eval.run_benchmark --config configs/benchmark.yaml
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from quantlab.core.graph import build_graph
from quantlab.core.seeding import DEFAULT_SEED, apply_seed
from quantlab.core.state import ResearchState

_COLUMNS: list[tuple[str, str]] = [
    ("config_name", "Config"),
    ("use_reflection", "Reflection"),
    ("sharpe", "Sharpe"),
    ("cagr", "CAGR"),
    ("max_drawdown", "Max DD"),
    ("win_rate", "Win Rate"),
    ("n_reflections", "# Reflections"),
    ("wall_clock_seconds", "Wall Clock (s)"),
]


def _format_cell(key: str, value: Any) -> str:
    if key == "use_reflection":
        return "Yes" if value else "No"
    if key in ("cagr", "max_drawdown", "win_rate"):
        return f"{value:.2%}"
    if key == "sharpe":
        return f"{value:.2f}"
    if key == "wall_clock_seconds":
        return f"{value:.2f}"
    return str(value)


def _run_once(
    name: str, objective: str, use_reflection: bool, seed: int
) -> dict[str, Any]:
    run_id = uuid.uuid4().hex[:8]
    apply_seed(seed)
    initial: ResearchState = {
        "run_id": run_id,
        "objective": objective,
        "run_config": {"seed": seed},
        "reflections": [],
        "tokens_used": 0,
        "usd_spent": 0.0,
    }

    graph = build_graph(use_reflection=use_reflection)
    t0 = time.perf_counter()
    final = graph.invoke(initial)
    elapsed = time.perf_counter() - t0

    m = final["metrics"]
    return {
        "config_name": name,
        "run_id": run_id,
        "objective": objective,
        "use_reflection": use_reflection,
        "sharpe": m.sharpe,
        "cagr": m.cagr,
        "max_drawdown": m.max_drawdown,
        "win_rate": m.win_rate,
        "n_reflections": len(final.get("reflections", [])),
        "wall_clock_seconds": round(elapsed, 3),
    }


def run_benchmark(config_path: Path) -> list[dict[str, Any]]:
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(spec.get("seed", DEFAULT_SEED))
    results = []
    for run_spec in spec["runs"]:
        print(
            f"[benchmark] running: {run_spec['name']} "
            f"(reflection={run_spec.get('use_reflection', True)})"
        )
        results.append(
            _run_once(
                name=run_spec["name"],
                objective=run_spec["objective"],
                use_reflection=run_spec.get("use_reflection", True),
                seed=seed,
            )
        )

    out_dir = Path(spec.get("output_dir", "./runs/benchmark"))
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "benchmark_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )

    keys = [k for k, _ in _COLUMNS]
    headers = [label for _, label in _COLUMNS]
    formatted_rows = [[_format_cell(k, r[k]) for k in keys] for r in results]

    md_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in formatted_rows:
        md_lines.append("| " + " | ".join(row) + " |")
    (out_dir / "benchmark_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in formatted_rows))
        for i in range(len(headers))
    ]

    def _pad_row(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    console_lines = [
        _pad_row(headers),
        "  ".join("-" * w for w in widths),
    ]
    console_lines.extend(_pad_row(row) for row in formatted_rows)

    print(f"\n[benchmark] done. Results in: {out_dir}")
    print("\n".join(console_lines))
    return results


def main() -> None:
    parser = argparse.ArgumentParser("quantlab.eval.run_benchmark")
    parser.add_argument(
        "--config",
        default="configs/benchmark.yaml",
        help="Path to a benchmark YAML config.",
    )
    args = parser.parse_args()
    run_benchmark(Path(args.config))


if __name__ == "__main__":
    main()
