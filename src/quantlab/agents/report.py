"""Research Report Agent."""

from __future__ import annotations

from pathlib import Path

from quantlab.core.state import ResearchState
from quantlab.report.writer import (
    render_markdown,
    save_equity_curve_png,
    save_markdown,
    save_metrics_json,
    save_reflections_jsonl,
    save_report_pdf,
)


def run(state: ResearchState) -> ResearchState:
    out_dir = Path(state.get("output_dir", "./runs")) / state["run_id"]
    chart_path = save_equity_curve_png(state["backtest"].equity_curve_path, out_dir)
    md = render_markdown(state, chart_filename=chart_path.name if chart_path else None)
    save_markdown(md, out_dir)
    save_metrics_json(state, out_dir)
    save_reflections_jsonl(state.get("reflections", []), out_dir)
    save_report_pdf(md, out_dir)
    state["report_md"] = md
    return state
