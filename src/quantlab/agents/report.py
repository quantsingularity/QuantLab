"""Research Report Agent.

Renders the deterministic markdown/PDF/JSON report bundle. When an LLM is
configured for this stage in run_config.models and available, an additional
short discussion paragraph is drafted and inserted into the Results section;
if the LLM path is unavailable, exceeds the run's budget, or returns
something that does not validate, the report is still rendered in full,
just without that paragraph.
"""

from __future__ import annotations

from pathlib import Path

from quantlab.core.llm import (
    LLM,
    BudgetExceededError,
    LLMUnavailableError,
    apply_usage,
    within_budget,
)
from quantlab.core.state import ResearchState
from quantlab.report.writer import (
    render_markdown,
    save_equity_curve_png,
    save_markdown,
    save_metrics_json,
    save_reflections_jsonl,
    save_report_pdf,
)

DISCUSSION_SYSTEM = """You write a short discussion paragraph for a quantitative finance
research report. Given the hypothesis, realised metrics, and any reflective critiques,
output a JSON object with a single string key "discussion": two to four sentences
interpreting the result in plain prose, written for a quantitative research audience.
Do not just repeat the numbers verbatim; interpret them."""


def _draft_discussion(state: ResearchState) -> str | None:
    cfg = state.get("run_config", {})
    model_name = cfg.get("models", {}).get("report")
    if not model_name or not LLM.available():
        return None
    budget = cfg.get("budget")
    if not within_budget(state, budget):
        return None

    h = state["hypothesis"]
    m = state["metrics"]
    refs = state.get("reflections", [])
    critiques = "; ".join(r.critique for r in refs) or "none"
    user = (
        f"Hypothesis: {h.statement}\n"
        f"Sharpe: {m.sharpe:.2f}, CAGR: {m.cagr:.2%}, Max drawdown: {m.max_drawdown:.2%}, "
        f"Win rate: {m.win_rate:.2%}\n"
        f"Reflective critiques: {critiques}"
    )
    try:
        llm = LLM(model=model_name)
        parsed, response = llm.complete_json(DISCUSSION_SYSTEM, user)
        apply_usage(state, response, budget)
    except (LLMUnavailableError, ValueError, BudgetExceededError):
        return None

    discussion = parsed.get("discussion")
    if isinstance(discussion, str) and discussion.strip():
        return discussion.strip()
    return None


def run(state: ResearchState) -> ResearchState:
    out_dir = Path(state.get("output_dir", "./runs")) / state["run_id"]
    chart_path = save_equity_curve_png(state["backtest"].equity_curve_path, out_dir)
    discussion = _draft_discussion(state)
    md = render_markdown(
        state,
        chart_filename=chart_path.name if chart_path else None,
        discussion=discussion,
    )
    save_markdown(md, out_dir)
    save_metrics_json(state, out_dir)
    save_reflections_jsonl(state.get("reflections", []), out_dir)
    save_report_pdf(md, out_dir)
    state["report_md"] = md
    return state
