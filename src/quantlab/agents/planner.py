"""Research Planner Agent: decomposes the objective into a task DAG."""

from __future__ import annotations

from quantlab.core.state import ResearchState

PLANNER_SYSTEM = """You are the research planner of QuantLab. Given a natural-language
research objective, output a JSON task DAG with keys: literature, hypothesis, data,
model, backtest, evaluate, report. Each entry has: `depends_on` (list) and `notes`.
Be concise and testable. Do not invent tools."""


def run(state: ResearchState) -> ResearchState:

    state["task_dag"] = {
        "literature": {
            "depends_on": [],
            "notes": "Retrieve 3 to 5 papers on the topic.",
        },
        "hypothesis": {
            "depends_on": ["literature"],
            "notes": "Formalise a testable hypothesis.",
        },
        "data": {
            "depends_on": ["hypothesis"],
            "notes": "Load prices and build features.",
        },
        "model": {"depends_on": ["data"], "notes": "Build ranked signal or ML model."},
        "backtest": {
            "depends_on": ["model"],
            "notes": "Vectorised backtest with costs.",
        },
        "evaluate": {
            "depends_on": ["backtest"],
            "notes": "Compute financial and quality metrics.",
        },
        "report": {"depends_on": ["evaluate"], "notes": "Compile markdown report."},
    }
    return state
