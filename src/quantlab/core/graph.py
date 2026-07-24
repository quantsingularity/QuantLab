"""LangGraph orchestration for the nine-agent QuantLab pipeline.

The graph is intentionally linear for the PoC; the Reflective Memory Agent
is called after every node via a shared side-channel rather than as an edge
so that adding reflection does not change the topology.
"""

from __future__ import annotations

from collections.abc import Callable

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from quantlab.agents import backtest as backtest_agent
from quantlab.agents import data_engineering as data_agent
from quantlab.agents import evaluation as eval_agent
from quantlab.agents import hypothesis as hypothesis_agent
from quantlab.agents import literature as literature_agent
from quantlab.agents import model as model_agent
from quantlab.agents import planner as planner_agent
from quantlab.agents import reflection as reflection_agent
from quantlab.agents import report as report_agent
from quantlab.core.state import ResearchState


def _with_reflection(
    node: Callable[[ResearchState], ResearchState], stage: str
) -> Callable[[ResearchState], ResearchState]:
    """Wrap a node so the Reflective Memory Agent critiques its output."""

    def wrapped(state: ResearchState) -> ResearchState:
        new_state = node(state)
        return reflection_agent.critique(new_state, stage=stage)

    return wrapped


def build_graph(use_reflection: bool = True) -> CompiledStateGraph:
    """Build the QuantLab agent graph.

    `use_reflection=False` skips the Reflective Memory Agent's post-stage
    critique at every node, producing the "non-reflective multi-agent
    baseline" referenced in docs/03_Evaluation_Framework.md's comparative
    evaluation section. Everything else about the topology is identical, so
    the comparison isolates the effect of the reflection layer.
    """
    g: StateGraph = StateGraph(ResearchState)

    def _wrap(
        node: Callable[[ResearchState], ResearchState], stage: str
    ) -> Callable[[ResearchState], ResearchState]:
        return _with_reflection(node, stage) if use_reflection else node

    g.add_node("planner", _wrap(planner_agent.run, "planner"))
    g.add_node("literature", _wrap(literature_agent.run, "literature"))
    g.add_node("hypothesis", _wrap(hypothesis_agent.run, "hypothesis"))
    g.add_node("data", _wrap(data_agent.run, "data"))
    g.add_node("model", _wrap(model_agent.run, "model"))
    g.add_node("backtest", _wrap(backtest_agent.run, "backtest"))
    g.add_node("evaluate", _wrap(eval_agent.run, "evaluate"))
    g.add_node("report", _wrap(report_agent.run, "report"))

    g.set_entry_point("planner")
    g.add_edge("planner", "literature")
    g.add_edge("literature", "hypothesis")
    g.add_edge("hypothesis", "data")
    g.add_edge("data", "model")
    g.add_edge("model", "backtest")
    g.add_edge("backtest", "evaluate")
    g.add_edge("evaluate", "report")
    g.add_edge("report", END)

    return g.compile()
