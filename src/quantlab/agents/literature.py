"""Literature Review Agent: retrieves and summarises papers.

Tries the public arXiv API first (a real HTTP call, stdlib only, no extra
dependency). If arXiv is unreachable, times out, or returns nothing
parseable, entirely plausible in a sandboxed or offline environment, this
falls back to a small bundled seed corpus of momentum papers so the
pipeline never crashes and always runs deterministically end to end. When
an LLM is configured for this stage and available, retrieved abstracts are
additionally condensed into a one-sentence key finding; the seed corpus
already carries curated key findings and is left untouched either way.

Query construction and relevance filtering
-------------------------------------------
arXiv's `all:` search field matches loosely against every metadata field, so
sending the raw objective sentence verbatim (for example "Develop a
momentum strategy for the NASDAQ 100") can surface physics papers that
happen to use the word "momentum" in its literal sense (particle momentum,
angular momentum, and so on) instead of the financial factor. The fix is
two-fold: (1) the query is scoped to the arXiv quantitative-finance category
(`cat:q-fin.*`) and built from finance-relevant keywords extracted from the
objective rather than the raw sentence, and (2) every candidate paper,
regardless of category, is additionally checked against a finance-domain
keyword filter before it is allowed into the report. Only if fewer than 3
papers survive both filters do we fall back to SEED_CORPUS.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

from quantlab.core.llm import (
    LLM,
    BudgetExceededError,
    LLMUnavailableError,
    apply_usage,
    within_budget,
)
from quantlab.core.state import PaperSummary, ResearchState

_ARXIV_API = "http://export.arxiv.org/api/query"
_ARXIV_TIMEOUT_SECONDS = 5
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

_QUERY_STOPWORDS = {
    "a",
    "an",
    "the",
    "for",
    "of",
    "on",
    "in",
    "using",
    "with",
    "to",
    "and",
    "or",
    "develop",
    "design",
    "build",
    "create",
    "implement",
    "explore",
    "test",
    "backtest",
    "strategy",
    "strategies",
    "approach",
    "model",
    "research",
    "study",
    "analysis",
    "analyze",
    "investigate",
}

_FINANCE_HINT_TERMS = (
    "moment",
    "portfolio",
    "equit",
    "stock",
    "asset",
    "return",
    "factor",
    "trading",
    "trader",
    "market",
    "financ",
    "invest",
    "price",
    "pricing",
    "volatilit",
    "sharpe",
    "alpha",
    "risk-adjusted",
    "arbitrage",
    "hedge",
    "capm",
    "cross-sectional",
    "cross sectional",
    "yield",
    "security",
    "securities",
)

SUMMARY_SYSTEM = """You summarise finance research papers for a research report. Given a
JSON list of objects with index, title, abstract, output a JSON object with a single
key "summaries": a list of objects with index and summary, where summary is a single
sentence stating the paper's key finding for a quantitative finance audience."""


def _extract_query_terms(objective: str, max_terms: int = 6) -> list[str]:
    """Pull a handful of finance-relevant keywords out of a free-text objective."""
    words = re.findall(r"[A-Za-z][A-Za-z\-]*", objective.lower())
    terms = [w for w in words if w not in _QUERY_STOPWORDS and len(w) > 2]
    seen: list[str] = []
    for w in terms:
        if w not in seen:
            seen.append(w)
    return seen[:max_terms] or ["momentum", "equities"]


def _is_finance_relevant(paper: PaperSummary) -> bool:
    haystack = f"{paper.title} {paper.abstract}".lower()
    return any(term in haystack for term in _FINANCE_HINT_TERMS)


def _build_search_query(topic: str) -> str:
    terms = _extract_query_terms(topic)
    abs_clause = " OR ".join(f'abs:"{t}"' for t in terms)
    return f"cat:q-fin.* AND ({abs_clause})"


def _query_arxiv(topic: str, max_results: int = 5) -> list[PaperSummary]:
    """Query the public arXiv API and return up to max_results finance papers.

    Returns an empty list, and never raises, if arXiv is unreachable, the
    response cannot be parsed, or no relevant entries come back. Callers
    are expected to fall back to SEED_CORPUS in that case.
    """
    params = urllib.parse.urlencode(
        {
            "search_query": _build_search_query(topic),
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
        }
    )
    url = f"{_ARXIV_API}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=_ARXIV_TIMEOUT_SECONDS) as resp:
            raw = resp.read()
        root = ET.fromstring(raw)
    except (urllib.error.URLError, TimeoutError, ET.ParseError, OSError):
        return []

    papers: list[PaperSummary] = []
    for i, entry in enumerate(root.findall("atom:entry", _ATOM_NS)):
        title_el = entry.find("atom:title", _ATOM_NS)
        summary_el = entry.find("atom:summary", _ATOM_NS)
        id_el = entry.find("atom:id", _ATOM_NS)
        published_el = entry.find("atom:published", _ATOM_NS)
        authors = [
            a.findtext("atom:name", default="", namespaces=_ATOM_NS)
            for a in entry.findall("atom:author", _ATOM_NS)
        ]
        if title_el is None or summary_el is None or id_el is None:
            continue
        year = (
            int(published_el.text[:4])
            if published_el is not None and published_el.text
            else 0
        )
        abstract = " ".join(summary_el.text.split()) if summary_el.text else ""
        paper = PaperSummary(
            title=" ".join(title_el.text.split()) if title_el.text else "Untitled",
            authors=[a for a in authors if a],
            year=year,
            url=id_el.text or "",
            abstract=abstract,
            key_findings=abstract[:280],
            relevance_score=round(1.0 - i * 0.1, 2),
        )
        if _is_finance_relevant(paper):
            papers.append(paper)
    return papers


SEED_CORPUS: list[PaperSummary] = [
    PaperSummary(
        title="Returns to Buying Winners and Selling Losers",
        authors=["Jegadeesh, N.", "Titman, S."],
        year=1993,
        url="https://doi.org/10.1111/j.1540-6261.1993.tb04702.x",
        abstract=(
            "Documents cross-sectional momentum: stocks that performed well over "
            "the past 3 to 12 months continue to outperform for the following "
            "3 to 12 months."
        ),
        key_findings=(
            "A zero-cost 12-1 momentum portfolio earned roughly 1 percent per "
            "month between 1965 and 1989 on US equities."
        ),
        relevance_score=1.0,
    ),
    PaperSummary(
        title="Momentum",
        authors=["Asness, C.", "Frazzini, A.", "Israel, R.", "Moskowitz, T."],
        year=2014,
        url="https://www.aqr.com/Insights/Research/Journal-Article/Momentum",
        abstract="Reviews momentum evidence across asset classes and addresses common critiques.",
        key_findings="Momentum survives out of sample, across countries, and across asset classes.",
        relevance_score=0.9,
    ),
    PaperSummary(
        title="Time Series Momentum",
        authors=["Moskowitz, T.", "Ooi, Y. H.", "Pedersen, L. H."],
        year=2012,
        url="https://doi.org/10.1016/j.jfineco.2011.11.003",
        abstract="Time-series momentum in 58 liquid instruments across four asset classes.",
        key_findings="One- to twelve-month past returns predict future returns of the same asset.",
        relevance_score=0.85,
    ),
]


def _summarize_with_llm(
    papers: list[PaperSummary], model_name: str, state: ResearchState
) -> None:
    cfg = state.get("run_config", {})
    budget = cfg.get("budget")
    if not within_budget(state, budget):
        return

    payload = [
        {"index": i, "title": p.title, "abstract": p.abstract}
        for i, p in enumerate(papers)
    ]
    try:
        llm = LLM(model=model_name)
        parsed, response = llm.complete_json(SUMMARY_SYSTEM, json.dumps(payload))
        apply_usage(state, response, budget)
    except (LLMUnavailableError, ValueError, BudgetExceededError):
        return

    summaries: Any = parsed.get("summaries")
    if not isinstance(summaries, list):
        return
    for entry in summaries:
        if not isinstance(entry, dict):
            continue
        index = entry.get("index")
        summary = entry.get("summary")
        if (
            isinstance(index, int)
            and 0 <= index < len(papers)
            and isinstance(summary, str)
            and summary.strip()
        ):
            papers[index].key_findings = summary.strip()


def run(state: ResearchState) -> ResearchState:
    topic = state.get("objective") or "momentum investing equities"
    retrieved = _query_arxiv(topic)
    used_seed_corpus = len(retrieved) < 3
    papers = SEED_CORPUS if used_seed_corpus else retrieved

    if not used_seed_corpus:
        model_name = state.get("run_config", {}).get("models", {}).get("summariser")
        if model_name and LLM.available():
            _summarize_with_llm(papers, model_name, state)

    state["literature"] = papers
    return state
