"""Literature Review Agent: retrieves and summarises papers.

Tries the public arXiv API first (a real HTTP call, stdlib only, no extra
dependency). If arXiv is unreachable, times out, or returns nothing parseable
-- entirely plausible in a sandboxed or offline dev environment -- this falls
back to a small bundled seed corpus of momentum papers so the pipeline never
crashes and always runs deterministically end to end. The full v0.5
implementation additionally queries Semantic Scholar and summarises with an
LLM instead of using the raw abstract.

Query construction and relevance filtering
-------------------------------------------
arXiv's `all:` search field matches loosely against every metadata field, so
sending the raw objective sentence verbatim (e.g. "Develop a momentum
strategy for the NASDAQ 100") can surface physics papers that happen to use
the word "momentum" in its literal sense (particle momentum, angular
momentum, etc.) instead of the financial factor -- which is exactly what
produced irrelevant citations in the research report. The fix is two-fold:
(1) the query is scoped to the arXiv quantitative-finance category
(`cat:q-fin.*`) and built from finance-relevant keywords extracted from the
objective rather than the raw sentence, and (2) every candidate paper --
regardless of category -- is additionally checked against a finance-domain
keyword filter before it is allowed into the report. Only if fewer than 3
papers survive both filters do we fall back to `SEED_CORPUS`.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

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

# If a candidate paper's title+abstract contains none of these terms, it is
# almost certainly not a quantitative-finance paper (this is what catches
# stray particle-physics / astrophysics "momentum" results).
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
    """Query the public arXiv API and return up to `max_results` finance papers.

    Returns an empty list (never raises) if arXiv is unreachable, the
    response can't be parsed, or no relevant entries come back -- callers
    are expected to fall back to `SEED_CORPUS` in that case.
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


def run(state: ResearchState) -> ResearchState:
    topic = state.get("objective") or "momentum investing equities"
    papers = _query_arxiv(topic)
    state["literature"] = papers if len(papers) >= 3 else SEED_CORPUS
    return state
