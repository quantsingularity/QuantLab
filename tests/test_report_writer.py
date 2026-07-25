"""Smoke tests for the markdown report renderer."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from quantlab.core.state import (
    BacktestResult,
    FeatureSpec,
    Hypothesis,
    MetricsBundle,
    ModelArtifact,
    ResearchState,
)
from quantlab.data.loaders import load_prices
from quantlab.report.writer import (
    _inline_to_reportlab_markup,
    _markdown_to_flowables,
    render_markdown,
)


def _state(tmp_path: Path) -> ResearchState:
    prices_path = load_prices(
        "nasdaq_100", "2015-01-01", "2020-01-01", tmp_path, seed=4
    )
    hypothesis = Hypothesis(
        statement="Momentum works.",
        null="Sharpe <= 0.",
        alternative="Sharpe > 0.",
        expected_sign="positive",
        horizon_days=21,
        universe="nasdaq_100",
        rationale="Because momentum.",
        supporting_papers=["Jegadeesh and Titman"],
    )
    metrics = MetricsBundle(
        sharpe=0.75, cagr=0.12, max_drawdown=-0.18, win_rate=0.55, turnover=6.0
    )
    backtest = BacktestResult(
        equity_curve_path=str(tmp_path / "equity.parquet"),
        trades_path=str(tmp_path / "trades.parquet"),
        start=datetime(2015, 1, 1, tzinfo=UTC),
        end=datetime(2020, 1, 1, tzinfo=UTC),
        transaction_cost_bps=5.0,
    )
    return {
        "run_id": "report_test",
        "objective": "Develop a momentum strategy for the NASDAQ 100",
        "run_config": {"oos_start": "2018-01-01"},
        "hypothesis": hypothesis,
        "metrics": metrics,
        "backtest": backtest,
        "model": ModelArtifact(
            kind="rank_signal", params={"lookback": 252, "top_pct": 0.1}
        ),
        "reflections": [],
        "literature": [],
        "features": [
            FeatureSpec(
                name="mom_12_1",
                formula="close[t-21] / close[t-252] - 1",
                lookback_days=252,
                parquet_path=str(prices_path),
                lineage=["close"],
            )
        ],
    }


def test_render_markdown_contains_all_numbered_sections(tmp_path):
    md = render_markdown(_state(tmp_path))
    for i in range(1, 10):
        assert f"## {i}." in md


def test_render_markdown_notes_synthetic_data_source(tmp_path):
    md = render_markdown(_state(tmp_path))
    assert "synthetic" in md.lower()


def test_render_markdown_includes_discussion_when_provided(tmp_path):
    md = render_markdown(
        _state(tmp_path),
        discussion="This result is broadly consistent with prior literature.",
    )
    assert "Discussion" in md
    assert "broadly consistent with prior literature" in md


def test_render_markdown_omits_discussion_when_not_provided(tmp_path):
    md = render_markdown(_state(tmp_path), discussion=None)
    assert "**Discussion:**" not in md


def test_render_markdown_reflects_model_kind(tmp_path):
    state = _state(tmp_path)
    state["model"] = ModelArtifact(
        kind="ridge", params={"lookback": 126, "top_pct": 0.2}
    )
    md = render_markdown(state)
    assert "`ridge`" in md


def test_section_heading_is_kept_together_with_its_first_content_block(tmp_path):
    """Regression test: a heading must never be left as the last flowable on a page.

    quantlab.report.writer used to append each heading straight to the PDF
    story, so ReportLab's automatic pagination could strand a heading alone
    at the bottom of a page with its content pushed to the next one. Every
    heading must now be grouped with the content that immediately follows
    it inside a single KeepTogether.
    """
    from reportlab.platypus import HRFlowable, KeepTogether, Paragraph

    md = "## 3. Data\n\nSome data description paragraph.\n"
    flowables = _markdown_to_flowables(md, tmp_path)

    groups = [f for f in flowables if isinstance(f, KeepTogether)]
    assert len(groups) == 1
    group_contents = groups[0]._content
    assert isinstance(group_contents[0], Paragraph)
    assert "3. Data" in group_contents[0].text
    assert any(isinstance(item, HRFlowable) for item in group_contents)
    assert isinstance(group_contents[-1], Paragraph)
    assert "Some data description paragraph" in group_contents[-1].text


def test_trailing_heading_with_no_following_content_is_still_grouped(tmp_path):
    from reportlab.platypus import KeepTogether

    md = "## 9. Reproducibility\n"
    flowables = _markdown_to_flowables(md, tmp_path)
    assert any(isinstance(f, KeepTogether) for f in flowables)


def test_multiple_sections_each_get_their_own_keep_together_group(tmp_path):
    from reportlab.platypus import KeepTogether

    md = "## Section A\n\nParagraph A.\n\n## Section B\n\nParagraph B.\n"
    flowables = _markdown_to_flowables(md, tmp_path)
    groups = [f for f in flowables if isinstance(f, KeepTogether)]
    assert len(groups) == 2


def test_pdf_styles_use_times_new_roman_family(tmp_path):
    from reportlab.platypus import KeepTogether, Paragraph

    md = "# Title\n\n### Subtitle\n\n## 1. Heading\n\nBody paragraph text.\n"
    flowables = _markdown_to_flowables(md, tmp_path)

    def _iter_paragraphs(items):
        for item in items:
            if isinstance(item, KeepTogether):
                yield from _iter_paragraphs(item._content)
            elif isinstance(item, Paragraph):
                yield item

    font_names = {p.style.fontName for p in _iter_paragraphs(flowables)}
    assert font_names
    assert font_names <= {"Times-Roman", "Times-Bold", "Times-Italic"}
    assert not any(name.startswith("Helvetica") for name in font_names)


def test_inline_code_renders_as_plain_text_not_a_separate_courier_span():
    """Regression test: backtick-wrapped terms must match the surrounding prose.

    quantlab.report.writer used to wrap inline code in a Courier <font> span.
    Courier's heavier, fixed-width strokes make text at an identical colour
    read as a visibly different shade, so the PDF now renders inline code
    as plain text in the same font and colour as everything around it.
    """
    markup = _inline_to_reportlab_markup("Model kind: `rank_signal`")
    assert "Courier" not in markup
    assert "<font" not in markup
    assert "rank_signal" in markup


def test_inline_code_backticks_are_stripped_in_pdf_table_cells(tmp_path):
    from reportlab.platypus import Table

    md = "| Parameter | Value |\n| --- | --- |\n| `lookback` | 252 |\n"
    flowables = _markdown_to_flowables(md, tmp_path)
    tables = [f for f in flowables if isinstance(f, Table)]
    assert tables
    cell_paragraph = tables[0]._cellvalues[1][0]
    assert "Courier" not in cell_paragraph.text
    assert "lookback" in cell_paragraph.text
