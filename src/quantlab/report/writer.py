"""Research report renderer.

Emits markdown, a PDF rendering of that same markdown (via reportlab -- no
system dependencies, unlike WeasyPrint), a machine-readable metrics.json, and
a reflections.jsonl audit trail of per-stage critiques. The thesis version
replaces the reportlab renderer with a Jinja + WeasyPrint pipeline for richer
typesetting, but the PoC output already matches what the README promises.

The markdown produced here is a deliberately small, known subset (title,
subtitle, numbered `##` sections, blockquote "panels" for callouts, pipe
tables, bullet/numbered lists, a single embedded image + caption, and
**bold**/`code`/[link](url) inline spans). `_markdown_to_flowables` below
renders exactly that subset to a styled PDF; it is not a general-purpose
markdown-to-PDF converter.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.core.seeding import DEFAULT_SEED
from quantlab.core.state import (
    Hypothesis,
    MetricsBundle,
    PaperSummary,
    Reflection,
    ResearchState,
)

# --- Formatting helpers ----------------------------------------------------


def _fmt_pct(x: float, signed: bool = False) -> str:
    if signed:
        return f"{x:+.2%}"
    return f"{x:.2%}"


def _fmt_param_value(v: Any) -> str:
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _count_universe_tickers(parquet_path: str) -> int | None:
    """Best-effort column count of the cached price parquet (i.e. #tickers)."""
    try:
        return int(pd.read_parquet(parquet_path).shape[1])
    except Exception:
        return None


def _hypothesis_verdict(h: Hypothesis, m: MetricsBundle) -> tuple[str, str]:
    """Evaluate the stated H0/H1 against the realised Sharpe ratio.

    Returns (verdict_label, verdict_detail) -- both plain sentences, no
    trailing period, ready to be dropped into the Executive Summary and
    Results sections.
    """
    positive_result = m.sharpe > 0
    expects_positive = h.expected_sign == "positive"
    supports_alt = positive_result if expects_positive else not positive_result
    direction = "positive" if positive_result else "non-positive"

    if supports_alt:
        alt = (h.alternative or "").rstrip(".")
        alt_lowered = (alt[0].lower() + alt[1:]) if alt else alt
        label = "Reject H0 in favour of H1"
        detail = (
            f"the out-of-sample Sharpe ratio is {direction} ({m.sharpe:.2f}), "
            f"consistent with H1 that {alt_lowered}"
        )
    else:
        null = (h.null or "").rstrip(".")
        null_lowered = (null[0].lower() + null[1:]) if null else null
        label = "Fail to reject H0"
        detail = (
            f"the out-of-sample Sharpe ratio is {direction} ({m.sharpe:.2f}), "
            f"which does not provide evidence against H0 that {null_lowered}"
        )
    return label, detail


def _literature_lines(papers: list[PaperSummary]) -> list[str]:
    if not papers:
        return ["No supporting literature was retrieved for this run."]
    lines = []
    for i, p in enumerate(papers, start=1):
        authors = ", ".join(p.authors) if p.authors else "Author unknown"
        link = f" [link]({p.url})" if p.url else ""
        lines.append(f"{i}. **{p.title}**, {authors} ({p.year}).{link}")
    return lines


def _params_table_lines(params: dict[str, Any]) -> list[str]:
    rows = ["| Parameter | Value |", "| --- | --- |"]
    for k, v in params.items():
        rows.append(f"| `{k}` | {_fmt_param_value(v)} |")
    return rows


def _results_table_lines(m: MetricsBundle) -> list[str]:
    return [
        "| Metric | Value |",
        "| --- | --- |",
        f"| Sharpe ratio | {m.sharpe:.2f} |",
        f"| Annualised return (CAGR) | {_fmt_pct(m.cagr, signed=True)} |",
        f"| Maximum drawdown | {_fmt_pct(m.max_drawdown)} |",
        f"| Win rate (monthly) | {_fmt_pct(m.win_rate)} |",
        f"| Annualised turnover | {m.turnover:.2f}\u00d7 |",
    ]


def _reflection_table_lines(reflections: list[Reflection]) -> list[str]:
    if not reflections:
        return ["No reflective critiques were raised for this run."]
    rows = ["| Severity | Agent / Stage | Critique |", "| --- | --- | --- |"]
    for r in reflections:
        rows.append(
            f"| **{r.severity.upper()}** | {r.agent} / {r.stage} | {r.critique} |"
        )
    return rows


def _limitations_lines(
    h: Hypothesis, m: MetricsBundle, n_tickers: int | None
) -> list[str]:
    lines = []
    if n_tickers:
        lines.append(
            f"- **Universe coverage:** the `{h.universe}` universe used here is a static "
            f"{n_tickers}-name slice, not a point-in-time reconstruction of the full index; "
            f"results may not generalise to the complete constituent list."
        )
    if m.max_drawdown <= -0.20:
        lines.append(
            f"- **Drawdown risk:** the strategy experienced a maximum drawdown of "
            f"{_fmt_pct(m.max_drawdown)}. A volatility-targeting or drawdown-control "
            f"overlay should be evaluated before any live deployment."
        )
    lines.append(
        "- **Backtest assumptions:** transaction costs are modelled as a flat per-side "
        "bps charge with no market-impact or slippage model, and the long-only "
        "implementation used here carries no borrowing cost."
    )
    lines.append(
        "- **Literature curation:** supporting papers are retrieved automatically from "
        "arXiv and are not a substitute for a systematic literature review."
    )
    lines.append(
        "- **Statistical significance:** the hypothesis test above is based on a single "
        "historical sample path and has not been corrected for multiple testing."
    )
    return lines


# --- Markdown assembly ------------------------------------------------------


def render_markdown(
    state: ResearchState,
    chart_filename: str | None = None,
    discussion: str | None = None,
) -> str:
    h: Hypothesis = state["hypothesis"]
    m: MetricsBundle = state["metrics"]
    bt = state["backtest"]
    mdl = state["model"]
    refs: list[Reflection] = state.get("reflections", [])
    papers: list[PaperSummary] = state.get("literature", [])

    generated_ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    oos_start = str(state.get("run_config", {}).get("oos_start", "2022-01-01"))

    n_tickers = None
    is_synthetic = False
    features = state.get("features")
    if features:
        n_tickers = _count_universe_tickers(features[0].parquet_path)
        is_synthetic = Path(features[0].parquet_path).stem.endswith("_synthetic")
    universe_desc = h.universe + (
        f" ({n_tickers} constituents, static PoC slice)" if n_tickers else ""
    )

    verdict_label, verdict_detail = _hypothesis_verdict(h, m)

    lines: list[str] = []
    lines.append("# QuantLab Research Report")
    lines.append(f"### {state['objective']}")
    lines.append("")
    lines.append(f"> **Run ID:** `{state['run_id']}`")
    lines.append(f"> **Generated:** {generated_ts}")
    lines.append(f"> **Universe:** {universe_desc}")
    lines.append(f"> **Evaluation window:** {oos_start} to {bt.end.date()}")
    lines.append("")

    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        f"> **{verdict_label}.** Over the out-of-sample window the strategy achieved a "
        f"Sharpe ratio of **{m.sharpe:.2f}**, an annualised return of "
        f"**{_fmt_pct(m.cagr, signed=True)}**, and a maximum drawdown of "
        f"**{_fmt_pct(m.max_drawdown)}**, with a monthly win rate of "
        f"**{_fmt_pct(m.win_rate)}**. In statistical terms, {verdict_detail}."
    )
    lines.append("")

    lines.append("## 1. Research Hypothesis")
    lines.append("")
    lines.append(h.statement)
    lines.append("")
    lines.append(f"- **H0:** {h.null}")
    lines.append(f"- **H1:** {h.alternative}")
    lines.append(f"- **Universe:** {h.universe}")
    lines.append(f"- **Horizon:** {h.horizon_days} trading days")
    lines.append("")

    lines.append("## 2. Literature Review")
    lines.append("")
    lines.extend(_literature_lines(papers))
    lines.append("")

    lines.append("## 3. Data")
    lines.append("")
    if is_synthetic:
        lines.append(
            f"Universe: {universe_desc}. Daily close prices between {bt.start.date()} and "
            f"{bt.end.date()}. A real market data provider was not reachable for this run, "
            f"so a deterministic synthetic price series was used instead; see "
            f"`quantlab.data.loaders` for the generation method. Results in this report "
            f"should be read as a pipeline validation, not a real-market backtest."
        )
    else:
        lines.append(
            f"Universe: {universe_desc}. Daily close prices between {bt.start.date()} and "
            f"{bt.end.date()}, sourced via yfinance and cached locally as parquet. No "
            f"survivorship correction is applied in this proof-of-concept."
        )
    lines.append("")

    lines.append("## 4. Signal & Model Specification")
    lines.append("")
    lines.append(f"**Model kind:** `{mdl.kind}`")
    lines.append("")
    lines.extend(_params_table_lines(mdl.params))
    lines.append("")

    lines.append("## 5. Backtest Methodology")
    lines.append("")
    lines.append(
        f"Vectorised backtest with a monthly rebalance and {bt.transaction_cost_bps:.1f} "
        f"bps of per-side transaction cost. Look-ahead bias is prevented by shifting "
        f"monthly weights forward by one trading day before applying them to daily returns."
    )
    lines.append("")

    lines.append(f"## 6. Results (out-of-sample, {oos_start[:4]} to {bt.end.year})")
    lines.append("")
    lines.extend(_results_table_lines(m))
    lines.append("")
    if chart_filename:
        lines.append(f"![Equity curve]({chart_filename})")
        lines.append("")
        lines.append(
            "*Figure 1. Growth of $1 invested, net of transaction costs, over the full sample.*"
        )
        lines.append("")
    lines.append(
        f"**Hypothesis test verdict:** {verdict_label}. Concretely, {verdict_detail}."
    )
    lines.append("")
    if discussion:
        lines.append(f"**Discussion:** {discussion}")
        lines.append("")

    lines.append("## 7. Reflective Critique")
    lines.append("")
    lines.extend(_reflection_table_lines(refs))
    lines.append("")

    lines.append("## 8. Limitations & Risk Disclosure")
    lines.append("")
    lines.extend(_limitations_lines(h, m, n_tickers))
    lines.append("")

    lines.append("## 9. Reproducibility")
    lines.append("")
    seed = state.get("run_config", {}).get("seed", DEFAULT_SEED)
    lines.append(
        f"Seed: {seed} (applied to Python's `random` and NumPy's global RNG at the "
        f"start of this run). Data snapshot: yfinance at run time. Configuration: "
        f"`configs/momentum_nasdaq.yaml`."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "*This report was generated automatically by the QuantLab agentic research "
        "pipeline. It is a research artefact, not investment advice.*"
    )

    return "\n".join(lines)


def save_markdown(md: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "report.md"
    p.write_text(md, encoding="utf-8")
    return p


def save_equity_curve_png(equity_curve_path: str, out_dir: Path) -> Path | None:
    """Render the equity curve with a highlighted max-drawdown episode."""
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    eq = pd.read_parquet(equity_curve_path)["equity"]
    if eq.empty:
        return None

    running_max = eq.cummax()
    drawdown = eq / running_max - 1.0
    trough_idx = drawdown.idxmin()
    peak_idx = eq[eq.index <= trough_idx].idxmax()

    x = mdates.date2num(eq.index)
    trough_x = float(mdates.date2num(trough_idx))
    trough_y = float(eq.loc[trough_idx])
    peak_x = float(mdates.date2num(peak_idx))

    line_color, dd_color, grid_color = "#2563eb", "#dc2626", "#94a3b8"

    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    ax.plot(x, eq.to_numpy(), color=line_color, linewidth=1.6, zorder=3)
    ax.fill_between(x, eq.to_numpy(), 1.0, color=line_color, alpha=0.07, zorder=1)
    ax.axhline(1.0, color=grid_color, linewidth=0.8, linestyle="--", zorder=2)

    if peak_idx != trough_idx:
        ax.axvspan(peak_x, trough_x, color=dd_color, alpha=0.06, zorder=0)
        ax.scatter([trough_x], [trough_y], color=dd_color, s=22, zorder=4)
        ax.annotate(
            f"Max DD {drawdown.loc[trough_idx]:.1%}",
            xy=(trough_x, trough_y),
            xytext=(10, -14),
            textcoords="offset points",
            fontsize=8.5,
            color=dd_color,
        )

    ax.xaxis_date()

    ax.set_title(
        "Equity Curve, Growth of $1 (net of costs)",
        fontsize=12,
        fontweight="bold",
        loc="left",
        color="#0f172a",
    )
    ax.set_ylabel("Growth of $1", fontsize=9.5)
    ax.tick_params(labelsize=8.5)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")

    out = out_dir / "equity_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_metrics_json(state: ResearchState, out_dir: Path) -> Path:
    """Write a machine-readable metrics.json alongside the human-readable report."""
    m = state["metrics"]
    bt = state["backtest"]
    payload = {
        "run_id": state["run_id"],
        "objective": state["objective"],
        "universe": state["hypothesis"].universe,
        "transaction_cost_bps": bt.transaction_cost_bps,
        "oos_start": state.get("run_config", {}).get("oos_start", "2022-01-01"),
        "metrics": {
            "sharpe": m.sharpe,
            "cagr": m.cagr,
            "max_drawdown": m.max_drawdown,
            "win_rate": m.win_rate,
            "turnover": m.turnover,
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "metrics.json"
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return p


def save_reflections_jsonl(reflections: list[Reflection], out_dir: Path) -> Path:
    """Write one JSON object per reflective-memory critique, in stage order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "reflections.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in reflections:
            f.write(
                json.dumps(
                    {
                        "agent": r.agent,
                        "stage": r.stage,
                        "severity": r.severity,
                        "critique": r.critique,
                    }
                )
                + "\n"
            )
    return p


# --- Minimal markdown -> reportlab renderer -------------------------------

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_CODE_RE = re.compile(r"`(.+?)`")
_LINK_RE = re.compile(r"\[(.+?)\]\((.+?)\)")
_IMAGE_RE = re.compile(r"^!\[(.*?)\]\((.*?)\)$")
_NUMBERED_RE = re.compile(r"^\d+\.\s")


def _palette(colors: Any) -> dict[str, Any]:
    """Colour tokens, built lazily so `reportlab` stays an optional import."""
    return {
        "ink": colors.HexColor("#0f172a"),
        "accent": colors.HexColor("#2563eb"),
        "accent_dark": colors.HexColor("#1e3a8a"),
        "muted": colors.HexColor("#64748b"),
        "panel_bg": colors.HexColor("#eff6ff"),
        "panel_border": colors.HexColor("#bfdbfe"),
        "row_alt": colors.HexColor("#f8fafc"),
        "border": colors.HexColor("#e2e8f0"),
        "white": colors.white,
    }


def _escape_xml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _inline_to_reportlab_markup(text: str) -> str:
    text = _escape_xml(text)
    text = _LINK_RE.sub(r'<link href="\2" color="#2563eb"><u>\1</u></link>', text)
    text = _BOLD_RE.sub(r"<b>\1</b>", text)
    text = _CODE_RE.sub(r'<font face="Courier" size="8.5">\1</font>', text)
    return text


def _markdown_to_flowables(md: str, out_dir: Path) -> list[Any]:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import (
        HRFlowable,
        Image,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )

    pal = _palette(colors)
    content_width = LETTER[0] - 108

    styles = {
        "title": ParagraphStyle(
            "QLTitle",
            fontName="Helvetica-Bold",
            fontSize=21,
            leading=25,
            textColor=pal["ink"],
            spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "QLSubtitle",
            fontName="Helvetica-Oblique",
            fontSize=11.5,
            leading=15,
            textColor=pal["muted"],
            spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "QLH2",
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=pal["accent_dark"],
            spaceBefore=14,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "QLBody",
            fontName="Helvetica",
            fontSize=9.6,
            leading=13.8,
            textColor=pal["ink"],
            spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "QLCaption",
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            leading=11,
            textColor=pal["muted"],
            alignment=TA_CENTER,
            spaceBefore=2,
            spaceAfter=10,
        ),
        "panel": ParagraphStyle(
            "QLPanel",
            fontName="Helvetica",
            fontSize=9.6,
            leading=13.8,
            textColor=pal["ink"],
            spaceAfter=2,
        ),
        "cell_head": ParagraphStyle(
            "QLCellHead",
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=pal["white"],
        ),
        "cell_body": ParagraphStyle(
            "QLCellBody",
            fontName="Helvetica",
            fontSize=9,
            leading=12.5,
            textColor=pal["ink"],
        ),
    }

    story: list[Any] = []
    table_buffer: list[list[str]] = []
    quote_buffer: list[str] = []

    def _flush_table() -> None:
        if not table_buffer:
            return
        data = []
        for ridx, row in enumerate(table_buffer):
            style = styles["cell_head"] if ridx == 0 else styles["cell_body"]
            data.append([Paragraph(_inline_to_reportlab_markup(c), style) for c in row])
        t = Table(data, hAlign="LEFT", repeatRows=1, colWidths=None)
        tstyle = [
            ("BACKGROUND", (0, 0), (-1, 0), pal["accent_dark"]),
            ("LINEBELOW", (0, 0), (-1, 0), 1, pal["accent_dark"]),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]
        for r in range(1, len(data)):
            if r % 2 == 0:
                tstyle.append(("BACKGROUND", (0, r), (-1, r), pal["row_alt"]))
            tstyle.append(("LINEBELOW", (0, r), (-1, r), 0.4, pal["border"]))
        t.setStyle(TableStyle(tstyle))
        story.append(t)
        story.append(Spacer(1, 12))
        table_buffer.clear()

    def _flush_quote() -> None:
        if not quote_buffer:
            return
        paras = [
            Paragraph(_inline_to_reportlab_markup(q), styles["panel"])
            for q in quote_buffer
        ]
        panel = Table([[paras]], colWidths=[content_width])
        panel.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), pal["panel_bg"]),
                    ("BOX", (0, 0), (-1, -1), 0.75, pal["panel_border"]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                    ("TOPPADDING", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                ]
            )
        )
        story.append(panel)
        story.append(Spacer(1, 12))
        quote_buffer.clear()

    for raw_line in md.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("|"):
            _flush_quote()
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if cells and all(set(c) <= {"-", " ", ":"} for c in cells):
                continue
            table_buffer.append(cells)
            continue

        if line.startswith("> "):
            _flush_table()
            quote_buffer.append(line[2:].strip())
            continue

        _flush_table()
        _flush_quote()

        if not stripped:
            story.append(Spacer(1, 6))
        elif stripped == "---":
            story.append(Spacer(1, 4))
            story.append(
                HRFlowable(
                    width="100%",
                    thickness=0.75,
                    color=pal["border"],
                    spaceBefore=2,
                    spaceAfter=10,
                )
            )
        elif stripped.startswith("# "):
            story.append(
                Paragraph(_inline_to_reportlab_markup(stripped[2:]), styles["title"])
            )
        elif stripped.startswith("### "):
            story.append(
                Paragraph(_inline_to_reportlab_markup(stripped[4:]), styles["subtitle"])
            )
        elif stripped.startswith("## "):
            story.append(
                Paragraph(_inline_to_reportlab_markup(stripped[3:]), styles["h2"])
            )
            story.append(
                HRFlowable(
                    width="100%",
                    thickness=1,
                    color=pal["accent_dark"],
                    spaceBefore=1,
                    spaceAfter=6,
                )
            )
        elif (m_img := _IMAGE_RE.match(stripped)) is not None:
            img_path = out_dir / m_img.group(2)
            if img_path.exists():
                try:
                    reader = ImageReader(str(img_path))
                    iw, ih = reader.getSize()
                    scale = min(1.0, content_width / iw) if iw else 1.0
                    story.append(
                        Image(str(img_path), width=iw * scale, height=ih * scale)
                    )
                except Exception as exc:
                    print(
                        f"[quantlab.report.writer] Skipping unreadable chart image: {exc}"
                    )
        elif (
            stripped.startswith("*")
            and stripped.endswith("*")
            and not stripped.startswith("**")
        ):
            story.append(
                Paragraph(
                    _inline_to_reportlab_markup(stripped.strip("*")), styles["caption"]
                )
            )
        elif stripped.startswith("- "):
            story.append(
                Paragraph(
                    "&bull;&nbsp;&nbsp;" + _inline_to_reportlab_markup(stripped[2:]),
                    styles["body"],
                )
            )
        elif _NUMBERED_RE.match(stripped):
            story.append(
                Paragraph(_inline_to_reportlab_markup(stripped), styles["body"])
            )
        else:
            story.append(
                Paragraph(_inline_to_reportlab_markup(stripped), styles["body"])
            )

    _flush_table()
    _flush_quote()
    return story


def save_report_pdf(md: str, out_dir: Path) -> Path | None:
    """Render report.md to report.pdf. Returns None if reportlab is unavailable."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import LETTER
        from reportlab.pdfgen import canvas as pdfcanvas
        from reportlab.platypus import SimpleDocTemplate
    except ImportError:
        return None

    pal = _palette(colors)
    page_w, page_h = LETTER

    class _NumberedCanvas(pdfcanvas.Canvas):
        """Draws a header accent bar + footer with 'Page X of Y' on every page.

        Standard two-pass reportlab recipe: buffer each page's drawing state
        via `showPage`, then replay them once the true page count is known
        at `save()` time.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._saved_page_states: list[dict[str, Any]] = []

        def showPage(self) -> None:
            self._saved_page_states.append(dict(self.__dict__))
            self._startPage()

        def save(self) -> None:
            total = len(self._saved_page_states)
            for state in self._saved_page_states:
                self.__dict__.update(state)
                self._draw_chrome(total)
                super().showPage()
            super().save()

        def _draw_chrome(self, total_pages: int) -> None:
            self.saveState()
            self.setFillColor(pal["accent_dark"])
            self.rect(0, page_h - 6, page_w, 6, fill=1, stroke=0)
            self.setFont("Helvetica", 8)
            self.setFillColor(pal["muted"])
            self.drawString(54, 28, "QuantLab Research Report")
            self.drawRightString(
                page_w - 54, 28, f"Page {self._pageNumber} of {total_pages}"
            )
            self.setStrokeColor(pal["border"])
            self.setLineWidth(0.75)
            self.line(54, 38, page_w - 54, 38)
            self.restoreState()

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "report.pdf"
    doc = SimpleDocTemplate(
        str(out),
        pagesize=LETTER,
        leftMargin=54,
        rightMargin=54,
        topMargin=46,
        bottomMargin=50,
        title="QuantLab Research Report",
    )
    story = _markdown_to_flowables(md, out_dir)
    doc.build(story, canvasmaker=_NumberedCanvas)
    return out
