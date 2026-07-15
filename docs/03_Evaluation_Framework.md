# QuantLab Research Evaluation Framework

Evaluation is stratified into four layers: financial performance, research quality, system efficiency, and comparative benchmarks. Each layer contributes to a composite score reported in the thesis.

## 1. Financial Performance (Primary)

All metrics are computed on a strictly held-out out-of-sample window (default: 2022-01-01 to 2025-12-31) using daily returns, net of a 5 bps per-side transaction cost and a 1 bps borrow cost on shorts.

| Metric             | Definition                                              | Threshold for "material" result |
| ------------------ | ------------------------------------------------------- | ------------------------------- |
| Sharpe ratio       | Annualised mean excess return divided by annualised std | > 0.75 net of costs             |
| Annualised return  | Geometric mean daily return, annualised                 | > risk-free rate + 3 percent    |
| Maximum drawdown   | Largest peak-to-trough loss on equity curve             | > -30 percent                   |
| Win rate (monthly) | Share of positive monthly returns                       | > 55 percent                    |

Auxiliary financial diagnostics: Sortino, Calmar, turnover, capacity estimate, exposure to Fama-French five factors.

## 2. Research Quality (Thesis Contribution)

| Metric             | Definition                                                                                                | Instrument                                                         |
| ------------------ | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Reproducibility    | Bit-exact re-run given the same seed and pinned data snapshot                                             | `pytest tests/test_reproducibility.py`                             |
| Code correctness   | Unit-test pass rate; static analysis (ruff, mypy strict)                                                  | CI badge; `coverage >= 85 percent`                                 |
| Report quality     | Rubric score across clarity, rigour, completeness, and honesty                                            | Three human graders plus LLM-as-judge, Krippendorff alpha reported |
| Hypothesis novelty | 1 minus max cosine similarity between the generated hypothesis embedding and any retrieved paper abstract | `sentence-transformers/all-mpnet-base-v2`                          |

## 3. System Efficiency (Operational)

| Metric                             | Definition                                  |
| ---------------------------------- | ------------------------------------------- |
| Execution time                     | Wall-clock seconds from objective to report |
| Token usage                        | Input plus output tokens per run, per agent |
| Cost per experiment                | USD per completed research cycle            |
| Completed research cycles per hour | Throughput at fixed budget                  |

All operational metrics are logged automatically via MLflow tags.

## 4. Comparative Evaluation (Benchmark)

Three baselines are run on the identical set of ten research objectives with an identical compute budget.

| Baseline                            | Description                                                                                                   |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Single-agent baseline               | One GPT-4o instance with function calling; no planner-worker decomposition                                    |
| Non-reflective multi-agent baseline | The full nine-agent QuantLab graph with the Reflective Memory Agent disabled                                  |
| Human-assisted workflow baseline    | A quant researcher (thesis author) uses ChatGPT plus manual coding for a fixed four-hour budget per objective |

**Primary comparison metric:** out-of-sample Sharpe ratio, aggregated across objectives by median and interquartile range, with a paired Wilcoxon signed-rank test at alpha = 0.05.

**Secondary comparison:** cost-adjusted Sharpe (Sharpe per USD spent).

## 5. Robustness Studies (Month 5)

| Study              | Description                                                                                                                        |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| Regime shift       | Split the out-of-sample window into low-volatility (VIX < 20) and high-volatility (VIX >= 20) sub-samples; report metrics on each. |
| Injected leakage   | Deliberately corrupt one feature with a one-day-ahead label; verify the Backtesting Agent's leakage guard flags it.                |
| Survivorship bias  | Rerun on a point-in-time NASDAQ 100 constituent list; report Sharpe delta.                                                         |
| Prompt sensitivity | Re-run each objective with three prompt paraphrases; report metric variance.                                                       |

## 6. Reporting Template

Every experiment emits a machine-readable `run.json` with the schema:

```json
{
  "objective": "...",
  "run_id": "...",
  "seed": 42,
  "financial": { "sharpe": 0.87, "cagr": 0.09, "mdd": -0.22, "win_rate": 0.57 },
  "quality": {
    "reproducibility": 1.0,
    "coverage": 0.88,
    "report": 4.2,
    "novelty": 0.34
  },
  "efficiency": { "wallclock_s": 612, "tokens": 184220, "cost_usd": 1.94 },
  "baseline": "quantlab_full"
}
```

These records are aggregated into the thesis results chapter and released alongside the codebase.
