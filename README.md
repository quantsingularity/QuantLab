# QuantLab

**An Agentic AI Framework for End-to-End Quantitative Research and Alpha Discovery**

QuantLab is a multi-agent AI system that autonomously performs literature review, hypothesis generation, data engineering, model development, backtesting, evaluation, and reporting for quantitative finance. It is developed as an MSc thesis proposal for the ETH Zurich Agentic Systems Lab.

---

## Table of Contents

1. [Motivation](#motivation)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quickstart Demo](#quickstart-demo)
5. [Repository Layout](#repository-layout)
6. [Configuration](#configuration)
7. [Evaluation](#evaluation)
8. [Development](#development)
9. [Roadmap](#roadmap)
10. [License](#license)

---

## Motivation

Systematic alpha discovery is bottlenecked by human throughput: a researcher can meaningfully test only a handful of hypotheses per week. QuantLab treats the full research pipeline as a graph of specialised LLM agents, adds a reflective memory layer for cross-run learning, and enforces leakage-aware backtesting so that the throughput gain does not translate into a p-hacking gain.

The thesis contribution is not "another autonomous agent"; it is a rigorous evaluation of whether reflection and multi-agent coordination measurably improve out-of-sample research quality in a high-noise, adversarial domain.

## Architecture

Nine agents cooperate over a shared, versioned state:

| Agent                       | Role                                                                   |
| --------------------------- | ---------------------------------------------------------------------- |
| Research Planner Agent      | Builds the task DAG that sequences the rest of the pipeline            |
| Literature Review Agent     | Retrieves and summarises related work for the objective                |
| Hypothesis Generation Agent | Formulates a testable hypothesis with an explicit null and alternative |
| Data Engineering Agent      | Loads, caches, and builds leakage-checked features                     |
| Model Development Agent     | Builds the signal: rank-based by default, or a walk-forward ridge/xgboost model |
| Backtesting Agent           | Runs the vectorised, leakage-guarded backtest                          |
| Evaluation Agent            | Computes financial and research-quality metrics                        |
| Research Report Agent       | Renders the markdown and PDF research report                           |
| Reflective Memory Agent     | Records per-stage critiques for cross-run learning                     |

```mermaid
flowchart LR
    U[Objective] --> P[Planner]
    P --> L[Literature]
    P --> H[Hypothesis]
    L --> H
    H --> D[Data Eng]
    D --> M[Model]
    M --> B[Backtest]
    B --> E[Evaluate]
    E --> R[Report]
    R --> U
    P & L & H & D & M & B & E & R <--> RM[(Reflective Memory)]
```

Full diagram and agent contracts: [`docs/01_System_Architecture.md`](docs/01_System_Architecture.md).

## Installation

```bash
git clone https://github.com/quantsingularity/QuantLab
cd QuantLab
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Requirements to run the demo: Python 3.11+. That's it. The pipeline is
offline-first: if a price provider or an LLM isn't reachable, every agent
that could use one falls back to a deterministic path automatically, so
nothing else needs to be running for a first run.

Two optional extras add real capability on top of that baseline:

```bash
pip install -e ".[dev,ml]"        # xgboost-backed walk-forward signal
pip install -e ".[dev,postgres]"  # PostgreSQL + pgvector reflective memory
```

Without `.[ml]`, requesting `model_kind: xgboost` in a config falls back to
ridge regression with a printed notice. Without `.[postgres]`, or without
`POSTGRES_DSN` set, the Reflective Memory Agent falls back to a local
`runs/memory.jsonl` file, embedded and recalled by cosine similarity with
no external dependency. To use a real Postgres instance instead:

```bash
docker compose up -d postgres
export POSTGRES_DSN=postgresql://quantlab:quantlab@localhost:5432/quantlab
```

`docker-compose.yml` also provisions a `redis` service for a future
task-queue and caching layer; no agent reads from it yet, so it is not
required for anything in this repository today.

Set `OPENAI_API_KEY` (see `.env.example`) and name a model under a config's
`models` section to let the Planner, Hypothesis, Literature, and Report
agents draft their text with an LLM instead of their deterministic
templates; every one of them validates the response and falls back
automatically if the call fails, times out, returns malformed JSON, or
would exceed the configured `budget`. This is optional, not required: the
pipeline is fully functional, and every claim in the generated report is
still true, without it.

If a real market data provider is unreachable, set `QUANTLAB_OFFLINE=1` (or
just let the pipeline detect the failure itself) to use a deterministic
synthetic price series instead; the generated report says so explicitly in
its Data section when that happens.

## Quickstart Demo

Run the pipeline end to end:

```bash
python -m quantlab.run \
    --objective "Develop a momentum strategy for the NASDAQ 100" \
    --out ./runs/momentum_nasdaq
```

Or drive it from a config file instead of flags (see [Configuration](#configuration)):

```bash
python -m quantlab.run --config configs/momentum_nasdaq.yaml --out ./runs/momentum_nasdaq
```

`configs/ml_momentum_nasdaq.yaml` runs the same objective through the
walk-forward ridge-regression signal instead of the plain rank-based one;
swap `model_kind: ridge` for `model_kind: xgboost` to use gradient-boosted
trees instead, if the `[ml]` extra is installed.

This produces `runs/momentum_nasdaq/<run_id>/` containing:

| File                   | Description                                          |
| ---------------------- | ---------------------------------------------------- |
| `report.md`            | Auto-generated research report                       |
| `report.pdf`           | PDF rendering (skipped if reportlab isn't installed) |
| `equity_curve.png`     | Cumulative returns figure                            |
| `equity_curve.parquet` | Raw equity curve                                     |
| `metrics.json`         | Machine-readable metrics                             |
| `trades.parquet`       | Trade log                                            |
| `reflections.jsonl`    | Per-stage critiques from the Reflective Memory Agent |
| `state.json`           | Full run snapshot                                    |

## Repository Layout

```
quantlab/
├── src/quantlab/
│   ├── agents/            # Nine agent implementations
│   ├── core/              # Graph, state, memory
│   ├── data/              # Loaders and caches
│   ├── strategies/        # Signal factories
│   ├── backtest/          # Vectorised engine + leakage guard
│   ├── evaluation/        # Financial + research-quality metrics
│   └── report/            # Markdown and PDF rendering
├── tests/                 # pytest + hypothesis
├── configs/               # yaml configs for objectives, budgets, models
├── docs/                  # Thesis exposé, architecture, evaluation
├── notebooks/             # Exploratory analysis
├── scripts/               # CLI entry points
└── docker-compose.yml
```

## Configuration

Objectives, universe, dates, and cost assumptions can be driven by a
declarative YAML file instead of CLI flags:

```yaml
# configs/momentum_nasdaq.yaml
objective: "Develop a momentum strategy for the NASDAQ 100"
universe: nasdaq_100
start: 2010-01-01
end: 2025-01-01
oos_start: 2022-01-01
transaction_cost_bps: 5
model_kind: rank_signal
model_params:
  lookback: 252
  skip: 21
  top_pct: 0.10
  long_only: true
models:
  planner: gpt-4o
  hypothesis: gpt-4o
  summariser: gpt-4o-mini
  report: gpt-4o
budget:
  max_tokens_per_run: 400000
  max_usd_per_run: 5.0
```

| Field                  | Purpose                                                        | Status                       |
| ----------------------- | --------------------------------------------------------------- | ----------------------------- |
| `objective`            | Free-text research objective passed to the pipeline            | Used today                   |
| `universe`             | Ticker universe to trade                                       | Used today                   |
| `start` / `end`        | Full sample date range                                         | Used today                   |
| `oos_start`            | Start of the out-of-sample evaluation window                   | Used today                   |
| `transaction_cost_bps` | Per-side transaction cost, in basis points                     | Used today                   |
| `model_kind`           | `rank_signal` (default), `ridge`, or `xgboost`                 | Used today                   |
| `model_params`         | `lookback`, `skip`, `top_pct`, `long_only`, `min_train_periods` | Used today                   |
| `models`               | Model routing per agent (planner, hypothesis, summariser, report) | Used today when set, otherwise skipped |
| `budget`               | Token and USD spend caps per run                                | Used today when `models` is set |

`objective`, `universe`, `start`, `end`, `oos_start`, `transaction_cost_bps`,
`model_kind`, and `model_params` are always read by the Data Engineering,
Model Development, Backtesting, and Evaluation agents. `models` and
`budget` only matter if `OPENAI_API_KEY` is set: naming a model for a stage
under `models` lets that agent draft its text with an LLM instead of its
deterministic template, and `budget` caps total token and USD spend across
the whole run, after which every remaining stage automatically falls back
to its deterministic behaviour. Leaving `models` out entirely, or running
without an API key, is a fully supported, fully deterministic mode, not a
degraded one.

## Evaluation

The evaluation framework covers financial performance, research quality, system efficiency, and comparative baselines. See [`docs/03_Evaluation_Framework.md`](docs/03_Evaluation_Framework.md).

Run the comparative benchmark:

```bash
python -m quantlab.eval.run_benchmark --config configs/benchmark.yaml
```

Honest scope note: `run_config.models` can point individual agents at an
LLM, but that is a per-stage text-drafting option, not a different system
architecture, so this harness still cannot reproduce the "single-agent
GPT-4o baseline" or "human-assisted workflow baseline" from the thesis
proposal. What it does give you today, end to end and not simulated, is a
genuine reflective-vs-non-reflective ablation (`use_reflection: true/false`
in `configs/benchmark.yaml`), which is the "non-reflective multi-agent
baseline" row in `docs/03_Evaluation_Framework.md`.

## Development

```bash
pytest -q
ruff check src tests
mypy src
```

## Roadmap

See [`docs/04_Six_Month_Plan.md`](docs/04_Six_Month_Plan.md) for the full six-month thesis plan.

| Milestone | Focus                                         |
| --------- | --------------------------------------------- |
| M1        | PoC and related-work chapter                  |
| M2        | Full nine-agent system with reflective memory |
| M3        | Evaluation harness and baselines              |
| M4        | Large-scale experiments and ablations         |
| M5        | Robustness studies                            |
| M6        | Thesis writing and open-source release        |

## License

MIT. See `LICENSE`.
