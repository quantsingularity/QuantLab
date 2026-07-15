# QuantLab Technology Stack

## 1. Recommended Stack

| Component                    | Technology                                                                          | Rationale                                                      |
| ---------------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Language                     | Python 3.11+                                                                        | Ecosystem parity with quant and ML tooling                     |
| Agent framework              | LangGraph (primary), CrewAI (fallback)                                              | Explicit state machine, typed edges, first-class checkpointing |
| LLM                          | OpenAI GPT-4o (planner, hypothesis, report); GPT-4o-mini (retrieval, summarisation) | Cost-quality trade-off                                         |
| Data                         | yfinance (PoC), Polygon.io (thesis scale)                                           | Free tier for PoC; institutional-grade for full study          |
| ML                           | scikit-learn, XGBoost, statsmodels                                                  | Standard, reproducible, well-understood baselines              |
| Backtesting                  | VectorBT (primary), Backtrader (event-driven check)                                 | Vectorised speed plus event-driven validation                  |
| API                          | FastAPI                                                                             | Async, typed, OpenAPI out of the box                           |
| Storage                      | PostgreSQL with pgvector, Redis                                                     | Structured artefacts plus vector recall plus cache             |
| Containerisation             | Docker, docker-compose                                                              | Reproducible dev and CI                                        |
| Orchestration (thesis scale) | Prefect                                                                             | Retry, scheduling, observability                               |
| Experiment tracking          | MLflow                                                                              | Model, metric, and artefact lineage                            |
| Testing                      | pytest, hypothesis                                                                  | Property-based tests for leakage guards                        |
| CI                           | GitHub Actions                                                                      | Public, free for open source                                   |
| Documentation                | MkDocs Material                                                                     | Thesis-quality developer docs                                  |

## 2. Why not a different stack

- **LangChain (classic).** Higher abstraction but lower observability; LangGraph gives explicit control-flow graphs suitable for a reproducibility-first thesis.
- **AutoGen.** Excellent for conversational agents, weaker for typed DAGs and checkpointing.
- **Local open-weight LLMs.** Considered for cost, but the reproducibility of a frozen `gpt-4o-2024-08-06` snapshot outweighs the marginal cost gain for a six-month thesis. Ablation with a strong open model (for example `Qwen2.5-72B`) is planned in Month 4.

## 3. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
docker compose up -d postgres redis
export OPENAI_API_KEY=...
```
