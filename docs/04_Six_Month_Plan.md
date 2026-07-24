# QuantLab: Six-Month Research Plan

The plan is organised as six monthly milestones, each with a concrete deliverable and a supervisor review checkpoint. Buffer time (approximately 15 percent) is folded into each month.

| Month | Goal                                                             | Key Artefacts                                                                         |
| ----- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| 1     | Establish the theoretical and engineering baseline               | Related-work chapter draft, agent-contract document, PoC repository tagged `v0.1`     |
| 2     | Bring the pipeline from PoC to research-grade                    | `v0.5` tag, integration test suite, internal design document on reflection mechanisms |
| 3     | Freeze the evaluation protocol and implement the three baselines | `v1.0` tag, evaluation harness, frozen benchmark specification                        |
| 4     | Run the main experimental campaign and the reflection ablation   | Experimental results database, interim thesis draft                                   |
| 5     | Stress the system on adversarial and realistic distortions       | Robustness chapter, annotated report dataset, third supervisor review                 |
| 6     | Consolidate contributions and publish                            | Final thesis, public GitHub release, paper draft                                      |

## Month 1: Foundation

**Goal.** Establish the theoretical and engineering baseline.

- Systematic literature survey on agentic AI (AutoGen, LangGraph, Reflexion, AI Scientist), automated machine learning for time series, and empirical asset pricing.
- Formal specification of the nine agent contracts (typed inputs and outputs, error surfaces).
- Implement and ship the two-day PoC (Deliverable 3).
- First supervisor review.

**Artefacts.** Related-work chapter draft (approximately 15 pages), agent-contract document, PoC repository tagged `v0.1`.

## Month 2: Full System

**Goal.** Bring the pipeline from PoC to research-grade.

- Replace deterministic PoC stubs with real LLM calls using LangGraph checkpointing.
- Implement PostgreSQL plus pgvector reflective memory.
- Implement the leakage-aware backtester with property-based tests.
- Add MLflow experiment tracking.

**Artefacts.** `v0.5` tag; integration test suite; internal design document on reflection mechanisms.

_Status note: optional LLM-backed drafting (with a validated, deterministic
fallback per stage) and a PostgreSQL plus pgvector reflective memory
backend (with a JSONL fallback) have already landed in the current
repository, ahead of this milestone. MLflow tracking and LangGraph
checkpointing have not._

## Month 3: Evaluation Harness and Baselines

**Goal.** Freeze the evaluation protocol and implement the three baselines.

- Implement the three-layer evaluation harness (financial, research-quality, system-efficiency).
- Implement the single-agent, non-reflective, and human-assisted baselines.
- Freeze the ten benchmark objectives (five in-distribution and five out-of-distribution relative to Month 1 literature).
- Second supervisor review.

**Artefacts.** `v1.0` tag; evaluation harness; frozen benchmark specification.

## Month 4: Large-Scale Experiments

**Goal.** Run the main experimental campaign and the reflection ablation.

- Execute all four systems on all ten benchmarks with three seeds each (120 runs total).
- Run the reflection ablation: full, no cross-run memory, no intra-run critic, neither.
- Analyse results with paired non-parametric tests and mixed-effects models.

**Artefacts.** Experimental results database (SQLite), interim thesis draft (approximately 45 pages).

## Month 5: Robustness

**Goal.** Stress the system on adversarial and realistic distortions.

- Regime-shift analysis (low vs. high volatility sub-samples).
- Injected leakage experiments to verify the guard.
- Survivorship-bias correction using a point-in-time constituent list.
- Prompt sensitivity study (three paraphrases per objective).
- Human-graded report study with three graders, Krippendorff alpha reported.

**Artefacts.** Robustness chapter; annotated report dataset; third supervisor review.

## Month 6: Thesis and Release

**Goal.** Consolidate contributions and publish.

- Complete thesis manuscript (target 70 to 90 pages).
- Prepare public open-source release with documentation and reproduction scripts.
- Draft a workshop-length paper (potential venues: NeurIPS ML for Finance, ICML AutoML, ICLR agent workshops).
- Thesis defence.

**Artefacts.** Final thesis; public GitHub release; paper draft.
