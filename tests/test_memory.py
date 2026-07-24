"""Unit tests for the reflective memory store (JSONL fallback backend)."""

from __future__ import annotations

from quantlab.core import memory


def test_append_and_recall_round_trip():
    memory.append({"run_id": "r1", "stage": "literature", "n_reflections": 1})
    results = memory.recall("literature", k=5)
    assert any(r["run_id"] == "r1" for r in results)


def test_recall_ranks_matching_stage_above_unrelated_stage():
    memory.append({"run_id": "r1", "stage": "literature", "n_reflections": 1})
    memory.append({"run_id": "r2", "stage": "literature", "n_reflections": 0})
    memory.append({"run_id": "r3", "stage": "evaluate", "n_reflections": 2})

    literature_hits = memory.recall("literature", k=5)
    assert [r["stage"] for r in literature_hits[:2]] == ["literature", "literature"]

    evaluate_hits = memory.recall("evaluate", k=5)
    assert evaluate_hits[0]["stage"] == "evaluate"


def test_recall_on_empty_store_returns_empty_list():
    assert memory.recall("anything", k=5) == []


def test_recall_respects_k():
    for i in range(10):
        memory.append({"run_id": f"r{i}", "stage": "model", "n_reflections": 0})
    results = memory.recall("model", k=3)
    assert len(results) == 3


def test_embed_is_deterministic_and_normalised():
    a = memory._embed("literature review momentum")
    b = memory._embed("literature review momentum")
    assert a == b
    norm = sum(x * x for x in a) ** 0.5
    assert abs(norm - 1.0) < 1e-9 or norm == 0.0


def test_tokenize_strips_json_punctuation():
    tokens = memory._tokenize('{"stage": "literature"}')
    assert "literature" in tokens
    assert "stage" in tokens
