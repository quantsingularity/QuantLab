"""Reflective memory: PostgreSQL for structured artefacts, pgvector for recall.

For the PoC this module falls back to a local JSONL file so the pipeline runs
without a database. The production path (thesis Month 2 onward) uses SQLAlchemy
plus pgvector.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

MEMORY_PATH = Path(os.environ.get("QUANTLAB_MEMORY", "./runs/memory.jsonl"))


def append(record: dict[str, Any]) -> None:
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MEMORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def recall(query: str, k: int = 5) -> list[dict[str, Any]]:
    """Very naive keyword recall for the PoC. Replaced by pgvector cosine search."""
    if not MEMORY_PATH.exists():
        return []
    hits: list[dict[str, Any]] = []
    q = query.lower()
    with MEMORY_PATH.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if q in json.dumps(rec).lower():
                hits.append(rec)
    return hits[-k:]
