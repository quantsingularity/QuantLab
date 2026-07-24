"""Reflective memory: PostgreSQL plus pgvector for structured recall.

Every stage of every run appends a record here. Records are embedded with a
small deterministic local embedding so that recall works without any
external API. When POSTGRES_DSN is set and reachable, records are stored in
PostgreSQL with a pgvector column and recalled by cosine distance. Otherwise
this module falls back to a local JSONL file at QUANTLAB_MEMORY (default
./runs/memory.jsonl) and recalls by cosine similarity computed in Python,
so the pipeline always runs end to end without any external service.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

EMBEDDING_DIM = 64
_POSTGRES_TABLE = "quantlab_memory"
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_DEFAULT_MEMORY_PATH = "./runs/memory.jsonl"

_postgres_unavailable_warned = False


def _memory_path() -> Path:
    """Resolve the JSONL fallback path from the environment on every call.

    Reading QUANTLAB_MEMORY fresh each time, rather than caching it once at
    import time, means a process that changes the environment variable
    partway through (as the test suite does, once per test) actually gets
    an isolated memory file instead of silently sharing whatever path was
    resolved the first time this module was imported.
    """
    return Path(os.environ.get("QUANTLAB_MEMORY", _DEFAULT_MEMORY_PATH))


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase alphanumeric tokens, punctuation stripped.

    JSON-serialised records carry words wrapped in quotes and followed by
    commas or braces (for example "stage": "literature"}), so a naive
    whitespace split would never produce a token equal to a plain query
    word like "literature". Extracting alphanumeric runs instead makes
    embeddings of free-text queries and embeddings of JSON payloads
    directly comparable.
    """
    return _TOKEN_RE.findall(text.lower())


def _embed(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Deterministically embed text into a fixed-size vector.

    This is a hashing-trick bag-of-words embedding: each token is hashed
    into one of dim buckets and the bucket is incremented, then the vector
    is L2-normalised. It requires no model download and no network access,
    and is stable across processes and machines, which keeps recall
    reproducible.
    """
    vector = [0.0] * dim
    tokens = _tokenize(text)
    if not tokens:
        return vector
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(v * v for v in vector))
    if norm > 1e-12:
        vector = [v / norm for v in vector]
    return vector


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def _record_text(record: dict[str, Any]) -> str:
    return json.dumps(record, default=str, sort_keys=True)


def _postgres_dsn() -> str | None:
    return os.environ.get("POSTGRES_DSN") or None


def _postgres_connect() -> Any:
    """Return a live psycopg connection with pgvector registered, or None.

    Any failure (missing packages, unreachable server, bad DSN) is treated
    as "Postgres is not available right now" rather than a fatal error, and
    is only logged once per process to avoid noisy repeated warnings.
    """
    global _postgres_unavailable_warned
    dsn = _postgres_dsn()
    if not dsn:
        return None
    try:
        import psycopg
        from pgvector.psycopg import register_vector

        conn = psycopg.connect(dsn, connect_timeout=3, autocommit=True)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_POSTGRES_TABLE} (
                id BIGSERIAL PRIMARY KEY,
                run_id TEXT,
                stage TEXT,
                payload JSONB NOT NULL,
                embedding vector({EMBEDDING_DIM}),
                created_at TIMESTAMPTZ DEFAULT now()
            )
            """
        )
        register_vector(conn)
        return conn
    except Exception as exc:
        if not _postgres_unavailable_warned:
            print(
                f"[quantlab.memory] Postgres unavailable, using JSONL fallback: {exc}"
            )
            _postgres_unavailable_warned = True
        return None


def _append_postgres(conn: Any, record: dict[str, Any]) -> None:
    from pgvector import Vector

    embedding = Vector(_embed(_record_text(record)))
    conn.execute(
        f"INSERT INTO {_POSTGRES_TABLE} (run_id, stage, payload, embedding) "
        "VALUES (%s, %s, %s, %s)",
        (
            record.get("run_id"),
            record.get("stage"),
            json.dumps(record, default=str),
            embedding,
        ),
    )


def _recall_postgres(conn: Any, query: str, k: int) -> list[dict[str, Any]]:
    from pgvector import Vector

    embedding = Vector(_embed(query))
    rows = conn.execute(
        f"SELECT payload FROM {_POSTGRES_TABLE} ORDER BY embedding <=> %s LIMIT %s",
        (embedding, k),
    ).fetchall()
    return [row[0] if isinstance(row[0], dict) else json.loads(row[0]) for row in rows]


def _append_jsonl(record: dict[str, Any]) -> None:
    memory_path = _memory_path()
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"payload": record, "embedding": _embed(_record_text(record))}
    with memory_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(envelope, default=str) + "\n")


def _recall_jsonl(query: str, k: int) -> list[dict[str, Any]]:
    memory_path = _memory_path()
    if not memory_path.exists():
        return []
    query_vector = _embed(query)
    scored: list[tuple[float, dict[str, Any]]] = []
    with memory_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            envelope = json.loads(line)
            similarity = _cosine_similarity(query_vector, envelope["embedding"])
            scored.append((similarity, envelope["payload"]))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [payload for _, payload in scored[:k]]


def append(record: dict[str, Any]) -> None:
    """Persist a record to the reflective memory store.

    Tries PostgreSQL plus pgvector first when POSTGRES_DSN is configured and
    reachable, and always falls back to the local JSONL file otherwise.
    """
    conn = _postgres_connect()
    if conn is not None:
        try:
            _append_postgres(conn, record)
            return
        except Exception as exc:
            print(
                f"[quantlab.memory] Postgres write failed, using JSONL fallback: {exc}"
            )
    _append_jsonl(record)


def recall(query: str, k: int = 5) -> list[dict[str, Any]]:
    """Return up to k prior records most semantically similar to query.

    Similarity is cosine similarity between hashing-trick embeddings, so
    recall works fully offline and is deterministic given the same memory
    contents.
    """
    conn = _postgres_connect()
    if conn is not None:
        try:
            return _recall_postgres(conn, query, k)
        except Exception as exc:
            print(
                f"[quantlab.memory] Postgres recall failed, using JSONL fallback: {exc}"
            )
    return _recall_jsonl(query, k)
