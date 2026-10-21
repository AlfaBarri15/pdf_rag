from __future__ import annotations

from typing import Any
from uuid import UUID

from psycopg import connect
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.config import DATABASE_URL
from app.ingestion.utils.chunker import Chunk

DEFAULT_DATASET = "default"


def store_document_chunks(
    *,
    source_path: str,
    chunks: list[Chunk],
    title: str | None = None,
    dataset: str = DEFAULT_DATASET,
    database_url: str = DATABASE_URL,
) -> dict[str, Any]:
    """Upsert a documents row and replace its chunks (embeddings written when present)."""
    with connect(database_url) as conn:
        with conn.transaction():
            document_id = _upsert_document(
                conn,
                source_path=source_path,
                title=title,
                dataset=dataset,
            )
            conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
            inserted = _insert_chunks(conn, document_id=document_id, chunks=chunks)

    return {
        "document_id": str(document_id),
        "source_path": source_path,
        "title": title,
        "dataset": dataset,
        "chunk_count": inserted,
    }


def _upsert_document(
    conn,
    *,
    source_path: str,
    title: str | None,
    dataset: str,
) -> UUID:
    """Insert or refresh a documents row keyed by source_path; return its id."""
    row = conn.execute(
        """
        INSERT INTO documents (source_path, title, dataset, ingested_at)
        VALUES (%s, %s, %s, now())
        ON CONFLICT (source_path) DO UPDATE
        SET title = EXCLUDED.title,
            dataset = EXCLUDED.dataset,
            ingested_at = now()
        RETURNING id
        """,
        (source_path, title, dataset),
    ).fetchone()
    assert row is not None
    return row[0]


def _insert_chunks(conn, *, document_id: UUID, chunks: list[Chunk]) -> int:
    """Bulk-insert chunk rows for a document (include embedding when already computed)."""
    if not chunks:
        return 0

    rows = [
        (
            document_id,
            chunk.chunk_index,
            chunk.content.replace("\x00", ""),
            Jsonb(chunk.metadata),
            _vector_literal(chunk.embedding) if chunk.embedding is not None else None,
        )
        for chunk in chunks
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks (document_id, chunk_index, content, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s::vector)
            """,
            rows,
        )
    return len(rows)


def fetch_document_summary(
    *,
    source_path: str,
    database_url: str = DATABASE_URL,
) -> dict[str, Any] | None:
    """Return document metadata and chunk_count, or None if missing."""
    with connect(database_url, row_factory=dict_row) as conn:
        document = conn.execute(
            """
            SELECT id, source_path, title, dataset, ingested_at
            FROM documents
            WHERE source_path = %s
            """,
            (source_path,),
        ).fetchone()
        if document is None:
            return None

        count_row = conn.execute(
            "SELECT count(*) AS chunk_count FROM chunks WHERE document_id = %s",
            (document["id"],),
        ).fetchone()
        assert count_row is not None

        strategy_row = conn.execute(
            """
            SELECT metadata ->> 'strategy' AS strategy
            FROM chunks
            WHERE document_id = %s
            ORDER BY chunk_index
            LIMIT 1
            """,
            (document["id"],),
        ).fetchone()
        return {
            "document_id": str(document["id"]),
            "source_path": document["source_path"],
            "title": document["title"],
            "dataset": document["dataset"],
            "ingested_at": document["ingested_at"].isoformat(),
            "chunk_count": count_row["chunk_count"],
            "chunk_strategy": (strategy_row or {}).get("strategy"),
        }


def _vector_literal(values: list[float]) -> str:
    """Format a float list as a pgvector literal: [0.1,0.2,...]."""
    return "[" + ",".join(str(float(v)) for v in values) + "]"


def embed_document_chunks(
    *,
    source_path: str,
    only_missing: bool = True,
    database_url: str = DATABASE_URL,
) -> dict[str, Any]:
    """Embed stored chunk text and write vectors into chunks.embedding."""
    from app.ingestion.utils.embeddings import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL_NAME, embed_texts

    with connect(database_url, row_factory=dict_row) as conn:
        document = conn.execute(
            """
            SELECT id, source_path, title
            FROM documents
            WHERE source_path = %s
            """,
            (source_path,),
        ).fetchone()
        if document is None:
            raise LookupError(f"No document stored for source_path={source_path!r}")

        if only_missing:
            rows = conn.execute(
                """
                SELECT id, content
                FROM chunks
                WHERE document_id = %s AND embedding IS NULL
                ORDER BY chunk_index
                """,
                (document["id"],),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, content
                FROM chunks
                WHERE document_id = %s
                ORDER BY chunk_index
                """,
                (document["id"],),
            ).fetchall()

        if not rows:
            return {
                "document_id": str(document["id"]),
                "source_path": source_path,
                "title": document["title"],
                "embedded_count": 0,
                "model": EMBEDDING_MODEL_NAME,
                "dimensions": EMBEDDING_DIMENSIONS,
                "only_missing": only_missing,
            }

        vectors = embed_texts([row["content"] for row in rows])
        with conn.transaction():
            with conn.cursor() as cur:
                for row, vector in zip(rows, vectors, strict=True):
                    if len(vector) != EMBEDDING_DIMENSIONS:
                        raise ValueError(
                            f"Expected {EMBEDDING_DIMENSIONS} dims, got {len(vector)}"
                        )
                    cur.execute(
                        """
                        UPDATE chunks
                        SET embedding = %s::vector
                        WHERE id = %s
                        """,
                        (_vector_literal(vector), row["id"]),
                    )

        return {
            "document_id": str(document["id"]),
            "source_path": source_path,
            "title": document["title"],
            "embedded_count": len(rows),
            "model": EMBEDDING_MODEL_NAME,
            "dimensions": EMBEDDING_DIMENSIONS,
            "only_missing": only_missing,
        }
