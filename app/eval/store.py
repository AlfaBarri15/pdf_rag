"""Write eval_runs and eval_cases."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from psycopg import connect
from psycopg.types.json import Jsonb

from app.config import DATABASE_URL


def create_run(
    *,
    dataset: str,
    embedding_model: str | None,
    rerank_model: str | None,
    chat_model: str | None,
    chunk_settings: dict[str, Any],
    database_url: str = DATABASE_URL,
) -> str:
    """Insert a run row and return its id."""
    with connect(database_url) as conn:
        row = conn.execute(
            """
            INSERT INTO eval_runs (
                dataset, embedding_model, rerank_model, chat_model, chunk_settings
            )
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (dataset, embedding_model, rerank_model, chat_model, Jsonb(chunk_settings)),
        ).fetchone()
        conn.commit()
    assert row is not None
    return str(row[0])


def insert_case(
    *,
    run_id: str,
    question_id: str,
    question: str,
    gold_answer: str,
    doc_name: str,
    doc_hit: bool,
    evidence_hit: bool,
    answer_hit: bool,
    top_score: float | None,
    retrieved_docs: list[str],
    model_answer: str,
    database_url: str = DATABASE_URL,
) -> None:
    """Insert one scored question."""
    with connect(database_url) as conn:
        conn.execute(
            """
            INSERT INTO eval_cases (
                run_id, question_id, question, gold_answer, doc_name,
                doc_hit, evidence_hit, answer_hit, top_score, retrieved_docs, model_answer
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                UUID(run_id),
                question_id,
                question,
                gold_answer,
                doc_name,
                doc_hit,
                evidence_hit,
                answer_hit,
                top_score,
                Jsonb(retrieved_docs),
                model_answer,
            ),
        )
        conn.commit()


def finish_run(
    *,
    run_id: str,
    case_count: int,
    doc_recall: float,
    evidence_recall: float,
    answer_recall: float,
    database_url: str = DATABASE_URL,
) -> None:
    """Store aggregate rates when the run completes."""
    with connect(database_url) as conn:
        conn.execute(
            """
            UPDATE eval_runs
            SET case_count = %s,
                doc_recall = %s,
                evidence_recall = %s,
                answer_recall = %s,
                finished_at = now()
            WHERE id = %s
            """,
            (case_count, doc_recall, evidence_recall, answer_recall, UUID(run_id)),
        )
        conn.commit()
