"""Postgres-backed ingest job queue (claim with FOR UPDATE SKIP LOCKED)."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from psycopg import connect
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.config import DATABASE_URL


def _serialize_job(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a DB row into a JSON-friendly job payload."""
    return {
        "job_id": str(row["id"]),
        "status": row["status"],
        "relative_path": row["relative_path"],
        "params": row["params"] or {},
        "progress": row["progress"] or {},
        "result": row["result"],
        "error": row["error"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "started_at": row["started_at"].isoformat() if row["started_at"] else None,
        "finished_at": row["finished_at"].isoformat() if row["finished_at"] else None,
    }


def create_job(
    *,
    relative_path: str,
    params: dict[str, Any],
    database_url: str = DATABASE_URL,
) -> dict[str, Any]:
    """Insert a queued ingest job and return it."""
    with connect(database_url, row_factory=dict_row) as conn:
        row = conn.execute(
            """
            INSERT INTO ingest_jobs (status, relative_path, params)
            VALUES ('queued', %s, %s)
            RETURNING *
            """,
            (relative_path, Jsonb(params)),
        ).fetchone()
        assert row is not None
        conn.commit()
        return _serialize_job(row)


def get_job(
    *,
    job_id: str | UUID,
    database_url: str = DATABASE_URL,
) -> dict[str, Any] | None:
    """Fetch one job by id, or None if missing."""
    with connect(database_url, row_factory=dict_row) as conn:
        row = conn.execute(
            "SELECT * FROM ingest_jobs WHERE id = %s",
            (str(job_id),),
        ).fetchone()
        return _serialize_job(row) if row else None


def claim_next_job(*, database_url: str = DATABASE_URL) -> dict[str, Any] | None:
    """Atomically claim the oldest queued job (safe with multiple workers)."""
    with connect(database_url, row_factory=dict_row) as conn:
        with conn.transaction():
            row = conn.execute(
                """
                UPDATE ingest_jobs
                SET status = 'running',
                    started_at = now(),
                    progress = %s
                WHERE id = (
                    SELECT id
                    FROM ingest_jobs
                    WHERE status = 'queued'
                    ORDER BY created_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                RETURNING *
                """,
                (Jsonb({"stage": "claimed"}),),
            ).fetchone()
        return _serialize_job(row) if row else None


def update_job_progress(
    *,
    job_id: str | UUID,
    stage: str,
    database_url: str = DATABASE_URL,
) -> None:
    """Record a lightweight stage hint for status polling."""
    with connect(database_url) as conn:
        conn.execute(
            """
            UPDATE ingest_jobs
            SET progress = %s
            WHERE id = %s AND status = 'running'
            """,
            (Jsonb({"stage": stage}), str(job_id)),
        )
        conn.commit()


def mark_job_succeeded(
    *,
    job_id: str | UUID,
    result: dict[str, Any],
    database_url: str = DATABASE_URL,
) -> dict[str, Any]:
    """Mark a running job as succeeded and store the ingest result."""
    with connect(database_url, row_factory=dict_row) as conn:
        row = conn.execute(
            """
            UPDATE ingest_jobs
            SET status = 'succeeded',
                result = %s,
                error = NULL,
                progress = %s,
                finished_at = now()
            WHERE id = %s
            RETURNING *
            """,
            (Jsonb(result), Jsonb({"stage": "done"}), str(job_id)),
        ).fetchone()
        assert row is not None
        conn.commit()
        return _serialize_job(row)


def mark_job_failed(
    *,
    job_id: str | UUID,
    error: str,
    database_url: str = DATABASE_URL,
) -> dict[str, Any]:
    """Mark a job as failed and store the error message."""
    with connect(database_url, row_factory=dict_row) as conn:
        row = conn.execute(
            """
            UPDATE ingest_jobs
            SET status = 'failed',
                error = %s,
                progress = %s,
                finished_at = now()
            WHERE id = %s
            RETURNING *
            """,
            (error, Jsonb({"stage": "failed"}), str(job_id)),
        ).fetchone()
        assert row is not None
        conn.commit()
        return _serialize_job(row)
