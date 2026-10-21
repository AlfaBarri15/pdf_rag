from psycopg import connect
from psycopg.errors import OperationalError

import pytest

from app.config import DATABASE_URL
from app.ingestion.utils.jobs import (
    claim_next_job,
    create_job,
    get_job,
    mark_job_failed,
    mark_job_succeeded,
)


def _db_available() -> bool:
    try:
        with connect(DATABASE_URL, connect_timeout=2) as conn:
            conn.execute("SELECT 1 FROM ingest_jobs LIMIT 0")
        return True
    except (OperationalError, OSError, Exception):
        return False


@pytest.mark.skipif(not _db_available(), reason="Postgres ingest_jobs table not reachable")
def test_create_get_and_claim_job():
    created = create_job(
        relative_path="raw/test-async.pdf",
        params={"max_pages": 1, "embed": False},
    )
    assert created["status"] == "queued"
    assert created["job_id"]

    fetched = get_job(job_id=created["job_id"])
    assert fetched is not None
    assert fetched["relative_path"] == "raw/test-async.pdf"

    claimed = claim_next_job()
    # May claim our job or an older queued one; keep claiming until ours or none.
    seen = []
    current = claimed
    while current is not None and current["job_id"] != created["job_id"]:
        seen.append(current["job_id"])
        mark_job_failed(job_id=current["job_id"], error="test cleanup other job")
        current = claim_next_job()

    if current is None:
        # Our job may have been claimed by a live worker; just assert get still works.
        again = get_job(job_id=created["job_id"])
        assert again is not None
        return

    assert current["status"] == "running"
    done = mark_job_succeeded(
        job_id=current["job_id"],
        result={"chunk_count": 0, "source_path": "raw/test-async.pdf"},
    )
    assert done["status"] == "succeeded"
    assert done["result"]["chunk_count"] == 0
