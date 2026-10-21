import pytest
from psycopg import connect
from psycopg.errors import OperationalError

from app.config import DATABASE_URL
from app.ingestion.utils.chunker import Chunk
from app.ingestion.utils.store import fetch_document_summary, store_document_chunks

SOURCE = "test/store-smoke.txt"


def _db_available() -> bool:
    try:
        with connect(DATABASE_URL, connect_timeout=2) as conn:
            conn.execute("SELECT 1")
        return True
    except (OperationalError, OSError):
        return False


@pytest.mark.skipif(not _db_available(), reason="Postgres not reachable")
def test_store_document_chunks_roundtrip():
    chunks = [
        Chunk(
            chunk_index=0,
            content="Hello chunk one",
            metadata={"source_path": SOURCE, "strategy": "paragraph"},
        ),
        Chunk(
            chunk_index=1,
            content="Hello chunk two",
            metadata={"source_path": SOURCE, "strategy": "paragraph"},
        ),
    ]

    first = store_document_chunks(
        source_path=SOURCE,
        title="Store smoke",
        chunks=chunks,
    )
    assert first["chunk_count"] == 2

    summary = fetch_document_summary(source_path=SOURCE)
    assert summary is not None
    assert summary["chunk_count"] == 2
    assert summary["title"] == "Store smoke"

    # Re-ingest replaces chunks (idempotent on source_path).
    replaced = store_document_chunks(
        source_path=SOURCE,
        title="Store smoke v2",
        chunks=chunks[:1],
    )
    assert replaced["document_id"] == first["document_id"]
    assert replaced["chunk_count"] == 1

    summary = fetch_document_summary(source_path=SOURCE)
    assert summary is not None
    assert summary["chunk_count"] == 1
    assert summary["title"] == "Store smoke v2"

    with connect(DATABASE_URL) as conn:
        conn.execute("DELETE FROM documents WHERE source_path = %s", (SOURCE,))
        conn.commit()
