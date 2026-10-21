from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Response, status
from psycopg.errors import Error as PsycopgError

from app.ingestion.utils.chunker import DEFAULT_MAX_CHARS, DEFAULT_OVERLAP
from app.ingestion.utils.jobs import create_job, get_job

router = APIRouter(prefix="/ingest", tags=["ingest"])

# Project-root/data — works on the host and in the api container when data is mounted.
DATA_ROOT = Path("/app/data")
if not DATA_ROOT.exists():
    DATA_ROOT = Path(__file__).resolve().parents[3] / "data"


def _resolve_data_path(relative_path: str) -> Path:
    """Resolve a path under data/ and reject path traversal outside that root."""
    path = (DATA_ROOT / relative_path).resolve()
    try:
        path.relative_to(DATA_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Path must stay under data/") from exc
    return path


def _source_path(path: Path) -> str:
    """Path relative to data/, used as documents.source_path."""
    return str(path.relative_to(DATA_ROOT.resolve()))


@router.post("/", status_code=status.HTTP_202_ACCEPTED)
def enqueue_ingest(
    response: Response,
    relative_path: str = Query(
        ...,
        description="Path relative to data/, e.g. raw/postgresql-19-A4.pdf",
    ),
    max_pages: int | None = Query(
        None,
        ge=1,
        le=10000,
        description="PDF page limit. Omit for the full document.",
    ),
    max_chars: int = Query(
        DEFAULT_MAX_CHARS,
        ge=100,
        le=10000,
        description="Max characters per chunk",
    ),
    overlap: int = Query(
        DEFAULT_OVERLAP,
        ge=0,
        le=1000,
        description="Overlap when splitting oversized paragraphs",
    ),
    embed: bool = Query(
        True,
        description="If true, batch-embed all chunks before store (MiniLM → VECTOR(384))",
    ),
):
    """Enqueue async ingest. Poll GET /ingest/jobs/{job_id} for status."""
    path = _resolve_data_path(relative_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {relative_path}")

    source_path = _source_path(path)
    params = {
        "max_pages": max_pages,
        "max_chars": max_chars,
        "overlap": overlap,
        "embed": embed,
    }

    try:
        job = create_job(relative_path=source_path, params=params)
    except (PsycopgError, OSError) as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}") from exc

    response.headers["Location"] = f"/ingest/jobs/{job['job_id']}"
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "relative_path": job["relative_path"],
        "params": job["params"],
    }


@router.get("/jobs/{job_id}")
def ingest_job_status(job_id: UUID):
    """Return status / progress / result for an ingest job."""
    try:
        job = get_job(job_id=job_id)
    except (PsycopgError, OSError) as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}") from exc

    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id={job_id}")
    return job
