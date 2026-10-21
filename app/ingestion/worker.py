"""Ingest worker: claim queued jobs from Postgres and run the offline pipeline.

Not an API — a long-running script. Run with:
  python -m app.ingestion.worker
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from app.ingestion.pipeline import run_ingest
from app.ingestion.utils.jobs import (
    claim_next_job,
    mark_job_failed,
    mark_job_succeeded,
    update_job_progress,
)

logger = logging.getLogger(__name__)

POLL_SECONDS = float(os.getenv("INGEST_WORKER_POLL_SECONDS", "2"))

DATA_ROOT = Path("/app/data")
if not DATA_ROOT.exists():
    DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def _resolve_data_path(relative_path: str) -> Path:
    path = (DATA_ROOT / relative_path).resolve()
    path.relative_to(DATA_ROOT.resolve())
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def process_job(job: dict) -> None:
    """Run one claimed job through run_ingest and update job status."""
    job_id = job["job_id"]
    relative_path = job["relative_path"]
    params = job.get("params") or {}

    def on_progress(stage: str) -> None:
        update_job_progress(job_id=job_id, stage=stage)
        logger.info("job=%s stage=%s", job_id, stage)

    try:
        path = _resolve_data_path(relative_path)
        result = run_ingest(
            path=path,
            source_path=relative_path,
            title=path.stem,
            max_pages=params.get("max_pages"),
            max_chars=params.get("max_chars", 1000),
            overlap=params.get("overlap", 100),
            embed=params.get("embed", True),
            dataset=params.get("dataset", "default"),
            strategy=params.get("strategy", "markdown_sections"),
            on_progress=on_progress,
        )
        mark_job_succeeded(job_id=job_id, result=result)
        logger.info(
            "job=%s succeeded chunks=%s",
            job_id,
            result.get("chunk_count"),
        )
    except Exception as exc:  # noqa: BLE001 — worker must never die on one bad job
        logger.exception("job=%s failed", job_id)
        mark_job_failed(job_id=job_id, error=str(exc))


def run_forever(*, poll_seconds: float = POLL_SECONDS) -> None:
    """Poll for queued jobs until the process is stopped."""
    logger.info(
        "ingest-worker started data_root=%s poll_seconds=%s",
        DATA_ROOT,
        poll_seconds,
    )
    while True:
        job = claim_next_job()
        if job is None:
            time.sleep(poll_seconds)
            continue
        logger.info("claimed job=%s path=%s", job["job_id"], job["relative_path"])
        process_job(job)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    run_forever()


if __name__ == "__main__":
    main()
