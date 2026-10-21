# `app/ingestion`

Offline pipeline: turn files under `data/` into searchable rows in Postgres.

## Layout

| Path                   | Role                                                                      |
|------------------------|---------------------------------------------------------------------------|
| `pipeline.py`          | Regroups parse → clean → chunk → batch-embed → store (`run_ingest`)       |
| `worker.py`            | Async script loop: claim job → `run_ingest` → update status               |
| `utils/jobs.py`        | Postgres job queue (`create` / `get` / `claim` / succeed / fail)          |
| `utils/parser.py`      | PDF → Markdown (`pymupdf4llm`); plain `pypdf` fallback; `parse_pdf_pages` for one block per page |
| `utils/clean.py`       | Format-agnostic clean for plain text + Markdown (TOC, trailing end matter, license boilerplate, junk) |
| `utils/chunker.py`     | H1–H3 sections by default; `strategy=pages` keeps each PDF page together    |
| `utils/store.py`       | Upsert `documents` / `chunks`, write embeddings when present              |
| `utils/embeddings.py`  | Local MiniLM (`sentence-transformers`) → 384-dim vectors (batched)        |

HTTP enqueues jobs; the **worker** runs `pipeline.py`. Step implementations live under `utils/`.

## Async ingest

```http
POST /ingest/?relative_path=raw/postgresql-19-A4.pdf&max_pages=50&embed=true
→ 202 {"job_id": "...", "status": "queued"}

GET /ingest/jobs/{job_id}
→ status: queued | running | succeeded | failed
```

Worker (Compose `ingest-worker`, same image as API):

```bash
python -m app.ingestion.worker
```

Existing DB volumes: apply [`docker/postgres/migrate_ingest_jobs.sql`](../../docker/postgres/migrate_ingest_jobs.sql).
