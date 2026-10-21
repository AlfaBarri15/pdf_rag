-- Additive migration for existing postgres_data volumes (init.sql only runs on first boot).
-- Apply:
--   docker exec -i rag-postgres psql -U rag -d rag < docker/postgres/migrate_ingest_jobs.sql

CREATE TABLE IF NOT EXISTS ingest_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'succeeded', 'failed')),
    relative_path TEXT NOT NULL,
    params JSONB NOT NULL DEFAULT '{}'::jsonb,
    progress JSONB NOT NULL DEFAULT '{}'::jsonb,
    result JSONB,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS ingest_jobs_status_created_idx
  ON ingest_jobs (status, created_at);
