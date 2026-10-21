-- Additive migration for existing postgres_data volumes (init.sql only runs on first boot).
-- Apply:
--   docker exec -i rag-postgres psql -U rag -d rag < docker/postgres/migrate_eval.sql

ALTER TABLE documents
  ADD COLUMN IF NOT EXISTS dataset TEXT NOT NULL DEFAULT 'default';

CREATE INDEX IF NOT EXISTS documents_dataset_idx ON documents (dataset);

CREATE TABLE IF NOT EXISTS eval_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset TEXT NOT NULL,
    embedding_model TEXT,
    rerank_model TEXT,
    chat_model TEXT,
    chunk_settings JSONB NOT NULL DEFAULT '{}'::jsonb,
    case_count INTEGER NOT NULL DEFAULT 0,
    doc_recall DOUBLE PRECISION,
    evidence_recall DOUBLE PRECISION,
    answer_recall DOUBLE PRECISION,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS eval_cases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES eval_runs (id) ON DELETE CASCADE,
    question_id TEXT NOT NULL,
    question TEXT NOT NULL,
    gold_answer TEXT,
    doc_name TEXT,
    doc_hit BOOLEAN NOT NULL,
    evidence_hit BOOLEAN NOT NULL,
    answer_hit BOOLEAN NOT NULL,
    top_score DOUBLE PRECISION,
    retrieved_docs JSONB NOT NULL DEFAULT '[]'::jsonb,
    model_answer TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS eval_cases_run_id_idx ON eval_cases (run_id);
