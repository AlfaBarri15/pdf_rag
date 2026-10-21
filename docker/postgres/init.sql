CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_path TEXT NOT NULL UNIQUE,
    title TEXT,
    dataset TEXT NOT NULL DEFAULT 'default',
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX documents_dataset_idx ON documents (dataset);

CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(content, ''))) STORED,
    embedding VECTOR(384),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX chunks_embedding_hnsw_idx
  ON chunks
  USING hnsw (embedding vector_cosine_ops)
  WHERE embedding IS NOT NULL;

CREATE INDEX chunks_content_tsv_idx ON chunks USING gin (content_tsv);

-- Async ingest queue (API enqueues; ingest-worker claims with SKIP LOCKED).
CREATE TABLE ingest_jobs (
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

CREATE INDEX ingest_jobs_status_created_idx
  ON ingest_jobs (status, created_at);

-- One row per eval invocation (FinanceBench or a later golden set).
CREATE TABLE eval_runs (
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

CREATE TABLE eval_cases (
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

CREATE INDEX eval_cases_run_id_idx ON eval_cases (run_id);