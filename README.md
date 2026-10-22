# Software Doc Search Tool

A **RAG-style documentation search engine** built end-to-end in Python: ingest software docs (PDF), chunk and embed them, and store vectors in Postgres with **pgvector**—served by a containerized **FastAPI** API.

Built as a hands-on data engineering / ML systems learning project, with a pipeline shaped like production RAG stacks (including Cohere-style **Embed → retrieve → Rerank → generate** stages). Local, free tooling today; provider APIs can plug in later without redesigning storage.

## Why this project

Recruiters and hiring managers can use this repo to see practical experience with:

- **Data pipelines** — parse → clean → chunk → embed → persist
- **Vector search foundations** — embeddings + Postgres/`pgvector`
- **API design** — coarse ingest endpoint, thin HTTP layer, modular domain code
- **Containers & local ops** — Docker Compose (API + Postgres + LLM), env-based config
- **Software craft** — tests, clear package layout, docs for each layer

## Features

| Area                         | Detail                                              |
|------------------------------|-----------------------------------------------------|
| Offline ingest               | Async jobs via `POST /ingest/` + worker             |
| Chunking                     | H1–H3 sections for manuals; one chunk per PDF page for FinanceBench |
| Vector index                 | HNSW on `chunks.embedding`                          |
| Hybrid retrieve              | pgvector cosine + Postgres full-text, then cross-encoder rerank (`top_n=8`) |
| Corpus filter                | `documents.dataset` plus company/year from the question title |
| Grounded answers             | `POST /answer/` and Open WebUI (`/v1/chat/completions`) with citations |
| Eval                         | FinanceBench runner writes `eval_runs` / `eval_cases` |
| LLM runtime                  | Compose `llm` + `model-sync`                        |

Sample corpus: PostgreSQL documentation PDF under `data/raw/`.

## Architecture

```text
OFFLINE                                ONLINE
───────                                ──────
Documents                              User question
   │                                        │
   ▼                                        ▼
Parse (PDF → Markdown or pages)        Query embedding + lexical terms
   │                                        │
   ▼                                        ▼
Clean                                  Dataset / company / year filter
   │                                        │
   ▼                                        ▼
Chunk (H1–H3 or one page)              Hybrid retrieve (pgvector + FTS)
   │                                   RRF merge → cross-encoder top 8
   ▼                                        │
Embed (MiniLM, 384-d)                       ▼
   │                                   LLM service (Compose)
   ▼                                        │
Postgres + pgvector (HNSW + GIN)            ▼
                                       Answer + citations
```

**Design choice:** public HTTP is small (`/ingest/`, `/answer/`, `/health`). Ingestion and retrieval stay as separate Python packages so they remain testable and can scale as workers. See `app/api/README.md` and `app/ingestion/README.md`.

## Design choices

### Embeddings (what we store)

Each chunk’s `content` is turned into a **384-dimensional vector** with `sentence-transformers` (`all-MiniLM-L6-v2`) and saved in `chunks.embedding` (`VECTOR(384)`).

We call `encode(..., normalize_embeddings=True)` so vectors sit on a unit sphere. That makes **cosine similarity** (angle between vectors) a natural “same meaning?” score for semantic search—aligned with how retrieval-oriented embed models are typically used (including Cohere-style Embed → retrieve flows).

### What an index is (and is not)

An **index** is a side structure Postgres builds to find rows faster, without scanning the whole table (like a book index → page number).

| Indexes are…                              | Indexes are not…                            |
|-------------------------------------------|---------------------------------------------|
| Extra structures next to table data       | A second business table for embeddings      |
| Speeds up lookups / similarity search     | Required for correctness (queries work OK)  |
| Cost some disk and slightly slower writes | A replacement for `documents` / `chunks`    |

We do **not** create a separate table for HNSW. Vectors stay on `chunks.embedding`; HNSW is an index **on that column**.

### HNSW vector index

**HNSW** (Hierarchical Navigable Small World) is a graph-based index for **approximate nearest-neighbor** search over vectors.

- **Without it:** compare the query embedding to every chunk (fine for tiny demos; slow at scale).
- **With it:** walk a multi-layer graph toward nearby vectors and return top-k much faster.

DDL (same `chunks` table):

```sql
CREATE INDEX chunks_embedding_hnsw_idx
  ON chunks
  USING hnsw (embedding vector_cosine_ops);
```

| Piece                 | Meaning                                      |
|-----------------------|----------------------------------------------|
| `USING hnsw`          | Use the HNSW algorithm                       |
| `embedding`           | Column being indexed                         |
| `vector_cosine_ops`   | “Nearby” = high cosine similarity            |

**Why cosine (`vector_cosine_ops`) and not L2?**  
Our embeddings are **normalized**. Cosine measures direction/alignment of meaning. pgvector also offers `vector_l2_ops` (Euclidean) and `vector_ip_ops` (inner product); with unit vectors, cosine and inner product rank similarly—we pick cosine to match the normalized semantic-search setup. Search queries should use the matching distance operator (e.g. cosine `<=>`) so Postgres can use this index.

### Hybrid top-k retrieval

An answer reads a short list of chunks, called **top-k**, not the whole corpus. `POST /answer/` asks each search for `top_k=20` chunks, fuses those two rankings, then the cross-encoder keeps `top_n=8` for the prompt.

**1. Two ranked lists, same filters**

Vector search and keyword search both apply the dataset filter and any company or year parsed from the question (`3M_2018_10K`-style titles).

| List | How it ranks | What the score means |
|------|----------------|----------------------|
| Vector | pgvector cosine (`<=>`) on `chunks.embedding`, via the HNSW index | Distance: lower is closer in meaning |
| Keyword | Postgres full-text on `chunks.content_tsv` (`to_tsquery`) | `ts_rank_cd`: higher means more of the question’s words |

Those two scores use different units, so the pipeline keeps each list’s **order** and throws the raw numbers away before the merge. `lexical_tsquery` turns the question into an OR of content words (`capital | expenditure | 2018`). With no such words, or if full-text is unavailable, the vector order is the result.

**2. Reciprocal rank fusion (`rrf_merge`)**

`app/retrieval/utils/search.py` scores a chunk by its place in each list. Rank starts at 1. The constant `rrf_k` is 60 (the usual reciprocal-rank-fusion constant):

```text
score(chunk) = Σ  1 / (60 + rank in that list)
```

Rank 1 in one list scores `1/61` ≈ 0.0164. Rank 2 scores `1/62` ≈ 0.0161. The gaps are small, so a chunk near the top of **both** lists beats a chunk that only one search found. The fused ids are cut back to 20 (`retrieve_k`).

The unit test uses two short lists:

| Chunk | Vector rank | Keyword rank | Score |
|-------|-------------|--------------|-------|
| `shared` | 2 | 2 | `1/62 + 1/62` ≈ 0.032 |
| `vec-only` | 1 | — | `1/61` ≈ 0.016 |
| `lex-only` | — | 1 | `1/61` ≈ 0.016 |

`shared` comes out first.

**3. Rerank to top-n**

The cross-encoder (`ms-marco-MiniLM-L-6-v2`) reads the question together with each of those 20 chunks and keeps the best 8. Those 8 are the citations in the prompt. Rerank is the slow step, so it runs on the fused top-k and not on every row in `chunks`.

That is the same shape as Cohere-style Embed → retrieve → Rerank: a cheap top-k, then a stronger model on a short list.

### API surface

- **Public:** `POST /ingest/` (async enqueue), `GET /ingest/jobs/{id}`, `POST /answer/`, `GET /health`.
- **Domain modules:** `parser` / `chunker` / `store` / `search` stay internal—not one public route per step.

## Tech stack

| Layer        | Choice                                                      |
|--------------|-------------------------------------------------------------|
| Language     | Python 3.13                                                 |
| API          | FastAPI + Uvicorn                                           |
| Packaging    | uv                                                          |
| Database     | PostgreSQL 17 + pgvector                                    |
| Embeddings   | `sentence-transformers` (`all-MiniLM-L6-v2`)                |
| Rerank       | `CrossEncoder` (`ms-marco-MiniLM-L-6-v2`)                   |
| LLM          | Ollama (`qwen2.5:7b`) in Compose `llm` service              |
| PDF parsing  | pymupdf4llm (Markdown) + pypdf (pages / fallback)           |
| Runtime      | Docker Compose (`api`, `ingest-worker`, `postgres`, `llm`, `model-sync`, `open-webui`) |

## Project layout

```text
app/
  api/
    __init__.py      # regroups HTTP routers
    utils/           # health, ingest, answer routes (mounted)
  ingestion/
    pipeline.py      # regroups parse → chunk → store → embed
    utils/           # parser, clean, chunker, store, embeddings
  retrieval/
    pipeline.py      # regroups search → (rerank) → answer
    search.py        # re-exports pipeline helpers
    utils/           # search, rerank, prompt, generate
  eval/
    download.py      # FinanceBench questions + PDFs
    runner.py        # ingest missing → run_answer → score
docker/
  api/
  postgres/          # init.sql + migrate_eval.sql + migrate_fts.sql
  open-webui/
data/raw/
data/eval/           # gitignored PDFs; questions.jsonl after download
tests/
```

## Quick start

```bash
cp .env.example .env
docker compose up --build -d
```

| Resource          | URL                            |
|-------------------|--------------------------------|
| OpenAPI / Swagger | http://localhost:8000/docs     |
| Health            | http://localhost:8000/health   |
| Chat (Open WebUI) | http://localhost:3000          |
| LLM API (debug)   | http://localhost:11434         |

First `docker compose up` syncs `qwen2.5:7b` via the one-shot `model-sync` job (can take a few minutes; weights persist in the `model_weights` volume). That 7B chat model is the local stand-in for a grounded generator (Cohere Command in production). If it OOMs on your laptop, set `OLLAMA_MODEL=llama3.2` in `.env` and re-run `docker compose run --rm model-sync`.

Optional: run only Postgres in Docker and the API with `uv`:

```bash
docker compose up -d postgres
uv sync --group dev
uv run uvicorn app.main:app --reload
```

```bash
uv run pytest
```

## API (summary)

Interactive OpenAPI UI: http://localhost:8000/docs

| Method | Endpoint            | Purpose                                                    |
|--------|---------------------|------------------------------------------------------------|
| `GET`  | `/health`           | API + database liveness                                    |
| `POST` | `/ingest/`          | Enqueue async ingest → **202** + `job_id`                  |
| `GET`  | `/ingest/jobs/{id}` | Poll ingest job status / result                            |
| `POST` | `/answer/`          | Grounded answer + citations (JSON `{"query": "..."}`)      |
| `GET`  | `/v1/models`        | Model list for Open WebUI (`rag`)                          |
| `POST` | `/v1/chat/completions` | OpenAI-shaped chat that runs the same RAG pipeline      |

**Ingest (async)**

```http
POST http://localhost:8000/ingest/?relative_path=raw/postgresql-19-A4.pdf&max_pages=50&embed=true
→ 202 {"job_id":"...", "status":"queued", ...}

GET http://localhost:8000/ingest/jobs/{job_id}
```

Omit `max_pages` for the whole PDF (handled by `ingest-worker`, not the API request).

**Answer (RAG)**

```http
POST http://localhost:8000/answer/
Content-Type: application/json

{"query": "what is postgresql"}
```

Returns `answer`, numbered `citations`, and underlying retrieval `results`.

**Chat UI**

Open WebUI (http://localhost:3000) is only the window. It calls `POST /v1/chat/completions` on this API. That route takes the latest user message, retrieves and reranks first, then streams Ollama tokens as server-sent events. A Sources footer is the last chunk. `POST /answer/` still returns one JSON body.

Inside Open WebUI, pick the **rag** model. Direct Ollama is disabled in Compose so a model choice cannot skip retrieval. Do not turn on Open WebUI’s built-in Knowledge upload — that would be a second, separate index.

The Compose image is `docker/open-webui` (upstream Open WebUI plus `cryptography==46.0.3`). Newer cryptography wheels crash with an illegal instruction on Docker Desktop for Apple Silicon.

## Eval

The eval runner uses the same ingest and answer code as the API. It only adds a dataset filter so benchmark filings are not mixed with other documents, then writes scores to Postgres.

```bash
uv run python -m app.eval.download
uv run python -m app.eval.runner --dataset financebench --limit 5
```

`download` pulls the public FinanceBench set (150 questions and their PDFs) into `data/eval/financebench/`. The runner ingests those filings as **one chunk per PDF page** (re-ingests if an older heading strategy is stored), calls `run_answer` with `dataset=financebench`, and stores one `eval_runs` row plus one `eval_cases` row per question.

Search is hybrid: cosine neighbors fused with full-text on `chunks.content_tsv`, then rerank. How those lists become one top-k is under [Hybrid top-k retrieval](#hybrid-top-k-retrieval). Existing databases need the migrate scripts below before the first run.

| Metric | Meaning |
|--------|---------|
| `doc_recall` | Share of questions whose gold filing was in the retrieved chunks |
| `evidence_recall` | Share whose labeled fact (overlapping words + a gold number) is in those chunks |
| `answer_recall` | Share whose gold answer, or its numbers, appeared in the model answer |

On the 150-question FinanceBench run, **retrieval is ahead of generation**: `evidence_hit` was ~97% while `answer_hit` was ~25%. The labeled table/page is usually in context; the chat model still misses the gold string. The next lever is a **stronger LLM** (`OLLAMA_MODEL` in `.env`, then `docker compose run --rm model-sync`), not another retrieve rewrite. Production analog: Cohere Command behind the same `generate.chat()` client.

Hit-type percentages for the latest run (True count / questions):

```bash
uv run python -m app.eval.summary
```

Per-question hits plus a short gold vs model preview:

```bash
uv run python -m app.eval.results
```

Pass `--run-id <uuid>` on either command to inspect an older run.

A later golden set is the same layout: `data/eval/<name>/questions.jsonl` and `pdfs/`. Do not send eval questions through Open WebUI.

Existing databases (volumes created before these tables existed):

```bash
docker exec -i rag-postgres psql -U rag -d rag < docker/postgres/migrate_eval.sql
docker exec -i rag-postgres psql -U rag -d rag < docker/postgres/migrate_fts.sql
```

## What’s next

Tracked in [`TODO.md`](TODO.md): **change the LLM to raise `answer_hit`**; optional `HF_TOKEN`; Cohere Chat behind the same `generate` client; CI; later a file-upload endpoint. Telemetry UI stays deferred.

## Author

Built by Alfa as a portfolio / learning project for data engineering and applied AI systems.
