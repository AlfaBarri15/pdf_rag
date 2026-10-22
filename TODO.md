# TODO

Local RAG pipeline checklist (free/local stack; Cohere-shaped stages).

## Done

- [x] Environment: uv, Docker Compose, Postgres + pgvector, API container
- [x] Ingest: parse PDF, chunk (skip TOC), store `documents` / `chunks`
- [x] Schema: `VECTOR(384)` for MiniLM embeddings
- [x] Add `sentence-transformers` + rebuild API image
- [x] Create `app/ingestion/embeddings.py` embedder module
- [x] Smoke-test `embed_text` in api container (384 dims)
- [x] Write embeddings into `chunks.embedding`
- [x] Regroup ingest API: `POST /ingest/` (parse → chunk → store → embed); keep `GET /ingest/chunk` preview
- [x] Add pgvector HNSW index on `chunks.embedding` (`vector_cosine_ops`, partial where not null)
- [x] Online: query embed + top-k retrieval API (`GET /search/`)
- [x] Local cross-encoder rerank (`GET /search/?rerank=true&top_n=3`)
- [x] LLM container in Compose (`llm` engine + one-shot `model-sync` Job for weights)
- [x] Retrieval `pipeline.py` (`run_search` / `run_answer`) + `POST /answer/` with citations
- [x] Full-document ingest (`max_pages` optional) + cleaner text + batched embeddings
- [x] Async ingest jobs (`POST /ingest/` → 202) + `ingest-worker` + `GET /ingest/jobs/{id}`
- [x] PDF → Markdown (`pymupdf4llm`) + chunk by H1–H3 sections (paragraph fallback)
- [x] Open WebUI chat via `POST /v1/chat/completions` (still runs RAG; Ollama direct disabled)
- [x] Stream Ollama tokens on `POST /v1/chat/completions` when `stream` is true
- [x] `documents.dataset` filter so FinanceBench filings are not mixed with software docs
- [x] FinanceBench eval runner (`uv run python -m app.eval.runner`) with stored `eval_runs` / `eval_cases`
- [x] Title filter: company + FY year parsed from the question (not gold `doc_name`)
- [x] Page-level chunks for eval 10-Ks (`strategy=pages`); strip PDF `NUL` bytes before store
- [x] Hybrid retrieve: pgvector + `content_tsv` GIN, RRF merge, rerank `top_n=8`
- [x] Deterministic scores: evidence overlap + gold numbers; grounded numeric answer prompt
- [x] Local chat model: Ollama `qwen2.5:7b` (replaced `llama3.2`; same `generate.chat()` interface)

## Next

- [ ] Improve **answer_hit** by changing the LLM. Retrieval is already ahead (`evidence_hit` ~97% vs `answer_hit` ~25% on FinanceBench 150). Try a stronger Ollama tag in `OLLAMA_MODEL` (then `docker compose run --rm model-sync`) or swap `generate.py` to Cohere Command. Re-score with `uv run python -m app.eval.runner --dataset financebench --limit 5` before a full 150.
- [ ] Optional: add `HF_TOKEN` to `.env` + docker-compose for Hugging Face Hub rate limits
- [ ] Optional: swap Ollama client for Cohere Chat behind the same `generate` interface
- [ ] Nice-to-have: CI (run pytest on PR)
- [ ] Later: PDF upload endpoint (Markdown later); telemetry / quality UI (explicitly deferred)

## Notes

- Prefer free/local: `sentence-transformers` + pgvector + Ollama (as the `llm` service).
- Keep the same pipeline shape as Cohere (Embed → retrieve → Rerank → Chat) so you can swap providers later.
- App code lives under `app/`; Docker files under `docker/api` and `docker/postgres`.
- API reaches the LLM at `http://llm:11434` inside Compose (service DNS). Engine image ≠ model weights (`model_weights` volume).
- Online flow mirrors offline: `app/ingestion/pipeline.py` ↔ `app/retrieval/pipeline.py`.
- Run eval from the project root with `uv run`, not system `python` (macOS CA bundle + project deps).
- Chat weights: `OLLAMA_MODEL` (default `qwen2.5:7b`). A 7B local model still under-extracts FinanceBench answers; bump the tag or use Cohere to move `answer_hit`.
- Existing volumes: `migrate_eval.sql` then `migrate_fts.sql`. Software-doc ingest stays `markdown_sections`; FinanceBench uses `pages`.
