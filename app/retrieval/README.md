# `app/retrieval`

Semantic search + optional cross-encoder rerank + grounded LLM answers
(same shape as Cohere Embed → retrieve → Rerank → Chat).

## Layout

| Path                   | Role                                                            |
|------------------------|-----------------------------------------------------------------|
| `pipeline.py`          | Public orchestration — `run_search`, `run_answer`               |
| `search.py`            | Re-exports pipeline helpers for a stable import path            |
| `utils/search.py`      | Embed query + cosine top-k SQL + optional rerank                |
| `utils/rerank.py`      | `CrossEncoder` (`ms-marco-MiniLM-L-6-v2`)                       |
| `utils/prompt.py`      | Numbered sources + citation list                                |
| `utils/generate.py`    | Ollama `/api/chat` client (`llm` Compose service)               |

## Pipeline

| Step       | What happens                                                      |
|------------|-------------------------------------------------------------------|
| Retrieve   | MiniLM embed + pgvector HNSW (`top_k`)                            |
| Rerank     | Optional cross-encoder; keep `top_n`                              |
| Generate   | Ollama answers from numbered sources; API returns `citations`     |

```http
POST /answer/
Content-Type: application/json

{"query": "..."}
```

`/answer/` keeps retrieval defaults internal (`top_k=20`, `rerank=true`, `top_n=8`). Vector search is fused with Postgres full-text (`content_tsv`) before rerank.
Domain helpers `run_search` / `run_answer` remain available in Python for tests.
