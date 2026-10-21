# `app/api`

HTTP surface for the RAG app. Routers are wired from `app/api/__init__.py` into `app/main.py`.

This package stays **thin**: validate params, call domain helpers, return JSON.

## Public layout

| Path               | Role                                                        |
|--------------------|-------------------------------------------------------------|
| `__init__.py`      | Regroups mounted routers (`health`, `ingest`, `answer`, `openai_chat`) |
| `utils/health.py`  | `GET /health`                                               |
| `utils/ingest.py`  | `POST /ingest/` (202), `GET /ingest/jobs/{id}`              |
| `utils/answer.py`  | `POST /answer/` (JSON `{"query": "..."}` → LLM + citations) |
| `utils/openai_chat.py` | `GET /v1/models`, `POST /v1/chat/completions` (Open WebUI → RAG) |

`utils/search.py` and `utils/chat.py` remain for optional remount/debug; they are **not** exposed.

## Docs

Interactive OpenAPI UI: http://localhost:8000/docs
