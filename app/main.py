from fastapi import FastAPI

from app.api import answer, health, ingest, openai_chat

app = FastAPI(title="RAG Search Engine")
app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(answer.router)
app.include_router(openai_chat.router)


def main() -> None:
    """CLI entrypoint: run uvicorn with reload for local development."""
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
