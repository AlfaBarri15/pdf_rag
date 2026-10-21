"""HTTP routers for the RAG API (imported by app.main).

Public surface: health, async ingest, grounded answer, OpenAI-compatible chat.
Search/chat/chunk helpers remain as modules for tests/debug but are not mounted.
"""

from app.api.utils import answer, health, ingest, openai_chat

__all__ = ["answer", "health", "ingest", "openai_chat"]
