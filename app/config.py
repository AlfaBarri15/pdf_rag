"""App settings loaded from the environment (Compose overrides URLs in Docker)."""

import os

# Host default for local uv runs; api service uses host `postgres` instead.
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://rag:rag@localhost:5432/rag",
)

# Host default for local uv runs; api service uses host `llm` instead.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
