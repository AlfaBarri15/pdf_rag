"""Local text → vector embeddings via sentence-transformers (MiniLM, 384 dims)."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384  # must match VECTOR(384) in Postgres


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the embedding model once per process (downloads weights on first call)."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: list[str], *, batch_size: int = 64) -> list[list[float]]:
    """Embed many strings in batches; returns one normalized float vector per input."""
    if not texts:
        return []
    model = get_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    return [vector.tolist() for vector in vectors]


def embed_text(text: str) -> list[float]:
    """Embed a single string into a normalized vector."""
    return embed_texts([text])[0]
