"""Local cross-encoder rerank (Cohere Rerank stand-in)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from sentence_transformers import CrossEncoder

RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Load the cross-encoder once per process (downloads weights on first call)."""
    return CrossEncoder(RERANK_MODEL_NAME)


def rerank_chunks(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    top_n: int = 3,
) -> list[dict[str, Any]]:
    """Re-score retrieved chunks with a cross-encoder and return the top_n."""
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    if not candidates:
        return []

    model = get_reranker()
    pairs = [(query, candidate["content"]) for candidate in candidates]
    scores = model.predict(pairs)

    ranked = sorted(
        zip(candidates, scores, strict=True),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    results: list[dict[str, Any]] = []
    for candidate, score in ranked[:top_n]:
        item = dict(candidate)
        item["rerank_score"] = float(score)
        results.append(item)
    return results
