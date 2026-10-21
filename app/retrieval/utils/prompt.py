"""Build the grounded RAG prompt and citation list from retrieved chunks."""

from __future__ import annotations

from typing import Any


def build_citations(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Number chunks for the prompt and for the API response (stable refs)."""
    citations: list[dict[str, Any]] = []
    for index, result in enumerate(results, start=1):
        citations.append(
            {
                "ref": index,
                "chunk_id": result["chunk_id"],
                "chunk_index": result["chunk_index"],
                "title": result.get("title"),
                "source_path": result.get("source_path"),
                "content": result["content"],
                "distance": result.get("distance"),
                "score": result.get("score"),
                "rerank_score": result.get("rerank_score"),
            }
        )
    return citations


def build_answer_prompt(*, query: str, citations: list[dict[str, Any]]) -> str:
    """Assemble the user message: numbered sources + question."""
    if not citations:
        return (
            "Sources:\n(none)\n\n"
            f"Question: {query}\n\n"
            "No sources were retrieved. Say you do not know."
        )

    blocks: list[str] = ["Sources:"]
    for citation in citations:
        header = (
            f"[{citation['ref']}] "
            f"(title={citation.get('title')}, "
            f"source={citation.get('source_path')}, "
            f"chunk_index={citation.get('chunk_index')})"
        )
        blocks.append(header)
        blocks.append(citation["content"])
        blocks.append("")

    blocks.append(f"Question: {query}")
    blocks.append("")
    blocks.append(
        "Reply with the answer first (the number or short fact in the units asked). "
        "Copy the line item from the sources. Do not compute a new total from other rows. "
        "Cite like [1]. No essay."
    )
    return "\n".join(blocks)
