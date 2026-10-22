"""Build the grounded RAG prompt and citation list from retrieved chunks."""

from __future__ import annotations

import re
from typing import Any

_QUERY_WORD = re.compile(r"[a-z0-9]{3,}")
_STOP = {
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "from",
    "what",
    "which",
    "when",
    "give",
    "response",
    "question",
    "answer",
    "using",
    "shown",
    "based",
    "assume",
    "primarily",
    "following",
    "please",
    "state",
    "explain",
    "exclude",
    "impact",
    "overall",
    "public",
    "equities",
    "analyst",
}
_MAX_EXCERPT_CHARS = 900
_BOOST = (
    "purchases of property",
    "property, plant and equipment net",
    "pp&e",
    "organic",
    "consumer",
    "cash flows from investing",
)
_PENALIZE = (
    "outlook",
    "expect",
    "expected",
    "guidance",
    "billion to",
    "million to",
)


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
        blocks.append(excerpt_for_query(citation["content"], query))
        blocks.append("")

    blocks.append(f"Question: {query}")
    blocks.append("")
    blocks.append(
        "Reply with the answer first (the number or short fact in the units asked). "
        "Copy the statement line item from the sources, not an outlook or range. "
        "If the question asks which segment dragged growth, pick the most negative organic change. "
        "Do not compute a new total from other rows. Cite like [1]. No essay."
    )
    return "\n".join(blocks)


def excerpt_for_query(content: str, query: str, *, max_chars: int = _MAX_EXCERPT_CHARS) -> str:
    """Keep lines that share terms with the question so a small LLM sees the row, not the whole page."""
    if not content.strip():
        return content
    terms = _query_terms(query)
    lines = content.splitlines()
    scored: list[tuple[int, int]] = []
    for index, line in enumerate(lines):
        low = line.lower()
        hits = sum(1 for term in terms if term in low)
        if not hits:
            continue
        boosts = sum(1 for phrase in _BOOST if phrase in low)
        penalties = sum(1 for phrase in _PENALIZE if phrase in low)
        score = hits + 4 * boosts - 5 * penalties
        if score > 0:
            scored.append((score, index))
    if not scored:
        return content[:max_chars].rstrip()

    keep: set[int] = set()
    for score, index in sorted(scored, key=lambda item: (-item[0], item[1]))[:10]:
        keep.add(index)
        low = lines[index].lower()
        if any(phrase in low for phrase in _BOOST) and not re.search(r"\d{2,}", lines[index]):
            keep.update(_number_tail(lines, index))
    excerpt = "\n".join(lines[i] for i in sorted(keep)).strip()
    if len(excerpt) > max_chars:
        return excerpt[:max_chars].rstrip()
    return excerpt


def _query_terms(query: str) -> set[str]:
    words = set(_QUERY_WORD.findall(query.lower()))
    extra: set[str] = set()
    blob = query.lower()
    if words & {"capex", "capital", "expenditure"} or "capital expenditure" in blob:
        extra.update({"purchases", "property", "plant", "equipment", "ppe", "capex", "pp&e"})
    if words & {"ppne", "ppe"} or "pp&e" in blob or "ppne" in blob:
        extra.update({"property", "plant", "equipment", "ppe", "ppne", "pp&e", "net"})
    if "margin" in words:
        extra.update({"operating", "margin", "gross", "sg&a"})
    if words & {"segment", "consumer", "organic"}:
        extra.update({"consumer", "organic", "segment", "sales"})
    if words & {"cash", "flow"}:
        extra.update({"cash", "investing", "purchases", "flows"})
    extra.update(re.findall(r"20\d{2}", query))
    return (words | extra) - _STOP


def _number_tail(lines: list[str], start: int) -> set[int]:
    """Include following lines until the first amount (PDF tables split label and value)."""
    extra: set[int] = set()
    for index in range(start + 1, min(len(lines), start + 8)):
        extra.add(index)
        if re.search(r"\d{2,}", lines[index]):
            break
    return extra
