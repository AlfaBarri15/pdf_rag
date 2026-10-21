"""Chunk cleaned document text for embedding.

Prefers Markdown heading sections (from PDF→Markdown parse). Falls back to
paragraph packing when few/no headings are present.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from typing import Any

from app.ingestion.utils.clean import clean_document_text, clean_page_text
from app.ingestion.utils.embeddings import embed_texts

DEFAULT_MAX_CHARS = 1000
DEFAULT_OVERLAP = 100
PAGE_MAX_CHARS = 12000
PAGE_OVERLAP = 200
STRATEGY_PAGES = "pages"
STRATEGY_MARKDOWN = "markdown_sections"
STRATEGY_PARAGRAPH = "paragraph"

# H1–H3 start a new section; deeper headings stay inside the body.
_SECTION_HEADING = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)

# Callout / box titles that should not become retrieval sections.
_CALLOUT_TITLE = re.compile(
    r"^(note|tip|caution|warning|important|see also)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Chunk:
    """One text piece ready for storage / embedding (matches chunks table shape)."""

    chunk_index: int
    content: str
    metadata: dict[str, Any]
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON API responses."""
        return asdict(self)


def chunk_text(
    text: str,
    *,
    source_path: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
    skip_toc: bool = True,
    embed: bool = False,
) -> list[Chunk]:
    """Clean then chunk by Markdown sections when possible."""
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    cleaned = clean_document_text(text, skip_toc=skip_toc)
    if not cleaned:
        return []

    sections = split_markdown_sections(cleaned)
    # Need real titled sections — otherwise paragraph strategy is safer.
    titled = [s for s in sections if s["title"]]
    if len(titled) >= 2:
        chunks = _chunks_from_sections(
            sections,
            source_path=source_path,
            max_chars=max_chars,
            overlap=overlap,
            skip_toc=skip_toc,
        )
    else:
        chunks = _chunks_from_paragraphs(
            cleaned,
            source_path=source_path,
            max_chars=max_chars,
            overlap=overlap,
            skip_toc=skip_toc,
        )

    if embed and chunks:
        vectors = embed_texts([chunk.content for chunk in chunks])
        chunks = [
            replace(chunk, embedding=vector)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

    return chunks


def chunk_pages(
    pages: list[dict[str, Any]],
    *,
    source_path: str,
    max_chars: int = PAGE_MAX_CHARS,
    overlap: int = PAGE_OVERLAP,
    embed: bool = False,
) -> list[Chunk]:
    """One chunk per PDF page. Split only when a page exceeds max_chars."""
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    chunks: list[Chunk] = []
    for page in pages:
        page_index = int(page.get("page_index", 0))
        cleaned = clean_page_text(str(page.get("text") or ""))
        if not cleaned:
            continue
        parts = _split_long(cleaned, max_chars=max_chars, overlap=overlap)
        for part_index, part in enumerate(parts):
            chunks.append(
                Chunk(
                    chunk_index=len(chunks),
                    content=part,
                    embedding=None,
                    metadata={
                        "source_path": source_path,
                        "strategy": STRATEGY_PAGES,
                        "page_index": page_index,
                        "page_part": part_index,
                        "max_chars": max_chars,
                        "overlap": overlap,
                        "char_count": len(part),
                        "skip_toc": False,
                        "cleaned": True,
                    },
                )
            )

    if embed and chunks:
        vectors = embed_texts([chunk.content for chunk in chunks])
        chunks = [
            replace(chunk, embedding=vector)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
    return chunks


def split_markdown_sections(text: str) -> list[dict[str, Any]]:
    """Split Markdown into sections on H1–H3 headings.

    Skips callout titles (Note/Tip/…) and repeated running headers so PDF
    page chrome does not create fake sections.
    """
    matches = list(_SECTION_HEADING.finditer(text))
    if not matches:
        return [{"level": 0, "title": None, "body": text.strip()}]

    title_counts: dict[str, int] = {}
    for match in matches:
        title = _normalize_heading_title(match.group(2))
        title_counts[title] = title_counts.get(title, 0) + 1

    # Keep only useful boundaries; fold the rest into the previous body.
    boundaries: list[re.Match[str]] = []
    for match in matches:
        title = _normalize_heading_title(match.group(2))
        if _CALLOUT_TITLE.match(title):
            continue
        # Running headers often repeat on every page (e.g. doc title).
        if title_counts.get(title, 0) >= 3:
            continue
        boundaries.append(match)

    if len(boundaries) < 2:
        return [{"level": 0, "title": None, "body": text.strip()}]

    sections: list[dict[str, Any]] = []
    preamble = text[: boundaries[0].start()].strip()
    if preamble:
        sections.append({"level": 0, "title": None, "body": preamble})

    for index, match in enumerate(boundaries):
        level = len(match.group(1))
        title = _normalize_heading_title(match.group(2))
        start = match.end()
        end = (
            boundaries[index + 1].start()
            if index + 1 < len(boundaries)
            else len(text)
        )
        body = text[start:end].strip()
        sections.append({"level": level, "title": title, "body": body})

    return sections


def _normalize_heading_title(raw: str) -> str:
    """Strip bold / nbsp noise from a Markdown heading title."""
    return raw.replace("**", "").replace("\xa0", " ").strip()


def _chunks_from_sections(
    sections: list[dict[str, Any]],
    *,
    source_path: str,
    max_chars: int,
    overlap: int,
    skip_toc: bool,
) -> list[Chunk]:
    """Turn each section into one or more sized chunks; keep section title in text."""
    chunks: list[Chunk] = []
    for section in sections:
        title = section["title"]
        body = section["body"]
        if not body and not title:
            continue

        pieces = _pieces_for_body(body, max_chars=max_chars, overlap=overlap)
        if not pieces and title:
            pieces = [title]
        elif not pieces:
            continue

        for piece in pieces:
            if title and not piece.startswith(title):
                content = f"{title}\n\n{piece}"
            else:
                content = piece
            # Re-split if title prefix pushed past max_chars.
            for part in _split_long(content, max_chars=max_chars, overlap=overlap):
                chunks.append(
                    Chunk(
                        chunk_index=len(chunks),
                        content=part,
                        embedding=None,
                        metadata={
                            "source_path": source_path,
                            "strategy": "markdown_sections",
                            "section_title": title,
                            "heading_level": section["level"],
                            "max_chars": max_chars,
                            "overlap": overlap,
                            "char_count": len(part),
                            "skip_toc": skip_toc,
                            "cleaned": True,
                        },
                    )
                )
    return chunks


def _chunks_from_paragraphs(
    cleaned: str,
    *,
    source_path: str,
    max_chars: int,
    overlap: int,
    skip_toc: bool,
) -> list[Chunk]:
    """Legacy paragraph-pack strategy when Markdown structure is weak."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    raw_pieces: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            raw_pieces.append(paragraph)
        else:
            raw_pieces.extend(
                _split_long(paragraph, max_chars=max_chars, overlap=overlap)
            )

    packed = _pack_pieces(raw_pieces, max_chars=max_chars)
    return [
        Chunk(
            chunk_index=index,
            content=content,
            embedding=None,
            metadata={
                "source_path": source_path,
                "strategy": "paragraph",
                "max_chars": max_chars,
                "overlap": overlap,
                "char_count": len(content),
                "skip_toc": skip_toc,
                "cleaned": True,
            },
        )
        for index, content in enumerate(packed)
    ]


def _pieces_for_body(body: str, *, max_chars: int, overlap: int) -> list[str]:
    """Pack section body paragraphs, then split any leftover long windows."""
    if not body.strip():
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    raw: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            raw.append(paragraph)
        else:
            raw.extend(_split_long(paragraph, max_chars=max_chars, overlap=overlap))
    return _pack_pieces(raw, max_chars=max_chars)


def _pack_pieces(pieces: list[str], *, max_chars: int) -> list[str]:
    """Merge short paragraphs until adding another would exceed max_chars."""
    packed: list[str] = []
    current = ""

    for piece in pieces:
        if not current:
            current = piece
            continue

        candidate = f"{current}\n\n{piece}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            packed.append(current)
            current = piece

    if current:
        packed.append(current)
    return packed


def _split_long(text: str, *, max_chars: int, overlap: int) -> list[str]:
    """Split an oversized string into overlapping windows of max_chars."""
    parts: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + max_chars, length)
        parts.append(text[start:end].strip())
        if end >= length:
            break
        start = max(0, end - overlap)

    return [p for p in parts if p]
