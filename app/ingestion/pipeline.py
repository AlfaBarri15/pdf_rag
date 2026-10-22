"""Offline ingest pipeline: parse → clean → chunk → embed → store."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from app.ingestion.utils.chunker import (
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP,
    PAGE_MAX_CHARS,
    PAGE_OVERLAP,
    STRATEGY_MARKDOWN,
    STRATEGY_PAGES,
    Chunk,
    chunk_pages,
    chunk_text,
)
from app.ingestion.utils.embeddings import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL_NAME,
    embed_texts,
)
from app.ingestion.utils.parser import parse_file, parse_pdf_pages, pdf_page_stats
from app.ingestion.utils.store import DEFAULT_DATASET, store_document_chunks


def build_chunks(
    path: Path,
    *,
    source_path: str,
    max_pages: int | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
    embed: bool = False,
    strategy: str = STRATEGY_MARKDOWN,
) -> list[Chunk]:
    """Parse a file, clean, and return chunks (preview or before store)."""
    if strategy == STRATEGY_PAGES:
        page_max = PAGE_MAX_CHARS if max_chars == DEFAULT_MAX_CHARS else max_chars
        page_overlap = PAGE_OVERLAP if overlap == DEFAULT_OVERLAP else overlap
        pages = parse_pdf_pages(path, max_pages=max_pages)
        return chunk_pages(
            pages,
            source_path=source_path,
            max_chars=page_max,
            overlap=page_overlap,
            embed=embed,
        )
    text = parse_file(path, max_pages=max_pages)
    return chunk_text(
        text,
        source_path=source_path,
        max_chars=max_chars,
        overlap=overlap,
        embed=embed,
    )


def attach_embeddings(chunks: list[Chunk]) -> list[Chunk]:
    """Batch-encode all chunk texts once (faster than embedding during chunking)."""
    if not chunks:
        return []
    vectors = embed_texts([chunk.content for chunk in chunks])
    return [
        replace(chunk, embedding=vector)
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]


def run_ingest(
    *,
    path: Path,
    source_path: str,
    title: str,
    max_pages: int | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
    embed: bool = True,
    dataset: str = DEFAULT_DATASET,
    strategy: str = STRATEGY_MARKDOWN,
    on_progress: Any | None = None,
) -> dict[str, Any]:
    """Full offline ingest for one document (omit max_pages for the whole PDF)."""

    def _progress(stage: str) -> None:
        """Forward the current ingest stage to the caller when a callback was passed."""
        if on_progress is not None:
            on_progress(stage)

    _progress("parse")
    page_stats = pdf_page_stats(path, max_pages=max_pages)
    _progress("clean_chunk")
    chunks = build_chunks(
        path,
        source_path=source_path,
        max_pages=max_pages,
        max_chars=max_chars,
        overlap=overlap,
        embed=False,
        strategy=strategy,
    )

    embedded_count = 0
    if embed and chunks:
        _progress("embed")
        chunks = attach_embeddings(chunks)
        embedded_count = len(chunks)

    chunks = [
        replace(chunk, metadata={**chunk.metadata, "dataset": dataset})
        for chunk in chunks
    ]

    _progress("store")
    stored = store_document_chunks(
        source_path=source_path,
        title=title,
        dataset=dataset,
        chunks=chunks,
    )

    return {
        "source_path": source_path,
        "title": title,
        "document_id": stored["document_id"],
        "chunk_count": stored["chunk_count"],
        "page_total": page_stats["page_total"],
        "pages_ingested": page_stats["pages_ingested"],
        "max_pages": max_pages,
        "max_chars": max_chars,
        "overlap": overlap,
        "embed": embed,
        "embedded_count": embedded_count,
        "embedding_model": EMBEDDING_MODEL_NAME if embed else None,
        "embedding_dimensions": EMBEDDING_DIMENSIONS if embed else None,
        "dataset": dataset,
        "strategy": strategy,
    }
