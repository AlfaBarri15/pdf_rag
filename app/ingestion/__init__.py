"""Ingestion package: pipeline (public) + utils (parse/chunk/store/embed)."""

from app.ingestion.pipeline import build_chunks, run_ingest
from app.ingestion.utils.chunker import (
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP,
    Chunk,
    chunk_text,
    split_markdown_sections,
)
from app.ingestion.utils.embeddings import EMBEDDING_DIMENSIONS, embed_text, embed_texts
from app.ingestion.utils.parser import parse_file, parse_pdf_pages
from app.ingestion.utils.store import (
    embed_document_chunks,
    fetch_document_summary,
    store_document_chunks,
)

__all__ = [
    "DEFAULT_MAX_CHARS",
    "DEFAULT_OVERLAP",
    "Chunk",
    "EMBEDDING_DIMENSIONS",
    "build_chunks",
    "chunk_text",
    "split_markdown_sections",
    "embed_document_chunks",
    "embed_text",
    "embed_texts",
    "fetch_document_summary",
    "parse_file",
    "parse_pdf_pages",
    "run_ingest",
    "store_document_chunks",
]
