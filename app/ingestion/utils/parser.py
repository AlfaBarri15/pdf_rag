"""Parse documents into text suitable for cleaning and chunking.

PDFs are converted to Markdown (pymupdf4llm) so we can chunk by headings.
Falls back to plain pypdf text if Markdown conversion fails.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".pdf"}


def parse_pdf_pages(
    path: str | Path, *, max_pages: int | None = None
) -> list[dict[str, object]]:
    """Extract one text block per PDF page (plain text, Markdown fallback)."""
    file_path = Path(path).resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type {file_path.suffix.lower()!r}. "
            f"Supported: {sorted(SUPPORTED_SUFFIXES)}"
        )

    reader = PdfReader(str(file_path))
    page_list = reader.pages if max_pages is None else reader.pages[:max_pages]
    pages: list[dict[str, object]] = []
    for index, page in enumerate(page_list):
        text = (page.extract_text() or "").replace("\x00", "").strip()
        if not text:
            text = _parse_pdf_markdown_page(file_path, index)
        if text:
            pages.append({"page_index": index, "text": text})
    return pages


def parse_file(path: str | Path, *, max_pages: int | None = None) -> str:
    """Extract document text. PDFs → Markdown when possible."""
    file_path = Path(path).resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type {suffix!r}. Supported: {sorted(SUPPORTED_SUFFIXES)}"
        )

    if suffix == ".pdf":
        return _parse_pdf_markdown(file_path, max_pages=max_pages)

    raise ValueError(f"No parser registered for {suffix!r}")


def pdf_page_stats(path: str | Path, *, max_pages: int | None = None) -> dict[str, int]:
    """Return total PDF pages and how many will be ingested."""
    file_path = Path(path).resolve()
    total = len(PdfReader(str(file_path)).pages)
    ingested = total if max_pages is None else min(max_pages, total)
    return {"page_total": total, "pages_ingested": ingested}


def _parse_pdf_markdown(path: Path, *, max_pages: int | None = None) -> str:
    """Convert PDF pages to Markdown via pymupdf4llm (structure-aware)."""
    try:
        import pymupdf4llm
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pymupdf4llm is required for PDF→Markdown parsing"
        ) from exc

    pages: list[int] | None = None
    if max_pages is not None:
        total = len(PdfReader(str(path)).pages)
        pages = list(range(min(max_pages, total)))

    markdown = pymupdf4llm.to_markdown(str(path), pages=pages)
    text = (markdown or "").strip()
    if text:
        return text

    # Rare empty conversion — fall back to plain extraction.
    return _parse_pdf_plain(path, max_pages=max_pages)


def _parse_pdf_markdown_page(path: Path, page_index: int) -> str:
    """Markdown for one 0-based page when pypdf extracted nothing."""
    try:
        import pymupdf4llm
    except ImportError:
        return ""
    markdown = pymupdf4llm.to_markdown(str(path), pages=[page_index])
    return (markdown or "").strip()


def _parse_pdf_plain(path: Path, *, max_pages: int | None = None) -> str:
    """Fallback: pull text from each PDF page and join with blank lines."""
    reader = PdfReader(str(path))
    page_list = reader.pages if max_pages is None else reader.pages[:max_pages]

    parts: list[str] = []
    for page in page_list:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text)

    return "\n\n".join(parts).strip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse a document into Markdown/text")
    parser.add_argument("path", help="Path to the file under data/")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Limit PDF pages (default: full document)",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=500,
        help="How many characters of text to print",
    )
    args = parser.parse_args()

    text = parse_file(args.path, max_pages=args.max_pages)
    stats = pdf_page_stats(args.path, max_pages=args.max_pages)
    preview = text[: args.preview_chars]
    print(
        f"chars={len(text)} pages_ingested={stats['pages_ingested']} "
        f"page_total={stats['page_total']}"
    )
    print("---")
    print(preview)
    if len(text) > args.preview_chars:
        print("...")
