import re
from pathlib import Path

import pytest

from app.ingestion.utils.parser import parse_file

PDF_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "postgresql-19-A4.pdf"


@pytest.mark.skipif(not PDF_PATH.is_file(), reason="sample PDF not present under data/raw")
def test_parse_pdf_smoke():
    text = parse_file(PDF_PATH, max_pages=1)
    assert isinstance(text, str)
    assert len(text) > 0


@pytest.mark.skipif(not PDF_PATH.is_file(), reason="sample PDF not present under data/raw")
def test_parse_pdf_returns_markdown_headings():
    """pymupdf4llm should emit # / ## headings we can chunk on."""
    text = parse_file(PDF_PATH, max_pages=20)
    assert re.search(r"^#{1,3}\s+\S", text, re.MULTILINE)


def test_parse_missing_file():
    with pytest.raises(FileNotFoundError):
        parse_file("data/raw/does-not-exist.pdf")
