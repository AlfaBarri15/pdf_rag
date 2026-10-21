from app.ingestion.utils.chunker import (
    STRATEGY_PAGES,
    chunk_pages,
    chunk_text,
    split_markdown_sections,
)
from app.ingestion.utils.clean import (
    clean_document_text,
    strip_boilerplate,
    strip_end_matter,
    strip_table_of_contents,
)
from app.ingestion.utils.embeddings import EMBEDDING_DIMENSIONS


def test_chunk_empty():
    assert chunk_text("", source_path="x.pdf") == []


def test_chunk_pages_keeps_one_page_together():
    pages = [
        {
            "page_index": 59,
            "text": (
                "Consolidated Statement of Cash Flows\n"
                "Purchases of property, plant and equipment (PP&E) (1,577) (1,373) (1,420)\n"
            ),
        },
        {"page_index": 60, "text": "Notes to consolidated financial statements continue here."},
    ]
    chunks = chunk_pages(pages, source_path="eval/3M_2018_10K.pdf")
    assert len(chunks) == 2
    assert chunks[0].metadata["strategy"] == STRATEGY_PAGES
    assert chunks[0].metadata["page_index"] == 59
    assert "1,577" in chunks[0].content
    assert chunks[1].metadata["page_index"] == 60


def test_chunk_pages_strips_nul_bytes():
    chunks = chunk_pages(
        [{"page_index": 0, "text": "Net sales were $34,229\x00 million."}],
        source_path="eval/3M_2022_10K.pdf",
    )
    assert len(chunks) == 1
    assert "\x00" not in chunks[0].content
    assert "34229" in chunks[0].content.replace(",", "")


def test_chunk_packs_paragraphs_without_embedding_by_default():
    text = (
        "First paragraph with enough letters to keep.\n\n"
        "Second paragraph with enough letters to keep.\n\n"
        "Third paragraph with enough letters to keep."
    )
    chunks = chunk_text(text, source_path="docs/x.pdf", max_chars=120, overlap=10)

    assert len(chunks) >= 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].embedding is None
    assert chunks[0].metadata["strategy"] == "paragraph"
    assert chunks[0].metadata["source_path"] == "docs/x.pdf"
    assert chunks[0].metadata["skip_toc"] is True
    assert "First paragraph" in chunks[0].content


def test_chunk_embed_true_attaches_vectors():
    text = "Useful paragraph about PostgreSQL indexing and vacuum."
    chunks = chunk_text(
        text,
        source_path="docs/x.pdf",
        max_chars=200,
        overlap=10,
        embed=True,
    )
    assert len(chunks) == 1
    assert chunks[0].embedding is not None
    assert len(chunks[0].embedding) == EMBEDDING_DIMENSIONS


def test_chunk_splits_long_paragraph():
    text = "word " * 600
    chunks = chunk_text(text, source_path="long.pdf", max_chars=1000, overlap=100)

    assert len(chunks) >= 2
    assert all(len(c.content) <= 1000 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_strip_table_of_contents():
    text = """Title Page with enough letters here.

Table of Contents
1. Intro ..................................... 1
2. Setup ..................................... 5

List of Figures
1.1 Architecture .............................. 9

Real chapter prose starts here with useful content about databases.
"""
    cleaned = strip_table_of_contents(text)
    assert "Table of Contents" not in cleaned
    assert "List of Figures" not in cleaned
    assert "1. Intro" not in cleaned
    assert "Real chapter prose" in cleaned
    assert "Title Page" in cleaned


def test_strip_markdown_toc_headers():
    text = """## **Table of Contents**

1. Intro ..................................... 1

## Real Chapter

Real chapter prose starts here with useful content about databases.
"""
    cleaned = strip_table_of_contents(text)
    assert "Table of Contents" not in cleaned
    assert "Real Chapter" in cleaned
    assert "Real chapter prose" in cleaned


def test_strip_markdown_link_toc():
    text = """# Guide

## Contents

- [Install](#install)
- [Usage](#usage)

## Install

Install steps with enough letters to keep for retrieval.
"""
    cleaned = strip_table_of_contents(text)
    assert "Contents" not in cleaned
    assert "[Install](#install)" not in cleaned
    assert "Install steps" in cleaned


def test_mid_document_references_chapter_is_kept():
    """A mid-manual References section must not be treated as back matter."""
    body = "Useful API notes about widgets and connectors with enough letters.\n\n"
    text = (
        "# Manual\n\n"
        + body * 8
        + "## References\n\n"
        + "See the widget API overview with enough letters here.\n\n"
        + body * 8
        + "## Index\n\n"
        + "widget ................................ 12\n"
    )
    cleaned = clean_document_text(text)
    assert "widget API overview" in cleaned
    assert "Index" not in cleaned
    assert "widget ................................" not in cleaned


def test_preserves_markdown_code_indentation():
    text = """## Example

Run this:

```python
def hello():
    return "world"
```

Prose after the example with enough letters to keep.
"""
    cleaned = clean_document_text(text)
    assert "def hello():" in cleaned
    assert '    return "world"' in cleaned


def test_strip_boilerplate_license_block():
    text = """Copyright (c) 2024 Example Authors. All rights reserved.
Permission to use, copy, modify and distribute this software is hereby granted
under the following conditions that appear in all copies of the software.

Actual chapter content about WAL and checkpoints starts here for readers.
"""
    cleaned = strip_boilerplate(text)
    assert "Permission to use" not in cleaned
    assert "Actual chapter content" in cleaned


def test_clean_document_text_fixes_hyphenation_and_toc():
    text = """Table of Contents
1. Intro ..................................... 1

This is a data-
base system used widely in production environments today.
"""
    cleaned = clean_document_text(text)
    assert "Table of Contents" not in cleaned
    assert "database system" in cleaned


def test_strip_end_matter_drops_index_keeps_glossary():
    text = """Chapter about VACUUM with enough letters for keepers.

Glossary
VACUUM
A maintenance command that reclaims storage from dead tuples in tables.

Index
absolute path ................................ 12
autovacuum .................................. 88

Bibliography
Some Paper. Journal of Databases, 2020.
"""
    cleaned = clean_document_text(text)
    assert "Chapter about VACUUM" in cleaned
    assert "Glossary" in cleaned
    assert "dead tuples" in cleaned
    assert "Index" not in cleaned
    assert "absolute path" not in cleaned
    assert "Bibliography" not in cleaned
    assert "Some Paper" not in cleaned


def test_strip_markdown_end_matter():
    text = """## Useful Chapter

Chapter about VACUUM with enough letters for keepers.

## Index

absolute path ................................ 12
"""
    cleaned = strip_end_matter(text)
    assert "Useful Chapter" in cleaned
    assert "VACUUM" in cleaned
    assert "Index" not in cleaned


def test_chunk_skips_toc_by_default():
    text = """Intro paragraph before TOC with enough letters.

Table of Contents
Preface ...................................... iii
1. Overview .................................. 1

Useful body paragraph about indexes and query planners in PostgreSQL.
"""
    chunks = chunk_text(text, source_path="doc.pdf", max_chars=500, overlap=20)
    joined = "\n".join(c.content for c in chunks)
    assert "Table of Contents" not in joined
    assert "Preface ...." not in joined
    assert "Useful body paragraph" in joined


def test_markdown_section_chunking_keeps_titles():
    text = """# Intro

Intro paragraph with enough letters to keep in the section body.

## Setup Guide

Setup paragraph explaining installation steps with enough letters.

## Usage

Usage paragraph about running queries with enough letters here.
"""
    chunks = chunk_text(text, source_path="docs/x.md", max_chars=500, overlap=20)
    assert len(chunks) >= 2
    assert all(c.metadata["strategy"] == "markdown_sections" for c in chunks)
    titles = {c.metadata.get("section_title") for c in chunks}
    assert "Setup Guide" in titles
    assert "Usage" in titles
    joined = "\n".join(c.content for c in chunks)
    assert "installation steps" in joined
    assert "running queries" in joined
    assert any(c.content.startswith("Setup Guide") for c in chunks)


def test_split_markdown_sections_finds_headings():
    sections = split_markdown_sections("# A\n\nbody a\n\n## B\n\nbody b\n")
    titled = [s for s in sections if s["title"]]
    assert [s["title"] for s in titled] == ["A", "B"]
    assert "body a" in titled[0]["body"]


def test_split_skips_callouts_and_repeated_headers():
    text = "\n".join(
        [
            "# Doc Title",
            "",
            "intro",
            "",
            "# Doc Title",
            "",
            "more",
            "",
            "# Doc Title",
            "",
            "still more",
            "",
            "## Real Chapter",
            "",
            "chapter body with enough letters here",
            "",
            "### Note",
            "",
            "callout body with enough letters here",
            "",
            "## Next Chapter",
            "",
            "next body with enough letters here",
        ]
    )
    sections = split_markdown_sections(text)
    titles = [s["title"] for s in sections if s["title"]]
    assert "Doc Title" not in titles
    assert "Note" not in titles
    assert titles == ["Real Chapter", "Next Chapter"]
