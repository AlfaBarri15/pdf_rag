"""Normalize and drop low-signal document text before chunking.

Works on plain text and Markdown (including PDF→Markdown from the parser).
Heuristics target common software-doc patterns — not a single vendor PDF.
"""

from __future__ import annotations

import re

# --- Front matter / navigation ---------------------------------------------

# Dotted leaders + page number (PDF TOCs and Markdown table rows).
_TOC_LINE = re.compile(
    r".+\.{4,}\s*([ivxlcdm]+|\d+)\s*\|?\s*$",
    re.IGNORECASE,
)
_TOC_HEADER = re.compile(
    r"^("
    r"contents|"
    r"table of contents|"
    r"list of (figures|tables|examples|listings)"
    r")\s*$",
    re.IGNORECASE,
)
# Markdown / HTML-ish TOC entries: "- [Title](#anchor)" or "* [Title](url)"
_MD_TOC_LINK = re.compile(
    r"^[\s>*\-•]*(?:\d+[\.\)]\s*)?\[[^\]]+\]\([^)]+\)\s*$"
)

# --- Back matter (navigation / citations; Glossary stays) ------------------

_END_MATTER_HEADER = re.compile(
    r"^("
    r"(subject |author )?index|"
    r"bibliography|"
    r"references|"
    r"works cited|"
    r"list of symbols|"
    r"colophon"
    r")\s*$",
    re.IGNORECASE,
)
# Only cut at end-matter headers that appear in this trailing fraction of the doc.
# Avoids dropping a mid-manual "References" / API chapter.
_END_MATTER_TAIL_FRACTION = 0.30

_PAGE_MARKER = re.compile(r"^([ivxlcdm]+|\d+)\s*$", re.IGNORECASE)

# Soft hyphenation across line wraps (PDF extract artifact): "data-\nbase" → "database"
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")

# Generic legal / license front matter (no vendor-specific phrases).
_BOILERPLATE_START = re.compile(
    r"("
    r"copyright\s+(\(c\)|©)|"
    r"all rights reserved|"
    r"permission to use,\s*copy,\s*modify|"
    r"redistribution and use in source|"
    r"this software is provided\b|"
    r"licensed under the|"
    r"spdx-license-identifier|"
    r"apache license|"
    r"mit license|"
    r"bsd license|"
    r"gnu (lesser )?general public license|"
    r"creative commons"
    r")",
    re.IGNORECASE,
)

_MIN_PARAGRAPH_CHARS = 40
_FENCE_LINE = re.compile(r"^(`{3,}|~{3,})")


def _strip_nul_bytes(text: str) -> str:
    """Postgres text cannot store NUL; PDF extractors sometimes emit it."""
    return text.replace("\x00", "")


def clean_page_text(text: str) -> str:
    """Light clean for one PDF page: keep tables, running headers, and figures."""
    if not text or not text.strip():
        return ""
    cleaned = _strip_nul_bytes(text)
    cleaned = cleaned.replace("\x0c", "\n\n")
    cleaned = _repair_hyphenation(cleaned)
    cleaned = _normalize_whitespace(cleaned)
    return cleaned.strip()


def clean_document_text(text: str, *, skip_toc: bool = True) -> str:
    """Full clean pipeline for software docs before chunking.

    Input may be plain text or Markdown. Steps are format-agnostic so the same
    path works for PDF→Markdown today and uploaded ``.md`` later.
    """
    if not text or not text.strip():
        return ""

    cleaned = _strip_nul_bytes(text)
    cleaned = cleaned.replace("\x0c", "\n\n")
    cleaned = _repair_hyphenation(cleaned)
    cleaned = _normalize_whitespace(cleaned)

    if skip_toc:
        cleaned = strip_table_of_contents(cleaned)

    cleaned = strip_end_matter(cleaned)
    cleaned = strip_boilerplate(cleaned)
    cleaned = drop_low_signal_paragraphs(cleaned)
    return cleaned.strip()


def strip_table_of_contents(text: str) -> str:
    """Remove TOC / list-of-* blocks (plain, Markdown headings, or link lists)."""
    if not text.strip():
        return ""

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    kept: list[str] = []
    skipping = False

    for paragraph in paragraphs:
        if _is_toc_header(paragraph):
            skipping = True
            continue

        if skipping:
            if _is_toc_paragraph(paragraph):
                continue
            skipping = False

        if _is_toc_paragraph(paragraph):
            continue

        kept.append(paragraph)

    return "\n\n".join(kept).strip()


def strip_end_matter(text: str, *, tail_fraction: float = _END_MATTER_TAIL_FRACTION) -> str:
    """Drop Index / Bibliography / trailing References-style back matter.

    Cuts when an end-matter heading is late in the doc **or** the text after it
    looks like an index/citation list. Mid-manual chapters titled \"References\"
    with real prose after them are kept. Glossary is never end matter.
    """
    if not text.strip():
        return ""

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return ""

    positions: list[int] = []
    cursor = 0
    for paragraph in paragraphs:
        positions.append(cursor)
        cursor += len(paragraph) + 2

    total = max(cursor, 1)
    cutoff = total * (1.0 - tail_fraction)

    kept: list[str] = []
    for index, (paragraph, start) in enumerate(
        zip(paragraphs, positions, strict=True)
    ):
        if _is_end_matter_header(paragraph):
            rest = paragraphs[index + 1 :]
            in_tail = start >= cutoff
            if in_tail or _rest_looks_like_back_matter(rest):
                break
        kept.append(paragraph)

    return "\n\n".join(kept).strip()


def strip_boilerplate(text: str) -> str:
    """Drop copyright / license-style paragraphs common in shipped manuals."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    kept: list[str] = []

    for paragraph in paragraphs:
        if _is_boilerplate_paragraph(paragraph):
            continue
        kept.append(paragraph)

    return "\n\n".join(kept).strip()


def drop_low_signal_paragraphs(
    text: str,
    *,
    min_chars: int = _MIN_PARAGRAPH_CHARS,
) -> str:
    """Drop tiny / page-number / symbol-only scraps that waste embedding slots."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    kept: list[str] = []

    for paragraph in paragraphs:
        if _is_low_signal(paragraph, min_chars=min_chars):
            continue
        kept.append(paragraph)

    return "\n\n".join(kept).strip()


def _repair_hyphenation(text: str) -> str:
    """Join soft hyphen line-breaks outside fenced code blocks."""
    out: list[str] = []
    in_fence = False
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if _FENCE_LINE.match(line.strip()):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue
        if (
            not in_fence
            and i + 1 < len(lines)
            and not _FENCE_LINE.match(lines[i + 1].strip())
        ):
            joined = _HYPHEN_BREAK.sub(r"\1\2", f"{line}\n{lines[i + 1]}")
            if "\n" not in joined:
                out.append(joined)
                i += 2
                continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _normalize_whitespace(text: str) -> str:
    """Normalize newlines; preserve Markdown indent and fenced code."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines: list[str] = []
    in_fence = False
    for line in text.split("\n"):
        if _FENCE_LINE.match(line.strip()):
            in_fence = not in_fence
            lines.append(line.rstrip())
            continue
        if in_fence or line.startswith(("    ", "\t")):
            # Keep code / indented blocks intact.
            lines.append(line.rstrip())
        else:
            # Soft-collapse odd multi-space runs from PDF extractors only.
            lines.append(re.sub(r"[ \t]{2,}", " ", line).rstrip())
    return "\n".join(lines).strip()


def _heading_plain_text(paragraph: str) -> str:
    """First-line heading text without Markdown markers (#, **, underlines)."""
    first_line = paragraph.splitlines()[0].strip()
    first_line = re.sub(r"^#{1,6}\s*", "", first_line)
    first_line = first_line.replace("**", "").replace("__", "").strip()
    first_line = first_line.strip("*_").strip()
    return first_line.replace("\xa0", " ").strip()


def _is_toc_header(paragraph: str) -> bool:
    """True if the paragraph starts a TOC / list-of-* section."""
    return bool(_TOC_HEADER.match(_heading_plain_text(paragraph)))


def _is_end_matter_header(paragraph: str) -> bool:
    """True if the paragraph starts Index / Bibliography-style back matter."""
    return bool(_END_MATTER_HEADER.match(_heading_plain_text(paragraph)))


def _rest_looks_like_back_matter(paragraphs: list[str]) -> bool:
    """True when following paragraphs look like an index or citation list."""
    if not paragraphs:
        return True

    sample = paragraphs[:30]
    hits = 0
    for paragraph in sample:
        if _is_end_matter_header(paragraph) or _is_toc_paragraph(paragraph):
            hits += 1
        elif _looks_like_citation(paragraph):
            hits += 1
    return hits >= max(1, (len(sample) + 1) // 2)


def _looks_like_citation(paragraph: str) -> bool:
    """Loose citation shape: [1] … or Author-ish line with a year."""
    compact = paragraph.strip()
    if re.match(r"^\[\d+\]", compact):
        return True
    if len(compact) <= 400 and re.search(r"\b(19|20)\d{2}\b", compact):
        # Prefer lines that also look bibliographic (period-separated bits).
        if compact.count(".") >= 1 or compact.count(",") >= 1:
            return True
    return False


def _is_toc_paragraph(paragraph: str) -> bool:
    """True if most lines look like TOC entries (leaders, tables, or MD links)."""
    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
    if not lines:
        return False

    if _TOC_HEADER.match(_heading_plain_text(paragraph)) and len(lines) == 1:
        return True

    if len(lines) == 1 and _PAGE_MARKER.match(lines[0]) and len(lines[0]) <= 6:
        return True

    toc_like = sum(1 for line in lines if _is_toc_line(line))
    return toc_like >= max(1, (len(lines) + 1) // 2)


def _is_toc_line(line: str) -> bool:
    """True for one TOC-looking line across plain / table / Markdown styles."""
    if _TOC_LINE.match(line) or _PAGE_MARKER.match(line):
        return True
    if _MD_TOC_LINK.match(line):
        return True
    if _is_markdown_toc_table_line(line):
        return True
    return False


def _is_markdown_toc_table_line(line: str) -> bool:
    """True for pipe-table TOC rows / separators (common after PDF→Markdown)."""
    stripped = line.strip()
    if re.match(r"^\|?[\s:\-|]+$", stripped):
        return True
    if "|" in stripped and re.search(
        r"\.{4,}\s*([ivxlcdm]+|\d+)\s*\|?\s*$", stripped, re.I
    ):
        return True
    return False


def _is_boilerplate_paragraph(paragraph: str) -> bool:
    """Heuristic: legal/license blocks and long ALL-CAPS legal walls."""
    # Never drop fenced code as "license" even if it mentions SPDX in a snippet.
    if _FENCE_LINE.search(paragraph):
        return False

    if _BOILERPLATE_START.search(paragraph):
        if len(paragraph) >= 120 or paragraph.count("\n") >= 2:
            return True

    letters = [c for c in paragraph if c.isalpha()]
    if len(letters) >= 80:
        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if upper_ratio >= 0.7:
            return True

    return False


def _is_low_signal(paragraph: str, *, min_chars: int) -> bool:
    """True for page markers, tiny scraps, or almost no letters."""
    stripped = paragraph.strip()
    # Keep Markdown headings — section chunking needs them as anchors.
    if re.match(r"^#{1,6}\s+\S", stripped):
        return False
    # Keep fenced code even if short.
    if _FENCE_LINE.match(stripped.splitlines()[0].strip()):
        return False

    compact = " ".join(paragraph.split())
    if _PAGE_MARKER.match(compact) and len(compact) <= 6:
        return True

    letters = sum(1 for c in compact if c.isalpha())
    if letters < 12:
        return True

    # Keep short Title-Case labels; drop tiny junk.
    if len(compact) < min_chars and "\n" not in paragraph:
        words = compact.split()
        if len(words) <= 3 and compact[:1].isupper():
            return False
        return True

    return False
