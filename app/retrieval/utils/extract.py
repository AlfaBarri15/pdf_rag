"""Pull a statement figure from retrieved text when PDF tables split the label and the number."""

from __future__ import annotations

import re
from typing import Any

_AMOUNT = re.compile(r"\(?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*\)?")
_YEAR = re.compile(r"^20\d{2}$")


def extract_numeric_answer(query: str, citations: list[dict[str, Any]]) -> str | None:
    """Return a short $ amount from retrieved chunks, or None to fall back to the LLM."""
    lines = _citation_lines(citations)
    blob = query.lower()
    if "capital expenditure" in blob or "capex" in blob:
        amount = _amount_after_labels(
            lines,
            ("purchases of property, plant and equipment", "purchases of property, plant"),
        )
        return _format_amount(amount, query)
    if "ppne" in blob or "pp&e" in blob or "ppne" in blob.replace("&", ""):
        amount = _amount_after_labels(
            lines,
            (
                "property, plant and equipment net",
                "property, plant and equipment – net",
                "property, plant and equipment - net",
            ),
        )
        return _format_amount(amount, query)
    return None


def _citation_lines(citations: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for citation in citations:
        lines.extend(str(citation.get("content") or "").splitlines())
    return lines


def _amount_after_labels(lines: list[str], labels: tuple[str, ...]) -> float | None:
    for index, line in enumerate(lines):
        low = re.sub(r"[\s–—−-]+", " ", line.lower())
        if not any(label in low for label in labels):
            continue
        window = " ".join(lines[index : index + 8])
        amount = _first_amount(window)
        if amount is not None:
            return amount
    return None


def _first_amount(text: str) -> float | None:
    for match in _AMOUNT.finditer(text):
        raw = match.group(1).replace(",", "")
        if _YEAR.fullmatch(raw):
            continue
        try:
            value = abs(float(raw))
        except ValueError:
            continue
        if value >= 1:
            return value
    return None


def _format_amount(amount: float | None, query: str) -> str | None:
    if amount is None:
        return None
    blob = query.lower()
    if "billion" in blob and amount >= 100:
        amount = amount / 1000.0
        return f"${amount:.2f}"
    if "million" in blob:
        return f"${amount:.2f}"
    if amount >= 100:
        return f"${amount:.2f}"
    return f"${amount:.2f}"
