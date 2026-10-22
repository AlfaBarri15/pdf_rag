"""Deterministic scores for one retrieved answer. No database and no LLM."""

from __future__ import annotations

import re
from typing import Any

_WORD = re.compile(r"[a-z]{4,}")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_EVIDENCE_WORD_OVERLAP = 0.20


def normalize_text(text: str) -> str:
    """Lowercase, drop thousands-separators, and collapse whitespace."""
    collapsed = text.lower().replace(",", "")
    collapsed = re.sub(r"\s+", " ", collapsed)
    return collapsed.strip()


def canonical_numbers(text: str) -> set[str]:
    """Digit strings from text, with trailing decimal zeros stripped."""
    found: set[str] = set()
    for raw in _NUMBER.findall(normalize_text(text)):
        if "." in raw:
            raw = raw.rstrip("0").rstrip(".")
        if raw:
            found.add(raw)
    return found


def doc_hit(*, doc_name: str, source_paths: list[str]) -> bool:
    """True when the gold filing name appears in a retrieved source path."""
    needle = doc_name.strip().lower()
    if not needle:
        return False
    return any(needle in path.lower() for path in source_paths)


def evidence_hit(*, evidence: str, contents: list[str]) -> bool:
    """True when the labeled fact is in the retrieved context.

    Full-page gold strings rarely match Markdown/plain extracts byte-for-byte, so
    a hit is either substring containment or enough overlapping words plus at
    least one shared number from the evidence.
    """
    needle = normalize_text(evidence)
    if not needle:
        return False
    haystack = normalize_text("\n".join(contents))
    if needle in haystack:
        return True

    evidence_words = set(_WORD.findall(needle))
    haystack_words = set(_WORD.findall(haystack))
    if not evidence_words:
        return bool(canonical_numbers(evidence) & canonical_numbers(haystack))

    overlap = len(evidence_words & haystack_words) / len(evidence_words)
    evidence_nums = canonical_numbers(evidence)
    haystack_nums = canonical_numbers(haystack)
    number_ok = True if not evidence_nums else bool(evidence_nums & haystack_nums)
    return overlap >= _EVIDENCE_WORD_OVERLAP and number_ok


def answer_hit(*, gold_answer: str, model_answer: str) -> bool:
    """True when the gold answer (or its numbers) appears in the model answer."""
    needle = normalize_text(gold_answer)
    if not needle:
        return False
    haystack = normalize_text(model_answer)
    if needle in haystack:
        return True
    gold_nums = canonical_numbers(gold_answer)
    if not gold_nums:
        return False
    model_nums = canonical_numbers(model_answer)
    matched = {gold for gold in gold_nums if _number_in_model(gold, model_nums)}
    if len(gold_nums) <= 2:
        return matched == gold_nums
    return bool(matched)


def _number_in_model(gold: str, model_nums: set[str]) -> bool:
    """True if gold appears in the model, including million/billion scale and rounding."""
    if gold in model_nums:
        return True
    try:
        target = float(gold)
    except ValueError:
        return False
    for raw in model_nums:
        try:
            value = float(raw)
        except ValueError:
            continue
        for candidate in (value, value / 1000.0, value * 1000.0):
            if abs(target) < 1e-9:
                if abs(candidate) < 1e-9:
                    return True
                continue
            if abs(candidate - target) / abs(target) <= 0.03:
                return True
    return False


def score_case(
    *,
    doc_name: str,
    evidence: str,
    gold_answer: str,
    model_answer: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score one question against the chunks that were shown to the model."""
    source_paths = [str(item.get("source_path") or "") for item in results]
    contents = [str(item.get("content") or "") for item in results]
    top_score = _top_score(results)
    return {
        "doc_hit": doc_hit(doc_name=doc_name, source_paths=source_paths),
        "evidence_hit": evidence_hit(evidence=evidence, contents=contents),
        "answer_hit": answer_hit(gold_answer=gold_answer, model_answer=model_answer),
        "top_score": top_score,
        "retrieved_docs": source_paths,
    }


def recall(hits: int, total: int) -> float:
    """Hit rate. Empty runs score 0."""
    if total < 1:
        return 0.0
    return hits / total


def _top_score(results: list[dict[str, Any]]) -> float | None:
    if not results:
        return None
    first = results[0]
    if first.get("rerank_score") is not None:
        return float(first["rerank_score"])
    if first.get("score") is not None:
        return float(first["score"])
    return None
