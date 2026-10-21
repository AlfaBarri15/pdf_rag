"""Load a dataset folder: questions.jsonl plus pdfs/."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalQuestion:
    """One labeled question. Same shape for FinanceBench and a later golden set."""

    id: str
    question: str
    gold_answer: str
    doc_name: str
    evidence: str


def dataset_dir(root: Path, dataset: str) -> Path:
    """Return data/eval/<dataset>."""
    return root / "eval" / dataset


def load_questions(path: Path) -> list[EvalQuestion]:
    """Read JSONL rows with id, question, gold_answer, doc_name, evidence."""
    if not path.is_file():
        raise FileNotFoundError(f"Questions file not found: {path}")

    questions: list[EvalQuestion] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        question = str(row.get("question") or "").strip()
        if not question:
            raise ValueError(f"{path}:{line_number} is missing question")
        questions.append(
            EvalQuestion(
                id=str(row.get("id") or line_number),
                question=question,
                gold_answer=str(row.get("gold_answer") or ""),
                doc_name=str(row.get("doc_name") or ""),
                evidence=str(row.get("evidence") or ""),
            )
        )
    return questions
