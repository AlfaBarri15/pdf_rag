"""Score a dataset folder through the same ingest and answer path as production.

Run:
  python -m app.eval.runner --dataset financebench
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.config import OLLAMA_MODEL
from app.eval.questions import dataset_dir, load_questions
from app.eval.score import recall, score_case
from app.eval.store import create_run, finish_run, insert_case
from app.ingestion.pipeline import run_ingest
from app.ingestion.utils.chunker import (
    PAGE_MAX_CHARS,
    PAGE_OVERLAP,
    STRATEGY_PAGES,
)
from app.ingestion.utils.embeddings import EMBEDDING_MODEL_NAME
from app.ingestion.utils.store import fetch_document_summary
from app.retrieval.pipeline import prepare_answer, run_answer
from app.retrieval.utils.generate import LLMError
from app.retrieval.utils.rerank import RERANK_MODEL_NAME

logger = logging.getLogger(__name__)

DATA_ROOT = Path("/app/data")
if not DATA_ROOT.exists():
    DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def ingest_missing_pdfs(
    folder: Path,
    *,
    dataset: str,
    doc_names: set[str],
) -> int:
    """Ingest the filings these questions cite, skipping ones already stored."""
    pdf_dir = folder / "pdfs"
    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"PDF folder not found: {pdf_dir}")

    ingested = 0
    for doc_name in sorted(doc_names):
        path = pdf_dir / f"{doc_name}.pdf"
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found for {doc_name}: {path}")
        source_path = path.relative_to(DATA_ROOT).as_posix()
        existing = fetch_document_summary(source_path=source_path)
        already_paged = (
            existing
            and existing.get("dataset") == dataset
            and existing.get("chunk_count", 0) > 0
            and existing.get("chunk_strategy") == STRATEGY_PAGES
        )
        if already_paged:
            logger.info("skip ingest source_path=%s", source_path)
            continue
        logger.info("ingest source_path=%s strategy=%s", source_path, STRATEGY_PAGES)
        run_ingest(
            path=path,
            source_path=source_path,
            title=path.stem,
            dataset=dataset,
            strategy=STRATEGY_PAGES,
            max_chars=PAGE_MAX_CHARS,
            overlap=PAGE_OVERLAP,
        )
        ingested += 1
    return ingested


def answer_question(*, query: str, dataset: str) -> dict[str, object]:
    """Answer one eval question. Keep retrieved chunks if the LLM process dies."""
    try:
        return run_answer(query=query, dataset=dataset)
    except LLMError as exc:
        logger.warning("llm failed query=%s: %s", query, exc)
        prepared = prepare_answer(query=query, dataset=dataset)
        return {
            "answer": "",
            "results": prepared["retrieval"]["results"],
        }


def run_eval(*, dataset: str, limit: int | None = None) -> dict[str, object]:
    """Ingest missing PDFs, answer each question, and store one eval run."""
    folder = dataset_dir(DATA_ROOT, dataset)
    questions = load_questions(folder / "questions.jsonl")
    if limit is not None:
        questions = questions[:limit]

    ingested = ingest_missing_pdfs(
        folder,
        dataset=dataset,
        doc_names={item.doc_name for item in questions if item.doc_name},
    )
    run_id = create_run(
        dataset=dataset,
        embedding_model=EMBEDDING_MODEL_NAME,
        rerank_model=RERANK_MODEL_NAME,
        chat_model=OLLAMA_MODEL,
        chunk_settings={
            "strategy": STRATEGY_PAGES,
            "max_chars": PAGE_MAX_CHARS,
            "overlap": PAGE_OVERLAP,
        },
    )

    doc_hits = 0
    evidence_hits = 0
    answer_hits = 0
    for item in questions:
        logger.info("question id=%s", item.id)
        answered = answer_question(query=item.question, dataset=dataset)
        scored = score_case(
            doc_name=item.doc_name,
            evidence=item.evidence,
            gold_answer=item.gold_answer,
            model_answer=answered["answer"],
            results=answered["results"],
        )
        doc_hits += int(scored["doc_hit"])
        evidence_hits += int(scored["evidence_hit"])
        answer_hits += int(scored["answer_hit"])
        insert_case(
            run_id=run_id,
            question_id=item.id,
            question=item.question,
            gold_answer=item.gold_answer,
            doc_name=item.doc_name,
            doc_hit=scored["doc_hit"],
            evidence_hit=scored["evidence_hit"],
            answer_hit=scored["answer_hit"],
            top_score=scored["top_score"],
            retrieved_docs=scored["retrieved_docs"],
            model_answer=answered["answer"],
        )

    total = len(questions)
    summary = {
        "run_id": run_id,
        "dataset": dataset,
        "ingested": ingested,
        "case_count": total,
        "doc_recall": recall(doc_hits, total),
        "evidence_recall": recall(evidence_hits, total),
        "answer_recall": recall(answer_hits, total),
    }
    finish_run(
        run_id=run_id,
        case_count=total,
        doc_recall=summary["doc_recall"],
        evidence_recall=summary["evidence_recall"],
        answer_recall=summary["answer_recall"],
    )
    return summary


def main() -> None:
    """Score a dataset through the RAG pipeline and print recall."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Score a dataset through the RAG pipeline")
    parser.add_argument("--dataset", default="financebench")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Score only the first N questions (smoke run)",
    )
    args = parser.parse_args()
    summary = run_eval(dataset=args.dataset, limit=args.limit)
    print(
        f"run_id={summary['run_id']} cases={summary['case_count']} "
        f"doc_recall={summary['doc_recall']:.3f} "
        f"evidence_recall={summary['evidence_recall']:.3f} "
        f"answer_recall={summary['answer_recall']:.3f}"
    )


if __name__ == "__main__":
    main()
