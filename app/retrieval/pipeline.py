"""Online retrieval pipeline: search → (rerank) → answer + citations."""

from __future__ import annotations

from typing import Any

from app.config import OLLAMA_MODEL
from app.retrieval.utils.extract import extract_numeric_answer
from app.retrieval.utils.generate import chat
from app.retrieval.utils.prompt import build_answer_prompt, build_citations
from app.retrieval.utils.search import search_chunks


def run_search(
    *,
    query: str,
    top_k: int = 5,
    rerank: bool = False,
    top_n: int = 3,
    dataset: str | None = None,
) -> dict[str, Any]:
    """Retrieve (and optionally rerank) chunks for a question."""
    return search_chunks(
        query=query,
        top_k=top_k,
        rerank=rerank,
        top_n=top_n,
        dataset=dataset,
    )


def prepare_answer(
    *,
    query: str,
    top_k: int = 20,
    rerank: bool = True,
    top_n: int = 8,
    dataset: str | None = None,
) -> dict[str, Any]:
    """Retrieve and build the grounded prompt. Does not call the LLM."""
    retrieval = run_search(
        query=query,
        top_k=top_k,
        rerank=rerank,
        top_n=top_n,
        dataset=dataset,
    )
    citations = build_citations(retrieval["results"])
    prompt = build_answer_prompt(query=retrieval["query"], citations=citations)
    return {
        "retrieval": retrieval,
        "citations": citations,
        "prompt": prompt,
    }


def run_answer(
    *,
    query: str,
    top_k: int = 20,
    rerank: bool = True,
    top_n: int = 8,
    model: str | None = None,
    dataset: str | None = None,
) -> dict[str, Any]:
    """Full online RAG: retrieve → optional rerank → LLM answer with citations.

    Citations are built from the chunks we put in the prompt (stable [1]..[n]),
    not by parsing the model output — same pattern as production grounded Chat.
    """
    prepared = prepare_answer(
        query=query,
        top_k=top_k,
        rerank=rerank,
        top_n=top_n,
        dataset=dataset,
    )
    retrieval = prepared["retrieval"]
    citations = prepared["citations"]
    extracted = extract_numeric_answer(retrieval["query"], citations)
    if extracted:
        answer = f"{extracted} [1]"
        model_name = "extract"
    else:
        llm = chat(user_prompt=prepared["prompt"], model=model or OLLAMA_MODEL)
        answer = llm["answer"]
        model_name = llm["model"]

    return {
        "query": retrieval["query"],
        "answer": answer,
        "model": model_name,
        "top_k": retrieval["top_k"],
        "rerank": retrieval["rerank"],
        "top_n": retrieval["top_n"],
        "embedding_model": retrieval["model"],
        "rerank_model": retrieval["rerank_model"],
        "result_count": retrieval["result_count"],
        "citations": citations,
        "results": retrieval["results"],
    }
