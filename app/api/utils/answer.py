from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException
from psycopg.errors import Error as PsycopgError

from app.retrieval.pipeline import run_answer
from app.retrieval.utils.generate import LLMError

router = APIRouter(prefix="/answer", tags=["answer"])


class AnswerRequest(BaseModel):
    """Ask a grounded question over ingested docs (retrieval knobs stay internal)."""

    query: str = Field(..., min_length=1, description="Natural-language question")


@router.post("/")
def answer(body: AnswerRequest):
    """Full online RAG: retrieve → rerank → LLM answer with citations.

    Uses fixed defaults: top_k=20, rerank=true, top_n=8.
    """
    try:
        return run_answer(query=body.query)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (PsycopgError, OSError) as exc:
        raise HTTPException(status_code=503, detail=f"Upstream error: {exc}") from exc
