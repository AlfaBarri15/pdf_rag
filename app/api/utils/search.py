from fastapi import APIRouter, HTTPException, Query
from psycopg.errors import Error as PsycopgError

from app.retrieval.pipeline import run_search

router = APIRouter(prefix="/search", tags=["search"])


@router.get("/")
def search(
    q: str = Query(..., min_length=1, description="Natural-language search question"),
    top_k: int = Query(
        5,
        ge=1,
        le=50,
        description="Vector-search candidate count (use ~20 when rerank=true)",
    ),
    rerank: bool = Query(
        False,
        description="If true, cross-encoder rerank candidates and return top_n",
    ),
    top_n: int = Query(
        3,
        ge=1,
        le=20,
        description="Final count after rerank (ignored when rerank=false)",
    ),
):
    """Embed the query, retrieve nearest chunks; optionally rerank with a cross-encoder."""
    try:
        return run_search(query=q, top_k=top_k, rerank=rerank, top_n=top_n)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (PsycopgError, OSError) as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}") from exc
