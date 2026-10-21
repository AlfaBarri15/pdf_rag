"""Online retrieval: embed a query and find nearest chunks in pgvector."""

from __future__ import annotations

import re
from typing import Any

from psycopg import connect
from psycopg.errors import SyntaxError as PgSyntaxError
from psycopg.errors import UndefinedColumn, UndefinedFunction
from psycopg.rows import dict_row

from app.config import DATABASE_URL
from app.ingestion.utils.embeddings import EMBEDDING_MODEL_NAME, embed_text
from app.retrieval.utils.rerank import RERANK_MODEL_NAME, rerank_chunks

_FY_YEAR = re.compile(r"(?i)\bFY\s*(20\d{2})\b")
_YEAR = re.compile(r"\b(20\d{2})\b")


def _vector_literal(values: list[float]) -> str:
    """Format a float list as a pgvector literal: [0.1,0.2,...]."""
    return "[" + ",".join(str(float(v)) for v in values) + "]"


def parse_title_constraints(query: str, titles: list[str]) -> dict[str, str | None]:
    """Read company and FY year from the question using stored document titles.

    Does not use gold doc_name. If a constraint is missing or ambiguous, it is
    left unset so search stays unfiltered on that axis.
    """
    return {
        "company": _company_from_query(query, titles),
        "year": _year_from_query(query),
    }


def title_matches_constraints(
    title: str,
    *,
    company: str | None,
    year: str | None,
) -> bool:
    """True when a `{company}_{year}_{form}` title matches parsed constraints."""
    if not title:
        return False
    if company is not None and not title.upper().startswith(f"{company.upper()}_"):
        return False
    if year is not None and f"_{year}_" not in title:
        return False
    return True


def _company_from_query(query: str, titles: list[str]) -> str | None:
    hits: list[str] = []
    for company in _companies_from_titles(titles):
        if re.search(rf"(?i)\b{re.escape(company)}\b", query):
            hits.append(company)
    if len(hits) == 1:
        return hits[0]
    return None


def _companies_from_titles(titles: list[str]) -> list[str]:
    seen: list[str] = []
    for title in titles:
        token = (title or "").split("_", 1)[0].strip()
        if not token or token in seen:
            continue
        seen.append(token)
    return seen


def _year_from_query(query: str) -> str | None:
    fy_years = list(dict.fromkeys(_FY_YEAR.findall(query)))
    if fy_years:
        return fy_years[0] if len(fy_years) == 1 else None
    years = list(dict.fromkeys(_YEAR.findall(query)))
    return years[0] if len(years) == 1 else None


def _like_prefix_underscore(value: str) -> str:
    """ILIKE pattern with ESCAPE '!': value then a literal underscore."""
    return value.replace("!", "!!") + "!_%"


def _like_underscore_wrapped(value: str) -> str:
    """ILIKE pattern with ESCAPE '!': literal `_value_` anywhere in the title."""
    escaped = value.replace("!", "!!")
    return f"%!_{escaped}!_%"


_RRF_K = 60
_LEXICAL_TOKEN = re.compile(r"[A-Za-z]{3,}|\d{2,}")


def lexical_tsquery(query: str) -> str | None:
    """OR-query of content words for Postgres `to_tsquery('english', ...)`."""
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in _LEXICAL_TOKEN.findall(query):
        key = raw.lower()
        if key in seen:
            continue
        seen.add(key)
        tokens.append(re.sub(r"[^A-Za-z0-9]", "", raw))
        if len(tokens) >= 16:
            break
    if not tokens:
        return None
    return " | ".join(tokens)


def rrf_merge(
    *ranked_id_lists: list[str],
    retrieve_k: int,
    rrf_k: int = _RRF_K,
) -> list[str]:
    """Reciprocal rank fusion over one or more ranked id lists."""
    scores: dict[str, float] = {}
    for ranked in ranked_id_lists:
        for rank, item_id in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rrf_k + rank)
    return [
        item_id
        for item_id, _score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[
            :retrieve_k
        ]
    ]


def search_chunks(
    *,
    query: str,
    top_k: int = 5,
    rerank: bool = False,
    top_n: int = 3,
    dataset: str | None = None,
    database_url: str = DATABASE_URL,
) -> dict[str, Any]:
    """Embed the query, retrieve top-k by cosine, optionally cross-encoder rerank to top_n."""
    cleaned = query.strip()
    if not cleaned:
        raise ValueError("query must not be empty")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if top_n < 1:
        raise ValueError("top_n must be >= 1")

    retrieve_k = top_k
    if rerank and retrieve_k < top_n:
        retrieve_k = top_n

    query_vector = embed_text(cleaned)
    literal = _vector_literal(query_vector)
    company: str | None = None
    year: str | None = None

    with connect(database_url, row_factory=dict_row) as conn:
        title_rows = conn.execute(
            """
            SELECT DISTINCT title
            FROM documents
            WHERE title IS NOT NULL AND title <> ''
              AND (%s::text IS NULL OR dataset = %s)
            """,
            (dataset, dataset),
        ).fetchall()
        titles = [str(row["title"]) for row in title_rows]
        constraints = parse_title_constraints(cleaned, titles)
        company = constraints["company"]
        year = constraints["year"]
        company_like = _like_prefix_underscore(company) if company else None
        year_like = _like_underscore_wrapped(year) if year else None

        vector_rows = conn.execute(
            """
            SELECT
                c.id,
                c.chunk_index,
                c.content,
                c.metadata,
                d.source_path,
                d.title,
                (c.embedding <=> %s::vector) AS distance
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.embedding IS NOT NULL
              AND (%s::text IS NULL OR d.dataset = %s)
              AND (%s::text IS NULL OR d.title ILIKE %s ESCAPE '!')
              AND (%s::text IS NULL OR d.title ILIKE %s ESCAPE '!')
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """,
            (
                literal,
                dataset,
                dataset,
                company,
                company_like,
                year,
                year_like,
                literal,
                retrieve_k,
            ),
        ).fetchall()
        lexical_rows = _lexical_rows(
            conn,
            query=cleaned,
            dataset=dataset,
            company=company,
            company_like=company_like,
            year=year,
            year_like=year_like,
            retrieve_k=retrieve_k,
        )

    by_id: dict[str, Any] = {}
    vector_ids: list[str] = []
    for row in vector_rows:
        item_id = str(row["id"])
        vector_ids.append(item_id)
        by_id[item_id] = row
    lexical_ids: list[str] = []
    for row in lexical_rows:
        item_id = str(row["id"])
        lexical_ids.append(item_id)
        by_id.setdefault(item_id, row)

    fused_ids = rrf_merge(vector_ids, lexical_ids, retrieve_k=retrieve_k) if lexical_ids else vector_ids
    rows = [by_id[item_id] for item_id in fused_ids]

    results = [
        {
            "chunk_id": str(row["id"]),
            "chunk_index": row["chunk_index"],
            "content": row["content"],
            "metadata": row["metadata"],
            "source_path": row["source_path"],
            "title": row["title"],
            # Cosine distance: 0 = identical direction; lower is better.
            "distance": float(row["distance"]) if row.get("distance") is not None else 1.0,
            "score": (
                1.0 - float(row["distance"]) if row.get("distance") is not None else 0.0
            ),
        }
        for row in rows
    ]

    if rerank:
        results = rerank_chunks(cleaned, results, top_n=top_n)

    return {
        "query": cleaned,
        "top_k": retrieve_k,
        "rerank": rerank,
        "top_n": top_n if rerank else None,
        "model": EMBEDDING_MODEL_NAME,
        "rerank_model": RERANK_MODEL_NAME if rerank else None,
        "dataset": dataset,
        "title_company": company,
        "title_year": year,
        "hybrid": bool(lexical_ids),
        "result_count": len(results),
        "results": results,
    }


def _lexical_rows(
    conn,
    *,
    query: str,
    dataset: str | None,
    company: str | None,
    company_like: str | None,
    year: str | None,
    year_like: str | None,
    retrieve_k: int,
) -> list[Any]:
    """Full-text candidates for the same title/dataset filter as vector search."""
    tsquery = lexical_tsquery(query)
    if not tsquery:
        return []
    try:
        return conn.execute(
            """
            SELECT
                c.id,
                c.chunk_index,
                c.content,
                c.metadata,
                d.source_path,
                d.title,
                NULL::double precision AS distance
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.content_tsv @@ to_tsquery('english', %s)
              AND (%s::text IS NULL OR d.dataset = %s)
              AND (%s::text IS NULL OR d.title ILIKE %s ESCAPE '!')
              AND (%s::text IS NULL OR d.title ILIKE %s ESCAPE '!')
            ORDER BY ts_rank_cd(c.content_tsv, to_tsquery('english', %s)) DESC
            LIMIT %s
            """,
            (
                tsquery,
                dataset,
                dataset,
                company,
                company_like,
                year,
                year_like,
                tsquery,
                retrieve_k,
            ),
        ).fetchall()
    except (UndefinedColumn, UndefinedFunction, PgSyntaxError):
        conn.rollback()
        return []
