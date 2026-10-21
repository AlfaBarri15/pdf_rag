from psycopg import connect
from psycopg.errors import OperationalError

import pytest

from app.config import DATABASE_URL
from app.retrieval.search import search_chunks
from app.retrieval.utils.search import (
    lexical_tsquery,
    parse_title_constraints,
    rrf_merge,
    title_matches_constraints,
)

_TITLES = ["3M_2018_10K", "3M_2022_10K", "3M_2023Q2_10Q"]
_FINANCEBENCH_LIMIT_5 = [
    (
        "What is the FY2018 capital expenditure amount (in USD millions) for 3M? "
        "Give a response to the question by relying on the details shown in the cash flow statement.",
        "3M",
        "2018",
        ["3M_2018_10K"],
    ),
    (
        "Assume that you are a public equities analyst. Answer the following question by "
        "primarily using information that is shown in the balance sheet: what is the year "
        "end FY2018 net PPNE for 3M? Answer in USD billions.",
        "3M",
        "2018",
        ["3M_2018_10K"],
    ),
    (
        "Is 3M a capital-intensive business based on FY2022 data?",
        "3M",
        "2022",
        ["3M_2022_10K"],
    ),
    (
        "What drove operating margin change as of FY2022 for 3M? If operating margin is "
        "not a useful metric for a company like this, then please state that and explain why.",
        "3M",
        "2022",
        ["3M_2022_10K"],
    ),
    (
        "If we exclude the impact of M&A, which segment has dragged down 3M's overall growth in 2022?",
        "3M",
        "2022",
        ["3M_2022_10K"],
    ),
]


def _db_available() -> bool:
    try:
        with connect(DATABASE_URL, connect_timeout=2) as conn:
            row = conn.execute(
                "SELECT count(*) FROM chunks WHERE embedding IS NOT NULL"
            ).fetchone()
            return row is not None and row[0] > 0
    except (OperationalError, OSError):
        return False


@pytest.mark.skipif(not _db_available(), reason="Postgres with embeddings not reachable")
def test_search_returns_results():
    payload = search_chunks(query="postgresql documentation", top_k=3)
    assert payload["result_count"] >= 1
    assert len(payload["results"]) <= 3
    assert payload["rerank"] is False
    first = payload["results"][0]
    assert "content" in first
    assert "distance" in first
    assert "score" in first
    assert first["score"] == pytest.approx(1.0 - first["distance"])


@pytest.mark.skipif(not _db_available(), reason="Postgres with embeddings not reachable")
def test_search_with_rerank():
    payload = search_chunks(
        query="postgresql documentation",
        top_k=3,
        rerank=True,
        top_n=2,
    )
    assert payload["rerank"] is True
    assert payload["result_count"] <= 2
    assert payload["rerank_model"] is not None
    assert all("rerank_score" in item for item in payload["results"])


def test_rrf_merge_prefers_items_in_both_lists():
    fused = rrf_merge(
        ["vec-only", "shared", "vec-third"],
        ["lex-only", "shared"],
        retrieve_k=3,
    )
    assert fused[0] == "shared"
    assert "lex-only" in fused


def test_lexical_tsquery_ors_content_words():
    q = lexical_tsquery("What is the FY2018 capital expenditure for 3M?")
    assert q is not None
    assert "2018" in q
    assert "capital" in q
    assert " | " in q


def test_search_rejects_empty_query():
    with pytest.raises(ValueError):
        search_chunks(query="   ", top_k=3)


@pytest.mark.parametrize(
    ("question", "company", "year", "expected_titles"),
    _FINANCEBENCH_LIMIT_5,
)
def test_title_constraints_first_five_financebench(
    question: str,
    company: str,
    year: str,
    expected_titles: list[str],
):
    constraints = parse_title_constraints(question, _TITLES)
    assert constraints == {"company": company, "year": year}
    matched = [
        title
        for title in _TITLES
        if title_matches_constraints(
            title, company=constraints["company"], year=constraints["year"]
        )
    ]
    assert matched == expected_titles


def test_title_constraints_skip_year_when_missing_or_ambiguous():
    titles = [*_TITLES, "ADOBE_2015_10K"]
    no_year = parse_title_constraints("Is 3M a capital-intensive business?", titles)
    assert no_year == {"company": "3M", "year": None}
    assert [
        title
        for title in titles
        if title_matches_constraints(
            title, company=no_year["company"], year=no_year["year"]
        )
    ] == ["3M_2018_10K", "3M_2022_10K", "3M_2023Q2_10Q"]

    two_years = parse_title_constraints(
        "Compare 3M results in 2018 versus 2022.", titles
    )
    assert two_years == {"company": "3M", "year": None}

    two_companies = parse_title_constraints(
        "Compare 3M and Adobe FY2018 capital expenditure.", titles
    )
    assert two_companies == {"company": None, "year": "2018"}
