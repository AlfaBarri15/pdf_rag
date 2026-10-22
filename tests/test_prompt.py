from app.retrieval.utils.prompt import (
    build_answer_prompt,
    build_citations,
    excerpt_for_query,
)


def test_build_citations_numbers_results():
    results = [
        {
            "chunk_id": "a",
            "chunk_index": 0,
            "title": "Doc",
            "source_path": "raw/doc.pdf",
            "content": "PostgreSQL is a database",
            "distance": 0.2,
            "score": 0.8,
        },
        {
            "chunk_id": "b",
            "chunk_index": 1,
            "title": "Doc",
            "source_path": "raw/doc.pdf",
            "content": "Other text",
            "distance": 0.4,
            "score": 0.6,
            "rerank_score": 1.5,
        },
    ]
    citations = build_citations(results)
    assert [c["ref"] for c in citations] == [1, 2]
    assert citations[0]["chunk_id"] == "a"
    assert citations[1]["rerank_score"] == 1.5


def test_build_answer_prompt_includes_sources_and_question():
    citations = build_citations(
        [
            {
                "chunk_id": "a",
                "chunk_index": 0,
                "title": "PG Docs",
                "source_path": "raw/pg.pdf",
                "content": "PostgreSQL is open source",
                "distance": 0.1,
                "score": 0.9,
            }
        ]
    )
    prompt = build_answer_prompt(query="What is PostgreSQL?", citations=citations)
    assert "[1]" in prompt
    assert "PostgreSQL is open source" in prompt
    assert "What is PostgreSQL?" in prompt
    assert "statement line item" in prompt
    assert "No essay" in prompt


def test_excerpt_keeps_ppe_row_and_drops_unrelated_notes():
    page = "\n".join(
        [
            "Goodwill increased during the year for unrelated reasons.",
            "The Company expects 2018 capital spending of $1.7 billion to $1.9 billion.",
            "Cash Flows from Investing Activities",
            "Purchases of property, plant and equipment (PP&E)",
            "$(1,577)",
            "Proceeds from sale of PP&E and other assets 262",
            "3M Belgium received a report from the Flemish inspectorate.",
        ]
    )
    excerpt = excerpt_for_query(
        page,
        "What is the FY2018 capital expenditure amount for 3M from the cash flow statement?",
    )
    assert "1,577" in excerpt
    assert "Purchases of property" in excerpt
    assert "Flemish inspectorate" not in excerpt
    assert "1.7 billion to $1.9 billion" not in excerpt
