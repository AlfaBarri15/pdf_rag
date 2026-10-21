from app.retrieval.utils.prompt import build_answer_prompt, build_citations


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
    assert "Copy the line item" in prompt
    assert "No essay" in prompt
