from app.retrieval.utils.rerank import rerank_chunks


def test_rerank_orders_by_score():
    candidates = [
        {"content": "unrelated text about cooking pasta", "distance": 0.9, "score": 0.1},
        {
            "content": "PostgreSQL is a relational database management system",
            "distance": 0.4,
            "score": 0.6,
        },
        {"content": "random gardening tips for tomatoes", "distance": 0.8, "score": 0.2},
    ]
    ranked = rerank_chunks("what is postgresql", candidates, top_n=2)
    assert len(ranked) == 2
    assert all("rerank_score" in item for item in ranked)
    assert ranked[0]["rerank_score"] >= ranked[1]["rerank_score"]
    assert "PostgreSQL" in ranked[0]["content"]


def test_rerank_rejects_bad_top_n():
    import pytest

    with pytest.raises(ValueError):
        rerank_chunks("q", [{"content": "x"}], top_n=0)
