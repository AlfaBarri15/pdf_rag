from app.eval.download import financebench_question
from app.eval.score import answer_hit, doc_hit, evidence_hit, recall, score_case


def test_doc_hit_matches_filename():
    assert doc_hit(
        doc_name="3M_2018_10K",
        source_paths=["eval/financebench/pdfs/3M_2018_10K.pdf"],
    )
    assert not doc_hit(doc_name="3M_2018_10K", source_paths=["raw/other.pdf"])


def test_evidence_and_answer_ignore_case_and_commas():
    assert evidence_hit(
        evidence="Revenue was $1,577",
        contents=["later text", "revenue was $1577 in the filing"],
    )
    assert answer_hit(gold_answer="$1,577.00", model_answer="The figure is $1577.00.")
    assert answer_hit(gold_answer="$1577.00", model_answer="Capital expenditure was 1577 million.")
    assert not answer_hit(gold_answer="$99", model_answer="about one hundred")


def test_evidence_hit_accepts_table_overlap_not_full_page():
    gold = (
        "Table of Contents 3M Company and Subsidiaries Consolidated Statement of Cash Flows "
        "Purchases of property plant and equipment (PP&E) (1,577) (1,373) (1,420)"
    )
    contents = [
        "Purchases of property, plant and equipment (PP&E) | **$(1,577)** | $ (1,373) | $ (1,420)"
    ]
    assert evidence_hit(evidence=gold, contents=contents)
    assert not evidence_hit(
        evidence=gold,
        contents=["Goodwill increased during the year for unrelated reasons."],
    )


def test_score_case_uses_retrieved_chunks():
    scored = score_case(
        doc_name="ACME_2020_10K",
        evidence="net sales were 10",
        gold_answer="10",
        model_answer="Net sales were 10.",
        results=[
            {
                "source_path": "eval/financebench/pdfs/ACME_2020_10K.pdf",
                "content": "Net sales were 10.",
                "rerank_score": 0.8,
            }
        ],
    )
    assert scored["doc_hit"] is True
    assert scored["evidence_hit"] is True
    assert scored["answer_hit"] is True
    assert scored["top_score"] == 0.8


def test_recall_empty_is_zero():
    assert recall(0, 0) == 0.0
    assert recall(1, 2) == 0.5


def test_financebench_question_maps_public_row():
    mapped = financebench_question(
        {
            "financebench_id": "financebench_id_03029",
            "question": "What was revenue?",
            "answer": "$1577.00",
            "doc_name": "3M_2018_10K",
            "evidence": [
                {"evidence_text": "Net sales were $1,577 million.", "doc_name": "3M_2018_10K"}
            ],
        }
    )
    assert mapped["id"] == "financebench_id_03029"
    assert mapped["gold_answer"] == "$1577.00"
    assert mapped["doc_name"] == "3M_2018_10K"
    assert "1,577" in mapped["evidence"]
