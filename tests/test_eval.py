from app.eval.download import financebench_question
from app.eval.results import format_report
from app.eval.summary import format_hit_percentages
from app.eval.runner import answer_question
from app.eval.score import answer_hit, doc_hit, evidence_hit, recall, score_case
from app.retrieval.utils.extract import extract_numeric_answer
from app.retrieval.utils.generate import LLMError


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
    assert answer_hit(gold_answer="$8.70", model_answer="Net PP&E was $8,738 million.")
    assert not answer_hit(gold_answer="$1577.00", model_answer="$1,400 million [1]")
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


def test_format_report_shows_hits_and_answers():
    text = format_report(
        {
            "id": "abc",
            "dataset": "financebench",
            "case_count": 1,
            "doc_recall": 1.0,
            "evidence_recall": 1.0,
            "answer_recall": 0.0,
        },
        [
            {
                "question_id": "financebench_id_03029",
                "doc_hit": True,
                "evidence_hit": True,
                "answer_hit": False,
                "gold_answer": "$1577.00",
                "model_answer": "About ninety million.",
            }
        ],
    )
    assert "doc_recall=1.000" in text
    assert "answer_recall=0.000" in text
    assert "financebench_id_03029" in text
    assert "$1577.00" in text
    assert "About ninety million." in text


def test_format_hit_percentages_counts_true_flags():
    text = format_hit_percentages(
        {"id": "abc"},
        [
            {"doc_hit": True, "evidence_hit": True, "answer_hit": True},
            {"doc_hit": True, "evidence_hit": True, "answer_hit": False},
            {"doc_hit": False, "evidence_hit": True, "answer_hit": False},
            {"doc_hit": False, "evidence_hit": False, "answer_hit": False},
        ],
    )
    assert "cases=4" in text
    assert "doc_hit=2/4 (50.0%)" in text
    assert "evidence_hit=3/4 (75.0%)" in text
    assert "answer_hit=1/4 (25.0%)" in text


def test_answer_question_keeps_chunks_when_llm_fails(monkeypatch):
    def boom(*, query: str, dataset: str):
        raise LLMError("model runner stopped")

    monkeypatch.setattr("app.eval.runner.run_answer", boom)
    monkeypatch.setattr(
        "app.eval.runner.prepare_answer",
        lambda *, query, dataset: {
            "retrieval": {"results": [{"source_path": "eval/financebench/pdfs/3M_2018_10K.pdf"}]}
        },
    )
    answered = answer_question(query="What was capex?", dataset="financebench")
    assert answered["answer"] == ""
    assert answered["results"][0]["source_path"].endswith("3M_2018_10K.pdf")


def test_extract_numeric_answer_reads_amount_on_following_lines():
    capex = extract_numeric_answer(
        "What is the FY2018 capital expenditure amount (in USD millions) for 3M?",
        [
            {
                "content": "Purchases of property, plant and equipment (PP&E)\n$\n(1,577)\n(1,373)",
            }
        ],
    )
    assert capex == "$1577.00"

    ppne = extract_numeric_answer(
        "what is the year end FY2018 net PPNE for 3M? Answer in USD billions.",
        [
            {
                "content": "Property, plant and equipment — net\n8,738\n8,866",
            }
        ],
    )
    assert ppne == "$8.74"
