"""Download the public FinanceBench 150 into data/eval/financebench/.

Questions become our JSONL shape. PDFs are saved as pdfs/{doc_name}.pdf.
"""

from __future__ import annotations

import json
import ssl
import urllib.request
from pathlib import Path

import certifi

QUESTIONS_URL = (
    "https://raw.githubusercontent.com/patronus-ai/financebench/main/"
    "data/financebench_open_source.jsonl"
)
PDF_URL = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/{doc_name}.pdf"


def financebench_question(row: dict) -> dict[str, str]:
    """Map one upstream row onto id, question, gold_answer, doc_name, evidence."""
    evidence_items = row.get("evidence") or []
    evidence = ""
    if evidence_items and isinstance(evidence_items, list):
        evidence = str(evidence_items[0].get("evidence_text") or "")
    return {
        "id": str(row.get("financebench_id") or ""),
        "question": str(row.get("question") or ""),
        "gold_answer": str(row.get("answer") or ""),
        "doc_name": str(row.get("doc_name") or ""),
        "evidence": evidence,
    }


def _opener() -> urllib.request.OpenerDirector:
    """HTTPS opener using certifi's CA bundle.

    The python.org macOS installer does not trust public CAs until its
    certificate command is run, so the default ssl context fails on GitHub.
    """
    context = ssl.create_default_context(cafile=certifi.where())
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))


def download_financebench(dest: Path) -> dict[str, int]:
    """Fetch questions and any missing PDFs. Existing PDFs are left in place."""
    dest.mkdir(parents=True, exist_ok=True)
    pdf_dir = dest / "pdfs"
    pdf_dir.mkdir(exist_ok=True)
    opener = _opener()

    with opener.open(QUESTIONS_URL, timeout=60) as response:
        raw = response.read().decode("utf-8")

    questions = [
        financebench_question(json.loads(line))
        for line in raw.splitlines()
        if line.strip()
    ]
    questions_path = dest / "questions.jsonl"
    questions_path.write_text(
        "".join(json.dumps(row) + "\n" for row in questions),
        encoding="utf-8",
    )

    downloaded = 0
    skipped = 0
    for doc_name in sorted({row["doc_name"] for row in questions if row["doc_name"]}):
        target = pdf_dir / f"{doc_name}.pdf"
        if target.is_file() and target.stat().st_size > 0:
            skipped += 1
            continue
        url = PDF_URL.format(doc_name=doc_name)
        with opener.open(url, timeout=120) as response:
            target.write_bytes(response.read())
        downloaded += 1

    return {
        "questions": len(questions),
        "pdfs_downloaded": downloaded,
        "pdfs_skipped": skipped,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2] / "data" / "eval" / "financebench"
    stats = download_financebench(root)
    print(
        f"wrote {root} questions={stats['questions']} "
        f"pdfs_downloaded={stats['pdfs_downloaded']} pdfs_skipped={stats['pdfs_skipped']}"
    )


if __name__ == "__main__":
    main()
