"""Print the latest (or a given) eval run from Postgres.

Run:
  uv run python -m app.eval.results
  uv run python -m app.eval.results --run-id <uuid>
"""

from __future__ import annotations

import argparse
import sys

from app.eval.store import fetch_cases, fetch_latest_run, fetch_run


def _flag(value: bool | None) -> str:
    return "t" if value else "f"


def _clip(text: str | None, width: int) -> str:
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= width:
        return collapsed
    return collapsed[: width - 1] + "…"


def format_report(run: dict, cases: list[dict]) -> str:
    """Human-readable summary: one header plus one line per question."""
    lines = [
        f"run_id={run['id']} dataset={run['dataset']} cases={run['case_count']}",
        (
            f"doc_recall={_rate(run['doc_recall'])} "
            f"evidence_recall={_rate(run['evidence_recall'])} "
            f"answer_recall={_rate(run['answer_recall'])}"
        ),
        "",
        f"{'question_id':<24} {'doc':<4} {'evid':<5} {'ans':<4} {'gold':<28} model",
    ]
    for case in cases:
        lines.append(
            f"{case['question_id']:<24} "
            f"{_flag(case['doc_hit']):<4} "
            f"{_flag(case['evidence_hit']):<5} "
            f"{_flag(case['answer_hit']):<4} "
            f"{_clip(case.get('gold_answer'), 28):<28} "
            f"{_clip(case.get('model_answer'), 80)}"
        )
    return "\n".join(lines)


def _rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Print scored eval cases from Postgres")
    parser.add_argument("--run-id", default=None, help="Eval run UUID (default: latest)")
    args = parser.parse_args()

    run = fetch_run(run_id=args.run_id) if args.run_id else fetch_latest_run()
    if run is None:
        print("No eval runs found. Run: uv run python -m app.eval.runner --dataset financebench --limit 5", file=sys.stderr)
        sys.exit(1)

    cases = fetch_cases(run_id=str(run["id"]))
    print(format_report(run, cases))


if __name__ == "__main__":
    main()
