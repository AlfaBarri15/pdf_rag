"""Print % True for each eval hit type on the latest (or a given) run.

Run:
  uv run python -m app.eval.summary
  uv run python -m app.eval.summary --run-id <uuid>
"""

from __future__ import annotations

import argparse
import sys

from app.eval.store import fetch_cases, fetch_latest_run, fetch_run

_HIT_TYPES = ("doc_hit", "evidence_hit", "answer_hit")


def format_hit_percentages(run: dict, cases: list[dict]) -> str:
    """One line per hit type: True count / total as a percent."""
    total = len(cases)
    lines = [f"run_id={run['id']} cases={total}"]
    for name in _HIT_TYPES:
        if total == 0:
            lines.append(f"{name}=n/a")
            continue
        hits = sum(1 for case in cases if case[name])
        lines.append(f"{name}={hits}/{total} ({100.0 * hits / total:.1f}%)")
    return "\n".join(lines)


def main() -> None:
    """Print the percent of True doc, evidence, and answer hits for one eval run."""
    parser = argparse.ArgumentParser(
        description="Print % True for doc_hit, evidence_hit, and answer_hit"
    )
    parser.add_argument("--run-id", default=None, help="Eval run UUID (default: latest)")
    args = parser.parse_args()

    run = fetch_run(run_id=args.run_id) if args.run_id else fetch_latest_run()
    if run is None:
        print(
            "No eval runs found. Run: uv run python -m app.eval.runner --dataset financebench",
            file=sys.stderr,
        )
        sys.exit(1)

    cases = fetch_cases(run_id=str(run["id"]))
    print(format_hit_percentages(run, cases))


if __name__ == "__main__":
    main()
