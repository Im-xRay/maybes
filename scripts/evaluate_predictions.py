"""Compute local exact-match diagnostics; this never reports official PA."""

from __future__ import annotations

import argparse
import json

from datamate.evaluation import evaluate_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", required=True, help="Approved gold answers JSONL")
    parser.add_argument(
        "--predictions",
        required=True,
        help="DataMate run-log or prediction JSONL",
    )
    parser.add_argument(
        "--questions",
        help="Optional questions JSONL for business-level breakdown",
    )
    parser.add_argument("--output", required=True, help="New report JSON path")
    args = parser.parse_args()
    report = evaluate_predictions(
        args.answers,
        args.predictions,
        args.output,
        questions_path=args.questions,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
