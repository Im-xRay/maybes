"""Prepare manually annotated QueryPlan examples without opening public-test data."""

from __future__ import annotations

import argparse
import json

from datamate.training_data import prepare_plan_training_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, help="Training questions JSONL")
    parser.add_argument(
        "--annotations",
        required=True,
        help="Manually reviewed JSONL with question_id, business_id, query_plan",
    )
    parser.add_argument(
        "--business-data-root",
        required=True,
        help="Directory containing one subdirectory per business_id",
    )
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = prepare_plan_training_data(
        args.questions,
        args.annotations,
        args.business_data_root,
        args.output,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    print(json.dumps(report.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
