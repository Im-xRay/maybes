"""Profile training data and write a sanitized JSON artifact and Markdown report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datamate.schema_profiler import (  # noqa: E402
    profile_training_dataset,
    render_schema_report,
    write_profile,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-root",
        type=Path,
        default=ROOT / "data" / "train" / "train",
        help="Root containing data/ and training metadata files.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "data" / "train" / "schema_profile.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "planning" / "15_SCHEMA_PROFILING.md",
    )
    parser.add_argument(
        "--max-rows-per-table",
        type=int,
        help="Optional bounded sample per table. Omit for a full streaming scan.",
    )
    args = parser.parse_args()

    profile = profile_training_dataset(
        args.training_root,
        max_rows_per_table=args.max_rows_per_table,
    )
    write_profile(profile, args.output_json)
    report = render_schema_report(profile)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8", newline="\n")
    summary = (
        f"Profile complete: {len(profile['businesses'])} businesses; "
        f"JSON={args.output_json}; report={args.report}"
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
