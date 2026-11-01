"""Tái chạy QueryPlan trong nhật ký và đối chiếu bằng chứng, đáp án."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datamate.replay import ReplayError, replay_run_record  # noqa: E402
from datamate.run_logging import RunLogError  # noqa: E402


def main() -> int:
    for output in (sys.stdout, sys.stderr):
        if hasattr(output, "reconfigure"):
            output.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path, help="Tệp run log JSONL")
    parser.add_argument("--question-id", required=True, help="ID câu hỏi cần tái chạy")
    parser.add_argument(
        "--business-data-root",
        required=True,
        type=Path,
        help="Thư mục chứa các thư mục con theo business_id",
    )
    parser.add_argument(
        "--occurrence",
        type=int,
        help="Lần xuất hiện (đếm từ 0) nếu question_id có nhiều bản ghi",
    )
    args = parser.parse_args()
    if args.occurrence is not None and args.occurrence < 0:
        parser.error("--occurrence phải là số nguyên không âm")
    try:
        result = replay_run_record(
            args.log,
            args.question_id,
            args.business_data_root,
            occurrence=args.occurrence,
        )
    except (ReplayError, RunLogError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["replay_matches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
