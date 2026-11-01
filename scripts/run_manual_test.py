"""Chạy câu hỏi do người dùng tự viết qua planner và executor offline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datamate.local_inference import LocalTransformersBackend
from datamate.manual_testing import ManualTestError, run_manual_test
from datamate.question_understanding import JsonPlanAdapter, QuestionUnderstandingError


def main() -> None:
    for output in (sys.stdout, sys.stderr):
        if hasattr(output, "reconfigure"):
            output.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "manual_test" / "questions.jsonl",
        help="JSONL câu hỏi tự viết",
    )
    parser.add_argument(
        "--business-data-root",
        type=Path,
        required=True,
        help="Thư mục chứa thư mục con theo business_id",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Checkpoint mô hình offline đã có sẵn trên máy",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT_ROOT / "manual_test" / "results.jsonl",
        help="JSONL kết quả (mặc định nằm trong manual_test/)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "manual_test" / "report.json",
        help="Báo cáo tổng hợp JSON",
    )
    parser.add_argument("--rulebook", type=Path, help="Rulebook dùng chung nếu có")
    parser.add_argument("--calendar", type=Path, help="Calendar CSV dùng chung nếu có")
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="bf16",
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=16_384)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--parameter-cap", type=int, default=15_000_000_000)
    parser.add_argument("--generation-timeout-seconds", type=float)
    args = parser.parse_args()
    if args.max_prompt_tokens < 1:
        parser.error("max-prompt-tokens must be positive")
    if args.generation_timeout_seconds is not None and (
        not math.isfinite(args.generation_timeout_seconds)
        or args.generation_timeout_seconds <= 0
    ):
        parser.error("generation-timeout-seconds must be finite and positive")

    backend = LocalTransformersBackend(
        args.model_path,
        precision=args.precision,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        parameter_cap=args.parameter_cap,
        generation_timeout_seconds=args.generation_timeout_seconds,
    )
    try:
        summary = run_manual_test(
            args.questions,
            args.business_data_root,
            JsonPlanAdapter(backend, model_name=backend.model_name),
            args.results,
            args.report,
            rulebook_path=args.rulebook,
            calendar_path=args.calendar,
        )
    except (QuestionUnderstandingError, ManualTestError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Kết quả chi tiết: {args.results}")
    print(f"Báo cáo tổng hợp: {args.report}")


if __name__ == "__main__":
    main()
