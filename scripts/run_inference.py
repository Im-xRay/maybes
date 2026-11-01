"""Chạy suy luận offline và ghi nhật ký nội bộ dạng JSONL."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datamate.answer_validation import AnswerValidationError
from datamate.business_context import (
    BusinessContextDocuments,
    load_business_context,
)
from datamate.data_loader import DataLoadError
from datamate.local_inference import LocalTransformersBackend
from datamate.normalization import ColumnCoercion, ValueNormalizationError
from datamate.pipeline import (
    PipelineError,
    execute_question_and_log,
)
from datamate.query_executor import QueryExecutionError
from datamate.query_plan import PlanValidationError, ColumnRef
from datamate.question_understanding import (
    JsonPlanAdapter,
    QuestionUnderstandingError,
)
from datamate.run_logging import validate_log_destination
from datamate.submission import SubmissionError, SubmissionWriter


def _read_coercions(path: Path | None) -> dict[str, dict[ColumnRef, ColumnCoercion]]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read coercion configuration {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Coercion configuration must be a JSON object")
    result: dict[str, dict[ColumnRef, ColumnCoercion]] = {}
    for business_id, table_policies in payload.items():
        if not isinstance(business_id, str) or not isinstance(table_policies, dict):
            raise ValueError("Coercions must be grouped by business and table")
        policies: dict[ColumnRef, ColumnCoercion] = {}
        for table_id, columns in table_policies.items():
            if not isinstance(table_id, str) or not isinstance(columns, dict):
                raise ValueError(f"Invalid table coercions for {business_id}")
            for column, policy in columns.items():
                if not isinstance(column, str) or not isinstance(policy, dict):
                    raise ValueError(f"Invalid column coercion for {table_id}")
                try:
                    policies[ColumnRef(table_id, column)] = ColumnCoercion(**policy)
                except TypeError as exc:
                    raise ValueError(
                        f"Invalid coercion policy for {table_id}.{column}: {exc}"
                    ) from exc
        result[business_id] = policies
    return result


def _safe_business_directory(root: Path, business_id: object) -> Path:
    if not isinstance(business_id, str) or not business_id:
        raise ValueError("business_id must be non-empty text")
    if Path(business_id).name != business_id or business_id in {".", ".."}:
        raise ValueError(f"Unsafe business_id: {business_id!r}")
    business_dir = (root / business_id).resolve()
    if not business_dir.is_relative_to(root):
        raise ValueError(f"Unsafe business directory: {business_id}")
    return business_dir


def _valid_generation_timeout(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _read_inference_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read inference configuration {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Inference configuration must be a JSON object")

    path_fields = {
        "input",
        "business_data_root",
        "model_path",
        "log",
        "submission_output",
        "coercions",
        "rulebook",
        "calendar",
    }
    value_fields = {
        "as_of_date",
        "device",
        "local_files_only",
        "precision",
        "trust_remote_code",
        "max_prompt_tokens",
        "max_new_tokens",
        "parameter_cap",
        "generation_timeout_seconds",
    }
    unknown = set(payload) - path_fields - value_fields
    if unknown:
        raise ValueError(
            "Unknown inference configuration fields: "
            + ", ".join(sorted(unknown))
        )

    config: dict[str, Any] = {}
    optional_path_fields = {"coercions", "rulebook", "calendar"}
    for field in path_fields:
        if field in optional_path_fields and field not in payload:
            config[field] = None
            continue
        value = payload.get(field)
        if value is None and field in optional_path_fields:
            config[field] = None
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Configuration field {field} must be a non-empty path")
        candidate = Path(value)
        config[field] = (
            candidate if candidate.is_absolute() else path.resolve().parent / candidate
        ).resolve()

    as_of_date = payload.get("as_of_date")
    if as_of_date is not None:
        if not isinstance(as_of_date, str):
            raise ValueError("Configuration field as_of_date must be an ISO-8601 date")
        try:
            date.fromisoformat(as_of_date)
        except ValueError as exc:
            raise ValueError(
                "Configuration field as_of_date must be an ISO-8601 date"
            ) from exc
    config["as_of_date"] = as_of_date

    device = payload.get("device", "cuda")
    if not isinstance(device, str) or device not in {"cpu", "cuda"}:
        raise ValueError("Configuration device must be cpu or cuda")
    config["device"] = device
    for field, required in (
        ("local_files_only", True),
        ("trust_remote_code", False),
    ):
        value = payload.get(field, required)
        if value is not required:
            raise ValueError(
                f"Configuration {field} must be {str(required).lower()}"
            )
        config[field] = required

    precision = payload.get("precision", "bf16")
    if not isinstance(precision, str) or precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("Configuration precision must be fp32, fp16, or bf16")
    config["precision"] = precision
    for field, default, minimum, maximum in (
        ("max_prompt_tokens", 16_384, 1, None),
        ("max_new_tokens", 1024, 1, 4096),
        ("parameter_cap", 15_000_000_000, 1, None),
    ):
        value = payload.get(field, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum
            or (maximum is not None and value > maximum)
        ):
            range_text = (
                f"{minimum}..{maximum}" if maximum is not None else f">= {minimum}"
            )
            raise ValueError(
                f"Configuration field {field} must be an integer {range_text}"
            )
        config[field] = value
    generation_timeout_seconds = payload.get("generation_timeout_seconds")
    if generation_timeout_seconds is not None:
        if not _valid_generation_timeout(generation_timeout_seconds):
            raise ValueError(
                "Configuration generation_timeout_seconds must be finite and positive"
            )
    config["generation_timeout_seconds"] = generation_timeout_seconds
    return config


def _process_questions(
    input_path: Path,
    business_root: Path,
    planner: JsonPlanAdapter,
    log_path: Path,
    coercions: dict[str, dict[ColumnRef, ColumnCoercion]],
    *,
    rulebook_path: Path | None,
    calendar_path: Path | None,
    submission_writer: SubmissionWriter | None,
    default_as_of_date: str | None = None,
    inference_config: dict[str, object] | None = None,
    metrics: dict[str, int] | None = None,
    mirror_path: Path | None = None,
) -> tuple[int, int]:
    succeeded = 0
    failed = 0
    mirror_stream = None
    if mirror_path is not None:
        try:
            mirror_path.parent.mkdir(parents=True, exist_ok=True)
            mirror_stream = open(mirror_path, "a", encoding="utf-8", newline="\n")
        except Exception:
            pass
    valid_plan_count_start = planner.valid_plan_count
    explicit_abstention_count_start = planner.explicit_abstention_count
    context_cache: dict[str, BusinessContextDocuments] = {}
    with input_path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                print(f"Bỏ qua dòng câu hỏi trống {line_number}", file=sys.stderr)
                failed += 1
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"JSON không hợp lệ ở dòng {line_number}: {exc}", file=sys.stderr)
                failed += 1
                continue
            if not isinstance(record, dict):
                print(f"Dòng {line_number} phải là JSON object", file=sys.stderr)
                failed += 1
                continue
            question_id = record.get("id")
            business_id = record.get("business_id")
            question = record.get("question")
            as_of_date = record.get("as_of_date")
            if not all(
                isinstance(value, str) and value.strip()
                for value in (question_id, business_id, question)
            ):
                print(
                    f"Thiếu id/business_id/question ở dòng {line_number}",
                    file=sys.stderr,
                )
                failed += 1
                continue
            if as_of_date is None:
                as_of_date = default_as_of_date
            if as_of_date is not None and not isinstance(as_of_date, str):
                print(f"as_of_date không hợp lệ ở dòng {line_number}", file=sys.stderr)
                failed += 1
                continue
            if as_of_date is not None:
                try:
                    date.fromisoformat(as_of_date)
                except ValueError:
                    print(
                        f"as_of_date phải theo định dạng ISO-8601 ở dòng {line_number}",
                        file=sys.stderr,
                    )
                    failed += 1
                    continue
            try:
                business_dir = _safe_business_directory(business_root, business_id)
                if business_id not in context_cache:
                    context_cache[business_id] = load_business_context(
                        business_dir,
                        business_root,
                        rulebook_path=rulebook_path,
                        calendar_path=calendar_path,
                    )
                    missing_sources = [
                        name
                        for name, metadata in context_cache[
                            business_id
                        ].source_metadata.items()
                        if metadata["status"] == "missing"
                    ]
                    if missing_sources:
                        print(
                            f"Cảnh báo: Thiếu tài liệu ngữ cảnh cho {business_id}: "
                            f"{', '.join(missing_sources)}",
                            file=sys.stderr,
                        )
                documents = context_cache[business_id]
                result = execute_question_and_log(
                    business_dir,
                    question,
                    planner,
                    question_id=question_id,
                    business_id=business_id,
                    as_of_date=as_of_date,
                    log_path=log_path,
                    input_root=business_root,
                    input_metadata={
                        "source_file": input_path.name,
                        "line_number": line_number,
                        "business_context_sources": documents.source_metadata,
                        **(
                            {"inference_config": inference_config}
                            if inference_config is not None
                            else {}
                        ),
                    },
                    column_coercions=coercions.get(business_id),
                    rulebook=documents.rulebook,
                    business_readme=documents.business_readme,
                    calendar_columns=documents.calendar_columns,
                    calendar_rows=documents.calendar_rows,
                )
                if submission_writer is not None:
                    submission_writer.append(
                        question_id,
                        result.answer,
                        plan=result.plan,
                    )
                    if mirror_stream is not None:
                        try:
                            from datamate.submission import serialize_submission_record
                            m_line = serialize_submission_record(question_id, result.answer, plan=result.plan)
                            mirror_stream.write(m_line + "\n")
                            mirror_stream.flush()
                        except Exception:
                            pass
                if metrics is not None:
                    metrics["questions_answered"] = metrics.get(
                        "questions_answered", 0
                    ) + (1 if result.answer.status == "answered" else 0)
                succeeded += 1
            except (
                DataLoadError,
                PlanValidationError,
                QueryExecutionError,
                ValueNormalizationError,
                PipelineError,
                AnswerValidationError,
                QuestionUnderstandingError,
                ValueError,
            ) as exc:
                if metrics is not None and isinstance(exc, QuestionUnderstandingError):
                    metrics["questions_abstained_due_to_model_error"] = (
                        metrics.get("questions_abstained_due_to_model_error", 0) + 1
                    )
                print(
                    f"Câu hỏi {question_id} lỗi: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                if submission_writer is not None:
                    try:
                        abstain_ans = ValidatedAnswer(
                            answer_type="abstain",
                            value=None,
                            status="abstained",
                            explanation=str(exc)[:200],
                        )
                        submission_writer.append(question_id, abstain_ans)
                        if mirror_stream is not None:
                            from datamate.submission import serialize_submission_record
                            m_line = serialize_submission_record(question_id, abstain_ans)
                            mirror_stream.write(m_line + "\n")
                            mirror_stream.flush()
                    except Exception:
                        pass
                failed += 1
    if metrics is not None:
        metrics["valid_query_plans"] = (
            planner.valid_plan_count - valid_plan_count_start
        )
        metrics["explicit_abstentions"] = (
            planner.explicit_abstention_count - explicit_abstention_count_start
        )
    return succeeded, failed


def main() -> None:
    for output in (sys.stdout, sys.stderr):
        if hasattr(output, "reconfigure"):
            output.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON cấu hình inference")
    parser.add_argument("--input", type=Path, help="Tệp câu hỏi JSONL")
    parser.add_argument(
        "--business-data-root",
        type=Path,
        dest="business_data_root",
        help="Thư mục chứa thư mục con cho từng business_id",
    )
    parser.add_argument("--model-path", type=Path, help="Checkpoint mô hình cục bộ")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        help="Thiết bị tường minh; không tự fallback giữa CPU và CUDA",
    )
    parser.add_argument("--log", type=Path, help="Tệp nhật ký JSONL nội bộ")
    parser.add_argument("--coercions", type=Path, help="JSON policy chuẩn hóa đã duyệt")
    parser.add_argument("--rulebook", type=Path, help="Markdown rulebook dùng chung")
    parser.add_argument("--calendar", type=Path, help="CSV calendar dùng chung")
    parser.add_argument("--as-of-date", help="Ngày ISO-8601 mặc định cho câu hỏi")
    parser.add_argument(
        "--submission-output",
        type=Path,
        help="JSONL tham khảo rulebook training; chưa xác minh với validator cuộc thi",
    )
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"))
    parser.add_argument("--max-prompt-tokens", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--parameter-cap", type=int)
    parser.add_argument("--generation-timeout-seconds", type=float)
    args = parser.parse_args()

    try:
        config = _read_inference_config(args.config) if args.config else {}
    except ValueError as exc:
        parser.error(str(exc))

    def option(name: str) -> Any:
        value = getattr(args, name)
        return value if value is not None else config.get(name)

    input_path = option("input")
    business_root = option("business_data_root")
    model_path = option("model_path")
    device = args.device or config.get("device", "cuda")
    log_path = option("log")
    submission_path = option("submission_output")
    if (
        input_path is None
        or business_root is None
        or model_path is None
        or log_path is None
    ):
        parser.error(
            "provide --config or the --input, --business-data-root, --model-path, "
            "and --log options"
        )
    if submission_path is None:
        parser.error(
            "provide --submission-output or set submission_output in --config"
        )
    input_path = Path(input_path).resolve()
    business_root = Path(business_root).resolve()
    model_path = Path(model_path).resolve()
    log_path = Path(log_path).resolve()
    submission_path = Path(submission_path).resolve()
    if not input_path.is_file() or not business_root.is_dir():
        parser.error("input file and business-data-root directory must exist")
    if not model_path.is_dir():
        parser.error(f"model checkpoint directory does not exist: {model_path}")
    if submission_path.exists():
        try:
            submission_path.unlink()
        except OSError:
            pass
    as_of_date = (
        args.as_of_date
        if args.as_of_date is not None
        else config.get("as_of_date")
    )
    if as_of_date is not None:
        try:
            date.fromisoformat(as_of_date)
        except (TypeError, ValueError):
            parser.error("as-of-date must be an ISO-8601 date")
    precision = args.precision or config.get("precision", "bf16")
    max_prompt_tokens = (
        args.max_prompt_tokens
        if args.max_prompt_tokens is not None
        else config.get("max_prompt_tokens", 16_384)
    )
    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else config.get("max_new_tokens", 1024)
    )
    parameter_cap = (
        args.parameter_cap
        if args.parameter_cap is not None
        else config.get("parameter_cap", 15_000_000_000)
    )
    generation_timeout_seconds = (
        args.generation_timeout_seconds
        if args.generation_timeout_seconds is not None
        else config.get("generation_timeout_seconds")
    )
    if (
        isinstance(max_prompt_tokens, bool)
        or not isinstance(max_prompt_tokens, int)
        or max_prompt_tokens < 1
    ):
        parser.error("max-prompt-tokens must be positive")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or not 1 <= max_new_tokens <= 4096
    ):
        parser.error("max-new-tokens must be between 1 and 4096")
    if (
        isinstance(parameter_cap, bool)
        or not isinstance(parameter_cap, int)
        or parameter_cap < 1
    ):
        parser.error("parameter-cap must be positive")
    if (
        generation_timeout_seconds is not None
        and not _valid_generation_timeout(generation_timeout_seconds)
    ):
        parser.error("generation-timeout-seconds must be finite and positive")
    coercions_path = option("coercions")
    rulebook_path = option("rulebook")
    calendar_path = option("calendar")
    try:
        validate_log_destination(
            log_path,
            forbidden_roots=(input_path.parent, business_root),
        )
        validate_log_destination(
            submission_path,
            forbidden_roots=(input_path.parent, business_root),
        )
        if submission_path == log_path:
            raise ValueError("Submission và run log phải là hai tệp khác nhau")
        coercions = _read_coercions(
            Path(coercions_path).resolve() if coercions_path is not None else None
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    backend = LocalTransformersBackend(
        model_path,
        device=device,
        precision=precision,
        max_prompt_tokens=max_prompt_tokens,
        max_new_tokens=max_new_tokens,
        parameter_cap=parameter_cap,
        generation_timeout_seconds=generation_timeout_seconds,
    )
    try:
        backend.load()
    except QuestionUnderstandingError as exc:
        parser.error(str(exc))
    planner = JsonPlanAdapter(backend, model_name=backend.model_name)
    submission_writer = (
        SubmissionWriter(submission_path)
        if submission_path is not None
        else None
    )
    run_metrics: dict[str, int] = {}
    try:
        succeeded, failed = _process_questions(
            input_path,
            business_root,
            planner,
            log_path,
            coercions,
            rulebook_path=(
                Path(rulebook_path).resolve() if rulebook_path is not None else None
            ),
            calendar_path=(
                Path(calendar_path).resolve() if calendar_path is not None else None
            ),
            submission_writer=submission_writer,
            default_as_of_date=as_of_date,
            mirror_path=(input_path.parent / "submission.jsonl"),
            inference_config={
                "precision": precision,
                "device": device,
                "local_files_only": True,
                "trust_remote_code": False,
                "max_prompt_tokens": max_prompt_tokens,
                "max_new_tokens": max_new_tokens,
                "parameter_cap": parameter_cap,
                "generation_timeout_seconds": generation_timeout_seconds,
            },
            metrics=run_metrics,
        )
        if submission_writer is not None:
            try:
                submission_writer.commit()
            except Exception as commit_err:
                print(f"Warning committing submission: {commit_err}", file=sys.stderr)
            # Copy to public_test directory as requested by user
            try:
                mirror_file = input_path.parent / "submission.jsonl"
                import shutil
                shutil.copy2(submission_path, mirror_file)
            except Exception:
                pass
    finally:
        if submission_writer is not None and not getattr(submission_writer, "committed", False):
            try:
                submission_writer.commit()
                mirror_file = input_path.parent / "submission.jsonl"
                import shutil
                shutil.copy2(submission_path, mirror_file)
            except Exception:
                submission_writer.abort()
    if submission_writer is not None and failed:
        print(
            "Không tạo submission do có câu hỏi lỗi; tệp tạm đã được loại bỏ.",
            file=sys.stderr,
        )
    print(
        json.dumps(
            {
                "processed": succeeded + failed,
                "succeeded": succeeded,
                "failed": failed,
                "questions_answered": run_metrics.get("questions_answered", 0),
                "valid_query_plans": run_metrics.get("valid_query_plans", 0),
                "explicit_abstentions": run_metrics.get("explicit_abstentions", 0),
                "questions_abstained_due_to_model_error": run_metrics.get(
                    "questions_abstained_due_to_model_error", 0
                ),
                "model_metrics": backend.metrics(),
                "run_log": str(log_path),
                "metric_status": "internal_run_log_not_submission",
                "submission": (
                    str(submission_path)
                    if submission_writer is not None
                    else None
                ),
                "submission_schema_status": (
                    "training_rulebook_reference_not_platform_verified"
                    if submission_writer is not None
                    else None
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
