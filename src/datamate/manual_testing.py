"""Run user-authored JSONL questions through the production planning pipeline."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path

from .answer_validation import ValidatedAnswer
from .business_context import BusinessContextDocuments, load_business_context
from .data_loader import DataLoadError
from .evaluation import (
    EvaluationError,
    answers_match,
    validate_answer_value,
)
from .pipeline import (
    PipelineError,
    PipelineResult,
    execute_question_and_log,
)
from .plan_serialization import QueryPlanFormatError
from .query_executor import QueryExecutionError
from .query_plan import PlanValidationError, QueryPlan
from .question_understanding import (
    QueryPlanner,
    QuestionUnderstandingError,
)
from .run_logging import _json_default
from .submission import serialize_submission_record

_INPUT_FIELDS = {"id", "business_id", "as_of_date", "question", "expected_answer"}
_EXPECTED_TYPES = {
    "number",
    "percent",
    "string",
    "date",
    "boolean",
    "list",
    "table",
    "unanswerable",
}
_ERROR_CATEGORIES = {
    "Không hiểu đúng câu hỏi",
    "Tạo QueryPlan sai",
    "Chọn sai bảng hoặc cột",
    "Lọc dữ liệu sai",
    "Thực thi truy vấn sai",
    "Tính toán sai",
    "Sai business rule",
    "Xuất đáp án sai định dạng",
    "Không đủ dữ liệu để trả lời",
    "Cần phân tích thủ công",
}


class ManualTestError(ValueError):
    """Raised when manual-test inputs or outputs cannot be safely handled."""


def _jsonable(value: object) -> object:
    return json.loads(json.dumps(value, ensure_ascii=False, default=_json_default))


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON constant không hợp lệ: {value}")


def _safe_business_directory(root: Path, business_id: str) -> Path:
    if Path(business_id).name != business_id or business_id in {".", ".."}:
        raise ManualTestError(f"business_id không an toàn: {business_id!r}")
    business_dir = (root / business_id).resolve()
    if not business_dir.is_relative_to(root) or not business_dir.is_dir():
        raise ManualTestError(f"Không tìm thấy thư mục doanh nghiệp: {business_id}")
    return business_dir


def _validate_expected_answer(value: object) -> tuple[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"type", "answer"}:
        raise ManualTestError(
            "expected_answer phải là null hoặc object chỉ gồm type và answer"
        )
    answer_type = value["type"]
    answer_value = value["answer"]
    if not isinstance(answer_type, str) or answer_type not in _EXPECTED_TYPES:
        raise ManualTestError(f"expected_answer.type không hỗ trợ: {answer_type!r}")
    try:
        validate_answer_value(answer_type, answer_value)
    except EvaluationError as exc:
        raise ManualTestError(f"expected_answer không hợp lệ: {exc}") from exc
    return answer_type, answer_value


def _validate_question_record(record: object) -> dict[str, object]:
    if not isinstance(record, dict):
        raise ManualTestError("Mỗi dòng phải là một JSON object")
    extra_fields = set(record) - _INPUT_FIELDS
    if extra_fields:
        raise ManualTestError(
            "Trường JSON không được hỗ trợ: " + ", ".join(sorted(extra_fields))
        )
    for field in ("id", "business_id", "question"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ManualTestError(f"{field} phải là chuỗi không rỗng")
    as_of_date = record.get("as_of_date")
    if as_of_date is not None:
        if not isinstance(as_of_date, str):
            raise ManualTestError("as_of_date phải là ngày ISO YYYY-MM-DD hoặc null")
        try:
            if date.fromisoformat(as_of_date).isoformat() != as_of_date:
                raise ValueError
        except ValueError as exc:
            raise ManualTestError(
                "as_of_date phải có định dạng ISO YYYY-MM-DD"
            ) from exc
    return record


def _answer_payload(
    question_id: str,
    answer: ValidatedAnswer,
    plan: QueryPlan,
) -> dict[str, object]:
    encoded = serialize_submission_record(question_id, answer, plan=plan)
    submission = json.loads(encoded)
    submission.pop("id", None)
    return submission


def _error_category(error: Exception) -> str:
    if isinstance(error, PlanValidationError | QueryPlanFormatError):
        return "Tạo QueryPlan sai"
    if isinstance(error, DataLoadError):
        return "Chọn sai bảng hoặc cột"
    if isinstance(error, QueryExecutionError):
        return "Thực thi truy vấn sai"
    if isinstance(error, QuestionUnderstandingError):
        return "Không hiểu đúng câu hỏi"
    if isinstance(error, PipelineError):
        return "Xuất đáp án sai định dạng"
    return "Cần phân tích thủ công"


def _failure_record(
    *,
    question_id: object,
    question: object,
    expected_answer: object,
    error: Exception,
    elapsed_ms: float,
) -> dict[str, object]:
    expected_valid = False
    if expected_answer is not None:
        try:
            expected_valid = _validate_expected_answer(expected_answer) is not None
        except ManualTestError:
            pass
    return {
        "id": question_id,
        "question": question,
        "generated_answer": None,
        "expected_answer": expected_answer,
        "expected_answer_valid": (
            expected_valid if expected_answer is not None else None
        ),
        "prediction_status": "error",
        "correctness": False if expected_valid else None,
        "query_plan": None,
        "evidence": [],
        "execution_trace": None,
        "error": f"{type(error).__name__}: {error}",
        "error_category": _error_category(error),
        "error_category_candidates": [],
        "elapsed_time": {"value": elapsed_ms, "unit": "ms"},
    }


def _run_one(
    record: dict[str, object],
    *,
    business_root: Path,
    planner: QueryPlanner,
    run_log: Path,
    input_root: Path,
    rulebook_path: Path | None,
    calendar_path: Path | None,
    context_cache: dict[str, BusinessContextDocuments],
) -> dict[str, object]:
    started = time.perf_counter()
    question_id = record["id"]
    business_id = record["business_id"]
    question = record["question"]
    expected_answer = record.get("expected_answer")
    try:
        business_dir = _safe_business_directory(business_root, business_id)
        documents = context_cache.get(business_id)
        if documents is None:
            documents = load_business_context(
                business_dir,
                business_root,
                rulebook_path=rulebook_path,
                calendar_path=calendar_path,
            )
            context_cache[business_id] = documents

        result: PipelineResult = execute_question_and_log(
            business_dir,
            question,
            planner,
            question_id=question_id,
            business_id=business_id,
            as_of_date=record.get("as_of_date"),
            log_path=run_log,
            input_root=input_root,
            input_metadata={"manual_test": True},
            rulebook=documents.rulebook,
            business_readme=documents.business_readme,
            calendar_columns=documents.calendar_columns,
            calendar_rows=documents.calendar_rows,
        )
        generated = _answer_payload(question_id, result.answer, result.plan)
        expected_validation_error = None
        try:
            expected = _validate_expected_answer(expected_answer)
        except ManualTestError as exc:
            expected = None
            expected_validation_error = str(exc)
        correctness = None
        error_category = (
            "expected_answer không hợp lệ"
            if expected_validation_error is not None
            else None
        )
        if expected is not None:
            predicted_type = result.answer.answer_type
            predicted_value = result.answer.value
            if predicted_type == "table":
                predicted_value = generated.get("answer")
            correctness = answers_match(
                expected[0],
                expected[1],
                predicted_type,
                predicted_value,
            )
            if correctness is False:
                error_category = "Cần phân tích thủ công"
        execution = (
            asdict(result.execution.trace)
            if result.execution is not None
            else None
        )
        output = {
            "id": question_id,
            "question": question,
            "generated_answer": generated,
            "expected_answer": expected_answer,
            "expected_answer_valid": (
                None if expected_answer is None else expected is not None
            ),
            "prediction_status": result.answer.status,
            "correctness": correctness,
            "query_plan": asdict(result.plan),
            "evidence": [asdict(item) for item in result.answer.evidence],
            "execution_trace": execution,
            "error": expected_validation_error,
            "error_category": error_category,
            "error_category_candidates": (
                sorted(_ERROR_CATEGORIES)
                if error_category == "Cần phân tích thủ công"
                else []
            ),
            "answer_type_matches": (
                result.answer.answer_type == expected[0]
                if expected is not None
                else None
            ),
            "elapsed_time": {
                "value": (time.perf_counter() - started) * 1000,
                "unit": "ms",
            },
        }
        return _jsonable(output)
    except (
        DataLoadError,
        PlanValidationError,
        QueryExecutionError,
        PipelineError,
        QuestionUnderstandingError,
        QueryPlanFormatError,
        EvaluationError,
        ManualTestError,
        ValueError,
        OSError,
    ) as exc:
        return _jsonable(
            _failure_record(
                question_id=question_id,
                question=question,
                expected_answer=expected_answer,
                error=exc,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
        )


def _read_input(path: Path) -> list[tuple[int, object, Exception | None]]:
    entries: list[tuple[int, object, Exception | None]] = []
    seen_ids: set[str] = set()
    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    entries.append(
                        (
                            line_number,
                            None,
                            ManualTestError(f"Dòng {line_number} để trống"),
                        )
                    )
                    continue
                record: object = None
                try:
                    record = json.loads(line, parse_constant=_reject_json_constant)
                    validated = _validate_question_record(record)
                    question_id = validated["id"]
                    if question_id in seen_ids:
                        raise ManualTestError(
                            f"id bị trùng ở dòng {line_number}: {question_id}"
                        )
                    seen_ids.add(question_id)
                    entries.append((line_number, validated, None))
                except (ValueError, ManualTestError) as exc:
                    entries.append((line_number, record, exc))
    except OSError as exc:
        raise ManualTestError(f"Không thể đọc questions JSONL: {exc}") from exc
    return entries


def _summary(records: list[dict[str, object]]) -> dict[str, object]:
    expected_valid = sum(
        record.get("expected_answer_valid") is True for record in records
    )
    correct = sum(record["correctness"] is True for record in records)
    wrong = sum(
        record["correctness"] is False
        for record in records
        if record.get("expected_answer") is not None
        and not (
            isinstance(record.get("generated_answer"), dict)
            and record["generated_answer"].get("type") == "abstain"
        )
    )
    errors = sum(record.get("error") is not None for record in records)
    prediction_errors = sum(
        record["prediction_status"] == "error" for record in records
    )
    unlabeled = sum(record.get("expected_answer") is None for record in records)
    abstentions = sum(
        isinstance(record.get("generated_answer"), dict)
        and record["generated_answer"].get("type") == "abstain"
        for record in records
    )
    return {
        "total_questions": len(records),
        "processed_successfully": len(records) - prediction_errors,
        "errors": errors,
        "prediction_errors": prediction_errors,
        "expected_answer_valid_count": expected_valid,
        "correct_answers": correct,
        "wrong_answers": wrong,
        "abstentions": abstentions,
        "without_expected_answer": unlabeled,
        "accuracy": correct / expected_valid if expected_valid else None,
        "accuracy_scope": (
            "So sánh local theo evaluation.answers_match; không phải điểm PA "
            "chính thức. Abstain chỉ đúng nếu type/value trùng đáp án chuẩn."
        ),
        "error_categories": sorted(_ERROR_CATEGORIES),
    }


def _write_jsonl_atomic(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                )
                stream.write("\n")
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError) as exc:
        temporary.unlink(missing_ok=True)
        raise ManualTestError(f"Không thể ghi kết quả {path}: {exc}") from exc


def run_manual_test(
    questions_path: Path | str,
    business_data_root: Path | str,
    planner: QueryPlanner,
    results_path: Path | str,
    report_path: Path | str,
    *,
    rulebook_path: Path | None = None,
    calendar_path: Path | None = None,
) -> dict[str, object]:
    """Run all manual questions and persist predictions plus an aggregate report."""
    questions = Path(questions_path).resolve()
    business_root = Path(business_data_root).resolve()
    results = Path(results_path).resolve()
    report = Path(report_path).resolve()
    if not questions.is_file():
        raise ManualTestError(f"Không tìm thấy file câu hỏi: {questions}")
    if not business_root.is_dir():
        raise ManualTestError(f"Không tìm thấy thư mục dữ liệu: {business_root}")
    if len({questions, results, report}) != 3:
        raise ManualTestError("questions, results và report phải là các file khác nhau")
    for destination in (results, report):
        if destination.is_relative_to(business_root):
            raise ManualTestError(
                f"Không ghi kết quả vào thư mục dữ liệu doanh nghiệp: {destination}"
            )

    entries = _read_input(questions)
    context_cache: dict[str, BusinessContextDocuments] = {}
    output_records: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="datamate-manual-test-") as temp_dir:
        run_log = Path(temp_dir) / "pipeline-runs.jsonl"
        for line_number, raw_record, validation_error in entries:
            if validation_error is not None:
                record = raw_record if isinstance(raw_record, dict) else {}
                expected = record.get("expected_answer")
                output_records.append(
                    _jsonable(
                        _failure_record(
                            question_id=record.get("id"),
                            question=record.get("question"),
                            expected_answer=expected,
                            error=ManualTestError(
                                f"Dòng {line_number}: {validation_error}"
                            ),
                            elapsed_ms=0.0,
                        )
                    )
                )
                output_records[-1]["error_category"] = "Định dạng đầu vào"
                try:
                    output_records[-1]["expected_answer_valid"] = (
                        None
                        if expected is None
                        else _validate_expected_answer(expected) is not None
                    )
                except ManualTestError:
                    output_records[-1]["expected_answer_valid"] = False
                continue
            assert isinstance(raw_record, dict)
            output_records.append(
                _run_one(
                    raw_record,
                    business_root=business_root,
                    planner=planner,
                    run_log=run_log,
                    input_root=business_root,
                    rulebook_path=rulebook_path,
                    calendar_path=calendar_path,
                    context_cache=context_cache,
                )
            )
    summary = _summary(output_records)
    _write_jsonl_atomic(results, output_records)
    report.parent.mkdir(parents=True, exist_ok=True)
    report_temp = report.with_name(f".{report.name}.tmp")
    try:
        report_temp.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(report_temp, report)
    except OSError as exc:
        report_temp.unlink(missing_ok=True)
        raise ManualTestError(f"Không thể ghi báo cáo {report}: {exc}") from exc
    return summary
