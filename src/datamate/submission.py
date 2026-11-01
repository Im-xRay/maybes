"""Đóng gói câu trả lời theo schema tham khảo trong training rulebook."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from .answer_validation import AnswerValidationError, ValidatedAnswer, validate_answer
from .query_plan import QueryPlan


class SubmissionError(ValueError):
    """Lỗi khi không thể tạo JSONL đầu ra đúng cấu trúc tham khảo."""


def _number_literal(value: int | float | Decimal) -> str:
    if isinstance(value, bool):
        raise SubmissionError("Giá trị số không được là boolean")
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise SubmissionError("Decimal không hữu hạn")
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise SubmissionError("Số thực không hữu hạn")
    return json.dumps(value, allow_nan=False)


def _encode_json(value: object) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float, Decimal)):
        return _number_literal(value)
    if isinstance(value, (date, datetime)):
        return json.dumps(value.isoformat(), ensure_ascii=False)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ",".join(_encode_json(item) for item in value) + "]"
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise SubmissionError("JSON object chỉ được có khóa dạng chuỗi")
        return (
            "{"
            + ",".join(
                f"{json.dumps(key, ensure_ascii=False)}:{_encode_json(item)}"
                for key, item in value.items()
            )
            + "}"
        )
    raise SubmissionError(f"Kiểu dữ liệu không thể xuất JSON: {type(value).__name__}")


def _date_value(value: object) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        raise SubmissionError("Đáp án date phải là chuỗi ISO-8601")
    try:
        if re.fullmatch(r"\d{4}-\d{2}", value):
            year, month = (int(part) for part in value.split("-"))
            date(year, month, 1)
            return value
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise SubmissionError("Đáp án date không hợp lệ") from exc
    raise SubmissionError("Đáp án date phải có dạng YYYY-MM hoặc YYYY-MM-DD")


def _table_column_labels(plan: QueryPlan | None) -> dict[str, str] | None:
    if plan is None:
        return None
    references = plan.selected_columns or plan.group_by
    keys = [f"{ref.table}.{ref.column}" for ref in references]
    labels = [ref.column for ref in references]
    aliases = [item.alias for item in plan.time_buckets]
    aliases.extend(item.alias for item in plan.aggregations)
    aliases.extend(item.alias for item in plan.calculations)
    duplicates = {label for label in labels if labels.count(label) > 1}
    labels = [
        key if label in duplicates or label in aliases else label
        for key, label in zip(keys, labels, strict=True)
    ]
    if len(set(labels + aliases)) != len(labels + aliases):
        raise SubmissionError("Cột đầu ra của table không có tên duy nhất")
    return dict(zip(keys + aliases, labels + aliases, strict=True))


def _table_value(
    value: object,
    columns_by_key: dict[str, str] | None,
) -> dict[str, object]:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise SubmissionError("Đáp án table phải là danh sách các dòng dạng object")
    if value:
        row_keys = list(value[0])
        if any(set(row) != set(row_keys) for row in value):
            raise SubmissionError("Các dòng trong table phải có cùng tập cột")
        if columns_by_key is not None and set(row_keys) != set(columns_by_key):
            raise SubmissionError("Cột trong table không khớp QueryPlan")
        columns = [
            columns_by_key.get(key, key) if columns_by_key else key
            for key in row_keys
        ]
        rows = [[row[key] for key in row_keys] for row in value]
    else:
        if not columns_by_key:
            raise SubmissionError(
                "Không thể suy ra tên cột cho table rỗng từ answer hiện có"
            )
        columns = list(columns_by_key.values())
        rows = []
    return {
        "columns": columns,
        "rows": rows,
    }


def serialize_submission_record(
    question_id: str,
    answer: ValidatedAnswer,
    *,
    plan: QueryPlan | None = None,
) -> str:
    """Tạo một dòng JSON theo schema rulebook, không tính metric PA."""
    if not isinstance(question_id, str) or not question_id:
        raise SubmissionError("question_id phải là chuỗi không rỗng")
    try:
        validate_answer(answer)
    except AnswerValidationError as exc:
        raise SubmissionError(f"Answer chưa hợp lệ: {exc}") from exc

    answer_type = answer.answer_type
    if answer_type == "abstain":
        record: dict[str, object] = {"id": question_id, "type": "abstain"}
    elif answer_type == "unanswerable":
        value: dict[str, object] = {"type": "unanswerable"}
        if answer.explanation is not None:
            value["reason"] = answer.explanation
        record = {
            "id": question_id,
            "type": "unanswerable",
            "answer": value,
        }
    else:
        value: object = answer.value
        if answer_type == "date":
            value = _date_value(value)
        elif answer_type == "table":
            value = _table_value(value, _table_column_labels(plan))
        record = {"id": question_id, "type": answer_type, "answer": value}
    return _encode_json(record)


class SubmissionWriter:
    """Ghi JSONL nguyên tử; lỗi giữa chừng không để lại submission hoàn chỉnh."""

    def __init__(self, destination: Path | str) -> None:
        self.destination = Path(destination).resolve()
        if self.destination.exists():
            raise SubmissionError(f"Tệp đầu ra đã tồn tại: {self.destination}")
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.destination.name}.",
            suffix=".tmp",
            dir=self.destination.parent,
        )
        self.temporary = Path(temporary_name)
        self.stream = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
        self.question_ids: set[str] = set()
        self.count = 0
        self.committed = False

    def append(
        self,
        question_id: str,
        answer: ValidatedAnswer,
        *,
        plan: QueryPlan | None = None,
    ) -> None:
        if self.committed or self.stream.closed:
            raise SubmissionError("SubmissionWriter đã đóng")
        if question_id in self.question_ids:
            raise SubmissionError(f"question_id bị trùng: {question_id}")
        line = serialize_submission_record(question_id, answer, plan=plan)
        self.stream.write(line + "\n")
        self.stream.flush()
        self.question_ids.add(question_id)
        self.count += 1

    def commit(self) -> None:
        if self.committed or self.stream.closed:
            raise SubmissionError("SubmissionWriter đã đóng")
        if self.count == 0:
            raise SubmissionError("Không thể tạo submission rỗng")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        if self.destination.exists():
            self.destination.unlink(missing_ok=True)
        os.replace(self.temporary, self.destination)
        self.committed = True

    def abort(self) -> None:
        if not self.stream.closed:
            self.stream.close()
        if not self.committed:
            self.temporary.unlink(missing_ok=True)

SubmissionWriter.write = SubmissionWriter.append
