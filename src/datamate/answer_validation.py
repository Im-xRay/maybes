"""Validation for typed answers produced by the deterministic pipeline."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Literal

AnswerType = Literal[
    "number",
    "percent",
    "string",
    "date",
    "boolean",
    "list",
    "table",
    "unanswerable",
    "abstain",
]
AnswerStatus = Literal[
    "answered",
    "no_data",
    "unanswerable",
    "ambiguous",
    "insufficient_data",
    "abstained",
]

_ANSWER_TYPES = {
    "number",
    "percent",
    "string",
    "date",
    "boolean",
    "list",
    "table",
    "unanswerable",
    "abstain",
}
_NON_ANSWER_STATUSES = {
    "no_data",
    "unanswerable",
    "ambiguous",
    "insufficient_data",
    "abstained",
}


@dataclass(frozen=True)
class Evidence:
    """A verifiable reference to the plan and rows supporting an answer."""

    source_table: str
    columns: tuple[str, ...]
    operation: str
    row_count: int
    filter_summary: str = ""


@dataclass(frozen=True)
class ValidatedAnswer:
    answer_type: AnswerType
    value: object
    status: AnswerStatus
    evidence: tuple[Evidence, ...] = ()
    explanation: str | None = None


class AnswerValidationError(ValueError):
    """Raised when an internal answer cannot safely be emitted."""


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return False
    if isinstance(value, Decimal):
        return value.is_finite()
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _is_valid_date(value: object) -> bool:
    if isinstance(value, datetime):
        return True
    if isinstance(value, date):
        return True
    if not isinstance(value, str):
        return False
    if re.fullmatch(r"\d{4}-\d{2}", value):
        try:
            year, month = (int(part) for part in value.split("-"))
            date(year, month, 1)
            return True
        except ValueError:
            return False
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _is_json_value(value: object) -> bool:
    if value is None or isinstance(value, (str, bool, int, Decimal, date, datetime)):
        return not isinstance(value, Decimal) or value.is_finite()
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_json_value(item)
            for key, item in value.items()
        )
    return False


def validate_answer(answer: ValidatedAnswer) -> None:
    """Validate type, status, value shape, and evidence before serialization."""
    issues: list[str] = []
    if answer.answer_type not in _ANSWER_TYPES:
        issues.append(f"unsupported answer type: {answer.answer_type}")
    if answer.status not in _NON_ANSWER_STATUSES | {"answered"}:
        issues.append(f"unsupported answer status: {answer.status}")

    if answer.status == "answered":
        if answer.answer_type in {"unanswerable", "abstain"}:
            issues.append("answered status cannot use unanswerable or abstain type")
        if not answer.evidence:
            issues.append("answerable result requires evidence")
    elif answer.answer_type not in {"unanswerable", "abstain"}:
        issues.append("non-answer status must use unanswerable or abstain type")
    elif answer.answer_type == "abstain" and answer.status != "abstained":
        issues.append("abstain type requires abstained status")
    elif answer.answer_type == "unanswerable" and answer.status == "abstained":
        issues.append("unanswerable type cannot use abstained status")

    if answer.answer_type in {"number", "percent"}:
        if not _is_finite_number(answer.value):
            issues.append(f"{answer.answer_type} value must be a finite number")
    elif answer.answer_type == "string" and not isinstance(answer.value, str):
        issues.append("string value must be text")
    elif answer.answer_type == "date" and not _is_valid_date(answer.value):
        issues.append("date value must be a date or ISO-8601 date string")
    elif answer.answer_type == "boolean" and not isinstance(answer.value, bool):
        issues.append("boolean value must be true or false")
    elif answer.answer_type == "list" and (
        not isinstance(answer.value, list) or not _is_json_value(answer.value)
    ):
        issues.append("list value must be a JSON-serializable array")
    elif answer.answer_type == "table" and (
        not isinstance(answer.value, list)
        or any(not isinstance(row, dict) for row in answer.value)
        or not _is_json_value(answer.value)
    ):
        issues.append("table value must be a JSON-serializable array of objects")
    elif answer.answer_type in {"unanswerable", "abstain"} and answer.value is not None:
        issues.append(f"{answer.answer_type} value must be null")

    for item in answer.evidence:
        if not item.source_table:
            issues.append("evidence source table must not be empty")
        if not item.operation:
            issues.append("evidence operation must not be empty")
        if (
            isinstance(item.row_count, bool)
            or not isinstance(item.row_count, int)
            or item.row_count < 0
        ):
            issues.append("evidence row count must be a non-negative integer")
    if answer.explanation is not None and not isinstance(answer.explanation, str):
        issues.append("answer explanation must be text")

    if issues:
        raise AnswerValidationError("; ".join(issues))
