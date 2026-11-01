"""Reproducible, append-only JSONL records for pipeline executions."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Iterator, Mapping

from .answer_validation import ValidatedAnswer
from .query_executor import ExecutionTrace
from .query_plan import QueryPlan


class RunLogError(RuntimeError):
    """Raised when a run record cannot be safely written or read."""


@dataclass(frozen=True)
class RunRecord:
    question_id: str
    business_id: str
    question: str
    model: str
    elapsed_ms: float
    plan: QueryPlan | None = None
    execution: ExecutionTrace | None = None
    answer: ValidatedAnswer | None = None
    confidence: float | None = None
    error: str | None = None
    input_metadata: Mapping[str, object] | None = None
    timestamp: str = ""


def _json_default(value: object) -> object:
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("Non-finite Decimal cannot be logged")
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, set | frozenset):
        return sorted(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def _record_data(record: RunRecord) -> dict[str, object]:
    if not record.question_id or not record.business_id or not record.model:
        raise RunLogError("question_id, business_id, and model are required")
    if not math.isfinite(record.elapsed_ms) or record.elapsed_ms < 0:
        raise RunLogError("elapsed_ms must be a finite non-negative number")
    if record.confidence is not None and (
        not math.isfinite(record.confidence) or not 0 <= record.confidence <= 1
    ):
        raise RunLogError("confidence must be between 0 and 1")
    answer = asdict(record.answer) if record.answer is not None else None
    plan = asdict(record.plan) if record.plan is not None else None
    execution = asdict(record.execution) if record.execution is not None else None
    return {
        "schema_version": 1,
        "timestamp": record.timestamp or datetime.now(timezone.utc).isoformat(),
        "question_id": record.question_id,
        "business_id": record.business_id,
        "input": {
            "question": record.question,
            "metadata": dict(record.input_metadata or {}),
        },
        "model": record.model,
        "query_plan": plan,
        "execution_trace": execution,
        "evidence": answer["evidence"] if answer is not None else [],
        "answer": answer,
        "answer_type": answer["answer_type"] if answer is not None else None,
        "confidence": record.confidence,
        "elapsed_ms": record.elapsed_ms,
        "error": record.error,
    }


def validate_log_destination(
    log_path: Path | str,
    *,
    forbidden_roots: tuple[Path | str, ...],
) -> Path:
    if not forbidden_roots:
        raise RunLogError("At least one input root is required to protect source data")
    destination = Path(log_path).resolve()
    for root in forbidden_roots:
        resolved_root = Path(root).resolve()
        if destination == resolved_root or destination.is_relative_to(resolved_root):
            raise RunLogError(
                f"Run log must not be written under input root: {resolved_root}"
            )
    return destination


def append_run_record(
    log_path: Path | str,
    record: RunRecord,
    *,
    forbidden_roots: tuple[Path | str, ...],
) -> None:
    """Append one complete JSON object, refusing writes under input data roots."""
    destination = validate_log_destination(
        log_path, forbidden_roots=forbidden_roots
    )
    data = _record_data(record)
    try:
        encoded = json.dumps(
            data,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            default=_json_default,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.write("\n")
    except (OSError, TypeError, ValueError) as exc:
        raise RunLogError(f"Cannot append run log {destination}: {exc}") from exc


def iter_run_records(log_path: Path | str) -> Iterator[dict[str, object]]:
    """Đọc và kiểm tra tuần tự từng bản ghi JSONL; không bỏ qua dòng lỗi."""
    source = Path(log_path)
    try:
        with source.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise RunLogError(f"Blank run-log line at {source}:{line_number}")
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RunLogError(
                        f"Invalid JSON at {source}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(value, dict) or value.get("schema_version") != 1:
                    raise RunLogError(
                        f"Unsupported run-log record at {source}:{line_number}"
                    )
                yield value
    except OSError as exc:
        raise RunLogError(f"Cannot read run log {source}: {exc}") from exc


def read_run_records(log_path: Path | str) -> list[dict[str, object]]:
    """Read and validate every JSONL record; malformed lines are never skipped."""
    return list(iter_run_records(log_path))
