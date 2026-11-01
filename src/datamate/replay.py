"""Deterministic replay of one successful pipeline record."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

from .answer_validation import AnswerValidationError
from .data_loader import DataLoadError
from .normalization import ColumnCoercion, ValueNormalizationError
from .pipeline import (
    PipelineError,
    PipelineResult,
    abstention_result,
    execute_business_plan,
)
from .query_executor import QueryExecutionError
from .query_plan import ColumnRef, PlanValidationError
from .plan_serialization import QueryPlanFormatError, query_plan_from_dict
from .run_logging import _json_default, iter_run_records

_LIMIT_NAMES = {
    "max_join_rows",
    "max_groups",
    "max_result_rows",
    "max_group_exclusions",
}
_COERCION_FIELDS = {
    "kind",
    "decimal_separator",
    "thousands_separator",
    "date_format",
    "fill_down_group",
    "deduplicate_group",
    "aliases",
}
_COERCION_KINDS = {
    "number",
    "integer",
    "date",
    "datetime",
    "boolean",
    "text",
    "identifier",
}


class ReplayError(RuntimeError):
    """Raised when a run record cannot be safely and faithfully replayed."""


def _business_directory(root: Path, business_id: object) -> Path:
    if (
        not isinstance(business_id, str)
        or not business_id
        or Path(business_id).name != business_id
        or business_id in {".", ".."}
    ):
        raise ReplayError("Run record has an unsafe business_id")
    business_dir = (root / business_id).resolve()
    if not business_dir.is_relative_to(root) or not business_dir.is_dir():
        raise ReplayError(f"Business directory is missing or unsafe: {business_id}")
    return business_dir


def _record_coercions(record: Mapping[str, object]) -> dict[ColumnRef, ColumnCoercion]:
    input_data = record.get("input")
    if not isinstance(input_data, dict):
        raise ReplayError("Run record has no valid input object")
    metadata = input_data.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ReplayError("Run record input metadata must be an object")
    raw_coercions = metadata.get("column_coercions", {})
    if not isinstance(raw_coercions, dict):
        raise ReplayError("Logged column coercions must be an object")

    coercions: dict[ColumnRef, ColumnCoercion] = {}
    for key, raw_policy in raw_coercions.items():
        if not isinstance(key, str) or not isinstance(raw_policy, dict):
            raise ReplayError("Invalid logged column coercion")
        table, separator, column = key.rpartition(".")
        if not separator or not table or not column:
            raise ReplayError(f"Invalid logged column reference: {key!r}")
        if (
            set(raw_policy) - _COERCION_FIELDS
            or raw_policy.get("kind") not in _COERCION_KINDS
        ):
            raise ReplayError(f"Invalid logged coercion policy for {key}")
        try:
            coercions[ColumnRef(table, column)] = ColumnCoercion(**raw_policy)
        except (TypeError, ValueError) as exc:
            raise ReplayError(
                f"Invalid logged coercion policy for {key}: {exc}"
            ) from exc
    return coercions


def _execution_limits(record: Mapping[str, object]) -> dict[str, int]:
    input_data = record.get("input")
    metadata = input_data.get("metadata", {}) if isinstance(input_data, dict) else {}
    if not isinstance(metadata, dict):
        raise ReplayError("Run record input metadata must be an object")
    raw_limits = metadata.get("execution_limits")
    if raw_limits is None:
        raise ReplayError(
            "Run record lacks execution limits; reliable replay is not possible"
        )
    if not isinstance(raw_limits, dict) or set(raw_limits) != _LIMIT_NAMES:
        raise ReplayError("Logged execution limits are incomplete or invalid")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in raw_limits.values()
    ):
        raise ReplayError("Logged execution limits must be positive integers")
    return {name: raw_limits[name] for name in sorted(_LIMIT_NAMES)}


def _find_record(
    log_path: Path,
    question_id: str,
    occurrence: int | None,
) -> dict[str, object]:
    matching = [
        record
        for record in iter_run_records(log_path)
        if record.get("question_id") == question_id
    ]
    if not matching:
        raise ReplayError(f"Question id not found in run log: {question_id}")
    if occurrence is None and len(matching) != 1:
        raise ReplayError(
            f"Found {len(matching)} records for {question_id}; select --occurrence"
        )
    index = 0 if occurrence is None else occurrence
    if index < 0 or index >= len(matching):
        raise ReplayError(
            f"Occurrence {index} is outside the {len(matching)} matching records"
        )
    return matching[index]


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
            default=_json_default,
        )
    except (TypeError, ValueError) as exc:
        raise ReplayError(f"Cannot compare replay values: {exc}") from exc


def replay_run_record(
    log_path: Path | str,
    question_id: str,
    business_data_root: Path | str,
    *,
    occurrence: int | None = None,
) -> dict[str, object]:
    """Re-execute one logged plan and compare its answer and trace."""
    root = Path(business_data_root).resolve()
    if not root.is_dir():
        raise ReplayError("Business-data root must be an existing directory")
    record = _find_record(Path(log_path), question_id, occurrence)
    if record.get("error") is not None:
        raise ReplayError("Failed runs cannot be replayed without the planner")
    raw_plan = record.get("query_plan")
    if not isinstance(raw_plan, dict):
        raise ReplayError("Run record does not contain a query plan")
    try:
        plan = query_plan_from_dict(raw_plan)
    except QueryPlanFormatError as exc:
        raise ReplayError(f"Logged query plan is invalid: {exc}") from exc

    business_dir = _business_directory(root, record.get("business_id"))
    coercions = _record_coercions(record)
    limits = _execution_limits(record)
    result: PipelineResult
    try:
        if plan.output_type in {"unanswerable", "abstain"}:
            result = abstention_result(plan)
            execution_trace = None
        else:
            result = execute_business_plan(
                business_dir,
                plan,
                column_coercions=coercions,
                **limits,
            )
            execution_trace = (
                asdict(result.execution.trace)
                if result.execution is not None
                else None
            )
    except (
        AnswerValidationError,
        DataLoadError,
        PipelineError,
        PlanValidationError,
        QueryExecutionError,
        ValueNormalizationError,
    ) as exc:
        raise ReplayError(
            f"Replay execution failed ({type(exc).__name__}): {exc}"
        ) from exc

    replayed_answer = asdict(result.answer)
    recorded_answer = record.get("answer")
    recorded_trace = record.get("execution_trace")
    answer_matches = _canonical_json(recorded_answer) == _canonical_json(
        replayed_answer
    )
    trace_matches = _canonical_json(recorded_trace) == _canonical_json(
        execution_trace
    )
    return {
        "question_id": question_id,
        "business_id": record.get("business_id"),
        "recorded_model": record.get("model"),
        "answer_matches": answer_matches,
        "execution_trace_matches": trace_matches,
        "replay_matches": answer_matches and trace_matches,
        "recorded_answer": recorded_answer,
        "replayed_answer": replayed_answer,
        "recorded_execution_trace": recorded_trace,
        "replayed_execution_trace": execution_trace,
    }
