"""Connect business data discovery, plan validation, execution, and answer checks."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Mapping

from .answer_validation import (
    AnswerValidationError,
    Evidence,
    ValidatedAnswer,
    validate_answer,
)
from .data_loader import BusinessDataLoader, DataLoadError, TableRef
from .normalization import (
    ColumnCoercion,
    ValueNormalizationError,
    normalize_rows,
)
from .question_understanding import (
    QueryPlanner,
    QuestionContext,
    QuestionUnderstandingError,
)
from .query_executor import ExecutionResult, QueryExecutionError, execute_query_plan
from .query_plan import ColumnRef, PlanValidationError, QueryPlan, TimeRange
from .run_logging import RunRecord, append_run_record, validate_log_destination


class PipelineError(RuntimeError):
    """Raised when an execution result cannot be mapped to a typed answer."""


_DEFAULT_MAX_JOIN_ROWS = 100_000
_DEFAULT_MAX_GROUPS = 100_000
_DEFAULT_MAX_RESULT_ROWS = 10_000
_DEFAULT_MAX_GROUP_EXCLUSIONS = 100_000


@dataclass(frozen=True)
class BusinessSchema:
    tables: dict[str, TableRef]
    columns: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class PipelineResult:
    answer: ValidatedAnswer
    execution: ExecutionResult | None
    plan: QueryPlan


def table_identifier(table: TableRef) -> str:
    """Return a stable table key unique within one business directory."""
    discriminator = table.sheet_name or table.sqlite_table or table.name
    return f"{table.relative_path}::{discriminator}"


def discover_business_schema(business_dir: Path | str) -> BusinessSchema:
    loader = BusinessDataLoader(business_dir)
    tables = {table_identifier(table): table for table in loader.discover()}
    return BusinessSchema(
        tables=tables,
        columns={identifier: table.columns for identifier, table in tables.items()},
    )


def _answer_value(
    result: ExecutionResult,
    answer_type: str,
    plan: QueryPlan,
) -> object:
    if plan.count_groups:
        if len(result.rows) != 1 or "group_count" not in result.rows[0]:
            raise PipelineError("group-count result is unavailable")
        return result.rows[0]["group_count"]
    if answer_type == "table":
        rows = list(result.rows)
        if plan.selected_columns:
            output_keys = [
                f"{ref.table}.{ref.column}"
                for ref in plan.selected_columns
            ]
            output_keys.extend(bucket.alias for bucket in plan.time_buckets)
            output_keys.extend(item.alias for item in plan.aggregations)
            output_keys.extend(item.alias for item in plan.calculations)
            try:
                return [
                    {key: row[key] for key in output_keys}
                    for row in rows
                ]
            except KeyError as exc:
                raise PipelineError(
                    f"table output column is unavailable: {exc.args[0]}"
                ) from exc
        return rows
    if answer_type == "list":
        if plan.selected_columns:
            if len(plan.selected_columns) != 1:
                raise PipelineError("list output requires exactly one selected column")
            key = (
                f"{plan.selected_columns[0].table}."
                f"{plan.selected_columns[0].column}"
            )
        elif len(plan.aggregations) == 1 and not plan.calculations:
            key = plan.aggregations[0].alias
        else:
            key = None
        if key is not None:
            try:
                return [row[key] for row in result.rows]
            except KeyError as exc:
                raise PipelineError(
                    f"list output column is unavailable: {key}"
                ) from exc
        if any(len(row) != 1 for row in result.rows):
            raise PipelineError("list output requires exactly one result column")
        return [next(iter(row.values())) for row in result.rows]
    if answer_type in {"date", "string", "boolean"} and len(result.rows) == 1:
        if len(plan.selected_columns) == 1:
            ref = plan.selected_columns[0]
            key = f"{ref.table}.{ref.column}"
        elif not plan.selected_columns and len(plan.time_buckets) == 1:
            key = plan.time_buckets[0].alias
        else:
            key = None
        if key is not None:
            try:
                return result.rows[0][key]
            except KeyError as exc:
                raise PipelineError(
                    f"scalar output column is unavailable: {key}"
                ) from exc
    if plan.calculations:
        if len(result.rows) != 1:
            raise PipelineError(
                f"{answer_type} output requires exactly one scalar result"
            )
        key = plan.calculations[-1].alias
        try:
            return result.rows[0][key]
        except KeyError as exc:
            raise PipelineError(
                f"scalar calculation output is unavailable: {key}"
            ) from exc
    if len(result.rows) != 1 or len(result.rows[0]) != 1:
        raise PipelineError(
            f"{answer_type} output requires exactly one scalar result"
        )
    return next(iter(result.rows[0].values()))


def _range_outside_coverage(
    time_range: TimeRange,
    execution: ExecutionResult,
) -> bool:
    column = time_range.column
    key = f"{column.table}.{column.column}"
    bounds = execution.time_coverage.get(key)
    if bounds is None or bounds[0] is None or bounds[1] is None:
        return True
    start = date.fromisoformat(time_range.start)
    end = date.fromisoformat(time_range.end)
    return end < bounds[0] or start > bounds[1]


def _evidence_for(plan: QueryPlan, execution: ExecutionResult) -> tuple[Evidence, ...]:
    references: set[ColumnRef] = {
        *plan.selected_columns,
        *plan.group_by,
        *plan.evidence_requirements,
    }
    references.update(
        aggregation.column
        for aggregation in plan.aggregations
        if aggregation.column is not None
    )
    references.update(item.column for item in plan.filters)
    references.update(bucket.column for bucket in plan.time_buckets)
    for aggregation in plan.aggregations:
        references.update(item.column for item in aggregation.filters)
        if aggregation.time_range is not None:
            references.add(aggregation.time_range.column)
        if aggregation.unit_conversion is not None:
            references.update(
                (
                    aggregation.unit_conversion.source_unit,
                    aggregation.unit_conversion.base_unit,
                    aggregation.unit_conversion.large_unit,
                    aggregation.unit_conversion.factor,
                )
            )
    for exclusion in plan.group_exclusions:
        references.add(exclusion.key)
        references.update(item.column for item in exclusion.filters)
    if plan.time_range is not None:
        references.add(plan.time_range.column)
    for join in plan.joins:
        references.update((join.left, join.right))

    evidence: list[Evidence] = []
    filter_count = len(plan.filters) + sum(
        len(aggregation.filters) for aggregation in plan.aggregations
    ) + sum(len(exclusion.filters) for exclusion in plan.group_exclusions)
    has_time_range = plan.time_range is not None or any(
        aggregation.time_range is not None for aggregation in plan.aggregations
    )
    has_time_bucket = bool(plan.time_buckets)
    for table in execution.trace.source_tables:
        columns = tuple(
            sorted(ref.column for ref in references if ref.table == table)
        )
        filter_summary = (
            f"{filter_count} filters; "
            f"time_range={'yes' if has_time_range else 'no'}"
        )
        if has_time_bucket:
            filter_summary += "; time_bucket=yes"
        if plan.group_exclusions:
            filter_summary += (
                f"; excluded_groups={len(plan.group_exclusions)}"
            )
        conversion_targets = sorted(
            {
                aggregation.unit_conversion.target_unit or "product_base_unit"
                for aggregation in plan.aggregations
                if aggregation.unit_conversion is not None
            }
        )
        if conversion_targets:
            filter_summary += (
                "; unit_conversion_to=" + ",".join(conversion_targets)
            )
        evidence.append(
            Evidence(
                source_table=table,
                columns=columns,
                operation="validated_query_plan",
                row_count=execution.trace.source_rows.get(table, 0),
                filter_summary=filter_summary,
            )
        )
    return tuple(evidence)


def execute_business_plan(
    business_dir: Path | str,
    plan: QueryPlan,
    *,
    max_join_rows: int = _DEFAULT_MAX_JOIN_ROWS,
    max_groups: int = _DEFAULT_MAX_GROUPS,
    max_result_rows: int = _DEFAULT_MAX_RESULT_ROWS,
    max_group_exclusions: int = _DEFAULT_MAX_GROUP_EXCLUSIONS,
    column_coercions: Mapping[ColumnRef, ColumnCoercion] | None = None,
) -> PipelineResult:
    """Discover the local schema and execute a plan using only approved readers."""
    loader = BusinessDataLoader(business_dir)
    discovered = loader.discover()
    tables = {table_identifier(table): table for table in discovered}
    if plan.output_type in {"unanswerable", "abstain"}:
        raise PipelineError(
            "Non-answer statuses must be returned through an explicit abstention path"
        )
    if plan.output_type is None:
        raise PipelineError("Query plan does not declare an output type")
    involved_tables = {plan.base_table}
    for join in plan.joins:
        involved_tables.update((join.left.table, join.right.table))
    catalog = {
        identifier: tables[identifier].columns
        for identifier in involved_tables
        if identifier in tables
    }
    policies_by_table: dict[str, dict[str, ColumnCoercion]] = {}
    for ref, policy in (column_coercions or {}).items():
        if ref.table not in tables or ref.column not in tables[ref.table].columns:
            raise PipelineError(
                f"Coercion references unknown column {ref.table}.{ref.column}"
            )
        if ref.table not in involved_tables:
            raise PipelineError(
                f"Coercion references a table outside the plan: {ref.table}"
            )
        policies_by_table.setdefault(ref.table, {})[ref.column] = policy

    sources = {}
    for identifier in involved_tables:
        if identifier not in tables:
            continue
        table = tables[identifier]
        policies = policies_by_table.get(identifier, {})
        sources[identifier] = lambda table=table, policies=policies: normalize_rows(
            loader.iter_rows(table),
            table.columns,
            policies,
        )
    execution = execute_query_plan(
        plan,
        catalog=catalog,
        sources=sources,
        max_join_rows=max_join_rows,
        max_groups=max_groups,
        max_result_rows=max_result_rows,
        max_group_exclusions=max_group_exclusions,
    )

    requested_ranges = [
        aggregation.time_range
        for aggregation in plan.aggregations
        if aggregation.time_range is not None
    ]
    if plan.time_range is not None:
        requested_ranges.append(plan.time_range)
    outside_coverage = bool(requested_ranges) and all(
        _range_outside_coverage(time_range, execution)
        for time_range in requested_ranges
    )

    if outside_coverage:
        value = None
    elif plan.count_groups:
        value = _answer_value(execution, plan.output_type, plan)
    elif execution.trace.filtered_rows == 0:
        has_scalar_count = (
            plan.output_type == "number"
            and not plan.group_by
            and not plan.time_buckets
            and not plan.calculations
            and any(
                item.function in {"count", "count_distinct"}
                for item in plan.aggregations
            )
        )
        has_scoped_sum = (
            plan.output_type == "number"
            and not plan.group_by
            and not plan.time_buckets
            and not plan.calculations
            and len(plan.aggregations) == 1
            and plan.aggregations[0].function == "sum"
            and bool(requested_ranges)
        )
        if has_scoped_sum:
            value = Decimal(0)
        elif has_scalar_count or plan.output_type in {"list", "table"}:
            value = _answer_value(execution, plan.output_type, plan)
        else:
            value = None
    else:
        value = _answer_value(execution, plan.output_type, plan)
    if value is None:
        no_rows_matched = outside_coverage or execution.trace.filtered_rows == 0
        answer = ValidatedAnswer(
            answer_type="unanswerable",
            value=None,
            status="no_data" if no_rows_matched else "insufficient_data",
            explanation=(
                "Không có dòng dữ liệu phù hợp với điều kiện truy vấn."
                if no_rows_matched
                else "Dữ liệu phù hợp không có giá trị cần thiết để tính."
            ),
            evidence=_evidence_for(plan, execution),
        )
    else:
        answer = ValidatedAnswer(
            answer_type=plan.output_type,
            value=value,
            status="answered",
            evidence=_evidence_for(plan, execution),
        )
    validate_answer(answer)
    return PipelineResult(answer=answer, execution=execution, plan=plan)


def abstention_result(plan: QueryPlan) -> PipelineResult:
    """Validate a planner-provided abstention without executing business data."""
    status = plan.metadata.get("status")
    reason = plan.metadata.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise PipelineError("Abstention plan requires a non-empty reason")
    if plan.output_type == "abstain":
        if status != "abstained":
            raise PipelineError("Abstain plan requires status 'abstained'")
        answer_status = "abstained"
    elif plan.output_type == "unanswerable":
        if status not in {"unanswerable", "ambiguous", "insufficient_data"}:
            raise PipelineError("Invalid unanswerable plan status")
        answer_status = status
    else:
        raise PipelineError("Plan is not an abstention plan")
    answer = ValidatedAnswer(
        answer_type=plan.output_type,
        value=None,
        status=answer_status,
        explanation=reason,
    )
    validate_answer(answer)
    return PipelineResult(answer=answer, execution=None, plan=plan)


def execute_and_log_business_plan(
    business_dir: Path | str,
    plan: QueryPlan,
    *,
    question_id: str,
    business_id: str,
    question: str,
    model: str,
    log_path: Path | str,
    input_root: Path | str,
    input_metadata: dict[str, object] | None = None,
    column_coercions: Mapping[ColumnRef, ColumnCoercion] | None = None,
) -> PipelineResult:
    """Execute one prepared plan and append success or known failure to JSONL."""
    validate_log_destination(log_path, forbidden_roots=(input_root,))
    started = time.perf_counter()
    try:
        result = execute_business_plan(
            business_dir,
            plan,
            column_coercions=column_coercions,
        )
    except (
        DataLoadError,
        PlanValidationError,
        QueryExecutionError,
        ValueNormalizationError,
        PipelineError,
        AnswerValidationError,
    ) as exc:
        append_run_record(
            log_path,
            RunRecord(
                question_id=question_id,
                business_id=business_id,
                question=question,
                model=model,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                plan=plan,
                error=f"{type(exc).__name__}: {exc}",
                input_metadata=_log_metadata(
                    input_metadata,
                    column_coercions,
                    execution_limits={
                        "max_join_rows": _DEFAULT_MAX_JOIN_ROWS,
                        "max_groups": _DEFAULT_MAX_GROUPS,
                        "max_result_rows": _DEFAULT_MAX_RESULT_ROWS,
                        "max_group_exclusions": _DEFAULT_MAX_GROUP_EXCLUSIONS,
                    },
                ),
            ),
            forbidden_roots=(input_root,),
        )
        raise

    append_run_record(
        log_path,
        RunRecord(
            question_id=question_id,
            business_id=business_id,
            question=question,
            model=model,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            plan=plan,
            execution=result.execution.trace,
            answer=result.answer,
            input_metadata=_log_metadata(
                input_metadata,
                column_coercions,
                execution_limits={
                    "max_join_rows": _DEFAULT_MAX_JOIN_ROWS,
                    "max_groups": _DEFAULT_MAX_GROUPS,
                    "max_result_rows": _DEFAULT_MAX_RESULT_ROWS,
                    "max_group_exclusions": _DEFAULT_MAX_GROUP_EXCLUSIONS,
                },
            ),
        ),
        forbidden_roots=(input_root,),
    )
    return result


def execute_question_and_log(
    business_dir: Path | str,
    question: str,
    planner: QueryPlanner,
    *,
    question_id: str,
    business_id: str,
    as_of_date: str | None,
    log_path: Path | str,
    input_root: Path | str,
    business_rules: tuple[str, ...] = (),
    rulebook: str | None = None,
    business_readme: str | None = None,
    calendar_columns: tuple[str, ...] = (),
    calendar_rows: tuple[Mapping[str, str], ...] = (),
    input_metadata: dict[str, object] | None = None,
    column_coercions: Mapping[ColumnRef, ColumnCoercion] | None = None,
    max_join_rows: int = _DEFAULT_MAX_JOIN_ROWS,
    max_groups: int = _DEFAULT_MAX_GROUPS,
    max_result_rows: int = _DEFAULT_MAX_RESULT_ROWS,
) -> PipelineResult:
    """Run question planning, deterministic execution, validation, and logging."""
    validate_log_destination(log_path, forbidden_roots=(input_root,))
    started = time.perf_counter()
    plan: QueryPlan | None = None
    metadata = dict(input_metadata or {})
    if as_of_date is not None:
        metadata["as_of_date"] = as_of_date
    execution_limits = {
        "max_join_rows": max_join_rows,
        "max_groups": max_groups,
        "max_result_rows": max_result_rows,
        "max_group_exclusions": _DEFAULT_MAX_GROUP_EXCLUSIONS,
    }
    metadata = _log_metadata(
        metadata,
        column_coercions,
        execution_limits=execution_limits,
    )
    try:
        schema = discover_business_schema(business_dir)
        context = QuestionContext(
            schema=schema.columns,
            as_of_date=as_of_date,
            business_rules=business_rules,
            rulebook=rulebook,
            business_readme=business_readme,
            calendar_columns=calendar_columns,
            calendar_rows=calendar_rows,
        )
        plan = planner.create_plan(question, context)
        if plan.output_type in {"unanswerable", "abstain"}:
            result = abstention_result(plan)
        else:
            result = execute_business_plan(
                business_dir,
                plan,
                max_join_rows=max_join_rows,
                max_groups=max_groups,
                max_result_rows=max_result_rows,
                column_coercions=column_coercions,
            )
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
        append_run_record(
            log_path,
            RunRecord(
                question_id=question_id,
                business_id=business_id,
                question=question,
                model=planner.model_name,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                plan=plan,
                error=f"{type(exc).__name__}: {exc}",
                input_metadata=metadata,
            ),
            forbidden_roots=(input_root,),
        )
        raise

    append_run_record(
        log_path,
        RunRecord(
            question_id=question_id,
            business_id=business_id,
            question=question,
            model=planner.model_name,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            plan=plan,
            execution=(
                result.execution.trace if result.execution is not None else None
            ),
            answer=result.answer,
            input_metadata=metadata,
        ),
        forbidden_roots=(input_root,),
    )
    return result


def _log_metadata(
    metadata: dict[str, object] | None,
    column_coercions: Mapping[ColumnRef, ColumnCoercion] | None,
    *,
    execution_limits: Mapping[str, int],
) -> dict[str, object]:
    result = dict(metadata or {})
    if column_coercions:
        result["column_coercions"] = {
            f"{ref.table}.{ref.column}": asdict(policy)
            for ref, policy in column_coercions.items()
        }
    result["execution_limits"] = dict(execution_limits)
    return result
