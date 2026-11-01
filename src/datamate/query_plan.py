"""Typed, allow-listed query plans for deterministic execution."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Literal, Mapping

AggregationKind = Literal["sum", "count", "count_distinct", "avg", "min", "max"]
CalculationKind = Literal["difference", "ratio", "ratio_percent", "percent_change"]
TimeBucketKind = Literal["day", "week", "month", "quarter", "year"]
FilterOperator = Literal[
    "eq",
    "ne",
    "lt",
    "lte",
    "gt",
    "gte",
    "in",
    "between",
    "contains",
    "is_null",
    "not_null",
]
JoinKind = Literal["inner", "left"]
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

_FILTER_OPERATORS = {
    "eq",
    "ne",
    "lt",
    "lte",
    "gt",
    "gte",
    "in",
    "between",
    "contains",
    "is_null",
    "not_null",
}
_AGGREGATIONS = {"sum", "count", "count_distinct", "avg", "min", "max"}
_CALCULATIONS = {"difference", "ratio", "ratio_percent", "percent_change"}
_JOIN_KINDS = {"inner", "left"}
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


def _safe_filter_scalar(value: object) -> bool:
    if value is None or isinstance(value, (str, bool, int, date, datetime)):
        return True
    if isinstance(value, Decimal):
        return value.is_finite()
    if isinstance(value, float):
        return math.isfinite(value)
    return False


@dataclass(frozen=True)
class ColumnRef:
    """A column identified within a discovered source table."""

    table: str
    column: str


@dataclass(frozen=True)
class Filter:
    column: ColumnRef
    operator: FilterOperator
    value: object = None


@dataclass(frozen=True)
class Join:
    left: ColumnRef
    right: ColumnRef
    kind: JoinKind = "inner"


@dataclass(frozen=True)
class Aggregation:
    function: AggregationKind
    column: ColumnRef | None
    alias: str
    filters: tuple[Filter, ...] = ()
    time_range: TimeRange | None = None
    unit_conversion: UnitConversion | None = None


@dataclass(frozen=True)
class UnitConversion:
    """Convert recorded row quantities using a product's explicit unit mapping."""

    source_unit: ColumnRef
    base_unit: ColumnRef
    large_unit: ColumnRef
    factor: ColumnRef
    target_unit: str | None


@dataclass(frozen=True)
class GroupFilter:
    """A numeric predicate applied to an aggregate or calculation per group."""

    alias: str
    operator: Literal["eq", "ne", "lt", "lte", "gt", "gte"]
    value: object


@dataclass(frozen=True)
class GroupExclusion:
    """Exclude every row sharing a key when any source row matches the filters."""

    key: ColumnRef
    filters: tuple[Filter, ...]


@dataclass(frozen=True)
class Calculation:
    """A fixed arithmetic operation over named aggregate results."""

    function: CalculationKind
    left: str
    right: str
    alias: str
    rounding_places: int | None = None


@dataclass(frozen=True)
class Sort:
    """Sort by a source column or an aggregation alias."""

    key: ColumnRef | str
    direction: Literal["asc", "desc"] = "asc"


@dataclass(frozen=True)
class TimeRange:
    column: ColumnRef
    start: str
    end: str
    inclusive_end: bool = True


@dataclass(frozen=True)
class TimeBucket:
    column: ColumnRef
    granularity: TimeBucketKind
    alias: str


@dataclass(frozen=True)
class QueryPlan:
    """Declarative plan; it contains no SQL or executable code."""

    base_table: str | None
    joins: tuple[Join, ...] = ()
    selected_columns: tuple[ColumnRef, ...] = ()
    filters: tuple[Filter, ...] = ()
    group_by: tuple[ColumnRef, ...] = ()
    group_filters: tuple[GroupFilter, ...] = ()
    group_exclusions: tuple[GroupExclusion, ...] = ()
    aggregations: tuple[Aggregation, ...] = ()
    calculations: tuple[Calculation, ...] = ()
    count_groups: bool = False
    sort: tuple[Sort, ...] = ()
    limit: int | None = None
    time_range: TimeRange | None = None
    time_buckets: tuple[TimeBucket, ...] = ()
    business_rules: tuple[str, ...] = ()
    output_type: AnswerType | None = None
    evidence_requirements: tuple[ColumnRef, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)


class PlanValidationError(ValueError):
    """Raised when a plan refers to data or operations outside the catalog."""

    def __init__(self, issues: list[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("; ".join(issues))


def validate_query_plan(
    plan: QueryPlan,
    catalog: Mapping[str, set[str] | frozenset[str]],
    *,
    max_limit: int = 10_000,
) -> None:
    """Validate all table and column references before any execution."""
    issues: list[str] = []
    if max_limit < 1:
        raise ValueError("max_limit must be positive")
    if plan.business_rules:
        issues.append(
            "business_rules are not executable; encode supported logic as validated "
            "filters/calculations or abstain"
        )

    included_tables = {plan.base_table} if isinstance(plan.base_table, str) else set()
    for join in plan.joins:
        if join.kind not in _JOIN_KINDS:
            issues.append(f"unsupported join kind: {join.kind}")
        left_known = join.left.table in included_tables
        right_known = join.right.table in included_tables
        if not left_known and not right_known:
            issues.append("each join must connect to a table already in the plan")
        elif left_known and right_known:
            issues.append("each join must introduce exactly one new table")
        included_tables.update((join.left.table, join.right.table))
    if not isinstance(plan.base_table, str) or not plan.base_table:
        issues.append("base_table must be a non-empty string")
    elif plan.base_table not in catalog:
        issues.append(f"unknown base table: {plan.base_table}")

    def check_column(ref: ColumnRef, context: str) -> None:
        if ref.table not in catalog:
            issues.append(f"{context}: unknown table {ref.table}")
        elif ref.table not in included_tables:
            issues.append(f"{context}: table is not part of the plan: {ref.table}")
        elif ref.column not in catalog[ref.table]:
            issues.append(f"{context}: unknown column {ref.table}.{ref.column}")

    for join in plan.joins:
        check_column(join.left, "join")
        check_column(join.right, "join")
        if join.left.table == join.right.table:
            issues.append("join must connect two different table references")
    for aggregation in plan.aggregations:
        conversion = aggregation.unit_conversion
        if conversion is None:
            continue
        lookup_tables = {
            conversion.base_unit.table,
            conversion.large_unit.table,
            conversion.factor.table,
        }
        if (
            aggregation.column is not None
            and conversion.source_unit.table != aggregation.column.table
        ):
            issues.append(
                "unit conversion source_unit must use the aggregation column's table"
            )
        if len(lookup_tables) != 1:
            issues.append(
                "unit conversion lookup columns must come from the same table"
            )
        elif (
            next(iter(lookup_tables)) != conversion.source_unit.table
            and next(iter(lookup_tables)) == plan.base_table
        ):
            issues.append(
                "unit conversion lookup table must be joined to the plan, not used "
                "as its base table"
            )
        joined_tables = {plan.base_table}
        for join in plan.joins:
            new_table = (
                join.right.table
                if join.right.table not in joined_tables
                else join.left.table
            )
            if (
                new_table in lookup_tables
                and new_table != conversion.source_unit.table
                and join.kind != "left"
            ):
                issues.append(
                    "unit conversion lookup joins must be left joins so unmapped "
                    "base units are preserved"
                )
            joined_tables.update((join.left.table, join.right.table))

    for ref in plan.selected_columns:
        check_column(ref, "selected column")
    for ref in plan.group_by:
        check_column(ref, "group by")
    bucket_aliases: set[str] = set()
    for bucket in plan.time_buckets:
        check_column(bucket.column, f"time bucket {bucket.alias!r}")
        if bucket.granularity not in {"day", "week", "month", "quarter", "year"}:
            issues.append(
                f"unsupported time bucket granularity: {bucket.granularity}"
            )
        if not bucket.alias or bucket.alias in bucket_aliases:
            issues.append(f"time bucket alias is empty or duplicated: {bucket.alias!r}")
        bucket_aliases.add(bucket.alias)

    def check_filter(filter_item: Filter, context: str) -> None:
        check_column(filter_item.column, context)
        if filter_item.operator not in _FILTER_OPERATORS:
            issues.append(f"unsupported filter operator: {filter_item.operator}")
        elif filter_item.operator == "between":
            if (
                not isinstance(filter_item.value, (tuple, list))
                or len(filter_item.value) != 2
            ):
                issues.append("between filter requires exactly two values")
            elif any(
                not _safe_filter_scalar(value) or value is None
                for value in filter_item.value
            ):
                issues.append("between filter values must be finite scalar values")
        elif filter_item.operator == "in":
            if not isinstance(filter_item.value, (tuple, list, set, frozenset)):
                issues.append("in filter requires a collection of values")
            elif any(not _safe_filter_scalar(value) for value in filter_item.value):
                issues.append("in filter values must be finite scalar values")
        elif filter_item.operator not in {"is_null", "not_null"} and filter_item.value is None:
            issues.append(f"{filter_item.operator} filter requires a non-null value")
        elif (
            filter_item.operator not in {"is_null", "not_null"}
            and not _safe_filter_scalar(filter_item.value)
        ):
            issues.append(f"{filter_item.operator} filter requires a safe scalar value")
        if filter_item.operator == "contains" and not isinstance(filter_item.value, str):
            issues.append("contains filter requires a text value")

    def check_time_range(time_range: TimeRange, context: str) -> None:
        check_column(time_range.column, context)
        if not isinstance(time_range.start, str) or not isinstance(
            time_range.end, str
        ):
            issues.append(f"{context} requires start and end values")
            return
        try:
            start = date.fromisoformat(time_range.start)
            end = date.fromisoformat(time_range.end)
        except ValueError:
            issues.append(f"{context} values must be ISO-8601 dates")
            return
        if start > end:
            issues.append(f"{context} start must not be after end")

    for filter_item in plan.filters:
        check_filter(filter_item, "filter")

    for index, exclusion in enumerate(plan.group_exclusions):
        context = f"group exclusion {index}"
        check_column(exclusion.key, context)
        if exclusion.key.table != plan.base_table:
            issues.append(
                f"{context} key must belong to the base table so source rows "
                "can be scanned without an ambiguous join"
            )
        if not exclusion.filters:
            issues.append(f"{context} requires at least one source-row filter")
        for filter_item in exclusion.filters:
            check_filter(filter_item, context)
            if filter_item.column.table != exclusion.key.table:
                issues.append(
                    f"{context} filters must use the same table as its group key"
                )

    aliases: set[str] = set(bucket_aliases)
    for aggregation in plan.aggregations:
        if aggregation.function not in _AGGREGATIONS:
            issues.append(f"unsupported aggregation: {aggregation.function}")
        if not aggregation.alias or aggregation.alias in aliases:
            issues.append(f"aggregation alias is empty or duplicated: {aggregation.alias!r}")
        aliases.add(aggregation.alias)
        if aggregation.column is None and aggregation.function != "count":
            issues.append(f"{aggregation.function} requires a column")
        elif aggregation.column is not None:
            check_column(aggregation.column, "aggregation")
        if aggregation.unit_conversion is not None:
            conversion = aggregation.unit_conversion
            if aggregation.function != "sum":
                issues.append("unit conversion is supported only for sum")
            for ref in (
                conversion.source_unit,
                conversion.base_unit,
                conversion.large_unit,
                conversion.factor,
            ):
                check_column(ref, "unit conversion")
            if conversion.target_unit is not None and (
                not isinstance(conversion.target_unit, str)
                or not conversion.target_unit.strip()
            ):
                issues.append(
                    "unit conversion target_unit must be null or non-empty text"
                )
        for filter_item in aggregation.filters:
            check_filter(filter_item, f"aggregation {aggregation.alias!r} filter")
        if aggregation.time_range is not None:
            check_time_range(
                aggregation.time_range,
                f"aggregation {aggregation.alias!r} time range",
            )

    for calculation in plan.calculations:
        if calculation.function not in _CALCULATIONS:
            issues.append(f"unsupported calculation: {calculation.function}")
        if calculation.left not in aliases or calculation.right not in aliases:
            issues.append(
                f"calculation {calculation.alias!r} must reference prior aggregate "
                "or calculation aliases"
            )
        if not calculation.alias or calculation.alias in aliases:
            issues.append(
                f"calculation alias is empty or duplicated: {calculation.alias!r}"
            )
        if calculation.rounding_places is not None:
            if (
                isinstance(calculation.rounding_places, bool)
                or not isinstance(calculation.rounding_places, int)
                or not 0 <= calculation.rounding_places <= 12
            ):
                issues.append("rounding_places must be an integer from 0 to 12")
            if calculation.function != "ratio":
                issues.append("rounding_places is supported only for ratio")
        aliases.add(calculation.alias)

    group_filter_aliases = {item.alias for item in plan.aggregations} | {
        item.alias for item in plan.calculations
    }
    for group_filter in plan.group_filters:
        if group_filter.alias not in group_filter_aliases:
            issues.append(
                f"group filter references unknown aggregate alias: {group_filter.alias}"
            )
        if group_filter.operator not in {"eq", "ne", "lt", "lte", "gt", "gte"}:
            issues.append(
                f"unsupported group filter operator: {group_filter.operator}"
            )
        value = group_filter.value
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, Decimal))
            or not _safe_filter_scalar(value)
        ):
            issues.append("group filter value must be a finite number")
    if plan.group_filters and not (plan.group_by or plan.time_buckets):
        issues.append("group filters require grouped rows")

    if not isinstance(plan.count_groups, bool):
        issues.append("count_groups must be a boolean")
    elif plan.count_groups:
        if not plan.group_by and not plan.time_buckets:
            issues.append("count_groups requires grouped rows")
        if plan.output_type != "number":
            issues.append("count_groups requires number output")
        if plan.selected_columns:
            issues.append("count_groups cannot select output columns")
        if plan.sort or plan.limit is not None:
            issues.append("count_groups cannot use sort or limit")

    for item in plan.sort:
        if item.direction not in {"asc", "desc"}:
            issues.append(f"unsupported sort direction: {item.direction}")
        if isinstance(item.key, ColumnRef):
            check_column(item.key, "sort")
        elif item.key not in aliases:
            issues.append(f"sort references unknown column or aggregation alias: {item.key}")

    if (
        not plan.selected_columns
        and not plan.aggregations
        and not plan.group_by
        and not plan.time_buckets
    ):
        issues.append("plan must select columns, group rows, or aggregate")
    if plan.aggregations or plan.group_by or plan.time_buckets:
        for ref in plan.selected_columns:
            if ref not in plan.group_by:
                issues.append(
                    f"selected column {ref.table}.{ref.column} must be in group_by "
                    "when grouping or aggregations are used"
                )
    for ordering in plan.sort:
        if isinstance(ordering.key, ColumnRef):
            valid_sort_columns = (
                set(plan.group_by)
                if plan.aggregations or plan.group_by
                else set(plan.selected_columns)
            )
            if ordering.key not in valid_sort_columns:
                issues.append(
                    f"sort column {ordering.key.table}.{ordering.key.column} "
                    "must be selected or grouped"
                )

    if plan.time_range is not None:
        check_time_range(plan.time_range, "time range")
    if plan.limit is not None and (
        isinstance(plan.limit, bool)
        or not isinstance(plan.limit, int)
        or not 1 <= plan.limit <= max_limit
    ):
        issues.append(f"limit must be between 1 and {max_limit}")
    if plan.output_type is not None and plan.output_type not in _ANSWER_TYPES:
        issues.append(f"unsupported output type: {plan.output_type}")
    for ref in plan.evidence_requirements:
        check_column(ref, "evidence requirement")

    if issues:
        raise PlanValidationError(issues)
