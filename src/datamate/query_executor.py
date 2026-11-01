"""Bounded deterministic query execution over streamed, typed table rows."""

from __future__ import annotations
import re

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from typing import Iterable, Mapping, Sequence

from .query_plan import (
    Calculation,
    ColumnRef,
    Filter,
    GroupFilter,
    Join,
    QueryPlan,
    TimeBucket,
    TimeRange,
    UnitConversion,
    validate_query_plan,
)

Row = Sequence[object]
TableRows = Iterable[Row] | Callable[[], Iterable[Row]]


class QueryExecutionError(RuntimeError):
    """Raised when a plan cannot be executed exactly within configured limits."""


@dataclass(frozen=True)
class ExecutionTrace:
    source_tables: tuple[str, ...]
    source_rows: Mapping[str, int]
    joined_rows: int
    filtered_rows: int
    group_count: int
    result_count: int


@dataclass(frozen=True)
class ExecutionResult:
    rows: tuple[dict[str, object], ...]
    trace: ExecutionTrace
    time_coverage: Mapping[str, tuple[date | None, date | None]]


@dataclass
class _AggregateState:
    count: int = 0
    scope_matches: int = 0
    total: Decimal = Decimal(0)
    extreme: object = None
    distinct_values: set[tuple[type, object]] | None = None


def _column_key(ref: ColumnRef) -> str:
    return f"{ref.table}.{ref.column}"


def _row_value(
    context: Mapping[str, Row | None],
    column_indexes: Mapping[str, Mapping[str, int]],
    ref: ColumnRef,
) -> object:
    row = context.get(ref.table)
    if row is None:
        return None
    try:
        index = column_indexes[ref.table][ref.column]
    except KeyError as exc:
        raise QueryExecutionError(
            f"Column is unavailable in source schema: {ref.table}.{ref.column}"
        ) from exc
    return row[index]


def _parse_flexible_date(val: object) -> date | None:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    if isinstance(val, (int, float)) and not isinstance(val, bool) and 30000 <= val <= 65000:
        try:
            return (datetime(1899, 12, 30) + timedelta(days=int(val))).date()
        except (ValueError, OverflowError):
            pass
    if isinstance(val, str):
        text = val.strip()
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            pass
        m = re.match(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})$", text)
        if m:
            d, mth, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                return date(y, mth, d)
            except ValueError:
                pass
    return None

def _matches_value(value: object, op: str, expected: object) -> bool:
    if op == "is_null":
        return value is None
    if op == "not_null":
        return value is not None
    if value is None:
        return False

    v_date = _parse_flexible_date(value)
    if v_date is not None:
        if op == "between" and isinstance(expected, (list, tuple)) and len(expected) == 2:
            low_d = _parse_flexible_date(expected[0])
            high_d = _parse_flexible_date(expected[1])
            if low_d is not None and high_d is not None:
                return low_d <= v_date <= high_d
        exp_date = _parse_flexible_date(expected)
        if exp_date is not None:
            if op == "eq":
                return v_date == exp_date
            if op == "ne":
                return v_date != exp_date
            if op == "lt":
                return v_date < exp_date
            if op == "lte":
                return v_date <= exp_date
            if op == "gt":
                return v_date > exp_date
            if op == "gte":
                return v_date >= exp_date

    if op == "eq":
        if isinstance(value, str) and isinstance(expected, str):
            return value.strip().casefold() == expected.strip().casefold()
        return value == expected
    if op == "ne":
        if isinstance(value, str) and isinstance(expected, str):
            return value.strip().casefold() != expected.strip().casefold()
        return value != expected
    if op == "in":
        try:
            return value in expected
        except TypeError as exc:
            raise QueryExecutionError("in filter contains unsupported values") from exc
    if op == "contains":
        s_val = str(value).strip().casefold()
        s_exp = str(expected).strip().casefold()
        if s_exp in s_val:
            return True
        if s_exp.replace("/", "-") in s_val.replace("/", "-"):
            return True
        if s_exp.replace("-", "/") in s_val.replace("-", "/"):
            return True
        if v_date is not None:
            v_iso = v_date.isoformat()
            if s_exp in v_iso or s_exp.replace("/", "-") in v_iso:
                return True
            # e.g. '05-2026' or '05/2026'
            v_my = f"{v_date.month:02d}-{v_date.year}"
            if s_exp in v_my or s_exp.replace("/", "-") in v_my:
                return True
        return False
    try:
        if op == "between":
            low, high = expected
            return low <= value <= high
        if op == "lt":
            return value < expected
        if op == "lte":
            return value <= expected
        if op == "gt":
            return value > expected
        if op == "gte":
            return value >= expected
    except TypeError as exc:
        raise QueryExecutionError(
            f"Incompatible filter value types for operator {op}"
        ) from exc
    raise QueryExecutionError(f"Unsupported filter operator: {op}")

def _matches(value: object, condition: Filter) -> bool:
    return _matches_value(value, condition.operator, condition.value)


def _matches_group_filter(row: Mapping[str, object], condition: GroupFilter) -> bool:
    value = row[condition.alias]
    if value is None:
        return False
    return _matches_value(
        _to_decimal(value, alias=condition.alias),
        condition.operator,
        _to_decimal(condition.value, alias=condition.alias),
    )


def _source_day(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        try:
            return date.fromisoformat(text)
        except ValueError:
            try:
                return datetime.fromisoformat(
                    text.replace("Z", "+00:00")
                ).date()
            except ValueError as exc:
                raise QueryExecutionError(
                    "Time-range source values must be an ISO-8601 date or datetime"
                ) from exc
    else:
        raise QueryExecutionError(
            "Time-range source values must be date, datetime, or ISO-8601 text"
        )

def _time_matches(value: object, time_range: TimeRange) -> bool:
    source_day = _source_day(value)
    if source_day is None:
        return False
    try:
        start = date.fromisoformat(time_range.start)
        end = date.fromisoformat(time_range.end)
    except ValueError as exc:
        raise QueryExecutionError("Time-range boundaries must be ISO-8601 dates") from exc

    if time_range.inclusive_end:
        return start <= source_day <= end
    return start <= source_day < end


def _time_bucket(value: object, bucket: TimeBucket) -> str | None:
    source_day = _source_day(value)
    if source_day is None:
        return None
    if bucket.granularity == "day":
        return source_day.isoformat()
    if bucket.granularity == "week":
        monday = date.fromordinal(source_day.toordinal() - source_day.weekday())
        return monday.isoformat()
    if bucket.granularity == "month":
        return f"{source_day.year:04d}-{source_day.month:02d}"
    if bucket.granularity == "quarter":
        quarter = (source_day.month - 1) // 3 + 1
        return f"{source_day.year:04d}-Q{quarter}"
    if bucket.granularity == "year":
        return f"{source_day.year:04d}"
    raise QueryExecutionError(
        f"Unsupported time bucket granularity: {bucket.granularity}"
    )


def _bucket_start(day: date, granularity: str) -> date:
    if granularity == "day":
        return day
    if granularity == "week":
        return date.fromordinal(day.toordinal() - day.weekday())
    if granularity == "month":
        return date(day.year, day.month, 1)
    if granularity == "quarter":
        month = ((day.month - 1) // 3) * 3 + 1
        return date(day.year, month, 1)
    if granularity == "year":
        return date(day.year, 1, 1)
    raise QueryExecutionError(f"Unsupported time bucket granularity: {granularity}")


def _next_bucket_start(day: date, granularity: str) -> date:
    if granularity == "day":
        return day + timedelta(days=1)
    if granularity == "week":
        return day + timedelta(days=7)
    if granularity == "month":
        return date(day.year + (day.month == 12), day.month % 12 + 1, 1)
    if granularity == "quarter":
        next_month = day.month + 3
        return date(day.year + (next_month > 12), (next_month - 1) % 12 + 1, 1)
    if granularity == "year":
        return date(day.year + 1, 1, 1)
    raise QueryExecutionError(f"Unsupported time bucket granularity: {granularity}")


def _to_decimal(value: object, *, alias: str) -> Decimal:
    if isinstance(value, bool):
        raise QueryExecutionError(f"Aggregation {alias!r} requires numeric values")
    if isinstance(value, (int, float, Decimal)):
        try:
            number = Decimal(str(value))
        except InvalidOperation as exc:
            raise QueryExecutionError(f"Invalid numeric value for {alias!r}") from exc
        if not number.is_finite():
            raise QueryExecutionError(f"Non-finite numeric value for {alias!r}")
        return number
    if isinstance(value, str):
        v = value.strip()
        # Vietnamese currency shorthand: '2tr756' -> 2,756,000, '1tr550' -> 1,550,000, '1tr5' -> 1,500,000, '2tr' -> 2,000,000
        m_tr = re.match(r"^([+-]?\d+)\s*tr\s*(\d+)?$", v.lower())
        if m_tr:
            trieu = int(m_tr.group(1))
            nghin_str = m_tr.group(2)
            if nghin_str:
                if len(nghin_str) == 1:
                    nghin = int(nghin_str) * 100_000
                elif len(nghin_str) == 2:
                    nghin = int(nghin_str) * 10_000
                elif len(nghin_str) == 3:
                    nghin = int(nghin_str) * 1_000
                else:
                    nghin = int(nghin_str)
                return Decimal(trieu * 1_000_000 + nghin)
            return Decimal(trieu * 1_000_000)

        # Check for unit suffix: '1 ly', '2 cái', '42k', '43.000đ'
        m_unit = re.match(r"^([+-]?[\d.,]+)\s*([a-zA-ZÀ-ỹ%]+)$", v)
        if m_unit:
            num_str, unit = m_unit.group(1), m_unit.group(2).lower()
            try:
                if unit in ("k", "nghìn"):
                    return Decimal(num_str.replace(",", "")) * 1000
                if unit in ("đ", "vnd", "dong"):
                    return Decimal(num_str.replace(".", "").replace(",", ""))
                return Decimal(num_str.replace(",", ""))
            except InvalidOperation:
                pass
        # Standard thousands comma with 3 digits: 21,000.0000 or 1,500,000
        if re.match(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$", v):
            try:
                return Decimal(v.replace(",", ""))
            except InvalidOperation:
                pass
    raise QueryExecutionError(f"Aggregation {alias!r} requires numeric values")


def _addition_precision(left: Decimal, right: Decimal) -> int:
    lowest_exponent = min(left.as_tuple().exponent, right.as_tuple().exponent)
    highest_digit = max(left.adjusted(), right.adjusted())
    return max(28, highest_digit - lowest_exponent + 2)


def _exact_add(left: Decimal, right: Decimal) -> Decimal:
    with localcontext() as decimal_context:
        decimal_context.prec = max(
            decimal_context.prec, _addition_precision(left, right)
        )
        return left + right


def _exact_subtract(left: Decimal, right: Decimal) -> Decimal:
    with localcontext() as decimal_context:
        decimal_context.prec = max(
            decimal_context.prec, _addition_precision(left, right)
        )
        return left - right


def _exact_multiply(left: Decimal, right: Decimal) -> Decimal:
    with localcontext() as decimal_context:
        decimal_context.prec = max(
            decimal_context.prec,
            len(left.as_tuple().digits) + len(right.as_tuple().digits) + 2,
        )
        return left * right


def _same_unit(left: object, right: object) -> bool:
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    return left.strip().casefold() == right.strip().casefold()


def _unit_matches_target(
    conversion: UnitConversion,
    context: Mapping[str, Row | None],
    column_indexes: Mapping[str, Mapping[str, int]],
) -> bool:
    base_unit = _row_value(context, column_indexes, conversion.base_unit)
    if base_unit is None:
        source_unit = _row_value(context, column_indexes, conversion.source_unit)
        if not isinstance(source_unit, str) or not source_unit.strip():
            raise QueryExecutionError("Missing source unit for unit conversion")
        if conversion.target_unit is None:
            return True
        return _same_unit(source_unit, conversion.target_unit)
    if conversion.target_unit is None:
        return True
    return _same_unit(base_unit, conversion.target_unit)


def _convert_quantity(
    value: object,
    conversion: UnitConversion,
    context: Mapping[str, Row | None],
    column_indexes: Mapping[str, Mapping[str, int]],
    *,
    alias: str,
) -> Decimal | None:
    if value is None:
        return None
    base_unit = _row_value(context, column_indexes, conversion.base_unit)
    source_unit = _row_value(context, column_indexes, conversion.source_unit)
    if base_unit is None:
        if not isinstance(source_unit, str) or not source_unit.strip():
            raise QueryExecutionError(
                f"Missing source unit for aggregation {alias!r}"
            )
        if conversion.target_unit is not None and not _same_unit(
            source_unit, conversion.target_unit
        ):
            raise QueryExecutionError(
                f"Missing unit conversion for aggregation {alias!r}"
            )
        factor = Decimal(1)
    elif (
        conversion.target_unit is not None
        and not _same_unit(base_unit, conversion.target_unit)
    ):
        return None
    elif _same_unit(source_unit, base_unit):
        factor = Decimal(1)
    elif _same_unit(
        source_unit,
        _row_value(context, column_indexes, conversion.large_unit),
    ):
        raw_factor = _row_value(context, column_indexes, conversion.factor)
        factor = _to_decimal(raw_factor, alias=f"{alias} unit factor")
        if factor <= 0:
            raise QueryExecutionError(
                f"Unit conversion factor for {alias!r} must be positive"
            )
    else:
        raise QueryExecutionError(
            f"Cannot convert source unit for aggregation {alias!r}"
        )
    return _exact_multiply(_to_decimal(value, alias=alias), factor)


def _aggregate(
    rows: Iterable[Mapping[str, Row | None]],
    plan: QueryPlan,
    column_indexes: Mapping[str, Mapping[str, int]],
    time_coverage: Mapping[str, Sequence[date | None]],
    *,
    max_groups: int,
    max_distinct_values: int,
) -> tuple[list[dict[str, object]], int]:
    grouped: dict[tuple[object, ...], tuple[tuple[object, ...], list[_AggregateState]]] = {}
    group_refs = plan.group_by
    time_buckets = plan.time_buckets
    distinct_value_count = 0

    for context in rows:
        matched_aggregations = [
            all(
                _matches(
                    _row_value(context, column_indexes, filter_item.column),
                    filter_item,
                )
                for filter_item in aggregation.filters
            )
            and (
                aggregation.time_range is None
                or _time_matches(
                    _row_value(
                        context,
                        column_indexes,
                        aggregation.time_range.column,
                    ),
                    aggregation.time_range,
                )
            )
            and (
                aggregation.unit_conversion is None
                or _unit_matches_target(
                    aggregation.unit_conversion,
                    context,
                    column_indexes,
                )
            )
            for aggregation in plan.aggregations
        ]
        if plan.aggregations and group_refs and not any(matched_aggregations):
            continue
        key = (
            *(
                _row_value(context, column_indexes, ref)
                for ref in group_refs
            ),
            *(
                _time_bucket(
                    _row_value(context, column_indexes, bucket.column),
                    bucket,
                )
                for bucket in time_buckets
            ),
        )
        try:
            if key not in grouped:
                if len(grouped) >= max_groups:
                    raise QueryExecutionError(
                        f"Group count exceeds configured limit {max_groups}"
                    )
                grouped[key] = (
                    key,
                    [_AggregateState() for _ in plan.aggregations],
                )
        except TypeError as exc:
            raise QueryExecutionError("Group-by values must be hashable scalars") from exc
        _, states = grouped[key]
        for aggregation, state, matches_scope in zip(
            plan.aggregations,
            states,
            matched_aggregations,
            strict=True,
        ):
            if not matches_scope:
                continue
            state.scope_matches += 1
            value = (
                1
                if aggregation.column is None
                else _row_value(context, column_indexes, aggregation.column)
            )
            if aggregation.unit_conversion is not None:
                value = _convert_quantity(
                    value,
                    aggregation.unit_conversion,
                    context,
                    column_indexes,
                    alias=aggregation.alias,
                )
            if aggregation.function == "count":
                if aggregation.column is None or value is not None:
                    state.count += 1
            elif aggregation.function == "count_distinct":
                if value is not None:
                    try:
                        distinct_key = (type(value), value)
                        if state.distinct_values is None:
                            state.distinct_values = set()
                        if distinct_key not in state.distinct_values:
                            if distinct_value_count >= max_distinct_values:
                                raise QueryExecutionError(
                                    "Distinct value count exceeds configured limit "
                                    f"{max_distinct_values}"
                                )
                            state.distinct_values.add(distinct_key)
                            state.count += 1
                            distinct_value_count += 1
                    except TypeError as exc:
                        raise QueryExecutionError(
                            f"Distinct values for {aggregation.alias!r} must be "
                            "hashable scalars"
                        ) from exc
            elif value is not None:
                if aggregation.function in {"sum", "avg"}:
                    number = _to_decimal(value, alias=aggregation.alias)
                    state.total = _exact_add(state.total, number)
                    state.count += 1
                else:
                    try:
                        if state.count == 0:
                            state.extreme = value
                        elif aggregation.function == "min":
                            state.extreme = min(state.extreme, value)
                        else:
                            state.extreme = max(state.extreme, value)
                    except TypeError as exc:
                        raise QueryExecutionError(
                            f"Mixed values cannot be compared for {aggregation.alias!r}"
                        ) from exc
                    state.count += 1

    if (
        len(time_buckets) == 1
        and plan.time_range is not None
        and time_buckets[0].column == plan.time_range.column
        and bool(plan.aggregations)
        and all(
            aggregation.time_range is None
            and aggregation.function in {"sum", "count", "count_distinct"}
            for aggregation in plan.aggregations
        )
    ):
        bucket = time_buckets[0]
        coverage = time_coverage.get(
            f"{bucket.column.table}.{bucket.column.column}"
        )
        if coverage is not None and coverage[0] is not None and coverage[1] is not None:
            requested_start = date.fromisoformat(plan.time_range.start)
            requested_end = date.fromisoformat(plan.time_range.end)
            empty_range = (
                not plan.time_range.inclusive_end
                and requested_start >= requested_end
            )
            if not plan.time_range.inclusive_end:
                if not empty_range:
                    requested_end -= timedelta(days=1)
            first_day = max(requested_start, coverage[0])
            last_day = min(requested_end, coverage[1])
            if not empty_range and first_day <= last_day:
                group_keys = (
                    {key[: len(group_refs)] for key in grouped}
                    if group_refs
                    else {()}
                )
                for group_key in group_keys:
                    bucket_day = _bucket_start(first_day, bucket.granularity)
                    while bucket_day <= last_day:
                        label = _time_bucket(bucket_day, bucket)
                        key = (*group_key, label)
                        if key not in grouped:
                            if len(grouped) >= max_groups:
                                raise QueryExecutionError(
                                    "Group count exceeds configured limit "
                                    f"{max_groups}"
                                )
                            states = []
                            for aggregation in plan.aggregations:
                                state = _AggregateState()
                                if aggregation.function == "sum":
                                    state.count = 1
                                states.append(state)
                            grouped[key] = (key, states)
                        try:
                            bucket_day = _next_bucket_start(
                                bucket_day,
                                bucket.granularity,
                            )
                        except (OverflowError, ValueError):
                            break

    if not grouped and not group_refs and not time_buckets:
        grouped[()] = ((), [_AggregateState() for _ in plan.aggregations])

    output: list[dict[str, object]] = []
    for key, states in grouped.values():
        result = {
            _column_key(ref): value
            for ref, value in zip(group_refs, key[: len(group_refs)], strict=True)
        }
        result.update(
            {
                bucket.alias: value
                for bucket, value in zip(
                    time_buckets,
                    key[len(group_refs) :],
                    strict=True,
                )
            }
        )
        for aggregation, state in zip(plan.aggregations, states, strict=True):
            if aggregation.function in {"count", "count_distinct"}:
                value: object = state.count
            elif state.count == 0 and state.scope_matches == 0:
                value = Decimal(0)
            elif state.count == 0:
                value = None
            elif aggregation.function == "sum":
                value = state.total
            elif aggregation.function == "avg":
                with localcontext() as decimal_context:
                    decimal_context.prec = max(decimal_context.prec, 50)
                    value = state.total / state.count
            else:
                value = state.extreme
            result[aggregation.alias] = value
        output.append(result)
    return output, len(grouped)


def _calculate(
    rows: list[dict[str, object]], calculations: Sequence[Calculation]
) -> None:
    for row in rows:
        for calculation in calculations:
            left = row.get(calculation.left)
            right = row.get(calculation.right)
            if left is None or right is None:
                row[calculation.alias] = None
                continue
            left_number = _to_decimal(left, alias=calculation.left)
            right_number = _to_decimal(right, alias=calculation.right)
            if calculation.function == "difference":
                row[calculation.alias] = _exact_subtract(left_number, right_number)
                continue
            if right_number == 0:
                row[calculation.alias] = None
                continue
            numerator = (
                _exact_subtract(left_number, right_number)
                if calculation.function == "percent_change"
                else left_number
            )
            multiplier = (
                Decimal(100)
                if calculation.function in {"ratio_percent", "percent_change"}
                else Decimal(1)
            )
            with localcontext() as decimal_context:
                decimal_context.prec = max(decimal_context.prec, 50)
                calculated = numerator / right_number * multiplier
                if calculation.function in {"ratio_percent", "percent_change"}:
                    decimal_context.prec = max(
                        decimal_context.prec,
                        max(0, calculated.adjusted()) + 3,
                    )
                    row[calculation.alias] = calculated.quantize(
                        Decimal("0.01"),
                        rounding=ROUND_HALF_UP,
                    )
                elif calculation.rounding_places is not None:
                    row[calculation.alias] = calculated.quantize(
                        Decimal(1).scaleb(-calculation.rounding_places),
                        rounding=ROUND_HALF_UP,
                    )
                else:
                    row[calculation.alias] = calculated


def execute_query_plan(
    plan: QueryPlan,
    *,
    catalog: Mapping[str, Sequence[str]],
    sources: Mapping[str, TableRows],
    max_join_rows: int = 100_000,
    max_groups: int = 100_000,
    max_result_rows: int = 10_000,
    max_distinct_values: int = 100_000,
    max_group_exclusions: int = 100_000,
) -> ExecutionResult:
    """Run allow-listed operations over streamed rows without executing SQL/code."""
    for name, value in (
        ("max_join_rows", max_join_rows),
        ("max_groups", max_groups),
        ("max_result_rows", max_result_rows),
        ("max_distinct_values", max_distinct_values),
        ("max_group_exclusions", max_group_exclusions),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive")
    schema_sets: dict[str, set[str]] = {}
    schemas: dict[str, tuple[str, ...]] = {}
    column_indexes: dict[str, dict[str, int]] = {}
    for table, columns in catalog.items():
        column_tuple = tuple(columns)
        if len(set(column_tuple)) != len(column_tuple):
            raise QueryExecutionError(
                f"Duplicate column names are ambiguous for {table}"
            )
        schemas[table] = column_tuple
        schema_sets[table] = set(column_tuple)
        column_indexes[table] = {
            column: index for index, column in enumerate(column_tuple)
        }
    validate_query_plan(plan, schema_sets)

    unit_conversion_tables = {
        ref.table
        for aggregation in plan.aggregations
        if aggregation.unit_conversion is not None
        for ref in (
            aggregation.unit_conversion.base_unit,
            aggregation.unit_conversion.large_unit,
            aggregation.unit_conversion.factor,
        )
        if ref.table != aggregation.unit_conversion.source_unit.table
    }

    involved_tables = {plan.base_table}
    for join in plan.joins:
        involved_tables.update((join.left.table, join.right.table))
    missing_sources = involved_tables - sources.keys()
    if missing_sources:
        raise QueryExecutionError(
            f"Missing row sources for tables: {', '.join(sorted(missing_sources))}"
        )

    opened_iterators: dict[str, Iterable[Row]] = {}

    def source_rows(table: str) -> Iterable[Row]:
        source = sources[table]
        rows = source() if callable(source) else source
        try:
            iterator = iter(rows)
        except TypeError as exc:
            raise QueryExecutionError(
                f"Row source for {table} is not iterable"
            ) from exc
        if opened_iterators.get(table) is iterator:
            raise QueryExecutionError(
                f"Row source for {table} must provide a fresh iterator when "
                "group exclusions require a second pass"
            )
        opened_iterators[table] = iterator
        return iterator

    def is_excluded(row: Row) -> bool:
        for key, excluded_keys in excluded_groups.items():
            group_key = row[key_indexes[key]]
            try:
                if group_key in excluded_keys:
                    return True
            except TypeError as exc:
                raise QueryExecutionError(
                    "Group exclusion keys must be hashable scalars"
                ) from exc
        return False

    excluded_groups: dict[ColumnRef, set[object]] = defaultdict(set)
    key_indexes: dict[ColumnRef, int] = {}
    if plan.group_exclusions:
        key_indexes = {
            exclusion.key: column_indexes[plan.base_table][exclusion.key.column]
            for exclusion in plan.group_exclusions
        }
        exclusions_by_key: dict[ColumnRef, list[object]] = defaultdict(list)
        for exclusion in plan.group_exclusions:
            exclusions_by_key[exclusion.key].append(exclusion.filters)
        excluded_group_count = 0
        for row_number, raw_row in enumerate(
            source_rows(plan.base_table),
            start=1,
        ):
            row = tuple(raw_row)
            if len(row) != len(schemas[plan.base_table]):
                raise QueryExecutionError(
                    f"Row width mismatch in {plan.base_table} at row {row_number}"
                )
            context = {plan.base_table: row}
            for key, exclusion_filters in exclusions_by_key.items():
                if not any(
                    all(
                        _matches(
                            _row_value(context, column_indexes, item.column),
                            item,
                        )
                        for item in filters
                    )
                    for filters in exclusion_filters
                ):
                    continue
                group_key = row[key_indexes[key]]
                if group_key is None:
                    raise QueryExecutionError(
                        "Cannot exclude a group when a matching source row has a "
                        f"null key: {key.table}.{key.column}"
                    )
                try:
                    excluded = excluded_groups[key]
                    if group_key not in excluded:
                        if excluded_group_count >= max_group_exclusions:
                            raise QueryExecutionError(
                                "Excluded group count exceeds configured limit "
                                f"{max_group_exclusions}"
                            )
                        excluded.add(group_key)
                        excluded_group_count += 1
                except TypeError as exc:
                    raise QueryExecutionError(
                        "Group exclusion keys must be hashable scalars"
                    ) from exc

    joined_tables = {plan.base_table}
    join_indexes: dict[str, dict[object, list[Row]]] = {}
    source_row_counts: dict[str, int] = {}
    time_columns: dict[str, set[str]] = defaultdict(set)
    if plan.time_range is not None:
        time_columns[plan.time_range.column.table].add(plan.time_range.column.column)
    for bucket in plan.time_buckets:
        time_columns[bucket.column.table].add(bucket.column.column)
    for aggregation in plan.aggregations:
        if aggregation.time_range is not None:
            time_columns[aggregation.time_range.column.table].add(
                aggregation.time_range.column.column
            )
    time_coverage: dict[str, list[date | None]] = {
        f"{table}.{column}": [None, None]
        for table, columns in time_columns.items()
        for column in columns
    }

    def record_time_coverage(table: str, row: Row) -> None:
        for column in time_columns.get(table, ()):
            day = _source_day(row[column_indexes[table][column]])
            if day is None:
                continue
            coverage = time_coverage[f"{table}.{column}"]
            if coverage[0] is None or day < coverage[0]:
                coverage[0] = day
            if coverage[1] is None or day > coverage[1]:
                coverage[1] = day

    for join in plan.joins:
        new_table = (
            join.right.table
            if join.right.table not in joined_tables
            else join.left.table
        )
        new_key = join.right if join.right.table == new_table else join.left
        schema = schemas[new_table]
        try:
            key_index = column_indexes[new_table][new_key.column]
        except KeyError as exc:
            raise QueryExecutionError(
                f"Join key is unavailable: {new_table}.{new_key.column}"
            ) from exc
        index: dict[object, list[Row]] = defaultdict(list)
        row_count = 0
        for row_count, raw_row in enumerate(source_rows(new_table), start=1):
            row = tuple(raw_row)
            if len(row) != len(schema):
                raise QueryExecutionError(
                    f"Row width mismatch in {new_table} at row {row_count}"
                )
            record_time_coverage(new_table, row)
            key = row[key_index]
            if key is not None:
                try:
                    if new_table in unit_conversion_tables and key in index:
                        raise QueryExecutionError(
                            "Unit-conversion lookup tables must have unique join keys"
                        )
                    index[key].append(row)
                except TypeError as exc:
                    raise QueryExecutionError(
                        f"Join key values must be hashable scalars: {new_table}"
                    ) from exc
            if row_count > max_join_rows:
                raise QueryExecutionError(
                    f"Join input {new_table} exceeds configured limit {max_join_rows}"
                )
        source_row_counts[new_table] = row_count
        join_indexes[new_table] = index
        joined_tables.add(new_table)

    joined_row_count = 0

    def contexts() -> Iterable[dict[str, Row | None]]:
        nonlocal joined_row_count
        base_row_count = 0
        for base_row_count, raw_row in enumerate(
            source_rows(plan.base_table),
            start=1,
        ):
            row = tuple(raw_row)
            if len(row) != len(schemas[plan.base_table]):
                raise QueryExecutionError(
                    f"Row width mismatch in {plan.base_table} at row {base_row_count}"
                )
            record_time_coverage(plan.base_table, row)
            if is_excluded(row):
                continue
            current: list[dict[str, Row | None]] = [{plan.base_table: row}]
            for join in plan.joins:
                new_table = (
                    join.right.table
                    if join.right.table not in current[0]
                    else join.left.table
                )
                probe_key = join.left if join.left.table != new_table else join.right
                expanded: list[dict[str, Row | None]] = []
                for context in current:
                    key = _row_value(context, column_indexes, probe_key)
                    matches = join_indexes[new_table].get(key, []) if key is not None else []
                    if matches:
                        for match in matches:
                            expanded.append({**context, new_table: match})
                    elif join.kind == "left":
                        expanded.append({**context, new_table: None})
                current = expanded
                if not current:
                    break
                if len(current) + joined_row_count > max_result_rows:
                    raise QueryExecutionError(
                        f"Intermediate joined rows exceed configured limit {max_result_rows}"
                    )
            for context in current:
                joined_row_count += 1
                yield context
        source_row_counts[plan.base_table] = base_row_count

    def filtered_contexts() -> Iterable[dict[str, Row | None]]:
        for context in contexts():
            if not all(
                _matches(_row_value(context, column_indexes, item.column), item)
                for item in plan.filters
            ):
                continue
            if plan.time_range and not _time_matches(
                _row_value(context, column_indexes, plan.time_range.column),
                plan.time_range,
            ):
                continue
            yield context

    filtered_count = 0

    def counted_contexts() -> Iterable[dict[str, Row | None]]:
        nonlocal filtered_count
        for context in filtered_contexts():
            filtered_count += 1
            yield context

    if (
        plan.aggregations
        or plan.calculations
        or plan.group_by
        or plan.time_buckets
    ):
        rows, group_count = _aggregate(
            counted_contexts(),
            plan,
            column_indexes,
            time_coverage,
            max_groups=max_groups,
            max_distinct_values=max_distinct_values,
        )
        _calculate(rows, plan.calculations)
        if plan.group_filters:
            rows = [
                row
                for row in rows
                if all(
                    _matches_group_filter(row, group_filter)
                    for group_filter in plan.group_filters
                )
            ]
        if plan.count_groups:
            rows = [{"group_count": len(rows)}]
    else:
        rows = []
        group_count = 0
        for context in counted_contexts():
            row = {
                _column_key(ref): _row_value(context, column_indexes, ref)
                for ref in plan.selected_columns
            }
            rows.append(row)
            if len(rows) > max_result_rows:
                raise QueryExecutionError(
                    f"Result row count exceeds configured limit {max_result_rows}"
                )

    for ordering in reversed(plan.sort):
        key = _column_key(ordering.key) if isinstance(ordering.key, ColumnRef) else ordering.key
        non_null = [row for row in rows if row[key] is not None]
        null_rows = [row for row in rows if row[key] is None]
        try:
            non_null.sort(
                key=lambda result: result[key],
                reverse=ordering.direction == "desc",
            )
        except TypeError as exc:
            raise QueryExecutionError(f"Values for sort key {key!r} are incomparable") from exc
        rows = non_null + null_rows
    if plan.limit is not None:
        rows = rows[: plan.limit]
    if len(rows) > max_result_rows:
        raise QueryExecutionError(
            f"Result row count exceeds configured limit {max_result_rows}"
        )
    return ExecutionResult(
        rows=tuple(rows),
        trace=ExecutionTrace(
            source_tables=tuple(sorted(involved_tables)),
            source_rows=dict(source_row_counts),
            joined_rows=joined_row_count,
            filtered_rows=filtered_count,
            group_count=group_count,
            result_count=len(rows),
        ),
        time_coverage={
            column: (bounds[0], bounds[1])
            for column, bounds in time_coverage.items()
        },
    )
