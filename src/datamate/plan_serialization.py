"""Strict JSON decoding for model-produced declarative query plans."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .query_plan import (
    Aggregation,
    Calculation,
    ColumnRef,
    Filter,
    GroupExclusion,
    GroupFilter,
    Join,
    QueryPlan,
    Sort,
    TimeBucket,
    TimeRange,
    UnitConversion,
)


class QueryPlanFormatError(ValueError):
    """Raised when a serialized plan is malformed or contains unknown fields."""


def _object(value: object, context: str, allowed: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise QueryPlanFormatError(f"{context} must be a JSON object")
    unknown = set(value) - allowed
    if unknown:
        raise QueryPlanFormatError(
            f"{context} has unknown fields: {', '.join(sorted(unknown))}"
        )
    return value


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise QueryPlanFormatError(f"{context} must be a non-empty string")
    return value


def _boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise QueryPlanFormatError(f"{context} must be a boolean")
    return value


def _sequence(value: object, context: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise QueryPlanFormatError(f"{context} must be a JSON array")
    return value


def _column_ref(value: object, context: str, default_table: str | None = None) -> ColumnRef:
    if isinstance(value, str) and value:
        return ColumnRef(table=default_table or "", column=value)
    if isinstance(value, dict):
        inner = value.get("column")
        if isinstance(inner, dict):
            tbl = inner.get("table") or value.get("table") or default_table or ""
            col = inner.get("column") or inner.get("field") or inner.get("column_name") or inner.get("name")
            return ColumnRef(table=str(tbl or ""), column=str(col or ""))
        elif isinstance(inner, str) and inner:
            tbl = value.get("table") or default_table or ""
            return ColumnRef(table=str(tbl or ""), column=inner)
        for alt in ("column_name", "field", "name"):
            if alt in value:
                value = dict(value)
                value["column"] = value.pop(alt)
                break
        if "table" not in value and default_table:
            value = dict(value)
            value["table"] = default_table
    data = _object(value, context, {"table", "column"})
    return ColumnRef(
        table=_string(data.get("table", default_table or ""), f"{context}.table"),
        column=_string(data.get("column"), f"{context}.column"),
    )


def _filter_value(value: object) -> object:
    if isinstance(value, list):
        return tuple(_filter_value(item) for item in value)
    if isinstance(value, dict):
        for k in ("value", "$date", "date", "v", "val"):
            if k in value:
                return _filter_value(value[k])
        if value:
            return _filter_value(next(iter(value.values())))
        return str(value)
    return value


def _filter_from_data(value: object, context: str, default_table: str | None = None) -> Filter:
    if isinstance(value, dict):
        if "column" not in value and "field" in value:
            value = dict(value)
            value["column"] = value.pop("field")
        if "operator" not in value:
            value = dict(value)
            value["operator"] = "eq"
    filter_data = _object(value, context, {"column", "operator", "value", "type", "field"})
    return Filter(
        column=_column_ref(filter_data.get("column"), f"{context}.column", default_table=default_table),
        operator=_string(filter_data.get("operator", "eq"), f"{context}.operator"),
        value=_filter_value(filter_data.get("value")),
    )


def _time_range_from_data(value: object, context: str) -> TimeRange:
    time_data = _object(
        value,
        context,
        {"column", "start", "end", "inclusive_end"},
    )
    return TimeRange(
        column=_column_ref(time_data.get("column"), f"{context}.column"),
        start=_string(time_data.get("start"), f"{context}.start"),
        end=_string(time_data.get("end"), f"{context}.end"),
        inclusive_end=_boolean(
            time_data.get("inclusive_end", True),
            f"{context}.inclusive_end",
        ),
    )


def _unit_conversion_from_data(value: object, context: str) -> UnitConversion:
    data = _object(
        value,
        context,
        {"source_unit", "base_unit", "large_unit", "factor", "target_unit"},
    )
    if "target_unit" not in data:
        raise QueryPlanFormatError(f"{context}.target_unit is required")
    target_unit_value = data.get("target_unit")
    if target_unit_value is not None and not isinstance(target_unit_value, str):
        raise QueryPlanFormatError(
            f"{context}.target_unit must be null or text"
        )
    return UnitConversion(
        source_unit=_column_ref(data.get("source_unit"), f"{context}.source_unit"),
        base_unit=_column_ref(data.get("base_unit"), f"{context}.base_unit"),
        large_unit=_column_ref(data.get("large_unit"), f"{context}.large_unit"),
        factor=_column_ref(data.get("factor"), f"{context}.factor"),
        target_unit=target_unit_value,
    )


def _reject_json_constant(value: str) -> Any:
    raise QueryPlanFormatError(f"non-finite JSON number is not allowed: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise QueryPlanFormatError(f"duplicate JSON field is not allowed: {key}")
        result[key] = value
    return result


def _strip_standalone_json_fence(raw: str) -> str:
    stripped = raw.strip()
    first_line, separator, body = stripped.partition("\n")
    if first_line not in {"```", "```json"} or not separator:
        return raw
    if not body.endswith("```"):
        return raw
    return body[:-3].strip()


def query_plan_from_json(raw: str) -> QueryPlan:
    """Decode JSON, allowing only an otherwise-standalone Markdown JSON fence."""
    try:
        data = json.loads(
            _strip_standalone_json_fence(raw),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except RecursionError as exc:
        raise QueryPlanFormatError("query plan is too deeply nested") from exc
    except (json.JSONDecodeError, TypeError) as exc:
        raise QueryPlanFormatError(f"query plan is not valid JSON: {exc}") from exc
    return query_plan_from_dict(data)


def query_plan_from_dict(value: object) -> QueryPlan:
    """Convert a JSON-compatible mapping to typed plan objects without inference."""
    fields = {
        "base_table",
        "joins",
        "selected_columns",
        "filters",
        "group_by",
        "group_filters",
        "group_exclusions",
        "aggregations",
        "calculations",
        "count_groups",
        "sort",
        "limit",
        "time_range",
        "time_buckets",
        "business_rules",
        "output_type",
        "evidence_requirements",
        "metadata",
    }
    data = _object(value, "query plan", fields)
    base_table_value = data.get("base_table")
    meta_val = data.get("metadata")
    meta_status = meta_val.get("status") if isinstance(meta_val, Mapping) else None
    if (
        base_table_value is None
        and (
            data.get("output_type") in {"unanswerable", "abstain"}
            or meta_status in {"unanswerable", "abstained", "ambiguous", "insufficient_data"}
            or (data.get("output_type") is None and not data.get("aggregations") and not data.get("selected_columns") and not data.get("group_by"))
        )
    ):
        base_table = None
        if data.get("output_type") not in {"unanswerable", "abstain"}:
            data["output_type"] = "abstain"
        if not isinstance(meta_val, Mapping):
            data["metadata"] = {"status": "abstained", "reason": "Không đủ dữ liệu để trả lời câu hỏi"}
        elif meta_val.get("status") not in {"abstained", "unanswerable"}:
            meta_dict = dict(meta_val)
            meta_dict["status"] = "abstained"
            data["metadata"] = meta_dict
    else:
        base_table = _string(base_table_value, "base_table")

    joins = []
    for index, item in enumerate(_sequence(data.get("joins", []), "joins")):
        context = f"joins[{index}]"
        join = _object(item, context, {"left", "right", "kind"})
        joins.append(
            Join(
                left=_column_ref(join.get("left"), f"{context}.left"),
                right=_column_ref(join.get("right"), f"{context}.right"),
                kind=_string(join.get("kind", "inner"), f"{context}.kind"),
            )
        )

    selected_columns = tuple(
        _column_ref(item, f"selected_columns[{index}]", default_table=base_table)
        for index, item in enumerate(
            _sequence(data.get("selected_columns", []), "selected_columns")
        )
    )
    filters = []
    for index, item in enumerate(_sequence(data.get("filters", []), "filters")):
        filters.append(_filter_from_data(item, f"filters[{index}]", default_table=base_table))

    group_by = tuple(
        _column_ref(item, f"group_by[{index}]", default_table=base_table)
        for index, item in enumerate(_sequence(data.get("group_by", []), "group_by"))
    )
    group_filters = []
    for index, item in enumerate(
        _sequence(data.get("group_filters", []), "group_filters")
    ):
        if isinstance(item, dict) and ("column" in item or "field" in item):
            filters.append(_filter_from_data(item, f"group_filters[{index}]", default_table=base_table))
            continue
        context = f"group_filters[{index}]"
        group_filter = _object(item, context, {"alias", "operator", "value"})
        group_filters.append(
            GroupFilter(
                alias=_string(group_filter.get("alias"), f"{context}.alias"),
                operator=_string(
                    group_filter.get("operator"), f"{context}.operator"
                ),
                value=group_filter.get("value"),
            )
        )
    aggregations = []
    for index, item in enumerate(
        _sequence(data.get("aggregations", []), "aggregations")
    ):
        context = f"aggregations[{index}]"
        aggregation = _object(
            item,
            context,
            {
                "function",
                "column",
                "alias",
                "filters",
                "time_range",
                "unit_conversion",
            },
        )
        column_value = aggregation.get("column")
        scoped_filters = tuple(
            _filter_from_data(filter_item, f"{context}.filters[{filter_index}]", default_table=base_table)
            for filter_index, filter_item in enumerate(
                _sequence(aggregation.get("filters", []), f"{context}.filters")
            )
        )
        time_range_value = aggregation.get("time_range")
        conversion_value = aggregation.get("unit_conversion")
        fn = _string(aggregation.get("function"), f"{context}.function")
        if fn == "count_distinct" and (column_value is None or column_value == ""):
            fn = "count"
        aggregations.append(
            Aggregation(
                function=fn,
                column=(
                    None
                    if column_value is None
                    else _column_ref(column_value, f"{context}.column", default_table=base_table)
                ),
                alias=_string(aggregation.get("alias") or f"agg_{index}", f"{context}.alias"),
                filters=scoped_filters,
                time_range=(
                    None
                    if time_range_value is None
                    else _time_range_from_data(
                        time_range_value,
                        f"{context}.time_range",
                    )
                ),
                unit_conversion=(
                    None
                    if conversion_value is None
                    else _unit_conversion_from_data(
                        conversion_value,
                        f"{context}.unit_conversion",
                    )
                ),
            )
        )

    group_exclusions = []
    for index, item in enumerate(
        _sequence(data.get("group_exclusions", []), "group_exclusions")
    ):
        context = f"group_exclusions[{index}]"
        exclusion = _object(item, context, {"key", "filters"})
        group_exclusions.append(
            GroupExclusion(
                key=_column_ref(exclusion.get("key"), f"{context}.key"),
                filters=tuple(
                    _filter_from_data(filter_item, f"{context}.filters[{filter_index}]", default_table=base_table)
                    for filter_index, filter_item in enumerate(
                        _sequence(
                            exclusion.get("filters", []),
                            f"{context}.filters",
                        )
                    )
                ),
            )
        )

    calculations = []
    for index, item in enumerate(
        _sequence(data.get("calculations", []), "calculations")
    ):
        context = f"calculations[{index}]"
        calculation = _object(
            item,
            context,
            {"function", "left", "right", "alias", "rounding_places"},
        )
        rounding_places = calculation.get("rounding_places")
        if rounding_places is not None and (
            isinstance(rounding_places, bool) or not isinstance(rounding_places, int)
        ):
            raise QueryPlanFormatError(
                f"{context}.rounding_places must be an integer or null"
            )
        calculations.append(
            Calculation(
                function=_string(
                    calculation.get("function"), f"{context}.function"
                ),
                left=_string(calculation.get("left"), f"{context}.left"),
                right=_string(calculation.get("right"), f"{context}.right"),
                alias=_string(calculation.get("alias"), f"{context}.alias"),
                rounding_places=rounding_places,
            )
        )

    sort = []
    for index, item in enumerate(_sequence(data.get("sort") or [], "sort")):
        context = f"sort[{index}]"
        if isinstance(item, str):
            item = {"key": item, "direction": "asc"}
        elif isinstance(item, dict):
            if "key" not in item:
                for alt in ("column", "column_name", "field", "name"):
                    if alt in item:
                        item = dict(item)
                        item["key"] = item.pop(alt)
                        break
        sort_data = _object(item, context, {"key", "direction"})
        key_value = sort_data.get("key")
        key = (
            _column_ref(key_value, f"{context}.key", default_table=base_table)
            if isinstance(key_value, dict)
            else _string(key_value, f"{context}.key")
        )
        sort.append(
            Sort(
                key=key,
                direction=_string(
                    sort_data.get("direction", "asc"), f"{context}.direction"
                ),
            )
        )

    limit = data.get("limit")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int)):
        raise QueryPlanFormatError("limit must be an integer or null")

    time_range_value = data.get("time_range")
    time_range = None
    if time_range_value is not None:
        time_range = _time_range_from_data(time_range_value, "time_range")

    time_buckets = []
    for index, item in enumerate(
        _sequence(data.get("time_buckets") or [], "time_buckets")
    ):
        context = f"time_buckets[{index}]"
        bucket = _object(item, context, {"column", "granularity", "alias"})
        time_buckets.append(
            TimeBucket(
                column=_column_ref(bucket.get("column"), f"{context}.column", default_table=base_table),
                granularity=_string(
                    bucket.get("granularity"),
                    f"{context}.granularity",
                ),
                alias=_string(bucket.get("alias"), f"{context}.alias"),
            )
        )

    business_rules = tuple(
        _string(item, f"business_rules[{index}]")
        for index, item in enumerate(
            _sequence(data.get("business_rules", []), "business_rules")
        )
    )
    output_type = data.get("output_type")
    type_alias = {
        "int": "number", "integer": "number", "float": "number", "decimal": "number",
        "str": "string", "text": "string",
        "array": "list",
        "bool": "boolean",
    }
    if output_type in type_alias:
        output_type = type_alias[output_type]
    if output_type is None:
        if bool(aggregations) and not bool(group_by):
            output_type = "number"
        elif bool(group_by) or len(selected_columns) > 1:
            output_type = "table"
        elif len(selected_columns) == 1:
            output_type = "list"
        else:
            output_type = "table"
    elif not isinstance(output_type, str):
        raise QueryPlanFormatError("output_type must be a string or null")
    evidence_requirements = tuple(
        _column_ref(item, f"evidence_requirements[{index}]")
        for index, item in enumerate(
            _sequence(data.get("evidence_requirements", []), "evidence_requirements")
        )
    )
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        raise QueryPlanFormatError("metadata must be a JSON object")

    return QueryPlan(
        base_table=base_table,
        joins=tuple(joins),
        selected_columns=selected_columns,
        filters=tuple(filters),
        group_by=group_by,
        group_filters=tuple(group_filters),
        group_exclusions=tuple(group_exclusions),
        aggregations=tuple(aggregations),
        calculations=tuple(calculations),
        count_groups=_boolean(data.get("count_groups", False), "count_groups"),
        sort=tuple(sort),
        limit=limit,
        time_range=time_range,
        time_buckets=tuple(time_buckets),
        business_rules=business_rules,
        output_type=output_type,
        evidence_requirements=evidence_requirements,
        metadata=metadata,
    )
