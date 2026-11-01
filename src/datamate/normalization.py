"""Explicit, fail-closed conversion of raw table cells to typed values."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Iterable, Literal, Mapping, Sequence

ValueKind = Literal["number", "integer", "date", "datetime", "boolean", "text", "identifier"]
_NULL_STRINGS = {"", "null", "none", "nan", "n/a", "na", "-"}
_TRUE_STRINGS = {"true", "yes", "y"}
_FALSE_STRINGS = {"false", "no", "n"}


class ValueNormalizationError(ValueError):
    """Raised when a source value cannot be converted without guessing."""


@dataclass(frozen=True)
class ColumnCoercion:
    kind: ValueKind
    decimal_separator: str = "."
    thousands_separator: str | None = None
    date_format: str | None = None
    fill_down_group: str | None = None
    deduplicate_group: str | None = None
    aliases: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        for field_name, group in (
            ("fill-down", self.fill_down_group),
            ("deduplication", self.deduplicate_group),
        ):
            if group is not None and (not isinstance(group, str) or not group):
                raise ValueError(f"{field_name} group must be non-empty text")
        if self.aliases is not None:
            if self.kind != "text":
                raise ValueError("aliases can only be configured for text columns")
            if not isinstance(self.aliases, Mapping):
                raise ValueError("aliases must be an object mapping text to text")
            normalized_aliases: set[str] = set()
            for alias, canonical in self.aliases.items():
                if (
                    not isinstance(alias, str)
                    or not alias.strip()
                    or not isinstance(canonical, str)
                    or not canonical.strip()
                ):
                    raise ValueError("aliases must map non-empty text to non-empty text")
                normalized_alias = alias.strip().casefold()
                if normalized_alias in normalized_aliases:
                    raise ValueError("aliases cannot contain duplicate case-insensitive keys")
                normalized_aliases.add(normalized_alias)
        if len(self.decimal_separator) != 1:
            raise ValueError("decimal separator must be one character")
        if (
            self.thousands_separator is not None
            and len(self.thousands_separator) != 1
        ):
            raise ValueError("thousands separator must be one character")
        if self.decimal_separator and self.decimal_separator == self.thousands_separator:
            raise ValueError("decimal and thousands separators must differ")
        if self.kind in {"date", "datetime"} and self.thousands_separator is not None:
            raise ValueError("date coercions do not use a thousands separator")


def _decimal(value: object, policy: ColumnCoercion, column: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueNormalizationError(f"{column}: boolean is not a numeric value")
    if isinstance(value, (int, float, Decimal)):
        text = str(value)
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise ValueNormalizationError(f"{column}: value is not numeric text")

    if not text or text.casefold() in _NULL_STRINGS:
        raise ValueNormalizationError(f"{column}: value is empty")
    if policy.thousands_separator and policy.thousands_separator in text:
        integer_part, decimal_mark, fractional_part = text.partition(
            policy.decimal_separator
        )
        if decimal_mark and policy.decimal_separator in fractional_part:
            raise ValueNormalizationError(f"{column}: invalid decimal separators")
        sign = ""
        if integer_part[:1] in {"+", "-"}:
            sign, integer_part = integer_part[0], integer_part[1:]
        groups = integer_part.split(policy.thousands_separator)
        if (
            not 1 <= len(groups[0]) <= 3
            or not groups[0].isdigit()
            or any(len(group) != 3 or not group.isdigit() for group in groups[1:])
        ):
            raise ValueNormalizationError(f"{column}: invalid thousands grouping")
        text = sign + "".join(groups)
        if decimal_mark:
            text += policy.decimal_separator + fractional_part

    if policy.decimal_separator != ".":
        if text.count(policy.decimal_separator) > 1:
            raise ValueNormalizationError(f"{column}: invalid decimal separators")
        text = text.replace(policy.decimal_separator, ".")
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise ValueNormalizationError(f"{column}: invalid numeric value") from exc
    if not number.is_finite():
        raise ValueNormalizationError(f"{column}: numeric value must be finite")
    return number


def normalize_value(
    value: object, policy: ColumnCoercion, *, column: str
) -> object:
    """Convert one cell according to an explicit policy; never infer locale."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().casefold() in _NULL_STRINGS:
        return None

    if policy.kind == "number":
        return _decimal(value, policy, column)
    if policy.kind == "integer":
        number = _decimal(value, policy, column)
        if number != number.to_integral_value():
            raise ValueNormalizationError(f"{column}: value is not an integer")
        return int(number)
    if policy.kind == "date":
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if not isinstance(value, str):
            raise ValueNormalizationError(f"{column}: value is not a date")
        try:
            return (
                datetime.strptime(value.strip(), policy.date_format).date()
                if policy.date_format
                else date.fromisoformat(value.strip())
            )
        except ValueError as exc:
            raise ValueNormalizationError(
                f"{column}: date does not match the declared format"
            ) from exc
    if policy.kind == "datetime":
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return (
                    datetime.strptime(value.strip(), policy.date_format)
                    if policy.date_format
                    else datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
                )
            except ValueError as exc:
                raise ValueNormalizationError(
                    f"{column}: datetime does not match the declared format"
                ) from exc
        raise ValueNormalizationError(f"{column}: value is not a datetime")
    if policy.kind == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().casefold()
            if normalized in _TRUE_STRINGS:
                return True
            if normalized in _FALSE_STRINGS:
                return False
        raise ValueNormalizationError(f"{column}: boolean token is not recognized")
    if policy.kind in {"text", "identifier"}:
        text = value if isinstance(value, str) else str(value)
        if policy.aliases is not None:
            normalized = text.strip().casefold()
            for alias, canonical in policy.aliases.items():
                if alias.strip().casefold() == normalized:
                    return canonical
        return text
    raise ValueNormalizationError(f"{column}: unsupported conversion policy")


def normalize_rows(
    rows: Iterable[Sequence[object]],
    columns: Sequence[str],
    policies: Mapping[str, ColumnCoercion],
    *,
    max_dedup_rows: int = 100_000,
) -> Iterable[tuple[object, ...]]:
    """Normalize each row lazily, preserving bounded-memory streaming."""
    if (
        isinstance(max_dedup_rows, bool)
        or not isinstance(max_dedup_rows, int)
        or max_dedup_rows < 1
    ):
        raise ValueError("max_dedup_rows must be positive")
    if len(set(columns)) != len(columns):
        raise ValueNormalizationError("Duplicate column names cannot be normalized safely")
    indexes = {column: index for index, column in enumerate(columns)}
    unknown_columns = set(policies) - indexes.keys()
    if unknown_columns:
        raise ValueNormalizationError(
            f"Coercion references unknown columns: {', '.join(sorted(unknown_columns))}"
        )
    fill_down_policies = {
        column: policy.fill_down_group
        for column, policy in policies.items()
        if policy.fill_down_group is not None
    }
    unknown_group_columns = set(fill_down_policies.values()) - indexes.keys()
    deduplicate_policies = {
        column: policy.deduplicate_group
        for column, policy in policies.items()
        if policy.deduplicate_group is not None
    }
    deduplicate_group_columns = set(deduplicate_policies.values())
    if len(deduplicate_group_columns) > 1:
        raise ValueNormalizationError(
            "Exact-row deduplication supports one group column per table"
        )
    unknown_group_columns.update(set(deduplicate_policies.values()) - indexes.keys())
    if unknown_group_columns:
        raise ValueNormalizationError(
            "Normalization group references unknown columns: "
            + ", ".join(sorted(unknown_group_columns))
        )
    self_group_columns = {
        column
        for column, group_column in fill_down_policies.items()
        if column == group_column
    }
    other_fill_down_columns = {
        column: group_column
        for column, group_column in fill_down_policies.items()
        if column != group_column
    }
    group_state: dict[str, object] = {}
    value_state: dict[str, tuple[object, object]] = {}
    seen_rows: dict[object, set[tuple[object, ...]]] = {}
    seen_row_count = 0
    for row_number, raw_row in enumerate(rows, start=1):
        row = tuple(raw_row)
        if len(row) != len(columns):
            raise ValueNormalizationError(
                f"Row width mismatch at row {row_number}: expected {len(columns)}, got {len(row)}"
            )
        values = list(row)
        for column, policy in policies.items():
            try:
                values[indexes[column]] = normalize_value(
                    values[indexes[column]], policy, column=column
                )
            except ValueNormalizationError as exc:
                raise ValueNormalizationError(
                    f"Row {row_number}: {exc}"
                ) from exc

        for column in self_group_columns:
            column_index = indexes[column]
            value = values[column_index]
            if value is not None:
                group_state[column] = value
            elif column in group_state:
                value = group_state[column]
                values[column_index] = value

        for column, group_column in other_fill_down_columns.items():
            group_value = values[indexes[group_column]]
            column_index = indexes[column]
            value = values[column_index]
            if group_value is None:
                value_state.pop(column, None)
                continue
            previous = value_state.get(column)
            if previous is None or previous[0] != group_value:
                if value is None:
                    value_state.pop(column, None)
                else:
                    value_state[column] = (group_value, value)
                continue
            if value is None:
                values[column_index] = previous[1]
            else:
                value_state[column] = (group_value, value)
        row_values = tuple(values)
        if deduplicate_group_columns:
            group_column = next(iter(deduplicate_group_columns))
            group_value = values[indexes[group_column]]
        else:
            group_value = None
        if group_value is not None:
            try:
                group_rows = seen_rows.setdefault(group_value, set())
                if row_values in group_rows:
                    continue
                else:
                    if seen_row_count >= max_dedup_rows:
                        raise ValueNormalizationError(
                            "Exact-row deduplication exceeded the configured "
                            f"row limit ({max_dedup_rows})"
                        )
                    group_rows.add(row_values)
                    seen_row_count += 1
            except TypeError as exc:
                raise ValueNormalizationError(
                    f"Row {row_number}: deduplication group and row values "
                    "must be hashable"
                ) from exc
        yield row_values
