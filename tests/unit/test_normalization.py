from datetime import date
from decimal import Decimal

import pytest

from datamate.normalization import (
    ColumnCoercion,
    ValueNormalizationError,
    normalize_rows,
    normalize_value,
)


def test_number_coercion_uses_declared_locale_and_exact_decimal() -> None:
    policy = ColumnCoercion(
        "number",
        decimal_separator=",",
        thousands_separator=".",
    )

    assert normalize_value("1.234,50", policy, column="amount") == Decimal("1234.50")
    assert normalize_value("2,50", policy, column="amount") == Decimal("2.50")
    assert normalize_value("", policy, column="amount") is None


def test_identifier_and_date_coercions_preserve_explicit_semantics() -> None:
    identifier = ColumnCoercion("identifier")
    day_first = ColumnCoercion("date", date_format="%d/%m/%Y")

    assert normalize_value("000012", identifier, column="code") == "000012"
    assert normalize_value("31/01/2026", day_first, column="date") == date(2026, 1, 31)


def test_normalization_fails_closed_for_bad_grouping_and_ambiguous_dates() -> None:
    number = ColumnCoercion("number", decimal_separator=",", thousands_separator=".")
    iso_date = ColumnCoercion("date")

    with pytest.raises(ValueNormalizationError, match="invalid thousands grouping"):
        normalize_value("12.34,50", number, column="amount")
    with pytest.raises(ValueNormalizationError, match="declared format"):
        normalize_value("31/01/2026", iso_date, column="date")


def test_rows_are_normalized_lazily_and_width_errors_are_explicit() -> None:
    rows = normalize_rows(
        [("12.50", "yes"), ("", "no")],
        ("amount", "active"),
        {
            "amount": ColumnCoercion("number"),
            "active": ColumnCoercion("boolean"),
        },
    )

    assert next(iter(rows)) == (Decimal("12.50"), True)
    with pytest.raises(ValueNormalizationError, match="Row width mismatch"):
        list(normalize_rows([("one",)], ("a", "b"), {}))


def test_normalization_rejects_duplicate_columns_and_unknown_policy_names() -> None:
    with pytest.raises(ValueNormalizationError, match="Duplicate column names"):
        list(normalize_rows([], ("amount", "amount"), {}))
    with pytest.raises(ValueNormalizationError, match="unknown columns"):
        list(
            normalize_rows(
                [],
                ("amount",),
                {"missing": ColumnCoercion("number")},
            )
        )


def test_rows_fill_sparse_group_keys_and_values_only_when_explicitly_configured() -> None:
    rows = normalize_rows(
        [
            ("O-1", "completed", "first"),
            (None, None, "second"),
            ("O-2", "returned", "third"),
            (None, None, "fourth"),
            ("O-2", "unknown", "ungrouped"),
            ("O-3", None, "new group without status"),
        ],
        ("order_id", "status", "note"),
        {
            "order_id": ColumnCoercion(
                "identifier",
                fill_down_group="order_id",
            ),
            "status": ColumnCoercion(
                "text",
                fill_down_group="order_id",
            ),
        },
    )

    assert list(rows) == [
        ("O-1", "completed", "first"),
        ("O-1", "completed", "second"),
        ("O-2", "returned", "third"),
        ("O-2", "returned", "fourth"),
        ("O-2", "unknown", "ungrouped"),
        ("O-3", None, "new group without status"),
    ]


def test_fill_down_rejects_unknown_group_column() -> None:
    rows = normalize_rows(
        [("value",)],
        ("status",),
        {
            "status": ColumnCoercion(
                "text",
                fill_down_group="order_id",
            )
        },
    )

    with pytest.raises(ValueNormalizationError, match="unknown columns"):
        list(rows)


def test_rows_deduplicate_exact_duplicates_only_within_configured_group() -> None:
    rows = normalize_rows(
        [
            ("O-1", "item-a", 2),
            ("O-1", "item-a", 2),
            ("O-1", "item-a", 3),
            ("O-2", "item-a", 2),
            (None, "item-a", 2),
            (None, "item-a", 2),
        ],
        ("order_id", "product", "quantity"),
        {
            "order_id": ColumnCoercion(
                "identifier",
                fill_down_group="order_id",
                deduplicate_group="order_id",
            )
        },
    )

    assert list(rows) == [
        ("O-1", "item-a", 2),
        ("O-1", "item-a", 3),
        ("O-2", "item-a", 2),
    ]
    ungrouped_rows = normalize_rows(
        [(None, "item-a", 2), (None, "item-a", 2)],
        ("order_id", "product", "quantity"),
        {
            "order_id": ColumnCoercion(
                "identifier",
                deduplicate_group="order_id",
            )
        },
    )
    assert list(ungrouped_rows) == [
        (None, "item-a", 2),
        (None, "item-a", 2),
    ]


def test_exact_row_deduplication_fails_closed_at_its_bound() -> None:
    rows = normalize_rows(
        [("O-1", "A"), ("O-1", "B")],
        ("order_id", "product"),
        {
            "order_id": ColumnCoercion(
                "identifier",
                deduplicate_group="order_id",
            )
        },
        max_dedup_rows=1,
    )

    with pytest.raises(ValueNormalizationError, match="row limit"):
        list(rows)


def test_text_aliases_are_canonicalized_only_when_explicitly_configured() -> None:
    rows = normalize_rows(
        [("latte",), ("CAFÉ LATTE ",), ("Latte macchiato",)],
        ("product",),
        {
            "product": ColumnCoercion(
                "text",
                aliases={
                    "Latte": "Cà phê sữa",
                    "Café latte": "Cà phê sữa",
                },
            )
        },
    )

    assert list(rows) == [
        ("Cà phê sữa",),
        ("Cà phê sữa",),
        ("Latte macchiato",),
    ]


def test_alias_policy_rejects_non_text_and_conflicting_keys() -> None:
    with pytest.raises(ValueError, match="text columns"):
        ColumnCoercion("identifier", aliases={"id": "canonical"})
    with pytest.raises(ValueError, match="duplicate case-insensitive keys"):
        ColumnCoercion("text", aliases={"alias": "A", "ALIAS ": "B"})
