from datetime import date, datetime
from decimal import Decimal

import pytest

from datamate.query_executor import QueryExecutionError, execute_query_plan
from datamate.query_plan import (
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


def test_streamed_aggregation_uses_decimal_and_null_semantics() -> None:
    table = "sales"
    product = ColumnRef(table, "product")
    amount = ColumnRef(table, "amount")
    order_date = ColumnRef(table, "date")
    plan = QueryPlan(
        base_table=table,
        selected_columns=(product,),
        filters=(Filter(amount, "gt", Decimal(0)),),
        group_by=(product,),
        aggregations=(
            Aggregation("sum", amount, "revenue"),
            Aggregation("avg", amount, "average"),
            Aggregation("count", amount, "count"),
        ),
        sort=(Sort("revenue", "desc"),),
        time_range=TimeRange(order_date, "2026-01-01", "2026-01-31"),
        limit=2,
    )

    result = execute_query_plan(
        plan,
        catalog={table: ("product", "amount", "date")},
        sources={
            table: [
                ("Tea", Decimal("10.10"), "2026-01-01"),
                ("Tea", None, "2026-01-02"),
                ("Coffee", Decimal("100"), "2026-01-03"),
                ("Tea", Decimal("2.30"), "2026-02-01"),
            ]
        },
    )

    assert result.rows == (
        {
            "sales.product": "Coffee",
            "revenue": Decimal("100"),
            "average": Decimal("100"),
            "count": 1,
        },
        {
            "sales.product": "Tea",
            "revenue": Decimal("10.10"),
            "average": Decimal("10.10"),
            "count": 1,
        },
    )
    assert result.trace.source_rows == {table: 4}
    assert result.trace.joined_rows == 4
    assert result.trace.filtered_rows == 2
    assert result.trace.group_count == 2


def test_count_distinct_counts_non_null_business_keys_once() -> None:
    table = "order_lines"
    order_id = ColumnRef(table, "order_id")
    plan = QueryPlan(
        base_table=table,
        aggregations=(
            Aggregation("count_distinct", order_id, "order_count"),
        ),
    )

    result = execute_query_plan(
        plan,
        catalog={table: ("order_id", "line_amount")},
        sources={
            table: [
                ("A", Decimal(10)),
                ("A", Decimal(20)),
                ("B", Decimal(30)),
                (None, Decimal(40)),
            ]
        },
    )

    assert result.rows == ({"order_count": 2},)


def test_group_exclusion_drops_every_row_of_a_matching_group() -> None:
    table = "order_lines"
    order_id = ColumnRef(table, "order_id")
    quantity = ColumnRef(table, "quantity")
    revenue = ColumnRef(table, "revenue")
    status = ColumnRef(table, "status")
    plan = QueryPlan(
        base_table=table,
        filters=(Filter(status, "ne", "cancelled"),),
        aggregations=(Aggregation("sum", revenue, "revenue"),),
        group_exclusions=(
            GroupExclusion(
                key=order_id,
                filters=(Filter(quantity, "lt", 0),),
            ),
        ),
    )

    result = execute_query_plan(
        plan,
        catalog={
            table: ("order_id", "quantity", "revenue", "status"),
        },
        sources={
            table: [
                ("A", 1, Decimal(20), "completed"),
                ("A", 1, Decimal(10), "completed"),
                ("B", 2, Decimal(40), "completed"),
                ("B", -2, Decimal(-40), "completed"),
                ("C", 1, Decimal(50), "cancelled"),
                ("D", 1, Decimal(5), "in_delivery"),
            ],
        },
    )

    assert result.rows == ({"revenue": Decimal(35)},)
    assert result.trace.filtered_rows == 3


def test_group_exclusion_is_bounded_and_requires_replayable_sources() -> None:
    table = "order_lines"
    order_id = ColumnRef(table, "order_id")
    quantity = ColumnRef(table, "quantity")
    plan = QueryPlan(
        base_table=table,
        aggregations=(
            Aggregation("count", None, "rows"),
        ),
        group_exclusions=(
            GroupExclusion(
                key=order_id,
                filters=(Filter(quantity, "lt", 0),),
            ),
        ),
    )
    catalog = {table: ("order_id", "quantity")}
    rows = [
        ("A", -1),
        ("B", -1),
    ]

    with pytest.raises(
        QueryExecutionError,
        match="Excluded group count exceeds configured limit 1",
    ):
        execute_query_plan(
            plan,
            catalog=catalog,
            sources={table: rows},
            max_group_exclusions=1,
        )

    with pytest.raises(
        QueryExecutionError,
        match="must provide a fresh iterator",
    ):
        execute_query_plan(
            plan,
            catalog=catalog,
            sources={table: iter(rows)},
        )


def test_group_exclusion_rejects_marker_rows_without_group_keys() -> None:
    table = "order_lines"
    plan = QueryPlan(
        base_table=table,
        aggregations=(Aggregation("count", None, "rows"),),
        group_exclusions=(
            GroupExclusion(
                key=ColumnRef(table, "order_id"),
                filters=(Filter(ColumnRef(table, "quantity"), "lt", 0),),
            ),
        ),
    )

    with pytest.raises(
        QueryExecutionError,
        match="matching source row has a null key",
    ):
        execute_query_plan(
            plan,
            catalog={table: ("order_id", "quantity")},
            sources={table: [(None, -1)]},
        )


def test_group_filters_count_distinct_groups_meeting_threshold() -> None:
    table = "order_lines"
    product = ColumnRef(table, "product")
    order_id = ColumnRef(table, "order_id")
    plan = QueryPlan(
        base_table=table,
        group_by=(product,),
        group_filters=(GroupFilter("orders", "gte", 2),),
        aggregations=(Aggregation("count_distinct", order_id, "orders"),),
        count_groups=True,
        output_type="number",
    )

    result = execute_query_plan(
        plan,
        catalog={table: ("product", "order_id")},
        sources={
            table: [
                ("A", "O-1"),
                ("A", "O-1"),
                ("A", "O-2"),
                ("B", "O-1"),
                ("B", "O-2"),
                ("C", "O-3"),
            ]
        },
    )

    assert result.rows == ({"group_count": 2},)
    assert result.trace.group_count == 3


def test_group_filter_compares_decimal_metric_to_json_float_exactly() -> None:
    table = "sales"
    product = ColumnRef(table, "product")
    amount = ColumnRef(table, "amount")
    plan = QueryPlan(
        base_table=table,
        group_by=(product,),
        group_filters=(GroupFilter("average", "gte", 1.1),),
        aggregations=(Aggregation("avg", amount, "average"),),
        count_groups=True,
        output_type="number",
    )

    result = execute_query_plan(
        plan,
        catalog={table: ("product", "amount")},
        sources={
            table: [
                ("boundary", Decimal("1.1")),
                ("below", Decimal("1.099")),
            ]
        },
    )

    assert result.rows == ({"group_count": 1},)


def test_count_distinct_is_bounded_and_reports_resource_limit() -> None:
    table = "order_lines"
    order_id = ColumnRef(table, "order_id")
    plan = QueryPlan(
        base_table=table,
        aggregations=(
            Aggregation("count_distinct", order_id, "order_count"),
        ),
    )

    with pytest.raises(
        QueryExecutionError,
        match="Distinct value count exceeds configured limit 2",
    ):
        execute_query_plan(
            plan,
            catalog={table: ("order_id",)},
            sources={table: [("A",), ("B",), ("C",)]},
            max_distinct_values=2,
        )


def test_left_join_keeps_unmatched_rows_and_aggregates_exactly() -> None:
    orders = "orders"
    customers = "customers"
    customer_name = ColumnRef(customers, "name")
    amount = ColumnRef(orders, "amount")
    plan = QueryPlan(
        base_table=orders,
        joins=(
            Join(
                ColumnRef(orders, "customer_id"),
                ColumnRef(customers, "id"),
                kind="left",
            ),
        ),
        group_by=(customer_name,),
        aggregations=(Aggregation("sum", amount, "revenue"),),
        sort=(Sort("revenue", "desc"),),
    )

    result = execute_query_plan(
        plan,
        catalog={
            orders: ("customer_id", "amount"),
            customers: ("id", "name"),
        },
        sources={
            orders: [(1, Decimal("10")), (2, Decimal("5")), (3, Decimal("7"))],
            customers: [(1, "An"), (2, "Bình")],
        },
    )

    assert [row["revenue"] for row in result.rows] == [
        Decimal("10"),
        Decimal("7"),
        Decimal("5"),
    ]
    assert result.rows[1][f"{customers}.name"] is None
    assert result.trace.source_rows == {customers: 2, orders: 3}
    assert result.trace.joined_rows == 3


def test_count_star_and_null_filter_work_for_empty_numeric_columns() -> None:
    table = "items"
    value = ColumnRef(table, "value")
    plan = QueryPlan(
        base_table=table,
        group_by=(value,),
        aggregations=(Aggregation("count", None, "rows"),),
    )

    result = execute_query_plan(
        plan,
        catalog={table: ("value",)},
        sources={table: [(None,), (None,), ("x",)]},
    )

    assert result.rows == (
        {f"{table}.value": None, "rows": 2},
        {f"{table}.value": "x", "rows": 1},
    )


def test_executor_rejects_non_numeric_values_and_configured_limit_overflow() -> None:
    table = "sales"
    amount = ColumnRef(table, "amount")
    aggregate = QueryPlan(
        base_table=table,
        aggregations=(Aggregation("sum", amount, "revenue"),),
    )
    kwargs = {"catalog": {table: ("amount",)}}

    with pytest.raises(QueryExecutionError, match="requires numeric values"):
        execute_query_plan(aggregate, sources={table: [("12,50",)]}, **kwargs)

    grouped = QueryPlan(base_table=table, group_by=(amount,))
    with pytest.raises(QueryExecutionError, match="Group count exceeds"):
        execute_query_plan(
            grouped,
            sources={table: [(1,), (2,)]},
            max_groups=1,
            **kwargs,
        )


def test_large_decimal_sum_does_not_round_to_default_context_precision() -> None:
    table = "ledger"
    amount = ColumnRef(table, "amount")
    plan = QueryPlan(
        base_table=table,
        aggregations=(Aggregation("sum", amount, "total"),),
    )

    result = execute_query_plan(
        plan,
        catalog={table: ("amount",)},
        sources={
            table: [
                (Decimal("9999999999999999999999999999"),),
                (Decimal("1"),),
            ]
        },
    )

    assert result.rows[0]["total"] == Decimal("10000000000000000000000000000")


def test_decimal_addition_preserves_small_terms_at_large_exponents() -> None:
    table = "ledger"
    amount = ColumnRef(table, "amount")
    plan = QueryPlan(
        base_table=table,
        aggregations=(Aggregation("sum", amount, "total"),),
    )

    result = execute_query_plan(
        plan,
        catalog={table: ("amount",)},
        sources={table: [(Decimal("1E+100"),), (Decimal("1"),)]},
    )

    assert result.rows[0]["total"] == Decimal("1" + "0" * 99 + "1")


def test_ratio_and_percent_change_are_deterministic_and_handle_zero_denominator() -> None:
    table = "periods"
    current = ColumnRef(table, "current")
    previous = ColumnRef(table, "previous")
    plan = QueryPlan(
        base_table=table,
        aggregations=(
            Aggregation("sum", current, "current_total"),
            Aggregation("sum", previous, "previous_total"),
        ),
        calculations=(
            Calculation(
                "percent_change",
                "current_total",
                "previous_total",
                "growth_percent",
            ),
            Calculation(
                "ratio",
                "current_total",
                "previous_total",
                "ratio",
            ),
        ),
    )
    arguments = {
        "catalog": {table: ("current", "previous")},
        "sources": {table: [(Decimal("120"), Decimal("100"))]},
    }

    result = execute_query_plan(plan, **arguments)

    assert result.rows[0]["growth_percent"] == Decimal("20")
    assert result.rows[0]["ratio"] == Decimal("1.2")
    decreasing = execute_query_plan(
        plan,
        catalog=arguments["catalog"],
        sources={table: [(Decimal("80"), Decimal("100"))]},
    )
    assert decreasing.rows[0]["growth_percent"] == Decimal("-20")
    zero_denominator = execute_query_plan(
        plan,
        catalog=arguments["catalog"],
        sources={table: [(Decimal("120"), Decimal("0"))]},
    )
    assert zero_denominator.rows[0]["growth_percent"] is None
    assert zero_denominator.rows[0]["ratio"] is None


def test_ratio_rounding_places_use_half_up_for_monetary_averages() -> None:
    table = "orders"
    total = ColumnRef(table, "total")
    count = ColumnRef(table, "count")
    plan = QueryPlan(
        base_table=table,
        aggregations=(
            Aggregation("sum", total, "revenue"),
            Aggregation("sum", count, "orders"),
        ),
        calculations=(
            Calculation(
                "ratio",
                "revenue",
                "orders",
                "average_order_value",
                rounding_places=0,
            ),
        ),
    )

    result = execute_query_plan(
        plan,
        catalog={table: ("total", "count")},
        sources={table: [(Decimal(35), Decimal(2))]},
    )

    assert result.rows[0]["average_order_value"] == Decimal(18)


def test_sum_converts_large_units_to_requested_product_base_unit() -> None:
    sales = "sales"
    products = "products"
    product_id = ColumnRef(sales, "product_id")
    quantity = ColumnRef(sales, "quantity")
    unit = ColumnRef(sales, "unit")
    plan = QueryPlan(
        base_table=sales,
        joins=(
            Join(product_id, ColumnRef(products, "product_id"), kind="left"),
        ),
        aggregations=(
            Aggregation(
                "sum",
                quantity,
                "quantity_in_base_units",
                unit_conversion=UnitConversion(
                    source_unit=unit,
                    base_unit=ColumnRef(products, "base_unit"),
                    large_unit=ColumnRef(products, "large_unit"),
                    factor=ColumnRef(products, "factor"),
                    target_unit="kg",
                ),
            ),
        ),
    )

    result = execute_query_plan(
        plan,
        catalog={
            sales: ("product_id", "quantity", "unit"),
            products: ("product_id", "base_unit", "large_unit", "factor"),
        },
        sources={
            sales: [
                ("A", Decimal(2), "Bag"),
                ("A", Decimal(3), "kg"),
                ("B", Decimal(4), "box"),
                ("B", Decimal(2), "pc"),
                ("C", Decimal(3), "kg"),
            ],
            products: [
                ("A", "kg", "bag", Decimal(50)),
                ("B", "pc", "box", Decimal(12)),
            ],
        },
    )

    assert result.rows == ({"quantity_in_base_units": Decimal(106)},)


def test_sum_converts_each_product_to_its_own_base_unit() -> None:
    sales = "sales"
    products = "products"
    plan = QueryPlan(
        base_table=sales,
        joins=(
            Join(
                ColumnRef(sales, "product_id"),
                ColumnRef(products, "product_id"),
                kind="left",
            ),
        ),
        aggregations=(
            Aggregation(
                "sum",
                ColumnRef(sales, "quantity"),
                "quantity_in_product_base_units",
                unit_conversion=UnitConversion(
                    source_unit=ColumnRef(sales, "unit"),
                    base_unit=ColumnRef(products, "base_unit"),
                    large_unit=ColumnRef(products, "large_unit"),
                    factor=ColumnRef(products, "factor"),
                    target_unit=None,
                ),
            ),
        ),
    )

    result = execute_query_plan(
        plan,
        catalog={
            sales: ("product_id", "quantity", "unit"),
            products: ("product_id", "base_unit", "large_unit", "factor"),
        },
        sources={
            sales: [
                ("A", Decimal(2), "bag"),
                ("A", Decimal(3), "kg"),
                ("B", Decimal(4), "box"),
                ("B", Decimal(2), "pc"),
                ("C", Decimal(5), "bottle"),
            ],
            products: [
                ("A", "kg", "bag", Decimal(50)),
                ("B", "pc", "box", Decimal(12)),
            ],
        },
    )

    assert result.rows == (
        {"quantity_in_product_base_units": Decimal(158)},
    )


def test_unit_conversion_rejects_missing_transaction_unit() -> None:
    sales = "sales"
    products = "products"
    plan = QueryPlan(
        base_table=sales,
        joins=(
            Join(
                ColumnRef(sales, "product_id"),
                ColumnRef(products, "product_id"),
                kind="left",
            ),
        ),
        aggregations=(
            Aggregation(
                "sum",
                ColumnRef(sales, "quantity"),
                "quantity_in_base_units",
                unit_conversion=UnitConversion(
                    source_unit=ColumnRef(sales, "unit"),
                    base_unit=ColumnRef(products, "base_unit"),
                    large_unit=ColumnRef(products, "large_unit"),
                    factor=ColumnRef(products, "factor"),
                    target_unit=None,
                ),
            ),
        ),
    )

    with pytest.raises(QueryExecutionError, match="Missing source unit"):
        execute_query_plan(
            plan,
            catalog={
                sales: ("product_id", "quantity", "unit"),
                products: ("product_id", "base_unit", "large_unit", "factor"),
            },
            sources={
                sales: [("unknown", Decimal(5), None)],
                products: [],
            },
        )


def test_unit_conversion_rejects_unknown_units_and_duplicate_lookups() -> None:
    sales = "sales"
    products = "products"
    product_id = ColumnRef(sales, "product_id")
    plan = QueryPlan(
        base_table=sales,
        joins=(
            Join(product_id, ColumnRef(products, "product_id"), kind="left"),
        ),
        aggregations=(
            Aggregation(
                "sum",
                ColumnRef(sales, "quantity"),
                "quantity",
                unit_conversion=UnitConversion(
                    source_unit=ColumnRef(sales, "unit"),
                    base_unit=ColumnRef(products, "base_unit"),
                    large_unit=ColumnRef(products, "large_unit"),
                    factor=ColumnRef(products, "factor"),
                    target_unit="kg",
                ),
            ),
        ),
    )
    catalog = {
        sales: ("product_id", "quantity", "unit"),
        products: ("product_id", "base_unit", "large_unit", "factor"),
    }

    with pytest.raises(QueryExecutionError, match="Cannot convert source unit"):
        execute_query_plan(
            plan,
            catalog=catalog,
            sources={
                sales: [("A", Decimal(2), "carton")],
                products: [("A", "kg", "bag", Decimal(50))],
            },
        )

    with pytest.raises(
        QueryExecutionError,
        match="Unit-conversion lookup tables must have unique join keys",
    ):
        execute_query_plan(
            plan,
            catalog=catalog,
            sources={
                sales: [("A", Decimal(2), "bag")],
                products: [
                    ("A", "kg", "bag", Decimal(50)),
                    ("A", "kg", "bag", Decimal(100)),
                ],
            },
        )

    with pytest.raises(
        QueryExecutionError,
        match="Unit conversion factor.*must be positive",
    ):
        execute_query_plan(
            plan,
            catalog=catalog,
            sources={
                sales: [("A", Decimal(2), "bag")],
                products: [("A", "kg", "bag", Decimal(0))],
            },
        )


def test_joined_measure_table_is_not_mistaken_for_unique_conversion_lookup() -> None:
    customers = "customers"
    sales = "sales"
    products = "products"
    plan = QueryPlan(
        base_table=customers,
        joins=(
            Join(
                ColumnRef(customers, "customer_id"),
                ColumnRef(sales, "customer_id"),
            ),
            Join(
                ColumnRef(sales, "product_id"),
                ColumnRef(products, "product_id"),
                kind="left",
            ),
        ),
        aggregations=(
            Aggregation(
                "sum",
                ColumnRef(sales, "quantity"),
                "quantity_in_kg",
                unit_conversion=UnitConversion(
                    source_unit=ColumnRef(sales, "unit"),
                    base_unit=ColumnRef(products, "base_unit"),
                    large_unit=ColumnRef(products, "large_unit"),
                    factor=ColumnRef(products, "factor"),
                    target_unit="kg",
                ),
            ),
        ),
    )

    result = execute_query_plan(
        plan,
        catalog={
            customers: ("customer_id",),
            sales: ("customer_id", "product_id", "quantity", "unit"),
            products: ("product_id", "base_unit", "large_unit", "factor"),
        },
        sources={
            customers: [(1,)],
            sales: [
                (1, "A", Decimal(2), "bag"),
                (1, "A", Decimal(3), "kg"),
            ],
            products: [("A", "kg", "bag", Decimal(50))],
        },
    )

    assert result.rows == ({"quantity_in_kg": Decimal(103)},)


def test_percentage_calculations_round_half_up_to_two_decimal_places() -> None:
    table = "periods"
    numerator = ColumnRef(table, "numerator")
    denominator = ColumnRef(table, "denominator")
    plan = QueryPlan(
        base_table=table,
        aggregations=(
            Aggregation("sum", numerator, "numerator"),
            Aggregation("sum", denominator, "denominator"),
        ),
        calculations=(
            Calculation("ratio_percent", "numerator", "denominator", "share"),
            Calculation(
                "percent_change",
                "numerator",
                "denominator",
                "growth_percent",
            ),
        ),
    )

    result = execute_query_plan(
        plan,
        catalog={table: ("numerator", "denominator")},
        sources={table: [(Decimal(1), Decimal(3))]},
    )

    assert result.rows[0]["share"] == Decimal("33.33")
    assert result.rows[0]["growth_percent"] == Decimal("-66.67")

    half_cent_boundary = execute_query_plan(
        plan,
        catalog={table: ("numerator", "denominator")},
        sources={table: [(Decimal(1), Decimal(32))]},
    )
    assert half_cent_boundary.rows[0]["share"] == Decimal("3.13")
    assert half_cent_boundary.rows[0]["growth_percent"] == Decimal("-96.88")


def test_min_max_and_top_k_honor_explicit_tie_break_sort() -> None:
    table = "sales"
    product_code = ColumnRef(table, "product_code")
    quantity = ColumnRef(table, "quantity")
    plan = QueryPlan(
        base_table=table,
        group_by=(product_code,),
        aggregations=(
            Aggregation("sum", quantity, "units_sold"),
            Aggregation("min", quantity, "minimum_sale"),
            Aggregation("max", quantity, "maximum_sale"),
        ),
        sort=(Sort("units_sold", "desc"), Sort(product_code, "asc")),
        limit=2,
    )

    result = execute_query_plan(
        plan,
        catalog={table: ("product_code", "quantity")},
        sources={
            table: [
                ("P-02", Decimal(5)),
                ("P-01", Decimal(5)),
                ("P-03", Decimal(4)),
                ("P-01", Decimal(2)),
            ]
        },
    )

    assert result.rows == (
        {
            f"{table}.product_code": "P-01",
            "units_sold": Decimal(7),
            "minimum_sale": Decimal(2),
            "maximum_sale": Decimal(5),
        },
        {
            f"{table}.product_code": "P-02",
            "units_sold": Decimal(5),
            "minimum_sale": Decimal(5),
            "maximum_sale": Decimal(5),
        },
    )


def test_scoped_aggregations_compare_two_periods_from_the_same_table() -> None:
    table = "sales"
    product = ColumnRef(table, "product")
    amount = ColumnRef(table, "amount")
    sold_at = ColumnRef(table, "sold_at")
    plan = QueryPlan(
        base_table=table,
        group_by=(product,),
        aggregations=(
            Aggregation(
                "sum",
                amount,
                "previous_revenue",
                time_range=TimeRange(sold_at, "2026-01-01", "2026-01-31"),
            ),
            Aggregation(
                "sum",
                amount,
                "current_revenue",
                time_range=TimeRange(sold_at, "2026-02-01", "2026-02-28"),
            ),
        ),
        calculations=(
            Calculation(
                "percent_change",
                "current_revenue",
                "previous_revenue",
                "growth_percent",
            ),
        ),
        sort=(Sort(product, "asc"),),
    )

    result = execute_query_plan(
        plan,
        catalog={table: ("product", "amount", "sold_at")},
        sources={
            table: [
                ("A", Decimal(60), date(2026, 1, 10)),
                ("A", Decimal(40), date(2026, 1, 31)),
                ("A", Decimal(120), datetime(2026, 2, 28, 23, 59)),
                ("B", Decimal(50), date(2026, 1, 15)),
                ("B", Decimal(25), date(2026, 2, 10)),
                ("C", Decimal(500), date(2026, 3, 10)),
            ]
        },
    )

    assert result.rows == (
        {
            f"{table}.product": "A",
            "previous_revenue": Decimal(100),
            "current_revenue": Decimal(120),
            "growth_percent": Decimal(20),
        },
        {
            f"{table}.product": "B",
            "previous_revenue": Decimal(50),
            "current_revenue": Decimal(25),
            "growth_percent": Decimal(-50),
        },
    )


@pytest.mark.parametrize(
    ("granularity", "expected"),
    [
        ("day", [("2026-03-01", Decimal(10)), ("2026-03-02", Decimal(15))]),
        ("week", [("2026-02-23", Decimal(10)), ("2026-03-02", Decimal(15))]),
        ("month", [("2026-03", Decimal(25))]),
        ("quarter", [("2026-Q1", Decimal(25))]),
        ("year", [("2026", Decimal(25))]),
    ],
)
def test_time_bucket_groups_calendar_values_and_sums(
    granularity: str,
    expected: list[tuple[str, Decimal]],
) -> None:
    table = "sales"
    sold_at = ColumnRef(table, "sold_at")
    amount = ColumnRef(table, "amount")
    plan = QueryPlan(
        base_table=table,
        time_range=TimeRange(sold_at, "2026-03-01", "2026-03-02"),
        time_buckets=(TimeBucket(sold_at, granularity, "period"),),
        aggregations=(Aggregation("sum", amount, "revenue"),),
        sort=(Sort("period", "asc"),),
    )

    result = execute_query_plan(
        plan,
        catalog={table: ("sold_at", "amount")},
        sources={
            table: [
                (date(2026, 3, 1), Decimal(10)),
                (datetime(2026, 3, 2, 23, 59), Decimal(15)),
                (date(2026, 3, 3), Decimal(100)),
            ]
        },
    )

    assert [
        (row["period"], row["revenue"])
        for row in result.rows
    ] == expected


@pytest.mark.parametrize(
    ("end", "expected"),
    [
        ("2026-03-01", [("2026-01", Decimal(10)), ("2026-02", Decimal(0))]),
        ("2026-01-01", []),
    ],
)
def test_time_bucket_fills_only_periods_before_exclusive_end(
    end: str,
    expected: list[tuple[str, Decimal]],
) -> None:
    table = "sales"
    sold_at = ColumnRef(table, "sold_at")
    amount = ColumnRef(table, "amount")
    plan = QueryPlan(
        base_table=table,
        time_range=TimeRange(
            sold_at,
            "2026-01-01",
            end,
            inclusive_end=False,
        ),
        time_buckets=(TimeBucket(sold_at, "month", "period"),),
        aggregations=(Aggregation("sum", amount, "revenue"),),
    )
    result = execute_query_plan(
        plan,
        catalog={table: ("sold_at", "amount")},
        sources={
            table: [
                (date(2026, 1, 1), Decimal(10)),
                (date(2026, 3, 1), Decimal(30)),
            ]
        },
    )

    assert [
        (row["period"], row["revenue"])
        for row in result.rows
    ] == expected


def test_time_bucket_enforces_group_limit_for_generated_empty_periods() -> None:
    table = "sales"
    sold_at = ColumnRef(table, "sold_at")
    amount = ColumnRef(table, "amount")
    plan = QueryPlan(
        base_table=table,
        time_range=TimeRange(sold_at, "2026-01-01", "2026-03-31"),
        time_buckets=(TimeBucket(sold_at, "month", "period"),),
        aggregations=(Aggregation("sum", amount, "revenue"),),
        sort=(Sort("period", "asc"),),
    )

    with pytest.raises(QueryExecutionError, match="Group count exceeds configured limit"):
        execute_query_plan(
            plan,
            catalog={table: ("sold_at", "amount")},
            sources={
                table: [
                    (date(2026, 1, 1), Decimal(10)),
                    (date(2026, 3, 1), Decimal(30)),
                ]
            },
            max_groups=2,
        )


def test_time_bucket_fills_all_supported_aggregations_for_empty_periods() -> None:
    table = "sales"
    sold_at = ColumnRef(table, "sold_at")
    amount = ColumnRef(table, "amount")
    order_id = ColumnRef(table, "order_id")
    plan = QueryPlan(
        base_table=table,
        time_range=TimeRange(sold_at, "2026-01-01", "2026-03-31"),
        time_buckets=(TimeBucket(sold_at, "month", "period"),),
        aggregations=(
            Aggregation("sum", amount, "revenue"),
            Aggregation("count_distinct", order_id, "orders"),
        ),
        sort=(Sort("period", "asc"),),
    )
    result = execute_query_plan(
        plan,
        catalog={table: ("sold_at", "amount", "order_id")},
        sources={
            table: [
                (date(2026, 1, 1), Decimal(10), "O-1"),
                (date(2026, 1, 2), Decimal(15), "O-1"),
                (date(2026, 3, 1), Decimal(20), "O-2"),
            ]
        },
    )

    assert result.rows == (
        {"period": "2026-01", "revenue": Decimal(25), "orders": 1},
        {"period": "2026-02", "revenue": Decimal(0), "orders": 0},
        {"period": "2026-03", "revenue": Decimal(20), "orders": 1},
    )


def test_time_bucket_fills_empty_periods_per_observed_group_within_limit() -> None:
    table = "sales"
    product = ColumnRef(table, "product")
    sold_at = ColumnRef(table, "sold_at")
    amount = ColumnRef(table, "amount")
    order_id = ColumnRef(table, "order_id")
    plan = QueryPlan(
        base_table=table,
        group_by=(product,),
        time_range=TimeRange(sold_at, "2026-01-01", "2026-03-31"),
        time_buckets=(TimeBucket(sold_at, "month", "month"),),
        aggregations=(
            Aggregation("sum", amount, "revenue"),
            Aggregation("count_distinct", order_id, "orders"),
        ),
        sort=(Sort(product, "asc"), Sort("month", "asc")),
    )
    sources = {
        table: [
            ("A", date(2026, 1, 1), Decimal(10), "A-1"),
            ("A", date(2026, 3, 1), Decimal(30), "A-2"),
            ("B", date(2026, 3, 1), Decimal(5), "B-1"),
        ]
    }

    result = execute_query_plan(
        plan,
        catalog={table: ("product", "sold_at", "amount", "order_id")},
        sources=sources,
    )

    assert result.rows == (
        {
            f"{table}.product": "A",
            "month": "2026-01",
            "revenue": Decimal(10),
            "orders": 1,
        },
        {
            f"{table}.product": "A",
            "month": "2026-02",
            "revenue": Decimal(0),
            "orders": 0,
        },
        {
            f"{table}.product": "A",
            "month": "2026-03",
            "revenue": Decimal(30),
            "orders": 1,
        },
        {
            f"{table}.product": "B",
            "month": "2026-01",
            "revenue": Decimal(0),
            "orders": 0,
        },
        {
            f"{table}.product": "B",
            "month": "2026-02",
            "revenue": Decimal(0),
            "orders": 0,
        },
        {
            f"{table}.product": "B",
            "month": "2026-03",
            "revenue": Decimal(5),
            "orders": 1,
        },
    )

    with pytest.raises(
        QueryExecutionError,
        match="Group count exceeds configured limit 5",
    ):
        execute_query_plan(
            plan,
            catalog={table: ("product", "sold_at", "amount", "order_id")},
            sources={table: iter(sources[table])},
            max_groups=5,
        )


def test_time_bucket_preserves_null_sum_for_periods_with_only_null_values() -> None:
    table = "sales"
    sold_at = ColumnRef(table, "sold_at")
    amount = ColumnRef(table, "amount")
    plan = QueryPlan(
        base_table=table,
        time_range=TimeRange(sold_at, "2026-01-01", "2026-03-31"),
        time_buckets=(TimeBucket(sold_at, "month", "period"),),
        aggregations=(Aggregation("sum", amount, "revenue"),),
        sort=(Sort("period", "asc"),),
    )
    result = execute_query_plan(
        plan,
        catalog={table: ("sold_at", "amount")},
        sources={
            table: [
                (date(2026, 1, 1), None),
                (date(2026, 3, 1), Decimal(30)),
            ]
        },
    )

    assert [
        (row["period"], row["revenue"])
        for row in result.rows
    ] == [
        ("2026-01", None),
        ("2026-02", Decimal(0)),
        ("2026-03", Decimal(30)),
    ]


@pytest.mark.parametrize(
    ("values", "inclusive_end", "expected"),
    [
        (
            [date(2026, 1, 1), date(2026, 1, 31), date(2026, 2, 1)],
            True,
            [date(2026, 1, 1), date(2026, 1, 31)],
        ),
        (
            [
                datetime(2026, 1, 1, 0, 0),
                datetime(2026, 1, 31, 23, 59, 59),
                datetime(2026, 2, 1, 0, 0),
            ],
            True,
            [datetime(2026, 1, 1, 0, 0), datetime(2026, 1, 31, 23, 59, 59)],
        ),
        (
            [
                datetime(2026, 1, 1, 0, 0),
                datetime(2026, 1, 31, 0, 0),
                datetime(2026, 2, 1, 0, 0),
            ],
            False,
            [datetime(2026, 1, 1, 0, 0)],
        ),
        (
            ["2026-01-01T00:00:00", "2026-01-31T23:59:59", "2026-02-01"],
            True,
            ["2026-01-01T00:00:00", "2026-01-31T23:59:59"],
        ),
    ],
)
def test_time_range_handles_date_and_datetime_cells_by_calendar_day(
    values: list[object],
    inclusive_end: bool,
    expected: list[object],
) -> None:
    table = "orders"
    order_date = ColumnRef(table, "order_date")
    plan = QueryPlan(
        base_table=table,
        selected_columns=(order_date,),
        time_range=TimeRange(
            order_date,
            "2026-01-01",
            "2026-01-31",
            inclusive_end=inclusive_end,
        ),
    )

    result = execute_query_plan(
        plan,
        catalog={table: ("order_date",)},
        sources={table: [(value,) for value in values]},
    )

    assert [row[f"{table}.order_date"] for row in result.rows] == expected


def test_time_range_rejects_non_iso_date_values_instead_of_lexically_filtering() -> None:
    table = "orders"
    order_date = ColumnRef(table, "order_date")
    plan = QueryPlan(
        base_table=table,
        selected_columns=(order_date,),
        time_range=TimeRange(order_date, "2026-01-01", "2026-01-31"),
    )

    with pytest.raises(QueryExecutionError, match="must be an ISO-8601 date"):
        execute_query_plan(
            plan,
            catalog={table: ("order_date",)},
            sources={table: [("31/01/2026",)]},
        )
