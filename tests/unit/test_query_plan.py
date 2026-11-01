import pytest

from datamate.query_plan import (
    Aggregation,
    Calculation,
    ColumnRef,
    Filter,
    GroupExclusion,
    GroupFilter,
    Join,
    PlanValidationError,
    QueryPlan,
    Sort,
    TimeBucket,
    TimeRange,
    UnitConversion,
    validate_query_plan,
)


CATALOG = {
    "orders.csv::orders": {"order_id", "customer_id", "amount", "order_date"},
    "customers.xlsx::Customers": {"customer_id", "name"},
}


def test_valid_plan_covers_joins_aggregation_time_and_evidence() -> None:
    orders = "orders.csv::orders"
    customers = "customers.xlsx::Customers"
    plan = QueryPlan(
        base_table=orders,
        joins=(
            Join(
                ColumnRef(orders, "customer_id"),
                ColumnRef(customers, "customer_id"),
                kind="left",
            ),
        ),
        filters=(Filter(ColumnRef(orders, "amount"), "gt", 0),),
        group_by=(ColumnRef(customers, "name"),),
        aggregations=(Aggregation("sum", ColumnRef(orders, "amount"), "revenue"),),
        calculations=(Calculation("ratio_percent", "revenue", "revenue", "share"),),
        sort=(Sort("share", "desc"),),
        limit=5,
        time_range=TimeRange(ColumnRef(orders, "order_date"), "2026-01-01", "2026-01-31"),
        evidence_requirements=(ColumnRef(orders, "order_id"),),
        output_type="table",
    )

    validate_query_plan(plan, CATALOG)


def test_invalid_plan_rejects_unknown_references_and_unsafe_limits() -> None:
    plan = QueryPlan(
        base_table="orders.csv::orders",
        selected_columns=(ColumnRef("orders.csv::orders", "missing"),),
        joins=(
            Join(
                ColumnRef("orders.csv::orders", "customer_id"),
                ColumnRef("orders.csv::orders; DROP TABLE orders", "id"),
            ),
        ),
        limit=100_001,
    )

    with pytest.raises(PlanValidationError) as error:
        validate_query_plan(plan, CATALOG)

    assert any("unknown table" in issue for issue in error.value.issues)
    assert any("unknown column" in issue for issue in error.value.issues)
    assert any("limit must be between" in issue for issue in error.value.issues)


def test_invalid_filters_aggregations_and_time_ranges_are_reported() -> None:
    orders = "orders.csv::orders"
    plan = QueryPlan(
        base_table=orders,
        filters=(
            Filter(ColumnRef(orders, "amount"), "between", [1]),
            Filter(ColumnRef(orders, "amount"), "gt", None),
        ),
        aggregations=(Aggregation("avg", None, "average"),),
        sort=(Sort("not_an_aggregate", "desc"),),
        time_range=TimeRange(ColumnRef(orders, "order_date"), "2026-02-01", "2026-01-01"),
    )

    with pytest.raises(PlanValidationError) as error:
        validate_query_plan(plan, CATALOG)

    assert any("between filter requires exactly two" in issue for issue in error.value.issues)
    assert any("requires a non-null value" in issue for issue in error.value.issues)
    assert any("avg requires a column" in issue for issue in error.value.issues)
    assert any("unknown column or aggregation alias" in issue for issue in error.value.issues)
    assert any("must not be after" in issue for issue in error.value.issues)


def test_join_must_be_connected_to_base_table() -> None:
    customers = "customers.xlsx::Customers"
    plan = QueryPlan(
        base_table=customers,
        joins=(
            Join(
                ColumnRef("orders.csv::orders", "customer_id"),
                ColumnRef("orders.csv::orders", "order_id"),
            ),
        ),
    )

    with pytest.raises(PlanValidationError, match="must connect"):
        validate_query_plan(plan, CATALOG)


def test_plan_rejects_business_rules_that_executor_cannot_apply() -> None:
    table = "orders.csv::orders"
    plan = QueryPlan(
        base_table=table,
        selected_columns=(ColumnRef(table, "order_id"),),
        business_rules=("Hủy đơn không tính doanh thu.",),
    )

    with pytest.raises(PlanValidationError, match="business_rules are not executable"):
        validate_query_plan(plan, CATALOG)


def test_plan_validates_scoped_aggregation_filters_and_dates() -> None:
    orders = "orders.csv::orders"
    plan = QueryPlan(
        base_table=orders,
        aggregations=(
            Aggregation(
                "count",
                None,
                "orders",
                filters=(
                    Filter(ColumnRef(orders, "missing_status"), "eq", "complete"),
                ),
                time_range=TimeRange(
                    ColumnRef(orders, "order_date"),
                    "2026-02-01",
                    "2026-01-01",
                ),
            ),
        ),
    )

    with pytest.raises(PlanValidationError) as error:
        validate_query_plan(plan, CATALOG)

    assert any(
        "unknown column orders.csv::orders.missing_status" in issue
        for issue in error.value.issues
    )
    assert any("start must not be after end" in issue for issue in error.value.issues)


def test_count_distinct_requires_a_valid_key_column() -> None:
    orders = "orders.csv::orders"
    plan = QueryPlan(
        base_table=orders,
        aggregations=(Aggregation("count_distinct", None, "orders"),),
    )

    with pytest.raises(PlanValidationError, match="count_distinct requires a column"):
        validate_query_plan(plan, CATALOG)


def test_group_count_requires_valid_aggregate_filter_and_numeric_output() -> None:
    orders = "orders.csv::orders"
    valid_plan = QueryPlan(
        base_table=orders,
        group_by=(ColumnRef(orders, "customer_id"),),
        aggregations=(Aggregation("count", None, "orders"),),
        group_filters=(GroupFilter("orders", "gte", 2),),
        count_groups=True,
        output_type="number",
    )
    validate_query_plan(valid_plan, CATALOG)

    invalid_plan = QueryPlan(
        base_table=orders,
        group_by=(ColumnRef(orders, "customer_id"),),
        group_filters=(GroupFilter("missing", "gte", True),),
        count_groups=True,
        output_type="string",
    )
    with pytest.raises(PlanValidationError) as error:
        validate_query_plan(invalid_plan, CATALOG)

    assert any("unknown aggregate alias" in issue for issue in error.value.issues)
    assert any("finite number" in issue for issue in error.value.issues)
    assert any("requires number output" in issue for issue in error.value.issues)


def test_ratio_rounding_places_are_bounded_and_only_apply_to_ratio() -> None:
    table = "orders.csv::orders"
    amount = ColumnRef(table, "amount")
    valid_plan = QueryPlan(
        base_table=table,
        aggregations=(
            Aggregation("sum", amount, "revenue"),
            Aggregation("count", None, "orders"),
        ),
        calculations=(
            Calculation("ratio", "revenue", "orders", "average", rounding_places=0),
        ),
    )
    validate_query_plan(valid_plan, CATALOG)

    invalid_plan = QueryPlan(
        base_table=table,
        aggregations=(Aggregation("sum", amount, "revenue"),),
        calculations=(
            Calculation(
                "difference",
                "revenue",
                "revenue",
                "difference",
                rounding_places=13,
            ),
        ),
    )
    with pytest.raises(PlanValidationError) as error:
        validate_query_plan(invalid_plan, CATALOG)

    assert any("integer from 0 to 12" in issue for issue in error.value.issues)
    assert any("only for ratio" in issue for issue in error.value.issues)


def test_group_exclusion_requires_base_key_and_same_table_filters() -> None:
    orders = "orders.csv::orders"
    customers = "customers.xlsx::Customers"
    plan = QueryPlan(
        base_table=orders,
        joins=(
            Join(
                ColumnRef(orders, "customer_id"),
                ColumnRef(customers, "customer_id"),
                kind="left",
            ),
        ),
        aggregations=(Aggregation("sum", ColumnRef(orders, "amount"), "total"),),
        group_exclusions=(
            GroupExclusion(
                key=ColumnRef(customers, "customer_id"),
                filters=(Filter(ColumnRef(orders, "amount"), "lt", 0),),
            ),
        ),
    )

    with pytest.raises(PlanValidationError) as error:
        validate_query_plan(plan, CATALOG)

    assert any("key must belong to the base table" in issue for issue in error.value.issues)
    assert any("filters must use the same table" in issue for issue in error.value.issues)


def test_group_exclusion_requires_marker_filters() -> None:
    orders = "orders.csv::orders"
    plan = QueryPlan(
        base_table=orders,
        aggregations=(Aggregation("sum", ColumnRef(orders, "amount"), "total"),),
        group_exclusions=(
            GroupExclusion(key=ColumnRef(orders, "order_id"), filters=()),
        ),
    )

    with pytest.raises(
        PlanValidationError,
        match="requires at least one source-row filter",
    ):
        validate_query_plan(plan, CATALOG)


def test_unit_conversion_requires_sum_and_valid_columns() -> None:
    orders = "orders.csv::orders"
    products = "products.xlsx::Units"
    catalog = {
        orders: {"customer_id", "amount", "unit"},
        products: {"customer_id", "base_unit", "large_unit", "factor"},
    }
    conversion = UnitConversion(
        source_unit=ColumnRef(orders, "unit"),
        base_unit=ColumnRef(products, "base_unit"),
        large_unit=ColumnRef(products, "large_unit"),
        factor=ColumnRef(products, "factor"),
        target_unit=None,
    )
    valid_plan = QueryPlan(
        base_table=orders,
        joins=(
            Join(
                ColumnRef(orders, "customer_id"),
                ColumnRef(products, "customer_id"),
                kind="left",
            ),
        ),
        aggregations=(
            Aggregation(
                "sum",
                ColumnRef(orders, "amount"),
                "quantity",
                unit_conversion=conversion,
            ),
        ),
    )
    validate_query_plan(valid_plan, catalog)

    invalid_plan = QueryPlan(
        base_table=orders,
        joins=(
            Join(
                ColumnRef(orders, "customer_id"),
                ColumnRef(products, "customer_id"),
            ),
        ),
        aggregations=(
            Aggregation(
                "count",
                None,
                "count",
                unit_conversion=conversion,
            ),
        ),
    )
    with pytest.raises(
        PlanValidationError,
        match="unit conversion is supported only for sum",
    ) as error:
        validate_query_plan(invalid_plan, catalog)
    assert any("must be left joins" in issue for issue in error.value.issues)


def test_unit_conversion_rejects_mismatched_source_and_lookup_tables() -> None:
    sales = "sales"
    products = "products"
    unrelated = "unrelated"
    catalog = {
        sales: {"product_id", "quantity", "unit"},
        products: {"product_id", "base_unit", "large_unit"},
        unrelated: {"product_id", "factor", "unit"},
    }
    plan = QueryPlan(
        base_table=sales,
        joins=(
            Join(
                ColumnRef(sales, "product_id"),
                ColumnRef(products, "product_id"),
                kind="left",
            ),
            Join(
                ColumnRef(sales, "product_id"),
                ColumnRef(unrelated, "product_id"),
                kind="left",
            ),
        ),
        aggregations=(
            Aggregation(
                "sum",
                ColumnRef(sales, "quantity"),
                "converted_quantity",
                unit_conversion=UnitConversion(
                    source_unit=ColumnRef(unrelated, "unit"),
                    base_unit=ColumnRef(products, "base_unit"),
                    large_unit=ColumnRef(products, "large_unit"),
                    factor=ColumnRef(unrelated, "factor"),
                    target_unit="kg",
                ),
            ),
        ),
    )

    with pytest.raises(PlanValidationError) as error:
        validate_query_plan(plan, catalog)

    assert any(
        "source_unit must use the aggregation column's table" in issue
        for issue in error.value.issues
    )
    assert any(
        "lookup columns must come from the same table" in issue
        for issue in error.value.issues
    )


def test_plan_rejects_unknown_time_bucket_columns_and_granularities() -> None:
    orders = "orders.csv::orders"
    plan = QueryPlan(
        base_table=orders,
        time_buckets=(
            TimeBucket(
                ColumnRef(orders, "missing_date"),
                "fortnight",
                "period",
            ),
        ),
        aggregations=(Aggregation("count", None, "rows"),),
    )

    with pytest.raises(PlanValidationError) as error:
        validate_query_plan(plan, CATALOG)

    assert any("unknown column" in issue for issue in error.value.issues)
    assert any(
        "unsupported time bucket granularity" in issue
        for issue in error.value.issues
    )


def test_plan_rejects_unsafe_literals_non_iso_dates_and_boolean_limits() -> None:
    orders = "orders.csv::orders"
    plan = QueryPlan(
        base_table=orders,
        selected_columns=(ColumnRef(orders, "order_id"),),
        filters=(
            Filter(ColumnRef(orders, "amount"), "eq", object()),
            Filter(ColumnRef(orders, "amount"), "contains", 1),
        ),
        limit=True,
        time_range=TimeRange(ColumnRef(orders, "order_date"), "yesterday", "tomorrow"),
    )

    with pytest.raises(PlanValidationError) as error:
        validate_query_plan(plan, CATALOG)

    assert any("safe scalar" in issue for issue in error.value.issues)
    assert any("contains filter requires a text value" in issue for issue in error.value.issues)
    assert any("ISO-8601 dates" in issue for issue in error.value.issues)
    assert any("limit must be between" in issue for issue in error.value.issues)
