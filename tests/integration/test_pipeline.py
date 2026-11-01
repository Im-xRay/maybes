import json
import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest

from datamate.pipeline import (
    execute_and_log_business_plan,
    execute_business_plan,
    execute_question_and_log,
    table_identifier,
)
from datamate.data_loader import BusinessDataLoader
from datamate.normalization import ColumnCoercion
from datamate.question_understanding import (
    JsonPlanAdapter,
    QuestionUnderstandingError,
)
from datamate.query_executor import QueryExecutionError
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
    UnitConversion,
    Sort,
    TimeBucket,
    TimeRange,
)
from datamate.run_logging import RunLogError, read_run_records
from datamate.submission import SubmissionWriter, serialize_submission_record


def _business_database(tmp_path: Path) -> Path:
    business = tmp_path / "business"
    business.mkdir(parents=True)
    with sqlite3.connect(business / "shop.sqlite") as connection:
        connection.execute(
            "CREATE TABLE sales (sale_id INTEGER PRIMARY KEY, amount NUMERIC)"
        )
        connection.executemany("INSERT INTO sales (amount) VALUES (?)", [(12,), (8,)])
    return business


def test_business_plan_runs_from_read_only_loader_through_answer_validation(
    tmp_path: Path,
) -> None:
    business = _business_database(tmp_path)
    loader = BusinessDataLoader(business)
    table = next(table for table in loader.discover() if table.name == "sales")
    table_id = table_identifier(table)
    amount = ColumnRef(table_id, "amount")
    plan = QueryPlan(
        base_table=table_id,
        aggregations=(Aggregation("sum", amount, "total"),),
        output_type="number",
    )

    result = execute_business_plan(business, plan)

    assert result.answer.value == 20
    assert result.answer.answer_type == "number"
    assert result.answer.status == "answered"
    assert result.answer.evidence[0].source_table == table_id
    assert result.execution.trace.source_rows == {table_id: 2}


def test_pipeline_compares_period_scoped_aggregations_and_records_evidence(
    tmp_path: Path,
) -> None:
    business = tmp_path / "period-business"
    business.mkdir()
    with sqlite3.connect(business / "shop.sqlite") as connection:
        connection.execute(
            "CREATE TABLE sales (product TEXT, amount NUMERIC, sold_at TEXT)"
        )
        connection.executemany(
            "INSERT INTO sales VALUES (?, ?, ?)",
            [
                ("A", 60, "2026-01-10"),
                ("A", 40, "2026-01-31"),
                ("A", 120, "2026-02-28T23:59:00"),
                ("B", 50, "2026-01-15"),
                ("B", 25, "2026-02-10"),
                ("C", 500, "2026-03-10"),
            ],
        )
    table = next(
        table for table in BusinessDataLoader(business).discover() if table.name == "sales"
    )
    table_id = table_identifier(table)
    product = ColumnRef(table_id, "product")
    amount = ColumnRef(table_id, "amount")
    sold_at = ColumnRef(table_id, "sold_at")
    plan = QueryPlan(
        base_table=table_id,
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
        output_type="table",
    )

    result = execute_business_plan(business, plan)

    assert result.answer.answer_type == "table"
    assert result.answer.value == [
        {
            f"{table_id}.product": "A",
            "previous_revenue": 100,
            "current_revenue": 120,
            "growth_percent": 20,
        },
        {
            f"{table_id}.product": "B",
            "previous_revenue": 50,
            "current_revenue": 25,
            "growth_percent": -50,
        },
    ]
    assert result.answer.evidence[0].columns == ("amount", "product", "sold_at")
    assert result.answer.evidence[0].filter_summary == "0 filters; time_range=yes"


def test_pipeline_groups_by_month_and_serializes_table_with_bucket_evidence(
    tmp_path: Path,
) -> None:
    business = tmp_path / "monthly-business"
    business.mkdir()
    with sqlite3.connect(business / "shop.sqlite") as connection:
        connection.execute(
            "CREATE TABLE sales (amount NUMERIC, sold_at TEXT)"
        )
        connection.executemany(
            "INSERT INTO sales VALUES (?, ?)",
            [
                (10, "2026-01-01"),
                (15, "2026-01-31T23:59:00"),
                (20, "2026-03-01"),
            ],
        )
        connection.execute(
            "CREATE TABLE product_sales (product TEXT, amount NUMERIC, sold_at TEXT)"
        )
        connection.executemany(
            "INSERT INTO product_sales VALUES (?, ?, ?)",
            [
                ("A", 10, "2026-01-01"),
                ("A", 30, "2026-03-01"),
                ("B", 5, "2026-03-01"),
            ],
        )
    table = next(
        table for table in BusinessDataLoader(business).discover() if table.name == "sales"
    )
    table_id = table_identifier(table)
    amount = ColumnRef(table_id, "amount")
    sold_at = ColumnRef(table_id, "sold_at")
    plan = QueryPlan(
        base_table=table_id,
        time_range=TimeRange(sold_at, "2026-01-01", "2026-03-31"),
        time_buckets=(TimeBucket(sold_at, "month", "month"),),
        aggregations=(Aggregation("sum", amount, "revenue"),),
        sort=(Sort("month", "asc"),),
        output_type="table",
    )

    result = execute_business_plan(business, plan)
    submission = json.loads(
        serialize_submission_record(
            "q-monthly",
            result.answer,
            plan=plan,
        )
    )

    assert submission == {
        "id": "q-monthly",
        "type": "table",
        "answer": {
            "columns": ["month", "revenue"],
            "rows": [
                ["2026-01", 25],
                ["2026-02", 0],
                ["2026-03", 20],
            ],
        },
    }
    assert result.answer.evidence[0].columns == ("amount", "sold_at")
    assert result.answer.evidence[0].filter_summary == (
        "0 filters; time_range=yes; time_bucket=yes"
    )

    list_plan = QueryPlan(
        base_table=table_id,
        time_range=TimeRange(sold_at, "2026-01-01", "2026-03-31"),
        time_buckets=(TimeBucket(sold_at, "month", "month"),),
        aggregations=(Aggregation("sum", amount, "revenue"),),
        sort=(Sort("month", "asc"),),
        output_type="list",
    )
    list_result = execute_business_plan(business, list_plan)
    assert list_result.answer.value == [25, 0, 20]
    list_submission = json.loads(
        serialize_submission_record(
            "q-monthly-series",
            list_result.answer,
            plan=list_plan,
        )
    )
    assert list_submission == {
        "id": "q-monthly-series",
        "type": "list",
        "answer": [25, 0, 20],
    }

    product_table = next(
        table
        for table in BusinessDataLoader(business).discover()
        if table.name == "product_sales"
    )
    product_table_id = table_identifier(product_table)
    product = ColumnRef(product_table_id, "product")
    product_sold_at = ColumnRef(product_table_id, "sold_at")
    grouped_plan = QueryPlan(
        base_table=product_table_id,
        selected_columns=(product,),
        group_by=(product,),
        time_range=TimeRange(product_sold_at, "2026-01-01", "2026-03-31"),
        time_buckets=(TimeBucket(product_sold_at, "month", "month"),),
        aggregations=(
            Aggregation(
                "sum",
                ColumnRef(product_table_id, "amount"),
                "revenue",
            ),
        ),
        sort=(Sort(product, "asc"), Sort("month", "asc")),
        output_type="table",
    )
    grouped_result = execute_business_plan(business, grouped_plan)
    grouped_submission = json.loads(
        serialize_submission_record(
            "q-product-monthly",
            grouped_result.answer,
            plan=grouped_plan,
        )
    )

    assert grouped_submission == {
        "id": "q-product-monthly",
        "type": "table",
        "answer": {
            "columns": ["product", "month", "revenue"],
            "rows": [
                ["A", "2026-01", 10],
                ["A", "2026-02", 0],
                ["A", "2026-03", 30],
                ["B", "2026-01", 0],
                ["B", "2026-02", 0],
                ["B", "2026-03", 5],
            ],
        },
    }


def test_pipeline_distinguishes_out_of_coverage_from_zero_within_coverage(
    tmp_path: Path,
) -> None:
    business = tmp_path / "coverage-business"
    business.mkdir()
    with sqlite3.connect(business / "shop.sqlite") as connection:
        connection.execute(
            "CREATE TABLE sales (amount NUMERIC, sold_at TEXT)"
        )
        connection.executemany(
            "INSERT INTO sales VALUES (?, ?)",
            [(10, "2026-01-01"), (25, "2026-01-31T23:59:00")],
        )
    table = next(
        table for table in BusinessDataLoader(business).discover() if table.name == "sales"
    )
    table_id = table_identifier(table)
    amount = ColumnRef(table_id, "amount")
    sold_at = ColumnRef(table_id, "sold_at")

    outside_plan = QueryPlan(
        base_table=table_id,
        time_range=TimeRange(sold_at, "2026-03-01", "2026-03-31"),
        aggregations=(Aggregation("sum", amount, "revenue"),),
        output_type="number",
    )
    outside = execute_business_plan(business, outside_plan)
    assert outside.answer.answer_type == "unanswerable"
    assert outside.answer.status == "no_data"

    covered_zero_plan = QueryPlan(
        base_table=table_id,
        filters=(Filter(amount, "gt", 1000),),
        time_range=TimeRange(sold_at, "2026-01-10", "2026-01-15"),
        aggregations=(Aggregation("sum", amount, "revenue"),),
        output_type="number",
    )
    covered_zero = execute_business_plan(business, covered_zero_plan)
    assert covered_zero.answer.answer_type == "number"
    assert covered_zero.answer.value == 0
    assert covered_zero.answer.status == "answered"

    partial_plan = QueryPlan(
        base_table=table_id,
        time_range=TimeRange(sold_at, "2025-12-01", "2026-01-31"),
        aggregations=(Aggregation("sum", amount, "revenue"),),
        output_type="number",
    )
    partial = execute_business_plan(business, partial_plan)
    assert partial.answer.answer_type == "number"
    assert partial.answer.value == 35


def test_pipeline_distinguishes_no_matching_data_from_internal_failure(
    tmp_path: Path,
) -> None:
    business = _business_database(tmp_path)
    table = next(
        table for table in BusinessDataLoader(business).discover() if table.name == "sales"
    )
    table_id = table_identifier(table)
    amount = ColumnRef(table_id, "amount")
    plan = QueryPlan(
        base_table=table_id,
        filters=(Filter(amount, "gt", 100),),
        aggregations=(Aggregation("sum", amount, "total"),),
        output_type="number",
    )

    result = execute_business_plan(business, plan)

    assert result.answer.answer_type == "unanswerable"
    assert result.answer.status == "no_data"
    assert result.answer.value is None
    assert result.answer.evidence[0].row_count == 2


def test_pipeline_preserves_exact_zero_count_for_no_matching_rows(
    tmp_path: Path,
) -> None:
    business = _business_database(tmp_path)
    table = next(
        table for table in BusinessDataLoader(business).discover() if table.name == "sales"
    )
    table_id = table_identifier(table)
    amount = ColumnRef(table_id, "amount")
    plan = QueryPlan(
        base_table=table_id,
        filters=(Filter(amount, "gt", 100),),
        aggregations=(Aggregation("count", None, "rows"),),
        output_type="number",
    )

    result = execute_business_plan(business, plan)

    assert result.answer.answer_type == "number"
    assert result.answer.value == 0
    assert result.answer.status == "answered"
    assert result.execution.trace.filtered_rows == 0


def test_pipeline_counts_distinct_orders_across_repeated_line_items(
    tmp_path: Path,
) -> None:
    business = tmp_path / "order-business"
    business.mkdir()
    with sqlite3.connect(business / "shop.sqlite") as connection:
        connection.execute(
            "CREATE TABLE order_lines (order_id TEXT, amount INTEGER)"
        )
        connection.executemany(
            "INSERT INTO order_lines VALUES (?, ?)",
            [("O-1", 10), ("O-1", 20), ("O-2", 30), (None, 40)],
        )
    table = next(
        table
        for table in BusinessDataLoader(business).discover()
        if table.name == "order_lines"
    )
    table_id = table_identifier(table)
    order_id = ColumnRef(table_id, "order_id")
    plan = QueryPlan(
        base_table=table_id,
        aggregations=(
            Aggregation("count_distinct", order_id, "order_count"),
        ),
        output_type="number",
    )

    result = execute_business_plan(business, plan)

    assert result.answer.answer_type == "number"
    assert result.answer.value == 2
    assert result.answer.status == "answered"
    assert result.execution.trace.filtered_rows == 4


def test_pipeline_calculates_cancelled_order_rate_over_all_unique_orders(
    tmp_path: Path,
) -> None:
    business = tmp_path / "cancellation-business"
    business.mkdir()
    with sqlite3.connect(business / "shop.sqlite") as connection:
        connection.execute(
            "CREATE TABLE order_lines "
            "(order_id TEXT, status TEXT, ordered_at TEXT, amount INTEGER)"
        )
        connection.executemany(
            "INSERT INTO order_lines VALUES (?, ?, ?, ?)",
            [
                ("O-1", "completed", "2026-01-03", 10),
                ("O-1", "completed", "2026-01-03", 20),
                ("O-2", "cancelled", "2026-01-04", 30),
                ("O-2", "cancelled", "2026-01-04", 40),
                ("O-3", "cancelled", "2026-01-05", 50),
                ("O-4", "returned", "2026-01-06", 60),
            ],
        )
    table = next(
        table
        for table in BusinessDataLoader(business).discover()
        if table.name == "order_lines"
    )
    table_id = table_identifier(table)
    order_id = ColumnRef(table_id, "order_id")
    status = ColumnRef(table_id, "status")
    ordered_at = ColumnRef(table_id, "ordered_at")
    plan = QueryPlan(
        base_table=table_id,
        time_range=TimeRange(ordered_at, "2026-01-01", "2026-01-31"),
        aggregations=(
            Aggregation(
                "count_distinct",
                order_id,
                "cancelled_orders",
                filters=(Filter(status, "eq", "cancelled"),),
            ),
            Aggregation("count_distinct", order_id, "all_orders"),
        ),
        calculations=(
            Calculation(
                "ratio_percent",
                "cancelled_orders",
                "all_orders",
                "cancellation_rate",
            ),
        ),
        output_type="percent",
    )

    result = execute_business_plan(business, plan)

    assert result.answer.answer_type == "percent"
    assert result.answer.value == Decimal("50.00")
    assert result.answer.status == "answered"


def test_pipeline_revenue_excludes_cancelled_and_returned_orders(
    tmp_path: Path,
) -> None:
    business = tmp_path / "revenue-business"
    business.mkdir()
    with sqlite3.connect(business / "shop.sqlite") as connection:
        connection.execute(
            "CREATE TABLE order_lines "
            "(order_id TEXT, status TEXT, order_date TEXT, line_total INTEGER, "
            "quantity INTEGER)"
        )
        connection.executemany(
            "INSERT INTO order_lines VALUES (?, ?, ?, ?, ?)",
            [
                ("O-1", "completed", "2026-01-01", 20, 1),
                ("O-1", "completed", "2026-01-01", 10, 1),
                ("O-2", "cancelled", "2026-01-02", 50, 1),
                ("O-3", "completed", "2026-01-03", 40, 2),
                ("O-3", "completed", "2026-01-03", -40, -2),
                ("O-4", "in_delivery", "2026-01-04", 5, 1),
            ],
        ),
    table = next(
        table
        for table in BusinessDataLoader(business).discover()
        if table.name == "order_lines"
    )
    table_id = table_identifier(table)
    status = ColumnRef(table_id, "status")
    order_date = ColumnRef(table_id, "order_date")
    line_total = ColumnRef(table_id, "line_total")
    quantity = ColumnRef(table_id, "quantity")
    order_id = ColumnRef(table_id, "order_id")
    plan = QueryPlan(
        base_table=table_id,
        time_range=TimeRange(order_date, "2026-01-01", "2026-01-31"),
        filters=(Filter(status, "ne", "cancelled"),),
        group_exclusions=(
            GroupExclusion(
                key=order_id,
                filters=(Filter(quantity, "lt", 0),),
            ),
        ),
        aggregations=(
            Aggregation("sum", line_total, "revenue"),
        ),
        output_type="number",
    )
    result = execute_business_plan(business, plan)

    assert result.answer.value == 35
    assert result.answer.status == "answered"
    evidence = result.answer.evidence[0]
    assert {"order_id", "quantity"} <= set(evidence.columns)
    assert "excluded_groups=1" in evidence.filter_summary


def test_pipeline_calculates_rounded_average_revenue_per_valid_order(
    tmp_path: Path,
) -> None:
    business = tmp_path / "average-order-business"
    business.mkdir()
    with sqlite3.connect(business / "shop.sqlite") as connection:
        connection.execute(
            "CREATE TABLE order_lines "
            "(order_id TEXT, status TEXT, order_date TEXT, line_total INTEGER)"
        )
        connection.executemany(
            "INSERT INTO order_lines VALUES (?, ?, ?, ?)",
            [
                ("O-1", "completed", "2026-01-01", 20),
                ("O-1", "completed", "2026-01-01", 10),
                ("O-2", "completed", "2026-01-02", 5),
                ("O-3", "cancelled", "2026-01-03", 100),
                ("O-4", "returned", "2026-01-04", 100),
            ],
        )
    table = next(
        table
        for table in BusinessDataLoader(business).discover()
        if table.name == "order_lines"
    )
    table_id = table_identifier(table)
    order_id = ColumnRef(table_id, "order_id")
    status = ColumnRef(table_id, "status")
    line_total = ColumnRef(table_id, "line_total")
    eligible_order_filters = (
        Filter(status, "ne", "cancelled"),
        Filter(status, "ne", "returned"),
    )
    plan = QueryPlan(
        base_table=table_id,
        aggregations=(
            Aggregation(
                "sum",
                line_total,
                "revenue",
                filters=eligible_order_filters,
            ),
            Aggregation(
                "count_distinct",
                order_id,
                "orders",
                filters=eligible_order_filters,
            ),
        ),
        calculations=(
            Calculation(
                "ratio",
                "revenue",
                "orders",
                "average_order_revenue",
                rounding_places=0,
            ),
        ),
        output_type="number",
    )

    result = execute_business_plan(business, plan)

    assert result.answer.value == 18
    assert result.answer.status == "answered"

    no_eligible_orders_plan = QueryPlan(
        base_table=table_id,
        aggregations=(
            Aggregation(
                "sum",
                line_total,
                "revenue",
                filters=(Filter(status, "eq", "unknown"),),
            ),
            Aggregation(
                "count_distinct",
                order_id,
                "orders",
                filters=(Filter(status, "eq", "unknown"),),
            ),
        ),
        calculations=plan.calculations,
        output_type="number",
    )
    no_eligible_orders = execute_business_plan(business, no_eligible_orders_plan)

    assert no_eligible_orders.answer.answer_type == "unanswerable"
    assert no_eligible_orders.answer.status == "insufficient_data"


def test_pipeline_converts_sales_units_using_joined_product_catalog(
    tmp_path: Path,
) -> None:
    business = tmp_path / "unit-conversion-business"
    business.mkdir()
    with sqlite3.connect(business / "shop.sqlite") as connection:
        connection.execute(
            "CREATE TABLE sales "
            "(product_id TEXT, quantity NUMERIC, unit TEXT)"
        )
        connection.execute(
            "CREATE TABLE products "
            "(product_id TEXT, base_unit TEXT, large_unit TEXT, factor NUMERIC)"
        )
        connection.executemany(
            "INSERT INTO sales VALUES (?, ?, ?)",
            [
                ("A", 2, "bag"),
                ("A", 3, "kg"),
                ("B", 4, "box"),
                ("B", 2, "pc"),
                ("C", 3, "kg"),
            ],
        )
        connection.executemany(
            "INSERT INTO products VALUES (?, ?, ?, ?)",
            [
                ("A", "kg", "bag", 50),
                ("B", "pc", "box", 12),
            ],
        )
    tables = {
        table.name: table
        for table in BusinessDataLoader(business).discover()
    }
    sales_id = table_identifier(tables["sales"])
    products_id = table_identifier(tables["products"])
    plan = QueryPlan(
        base_table=sales_id,
        joins=(
            Join(
                ColumnRef(sales_id, "product_id"),
                ColumnRef(products_id, "product_id"),
                kind="left",
            ),
        ),
        aggregations=(
            Aggregation(
                "sum",
                ColumnRef(sales_id, "quantity"),
                "quantity_in_product_base_units",
                unit_conversion=UnitConversion(
                    source_unit=ColumnRef(sales_id, "unit"),
                    base_unit=ColumnRef(products_id, "base_unit"),
                    large_unit=ColumnRef(products_id, "large_unit"),
                    factor=ColumnRef(products_id, "factor"),
                    target_unit=None,
                ),
            ),
        ),
        output_type="number",
    )

    result = execute_business_plan(business, plan)

    assert result.answer.value == 156
    assert result.answer.status == "answered"
    assert result.answer.evidence[0].columns
    assert any(
        "unit_conversion_to=product_base_unit" in item.filter_summary
        for item in result.answer.evidence
    )

    product_id = ColumnRef(sales_id, "product_id")
    ranking_plan = QueryPlan(
        base_table=sales_id,
        joins=plan.joins,
        selected_columns=(product_id,),
        group_by=(product_id,),
        aggregations=plan.aggregations,
        sort=(
            Sort("quantity_in_product_base_units", "desc"),
            Sort(product_id, "asc"),
        ),
        limit=1,
        output_type="list",
    )
    ranking_result = execute_business_plan(business, ranking_plan)

    assert ranking_result.answer.value == ["A"]


def test_pipeline_counts_groups_meeting_distinct_order_threshold(
    tmp_path: Path,
) -> None:
    business = tmp_path / "threshold-business"
    business.mkdir()
    with sqlite3.connect(business / "shop.sqlite") as connection:
        connection.execute(
            "CREATE TABLE sales "
            "(product TEXT, order_id TEXT, quantity INTEGER, line_total INTEGER)"
        )
        connection.executemany(
            "INSERT INTO sales VALUES (?, ?, ?, ?)",
            [
                ("A", "O-1", 1, 20),
                ("A", "O-1", 2, 30),
                ("A", "O-2", 1, 25),
                ("B", "O-1", 1, 10),
                ("B", "O-2", 1, 15),
                ("C", "O-3", 1, 60),
            ],
        )
    table = next(
        table
        for table in BusinessDataLoader(business).discover()
        if table.name == "sales"
    )
    table_id = table_identifier(table)
    product = ColumnRef(table_id, "product")
    order_id = ColumnRef(table_id, "order_id")
    plan = QueryPlan(
        base_table=table_id,
        group_by=(product,),
        group_filters=(GroupFilter("orders", "gte", 2),),
        aggregations=(Aggregation("count_distinct", order_id, "orders"),),
        count_groups=True,
        output_type="number",
    )

    result = execute_business_plan(business, plan)

    assert result.answer.value == 2
    assert result.answer.status == "answered"
    assert result.execution.trace.group_count == 3

    order_id = ColumnRef(table_id, "order_id")
    line_total = ColumnRef(table_id, "line_total")
    qualifying_orders_plan = QueryPlan(
        base_table=table_id,
        group_by=(order_id,),
        group_filters=(GroupFilter("revenue", "gte", 60),),
        aggregations=(Aggregation("sum", line_total, "revenue"),),
        count_groups=True,
        output_type="number",
    )
    qualifying_orders = execute_business_plan(
        business,
        qualifying_orders_plan,
    )

    assert qualifying_orders.answer.value == 2
    assert qualifying_orders.answer.status == "answered"

    no_match_plan = QueryPlan(
        base_table=table_id,
        filters=(Filter(ColumnRef(table_id, "quantity"), "gt", 10),),
        group_by=(product,),
        group_filters=(GroupFilter("orders", "gte", 2),),
        aggregations=(Aggregation("count_distinct", order_id, "orders"),),
        count_groups=True,
        output_type="number",
    )
    no_match_result = execute_business_plan(business, no_match_plan)

    assert no_match_result.answer.value == 0
    assert no_match_result.answer.status == "answered"


def test_pipeline_returns_ranked_group_key_as_scalar_answer(tmp_path: Path) -> None:
    business = tmp_path / "top-day-business"
    business.mkdir()
    with sqlite3.connect(business / "shop.sqlite") as connection:
        connection.execute(
            "CREATE TABLE sales (sold_at TEXT, product TEXT, amount INTEGER)"
        )
        connection.executemany(
            "INSERT INTO sales VALUES (?, ?, ?)",
            [
                ("2026-01-01", "Alpha", 20),
                ("2026-01-02", "Beta", 35),
                ("2026-01-03", "Gamma", 50),
                ("2026-01-03", "Gamma", 10),
            ],
        )
    table = next(
        table
        for table in BusinessDataLoader(business).discover()
        if table.name == "sales"
    )
    table_id = table_identifier(table)
    sold_at = ColumnRef(table_id, "sold_at")
    amount = ColumnRef(table_id, "amount")
    plan = QueryPlan(
        base_table=table_id,
        time_buckets=(TimeBucket(sold_at, "day", "sale_day"),),
        aggregations=(Aggregation("sum", amount, "revenue"),),
        sort=(Sort("revenue", "desc"),),
        limit=1,
        output_type="date",
    )

    result = execute_business_plan(business, plan)

    assert result.answer.answer_type == "date"
    assert result.answer.value == "2026-01-03"
    assert result.answer.status == "answered"

    product = ColumnRef(table_id, "product")
    product_plan = QueryPlan(
        base_table=table_id,
        selected_columns=(product,),
        group_by=(product,),
        aggregations=(Aggregation("sum", amount, "revenue"),),
        sort=(Sort("revenue", "desc"),),
        limit=1,
        output_type="string",
    )

    product_result = execute_business_plan(business, product_plan)

    assert product_result.answer.answer_type == "string"
    assert product_result.answer.value == "Gamma"
    assert product_result.answer.status == "answered"


def test_table_output_hides_grouping_key_used_only_for_tie_breaks(
    tmp_path: Path,
) -> None:
    business = tmp_path / "ranked-business"
    business.mkdir()
    with sqlite3.connect(business / "shop.sqlite") as connection:
        connection.execute(
            "CREATE TABLE products (product_code TEXT, name TEXT, revenue INTEGER)"
        )
        connection.executemany(
            "INSERT INTO products VALUES (?, ?, ?)",
            [
                ("Z-2", "Apple", 100),
                ("A-1", "Zulu", 100),
            ],
        )
    table = next(
        table
        for table in BusinessDataLoader(business).discover()
        if table.name == "products"
    )
    table_id = table_identifier(table)
    code = ColumnRef(table_id, "product_code")
    name = ColumnRef(table_id, "name")
    revenue = ColumnRef(table_id, "revenue")
    plan = QueryPlan(
        base_table=table_id,
        selected_columns=(name,),
        group_by=(code, name),
        aggregations=(Aggregation("sum", revenue, "total_revenue"),),
        sort=(Sort("total_revenue", "desc"), Sort(code, "asc")),
        output_type="table",
    )

    result = execute_business_plan(business, plan)

    assert result.answer.value == [
        {
            f"{table_id}.name": "Zulu",
            "total_revenue": 100,
        },
        {
            f"{table_id}.name": "Apple",
            "total_revenue": 100,
        },
    ]
    submission = json.loads(
        serialize_submission_record("q-ranked", result.answer, plan=plan)
    )
    assert submission["answer"] == {
        "columns": ["name", "total_revenue"],
        "rows": [["Zulu", 100], ["Apple", 100]],
    }


def test_csv_plan_uses_explicit_locale_normalization_without_guessing(
    tmp_path: Path,
) -> None:
    business = tmp_path / "csv-business"
    business.mkdir()
    (business / "sales.csv").write_text(
        "amount;note\n1.234,50;first\n2,50;second\n",
        encoding="utf-8",
    )
    table = next(
        table for table in BusinessDataLoader(business).discover() if table.name == "sales"
    )
    table_id = table_identifier(table)
    amount = ColumnRef(table_id, "amount")
    plan = QueryPlan(
        base_table=table_id,
        aggregations=(Aggregation("sum", amount, "total"),),
        output_type="number",
    )

    with pytest.raises(QueryExecutionError, match="requires numeric values"):
        execute_business_plan(business, plan)

    result = execute_business_plan(
        business,
        plan,
        column_coercions={
            amount: ColumnCoercion(
                "number",
                decimal_separator=",",
                thousands_separator=".",
            )
        },
    )

    assert result.answer.value == 1237
    assert result.answer.status == "answered"


def test_pipeline_logs_success_and_rejects_logs_inside_input_root(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    business = _business_database(input_root)
    table = next(
        table for table in BusinessDataLoader(business).discover() if table.name == "sales"
    )
    table_id = table_identifier(table)
    amount = ColumnRef(table_id, "amount")
    plan = QueryPlan(
        base_table=table_id,
        aggregations=(Aggregation("sum", amount, "total"),),
        output_type="number",
    )
    log_path = tmp_path / "logs" / "runs.jsonl"

    result = execute_and_log_business_plan(
        business,
        plan,
        question_id="q-001",
        business_id="biz-001",
        question="Tổng tiền là bao nhiêu?",
        model="manual-query-plan",
        log_path=log_path,
        input_root=input_root,
        input_metadata={"as_of_date": "2026-01-31"},
        column_coercions={amount: ColumnCoercion("number")},
    )

    (record,) = read_run_records(log_path)
    assert result.answer.value == 20
    assert record["question_id"] == "q-001"
    assert record["answer"]["value"] == "20"
    assert record["execution_trace"]["result_count"] == 1
    assert record["input"]["metadata"]["column_coercions"][
        f"{table_id}.amount"
    ] == {
        "kind": "number",
        "decimal_separator": ".",
        "thousands_separator": None,
        "date_format": None,
        "fill_down_group": None,
        "deduplicate_group": None,
        "aliases": None,
    }
    with pytest.raises(RunLogError, match="must not be written under input root"):
        execute_and_log_business_plan(
            business,
            plan,
            question_id="q-002",
            business_id="biz-001",
            question="Tổng tiền?",
            model="manual-query-plan",
            log_path=input_root / "runs.jsonl",
            input_root=input_root,
        )


def test_question_to_plan_to_answer_is_integrated_and_logged(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    business = _business_database(input_root)
    table = next(
        table for table in BusinessDataLoader(business).discover() if table.name == "sales"
    )
    table_id = table_identifier(table)
    generated_plan = {
        "base_table": table_id,
        "aggregations": [
            {
                "function": "sum",
                "column": {"table": table_id, "column": "amount"},
                "alias": "total",
            }
        ],
        "output_type": "number",
    }
    prompts: list[str] = []
    planner = JsonPlanAdapter(
        lambda prompt: prompts.append(prompt) or json.dumps(generated_plan),
        model_name="local-stub",
    )
    log_path = tmp_path / "logs" / "question.jsonl"
    submission_path = tmp_path / "result" / "submission.jsonl"

    result = execute_question_and_log(
        business,
        "Tổng doanh thu là bao nhiêu?",
        planner,
        question_id="q-003",
        business_id="biz-test",
        as_of_date="2026-01-31",
        log_path=log_path,
        input_root=input_root,
        rulebook="# Quy tắc doanh thu",
        business_readme="amount tính bằng VND",
        calendar_columns=("solar_date",),
        calendar_rows=({"solar_date": "2026-01-31"},),
        input_metadata={
            "business_context_sources": {
                "rulebook": {"status": "loaded", "sha256": "abc123"}
            }
        },
    )
    writer = SubmissionWriter(submission_path)
    writer.append("q-003", result.answer, plan=result.plan)
    writer.commit()

    (record,) = read_run_records(log_path)
    assert result.answer.value == 20
    assert record["model"] == "local-stub"
    assert record["input"]["metadata"]["as_of_date"] == "2026-01-31"
    assert record["input"]["metadata"]["business_context_sources"]["rulebook"][
        "sha256"
    ] == "abc123"
    assert record["query_plan"]["base_table"] == table_id
    assert prompts and "# Quy tắc doanh thu" in prompts[0]
    assert "amount tính bằng VND" in prompts[0]
    assert "2026-01-31" in prompts[0]
    assert json.loads(submission_path.read_text(encoding="utf-8")) == {
        "id": "q-003",
        "type": "number",
        "answer": 20,
    }


def test_question_planning_failure_is_recorded_without_a_fake_plan(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    business = _business_database(input_root)
    log_path = tmp_path / "logs" / "question-failure.jsonl"

    with pytest.raises(QuestionUnderstandingError, match="Generated query plan is invalid"):
        execute_question_and_log(
            business,
            "Câu hỏi không thể lập kế hoạch",
            JsonPlanAdapter(lambda _: "not JSON", model_name="local-stub"),
            question_id="q-bad-plan",
            business_id="biz-test",
            as_of_date=None,
            log_path=log_path,
            input_root=input_root,
        )

    (record,) = read_run_records(log_path)
    assert record["query_plan"] is None
    assert record["execution_trace"] is None
    assert "QuestionUnderstandingError" in record["error"]
    assert "model_output_prefix='not JSON'" in record["error"]


@pytest.mark.parametrize(
    ("output_type", "status"),
    [("unanswerable", "ambiguous"), ("abstain", "abstained")],
)
def test_question_pipeline_records_explicit_abstention_without_execution(
    tmp_path: Path,
    output_type: str,
    status: str,
) -> None:
    input_root = tmp_path / "input"
    business = _business_database(input_root)
    table = next(
        table for table in BusinessDataLoader(business).discover() if table.name == "sales"
    )
    table_id = table_identifier(table)
    abstention = {
        "base_table": table_id,
        "output_type": output_type,
        "metadata": {
            "status": status,
            "reason": "Chưa rõ người hỏi muốn tổng tiền hay số hóa đơn.",
        },
    }
    log_path = tmp_path / "logs" / "abstention.jsonl"

    result = execute_question_and_log(
        business,
        "Tháng này thế nào?",
        JsonPlanAdapter(
            lambda _: json.dumps(abstention),
            model_name="local-stub",
        ),
        question_id="q-ambiguous",
        business_id="biz-test",
        as_of_date="2026-01-31",
        log_path=log_path,
        input_root=input_root,
    )

    (record,) = read_run_records(log_path)
    assert result.answer.status == status
    assert result.answer.answer_type == output_type
    assert result.execution is None
    assert record["execution_trace"] is None
    assert record["answer"]["value"] is None
    assert record["answer"]["explanation"] == abstention["metadata"]["reason"]


def test_pipeline_records_plan_validation_failures(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    business = _business_database(input_root)
    log_path = tmp_path / "logs" / "failures.jsonl"
    plan = QueryPlan(base_table="missing.csv::table", output_type="number")

    with pytest.raises(PlanValidationError):
        execute_and_log_business_plan(
            business,
            plan,
            question_id="q-bad",
            business_id="biz-001",
            question="Câu hỏi không thể lập plan",
            model="manual-query-plan",
            log_path=log_path,
            input_root=input_root,
        )

    (record,) = read_run_records(log_path)
    assert record["answer"] is None
    assert "PlanValidationError" in record["error"]
