import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from datamate.answer_validation import Evidence, ValidatedAnswer
from datamate.query_plan import Aggregation, ColumnRef, QueryPlan, TimeBucket
from datamate.submission import (
    SubmissionError,
    SubmissionWriter,
    serialize_submission_record,
)


EVIDENCE = (Evidence("sales", ("amount",), "sum", 2),)


def test_submission_keeps_decimal_as_json_number_and_date_as_iso_text() -> None:
    number_line = serialize_submission_record(
        "q-1",
        ValidatedAnswer("number", Decimal("4820000"), "answered", EVIDENCE),
    )
    date_line = serialize_submission_record(
        "q-2",
        ValidatedAnswer("date", date(2026, 8, 15), "answered", EVIDENCE),
    )

    assert '"answer":4820000' in number_line
    assert json.loads(number_line) == {
        "id": "q-1",
        "type": "number",
        "answer": 4820000,
    }
    assert json.loads(date_line)["answer"] == "2026-08-15"


def test_submission_formats_unanswerable_and_abstain_separately() -> None:
    unanswerable = serialize_submission_record(
        "q-1",
        ValidatedAnswer(
            "unanswerable",
            None,
            "ambiguous",
            explanation="Chưa rõ kỳ cần tra cứu",
        ),
    )
    abstain = serialize_submission_record(
        "q-2",
        ValidatedAnswer("abstain", None, "abstained"),
    )

    assert json.loads(unanswerable) == {
        "id": "q-1",
        "type": "unanswerable",
        "answer": {
            "type": "unanswerable",
            "reason": "Chưa rõ kỳ cần tra cứu",
        },
    }
    assert json.loads(abstain) == {"id": "q-2", "type": "abstain"}


def test_submission_allows_unanswerable_without_optional_reason() -> None:
    record = serialize_submission_record(
        "q-1",
        ValidatedAnswer("unanswerable", None, "no_data"),
    )

    assert json.loads(record) == {
        "id": "q-1",
        "type": "unanswerable",
        "answer": {"type": "unanswerable"},
    }


def test_submission_table_uses_plan_columns_and_keeps_numeric_cells(
) -> None:
    table = "sales"
    name = ColumnRef(table, "name")
    amount = ColumnRef(table, "amount")
    plan = QueryPlan(
        base_table=table,
        selected_columns=(name,),
        group_by=(name,),
        aggregations=(Aggregation("sum", amount, "revenue"),),
        output_type="table",
    )

    line = serialize_submission_record(
        "q-1",
        ValidatedAnswer(
            "table",
            [{"sales.name": "Trà", "revenue": Decimal("12.50")}],
            "answered",
            EVIDENCE,
        ),
        plan=plan,
    )

    assert json.loads(line, parse_float=Decimal) == {
        "id": "q-1",
        "type": "table",
        "answer": {
            "columns": ["name", "revenue"],
            "rows": [["Trà", Decimal("12.50")]],
        },
    }


def test_empty_table_uses_plan_to_emit_headers() -> None:
    table = "sales"
    name = ColumnRef(table, "name")
    plan = QueryPlan(
        base_table=table,
        selected_columns=(name,),
        output_type="table",
    )

    line = serialize_submission_record(
        "q-1",
        ValidatedAnswer("table", [], "answered", EVIDENCE),
        plan=plan,
    )

    assert json.loads(line) == {
        "id": "q-1",
        "type": "table",
        "answer": {"columns": ["name"], "rows": []},
    }


def test_empty_table_omits_grouping_key_used_only_for_tie_breaks() -> None:
    table = "sales"
    code = ColumnRef(table, "product_code")
    name = ColumnRef(table, "name")
    amount = ColumnRef(table, "amount")
    plan = QueryPlan(
        base_table=table,
        selected_columns=(name,),
        group_by=(code, name),
        aggregations=(Aggregation("sum", amount, "revenue"),),
        output_type="table",
    )

    line = serialize_submission_record(
        "q-empty-ranked",
        ValidatedAnswer("table", [], "answered", EVIDENCE),
        plan=plan,
    )

    assert json.loads(line)["answer"] == {
        "columns": ["name", "revenue"],
        "rows": [],
    }


def test_empty_time_bucket_table_keeps_period_column_before_measure() -> None:
    table = "sales"
    sold_at = ColumnRef(table, "sold_at")
    plan = QueryPlan(
        base_table=table,
        time_buckets=(TimeBucket(sold_at, "month", "month"),),
        aggregations=(
            Aggregation("sum", ColumnRef(table, "amount"), "revenue"),
        ),
        output_type="table",
    )

    line = serialize_submission_record(
        "q-months",
        ValidatedAnswer("table", [], "answered", EVIDENCE),
        plan=plan,
    )

    assert json.loads(line)["answer"] == {
        "columns": ["month", "revenue"],
        "rows": [],
    }


def test_submission_writer_aborts_on_duplicate_question_id(tmp_path: Path) -> None:
    destination = tmp_path / "submission.jsonl"
    writer = SubmissionWriter(destination)
    answer = ValidatedAnswer("number", 1, "answered", EVIDENCE)
    writer.append("q-1", answer)

    with pytest.raises(SubmissionError, match="bị trùng"):
        writer.append("q-1", answer)
    writer.abort()

    assert not destination.exists()
    assert not list(tmp_path.glob("*.tmp"))
