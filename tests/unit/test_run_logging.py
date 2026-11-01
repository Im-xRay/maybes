import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from datamate.answer_validation import Evidence, ValidatedAnswer
from datamate.query_executor import ExecutionTrace
from datamate.query_plan import Aggregation, ColumnRef, QueryPlan
from datamate.run_logging import (
    RunLogError,
    RunRecord,
    append_run_record,
    read_run_records,
)


def test_run_log_round_trips_plan_trace_and_decimal_evidence(tmp_path: Path) -> None:
    table = "sales.csv::sales"
    amount = ColumnRef(table, "amount")
    record = RunRecord(
        question_id="q-1",
        business_id="biz-1",
        question="Doanh thu ngày nào cao nhất?",
        model="local-unselected",
        elapsed_ms=12.5,
        plan=QueryPlan(
            base_table=table,
            aggregations=(Aggregation("sum", amount, "revenue"),),
        ),
        execution=ExecutionTrace((table,), {table: 3}, 3, 3, 1, 1),
        answer=ValidatedAnswer(
            "number",
            Decimal("123.45"),
            "answered",
            (Evidence(table, ("amount",), "sum", 3),),
        ),
        confidence=0.8,
        input_metadata={"as_of_date": date(2026, 1, 31)},
        timestamp="2026-01-31T12:00:00+00:00",
    )
    log_path = tmp_path / "logs" / "runs.jsonl"

    append_run_record(log_path, record, forbidden_roots=(tmp_path / "data",))
    (loaded,) = read_run_records(log_path)

    assert loaded["query_plan"]["base_table"] == table
    assert loaded["execution_trace"]["source_rows"] == {table: 3}
    assert loaded["answer"]["value"] == "123.45"
    assert loaded["input"]["metadata"]["as_of_date"] == "2026-01-31"


def test_run_log_refuses_data_root_and_rejects_malformed_records(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    record = RunRecord("q", "biz", "question", "none", 0)

    with pytest.raises(RunLogError, match="must not be written under input root"):
        append_run_record(data_root / "run.jsonl", record, forbidden_roots=(data_root,))

    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"schema_version":1}\nnot-json\n', encoding="utf-8")
    with pytest.raises(RunLogError, match="Invalid JSON"):
        read_run_records(broken)
