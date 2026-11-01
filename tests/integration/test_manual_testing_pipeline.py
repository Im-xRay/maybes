import json
import sqlite3
from pathlib import Path

from datamate.data_loader import BusinessDataLoader
from datamate.manual_testing import run_manual_test
from datamate.query_plan import Aggregation, ColumnRef, QueryPlan
from datamate.question_understanding import JsonPlanAdapter


def test_manual_test_runs_original_questions_and_compares_after_prediction(
    tmp_path: Path,
) -> None:
    business_root = tmp_path / "businesses"
    business = business_root / "biz_1"
    business.mkdir(parents=True)
    with sqlite3.connect(business / "shop.sqlite") as connection:
        connection.execute("CREATE TABLE sales (amount INTEGER)")
        connection.executemany("INSERT INTO sales VALUES (?)", [(12,), (8,)])
    table = next(
        item
        for item in BusinessDataLoader(business).discover()
        if item.name == "sales"
    )
    table_id = f"{table.relative_path}::{table.sqlite_table}"
    plan = QueryPlan(
        base_table=table_id,
        aggregations=(
            Aggregation("sum", ColumnRef(table_id, "amount"), "total"),
        ),
        output_type="number",
    )
    prompts: list[str] = []
    planner = JsonPlanAdapter(
        lambda prompt: prompts.append(prompt)
        or json.dumps(
            {
                "base_table": plan.base_table,
                "aggregations": [
                    {
                        "function": "sum",
                        "column": {"table": table_id, "column": "amount"},
                        "alias": "total",
                    }
                ],
                "output_type": "number",
            }
        ),
        model_name="manual-test-stub",
    )
    questions = tmp_path / "manual_test" / "questions.jsonl"
    questions.parent.mkdir()
    questions.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "q-correct",
                        "business_id": "biz_1",
                        "question": "Câu hỏi tôi tự viết: tổng tiền?",
                        "expected_answer": {"type": "number", "answer": 20},
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "id": "q-wrong",
                        "business_id": "biz_1",
                        "question": "Câu hỏi thứ hai do tôi tự viết",
                        "expected_answer": {
                            "type": "number",
                            "answer": 987654321,
                        },
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "id": "q-unlabeled",
                        "business_id": "biz_1",
                        "question": "Chỉ xem dự đoán",
                        "expected_answer": None,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "id": "q-invalid-expected",
                        "business_id": "biz_1",
                        "question": "Đáp án chuẩn sai định dạng vẫn không được lộ",
                        "expected_answer": {
                            "type": "number",
                            "answer": "secret-expected-value",
                        },
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    results = tmp_path / "manual_test" / "results.jsonl"
    report = tmp_path / "manual_test" / "report.json"

    summary = run_manual_test(
        questions,
        business_root,
        planner,
        results,
        report,
    )

    result_records = [
        json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()
    ]
    assert summary["total_questions"] == 4
    assert summary["processed_successfully"] == 4
    assert summary["errors"] == 1
    assert summary["prediction_errors"] == 0
    assert summary["correct_answers"] == 1
    assert summary["wrong_answers"] == 1
    assert summary["without_expected_answer"] == 1
    assert summary["accuracy"] == 0.5
    assert result_records[0]["generated_answer"] == {"type": "number", "answer": 20}
    assert result_records[0]["correctness"] is True
    assert result_records[1]["correctness"] is False
    assert result_records[1]["error_category"] == "Cần phân tích thủ công"
    assert result_records[2]["correctness"] is None
    assert result_records[2]["expected_answer_valid"] is None
    assert result_records[0]["query_plan"]["base_table"] == table_id
    assert result_records[0]["evidence"]
    assert result_records[0]["execution_trace"]
    assert "987654321" not in "\n".join(prompts)
    assert "secret-expected-value" not in "\n".join(prompts)
    assert result_records[3]["generated_answer"] == {"type": "number", "answer": 20}
    assert result_records[3]["expected_answer_valid"] is False
    assert json.loads(report.read_text(encoding="utf-8")) == summary


def test_manual_test_writes_error_record_for_bad_input_without_skipping(
    tmp_path: Path,
) -> None:
    business_root = tmp_path / "businesses"
    business_root.mkdir()
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        '{invalid json\n'
        + json.dumps(
            {
                "id": "bad",
                "business_id": "biz",
                "question": "Lỗi schema",
                "unexpected": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    results = tmp_path / "results.jsonl"
    report = tmp_path / "report.json"

    class PlannerThatMustNotRun:
        model_name = "must-not-run"

        def create_plan(self, question, context):
            raise AssertionError("Invalid input must not reach planner")

    summary = run_manual_test(
        questions,
        business_root,
        PlannerThatMustNotRun(),
        results,
        report,
    )

    output = [
        json.loads(line)
        for line in results.read_text(encoding="utf-8").splitlines()
    ]
    assert len(output) == 2
    assert all(item["prediction_status"] == "error" for item in output)
    assert summary["total_questions"] == 2
    assert summary["prediction_errors"] == 2
    assert summary["errors"] == 2
