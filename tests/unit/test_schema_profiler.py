import json
import sqlite3
from pathlib import Path

from datamate.schema_profiler import (
    profile_training_dataset,
    render_schema_report,
    summarize_profile,
    write_profile,
)


def test_profile_reports_types_nulls_and_sampled_candidate_keys(tmp_path: Path) -> None:
    root = tmp_path / "train"
    business = root / "data" / "biz_001"
    business.mkdir(parents=True)
    (business / "README.md").write_text("# Test shop\n", encoding="utf-8")
    (business / "orders.csv").write_text(
        "order_id,amount,order_date,status\n"
        "A001,12.5,2026-01-02,paid\n"
        "A002,20,2026-01-03,paid\n"
        "A003,,2026-01-04,open\n",
        encoding="utf-8",
    )
    (root / "questions.jsonl").write_text(
        json.dumps(
            {
                "id": "q1",
                "business_id": "biz_001",
                "question": "Doanh thu tháng này so với tháng trước?",
                "as_of_date": "2026-01-31",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "answers.jsonl").write_text(
        json.dumps({"id": "q1", "type": "percent", "answer": 2.5}) + "\n",
        encoding="utf-8",
    )
    (root / "calendar.csv").write_text("date\n2026-01-01\n", encoding="utf-8")
    (root / "rulebook.md").write_text("# Rules\n## Revenue\n", encoding="utf-8")

    profile = profile_training_dataset(root)
    table = profile["businesses"][0]["tables"][0]
    columns = {column["name"]: column for column in table["columns"]}

    assert table["row_count"] == 3
    assert columns["order_id"]["candidate_key"] is True
    assert columns["amount"]["null_count"] == 1
    assert columns["amount"]["observed_types"] == {"decimal": 1, "integer": 1}
    assert "possible_currency" in columns["amount"]["semantic_tags"]
    assert columns["order_date"]["observed_types"] == {"date_text": 3}
    assert profile["training_metadata"]["question_answer_id_alignment"] == {
        "unmatched_question_ids": 0,
        "extra_answer_ids": 0,
    }
    assert profile["training_metadata"]["answers"]["types"] == {"percent": 1}
    assert profile["training_metadata"]["questions"]["overlapping_keyword_signals"][
        "time_comparison"
    ] == 1


def test_sqlite_primary_key_is_detected_and_foreign_key_is_only_a_candidate(
    tmp_path: Path,
) -> None:
    root = tmp_path / "train"
    business = root / "data" / "biz_002"
    business.mkdir(parents=True)
    (business / "README.md").write_text("# Another shop\n", encoding="utf-8")
    with sqlite3.connect(business / "shop.sqlite") as connection:
        connection.execute("CREATE TABLE customer (customer_id INTEGER PRIMARY KEY)")
        connection.execute(
            "CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER)"
        )
        connection.executemany("INSERT INTO customer VALUES (?)", [(1,), (2,)])
        connection.executemany(
            "INSERT INTO orders VALUES (?, ?)", [(10, 1), (11, 2), (12, 1)]
        )

    profile = profile_training_dataset(root)
    summary = summarize_profile(profile)
    relationship = summary["relationship_candidates"][0]

    assert relationship["source_table"] == "orders"
    assert relationship["source_column"] == "customer_id"
    assert relationship["target_table"] == "customer"
    assert relationship["status"] == "unverified_candidate"
    assert relationship["evidence"] == "normalized_column_name_only"


def test_profile_is_sanitized_and_report_states_heuristic_limits(tmp_path: Path) -> None:
    root = tmp_path / "train"
    business = root / "data" / "biz_003"
    business.mkdir(parents=True)
    (business / "README.md").write_text("# Shop\n", encoding="utf-8")
    (business / "private.csv").write_text(
        "customer_name,amount\nNguyen Van A,100\n", encoding="utf-8"
    )

    profile = profile_training_dataset(root)
    destination = tmp_path / "profile.json"
    write_profile(profile, destination)
    serialized = destination.read_text(encoding="utf-8")
    report = render_schema_report(profile)

    assert "Nguyen Van A" not in serialized
    assert "Khóa ngoại chưa được xác nhận tự động" in report
