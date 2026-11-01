import csv
import hashlib
from pathlib import Path

import pytest

from datamate.business_context import (
    BusinessContextError,
    load_business_context,
)


def test_loads_business_documents_and_records_hashes_without_contents(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "dataset" / "data"
    business_dir = data_root / "biz_1"
    business_dir.mkdir(parents=True)
    rulebook = data_root.parent / "rulebook.md"
    calendar = data_root.parent / "calendar.csv"
    rulebook.write_text("# Quy tắc\nHủy đơn không tính doanh thu.\n", encoding="utf-8")
    (business_dir / "README.md").write_text(
        "amount có đơn vị VND.\n",
        encoding="utf-8",
    )
    with calendar.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["solar_date", "lunar_date"])
        writer.writerow(["2026-01-01", "2025-11-13"])

    documents = load_business_context(business_dir, data_root)

    assert documents.rulebook == "# Quy tắc\nHủy đơn không tính doanh thu.\n"
    assert documents.business_readme == "amount có đơn vị VND.\n"
    assert documents.calendar_columns == ("solar_date", "lunar_date")
    assert documents.calendar_rows == (
        {"solar_date": "2026-01-01", "lunar_date": "2025-11-13"},
    )
    assert documents.source_metadata["rulebook"]["sha256"] == hashlib.sha256(
        rulebook.read_bytes()
    ).hexdigest()
    assert documents.source_metadata["calendar"]["status"] == "loaded"
    assert documents.source_metadata["business_readme"]["status"] == "loaded"
    assert all(
        "Hủy đơn" not in repr(metadata)
        for metadata in documents.source_metadata.values()
    )


def test_missing_context_sources_are_reported_explicitly(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset" / "data"
    business_dir = data_root / "biz_1"
    business_dir.mkdir(parents=True)

    documents = load_business_context(business_dir, data_root)

    assert documents.rulebook is None
    assert documents.calendar_columns == ()
    assert documents.source_metadata == {
        "business_readme": {"status": "missing"},
        "rulebook": {"status": "missing"},
        "calendar": {"status": "missing"},
    }


def test_malformed_calendar_fails_instead_of_being_silently_truncated(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "dataset" / "data"
    business_dir = data_root / "biz_1"
    business_dir.mkdir(parents=True)
    calendar = data_root.parent / "calendar.csv"
    calendar.write_text("date,date\n2026-01-01,2025-11-13\n", encoding="utf-8")

    with pytest.raises(BusinessContextError, match="tiêu đề duy nhất và không rỗng"):
        load_business_context(business_dir, data_root)
