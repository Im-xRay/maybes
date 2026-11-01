"""Memory-bounded schema profiling for heterogeneous business data."""

from __future__ import annotations

import csv
import json
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable

from .data_loader import BusinessDataLoader, DataLoadError, TableRef

_DATE_PATTERNS = (
    re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$"),
    re.compile(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$"),
    re.compile(r"^\d{4}-\d{1,2}$"),
)
_INTEGER = re.compile(r"^[+-]?\d+$")
_NUMBER = re.compile(r"^[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)$")
_MONEY_NAME = re.compile(
    r"tien|doanh.?thu|thanh.?toan|gia.?von|gia.?ban|price|amount|revenue|cost",
    re.IGNORECASE,
)
_QUANTITY_NAME = re.compile(
    r"so.?luong|sl$|quantity|count|ton.?kho|he.?so|quy.?doi",
    re.IGNORECASE,
)
_DATE_NAME = re.compile(
    r"ngay|date|time|luc|han.?su.?dung|thoi.?gian|timestamp",
    re.IGNORECASE,
)
_STATUS_NAME = re.compile(r"trang.?thai|status|ket.?qua", re.IGNORECASE)
_ID_NAME = re.compile(r"(^id$|^ma|_id$|id$|code$|^so.?hoa.?don$)", re.IGNORECASE)
_NULL_STRINGS = {"", "null", "none", "nan", "n/a", "na", "-"}
_DISTINCT_CAP = 10_000
_KEY_SAMPLE_CAP = 10_000
_VALUE_SAMPLE_CAP = 2_000
_QUESTION_SIGNALS = {
    "sales_or_revenue": re.compile(r"doanh thu|bán được|bán ra|tiền bán", re.IGNORECASE),
    "inventory": re.compile(r"tồn kho|còn hàng|hết hàng|sắp hết", re.IGNORECASE),
    "debt": re.compile(r"công nợ|còn nợ|trả nợ|tuổi nợ", re.IGNORECASE),
    "ranking": re.compile(r"nhất|top\s*\d|nhiều nhất|ít nhất|cao nhất|thấp nhất", re.IGNORECASE),
    "time_comparison": re.compile(r"so với|tăng|giảm|chênh lệch|kỳ trước", re.IGNORECASE),
    "date_or_period": re.compile(r"hôm nay|hôm qua|tuần|tháng|quý|năm|tết", re.IGNORECASE),
    "list_or_table": re.compile(r"liệt kê|danh sách|bảng", re.IGNORECASE),
    "customer": re.compile(r"khách hàng|khách nào|đối tác", re.IGNORECASE),
}


def _normalize_name(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    ascii_name = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "", ascii_name)


def _is_null(value: object) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and value.strip().casefold() in _NULL_STRINGS


def _value_kind(value: object) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, datetime):
        return "datetime"
    if isinstance(value, date):
        return "date"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, (float, Decimal)):
        return "decimal"
    if not isinstance(value, str):
        return "other"

    text = value.strip()
    if not text:
        return "empty"
    if text.casefold() in {"true", "false", "yes", "no"}:
        return "boolean"
    if any(pattern.fullmatch(text) for pattern in _DATE_PATTERNS):
        return "date_text"
    numeric = text.replace(" ", "")
    if re.fullmatch(r"[+-]?\d{1,3}(?:[.,]\d{3})+", numeric):
        numeric = numeric.replace(".", "").replace(",", "")
    elif "," in numeric and "." not in numeric:
        numeric = numeric.replace(",", ".")
    if _NUMBER.fullmatch(numeric):
        try:
            Decimal(numeric)
            return "integer" if _INTEGER.fullmatch(numeric) else "decimal"
        except InvalidOperation:
            pass
    return "text"


def _semantic_tags(column: str, observed: Counter[str]) -> list[str]:
    normalized_name = _normalize_name(column)
    tags: list[str] = []
    if _DATE_NAME.search(normalized_name) or (
        observed.get("date", 0)
        + observed.get("datetime", 0)
        + observed.get("date_text", 0)
    ):
        tags.append("date_or_time")
    if _MONEY_NAME.search(normalized_name):
        tags.append("possible_currency")
    if _QUANTITY_NAME.search(normalized_name):
        tags.append("possible_quantity_or_measure")
    if _STATUS_NAME.search(normalized_name):
        tags.append("possible_status")
    if _ID_NAME.search(normalized_name):
        tags.append("possible_identifier")
    return tags


@dataclass
class ColumnAccumulator:
    """Bounded per-column aggregates; raw values are never written to reports."""

    values: Counter[str] = field(default_factory=Counter)
    sample_distinct: set[str] = field(default_factory=set)
    sample_key_values: set[str] = field(default_factory=set)
    sample_non_null: int = 0
    distinct_saturated: bool = False
    key_saturated: bool = False

    def observe(self, value: object, *, row_index: int) -> None:
        if _is_null(value):
            self.values["null"] += 1
            return
        kind = _value_kind(value)
        self.values[kind] += 1
        if self.sample_non_null < _VALUE_SAMPLE_CAP:
            self.sample_non_null += 1
            if len(self.sample_distinct) < _DISTINCT_CAP:
                self.sample_distinct.add(str(value))
            else:
                self.distinct_saturated = True
        if row_index <= _KEY_SAMPLE_CAP:
            if len(self.sample_key_values) < _DISTINCT_CAP:
                self.sample_key_values.add(str(value))
            else:
                self.key_saturated = True


def _profile_column(
    name: str,
    accumulator: ColumnAccumulator,
    *,
    row_count: int,
    sqlite_primary_key: bool = False,
) -> dict[str, object]:
    counts = accumulator.values
    nulls = counts.get("null", 0)
    non_null = row_count - nulls
    types = {key: value for key, value in counts.items() if key != "null"}
    dominant = max(types, key=types.get) if types else "empty"
    inferred = dominant if len(types) <= 1 else "mixed"
    sample_rows = min(row_count, _KEY_SAMPLE_CAP)
    sample_distinct = len(accumulator.sample_key_values)
    candidate_key = (
        sqlite_primary_key
        or (
            sample_rows > 0
            and sample_distinct == sample_rows
            and nulls == 0
            and not accumulator.key_saturated
        )
    )
    return {
        "name": name,
        "inferred_type": inferred,
        "observed_types": dict(sorted(types.items())),
        "null_count": nulls,
        "non_null_count": non_null,
        "null_fraction": round(nulls / row_count, 6) if row_count else None,
        "sample_distinct_non_null": len(accumulator.sample_distinct),
        "distinct_sample_size": accumulator.sample_non_null,
        "distinct_sample_saturated": accumulator.distinct_saturated,
        "candidate_key": candidate_key,
        "candidate_key_basis": (
            "sqlite_primary_key"
            if sqlite_primary_key
            else (
                f"unique_non_null_in_first_{sample_rows}_rows"
                if candidate_key
                else (
                    "not_unique_in_sample"
                    if sample_rows
                    else "empty_table"
                )
            )
        ),
        "semantic_tags": _semantic_tags(name, counts),
    }


def _sqlite_primary_keys(
    business_dir: Path, table: TableRef
) -> set[str]:
    if table.source_format != "sqlite" or table.sqlite_table is None:
        return set()
    path = (business_dir / Path(*Path(table.relative_path).parts)).resolve()
    try:
        with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True) as connection:
            rows = connection.execute(
                f'PRAGMA table_info("{table.sqlite_table.replace(chr(34), chr(34) * 2)}")'
            ).fetchall()
    except sqlite3.Error:
        return set()
    return {row[1] for row in rows if row[5] > 0}


def _profile_table(
    business_dir: Path,
    table: TableRef,
    rows: Iterable[tuple[object, ...]],
    *,
    max_rows: int | None,
) -> dict[str, object]:
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be greater than zero")
    accumulators = [ColumnAccumulator() for _ in table.columns]
    row_count = 0
    errors: list[str] = []
    try:
        for row_count, row in enumerate(rows, start=1):
            if len(row) != len(table.columns):
                raise DataLoadError(
                    f"Column count mismatch in {table.relative_path}:{table.name} "
                    f"at data row {row_count}"
                )
            for accumulator, value in zip(accumulators, row, strict=True):
                accumulator.observe(value, row_index=row_count)
            if max_rows is not None and row_count >= max_rows:
                break
    except (DataLoadError, csv.Error, sqlite3.Error, OSError, UnicodeError) as exc:
        errors.append(str(exc))

    primary_keys = _sqlite_primary_keys(business_dir, table)
    columns = [
        _profile_column(
            column,
            accumulator,
            row_count=row_count,
            sqlite_primary_key=column in primary_keys,
        )
        for column, accumulator in zip(table.columns, accumulators, strict=True)
    ]
    return {
        "source": table.relative_path,
        "table": table.name,
        "format": table.source_format,
        "sheet": table.sheet_name,
        "header_row": table.header_row if table.source_format == "xlsx" else None,
        "row_count": row_count,
        "row_count_is_sampled": max_rows is not None and row_count >= max_rows,
        "columns": columns,
        "warnings": list(table.warnings),
        "errors": errors,
    }


def _business_label(readme_path: Path) -> tuple[str | None, str | None]:
    try:
        lines = readme_path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return None, None
    title = next((line.lstrip("# ").strip() for line in lines if line.startswith("#")), None)
    text = "\n".join(lines).casefold()
    categories = {
        "pharmacy": ("nhà thuốc", "thuốc kê đơn", "dược"),
        "construction_materials": ("vật liệu xây dựng", "xi măng", "sắt thép"),
        "cafe_or_food_service": ("cà phê", "quán ăn", "nhà hàng", "đồ uống"),
        "grocery_or_retail": ("tạp hóa", "siêu thị", "bán lẻ"),
        "other_or_unknown": (),
    }
    matches = [
        category
        for category, terms in categories.items()
        if terms and any(term in text for term in terms)
    ]
    if len(matches) == 1:
        return title, matches[0]
    return title, "other_or_unknown" if not matches else "ambiguous"


def _profile_package_files(root: Path) -> dict[str, object]:
    questions_path = root / "questions.jsonl"
    answers_path = root / "answers.jsonl"
    question_count = 0
    question_ids: set[str] = set()
    business_counts: Counter[str] = Counter()
    question_signals: Counter[str] = Counter()
    as_of_dates: list[str] = []
    malformed_questions = 0
    if questions_path.is_file():
        try:
            with questions_path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            raise ValueError("record is not an object")
                        question_count += 1
                        identifier = record.get("id")
                        if isinstance(identifier, str):
                            question_ids.add(identifier)
                        business_id = record.get("business_id")
                        if isinstance(business_id, str):
                            business_counts[business_id] += 1
                        as_of = record.get("as_of_date")
                        if isinstance(as_of, str):
                            as_of_dates.append(as_of)
                        question = record.get("question")
                        if isinstance(question, str):
                            for category, pattern in _QUESTION_SIGNALS.items():
                                if pattern.search(question):
                                    question_signals[category] += 1
                    except (json.JSONDecodeError, ValueError):
                        malformed_questions += 1
        except (OSError, UnicodeError) as exc:
            malformed_questions += 1
            question_error = str(exc)
        else:
            question_error = None
    else:
        question_error = "questions.jsonl is missing"

    answer_count = 0
    answer_ids: set[str] = set()
    answer_types: Counter[str] = Counter()
    alias_count = 0
    malformed_answers = 0
    if answers_path.is_file():
        try:
            with answers_path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            raise ValueError("record is not an object")
                        answer_count += 1
                        identifier = record.get("id")
                        if isinstance(identifier, str):
                            answer_ids.add(identifier)
                        answer_type = record.get("type")
                        if isinstance(answer_type, str):
                            answer_types[answer_type] += 1
                        if record.get("alias"):
                            alias_count += 1
                    except (json.JSONDecodeError, ValueError):
                        malformed_answers += 1
        except (OSError, UnicodeError) as exc:
            malformed_answers += 1
            answer_error = str(exc)
        else:
            answer_error = None
    else:
        answer_error = "answers.jsonl is missing"

    calendar_path = root / "calendar.csv"
    calendar_summary: dict[str, object] | None = None
    if calendar_path.is_file():
        try:
            with calendar_path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                dates = [row.get("date", "") for row in reader if row.get("date")]
            calendar_summary = {
                "row_count": len(dates),
                "min_date": min(dates) if dates else None,
                "max_date": max(dates) if dates else None,
            }
        except (OSError, UnicodeError, csv.Error) as exc:
            calendar_summary = {"error": str(exc)}
    rulebook_path = root / "rulebook.md"
    headings: list[str] = []
    if rulebook_path.is_file():
        try:
            headings = [
                line.lstrip("# ").strip()
                for line in rulebook_path.read_text(encoding="utf-8-sig").splitlines()
                if line.startswith("#")
            ]
        except (OSError, UnicodeError):
            headings = []
    return {
        "questions": {
            "record_count": question_count,
            "unique_id_count": len(question_ids),
            "business_counts": dict(sorted(business_counts.items())),
            "as_of_date_min": min(as_of_dates) if as_of_dates else None,
            "as_of_date_max": max(as_of_dates) if as_of_dates else None,
            "overlapping_keyword_signals": dict(question_signals),
            "malformed_records": malformed_questions,
            "error": question_error,
        },
        "answers": {
            "record_count": answer_count,
            "unique_id_count": len(answer_ids),
            "types": dict(sorted(answer_types.items())),
            "records_with_alias": alias_count,
            "malformed_records": malformed_answers,
            "error": answer_error,
        },
        "question_answer_id_alignment": {
            "unmatched_question_ids": len(question_ids - answer_ids),
            "extra_answer_ids": len(answer_ids - question_ids),
        },
        "calendar": calendar_summary,
        "rulebook_headings": headings,
    }


def profile_training_dataset(
    training_root: Path | str,
    *,
    max_rows_per_table: int | None = None,
) -> dict[str, object]:
    """Profile every business table with bounded memory and no value samples."""
    root = Path(training_root).resolve()
    data_root = root / "data"
    if not data_root.is_dir():
        raise DataLoadError(f"Training business directory is missing: {data_root}")

    businesses: list[dict[str, object]] = []
    for business_dir in sorted(
        (path for path in data_root.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    ):
        loader = BusinessDataLoader(business_dir)
        title, category = _business_label(business_dir / "README.md")
        try:
            tables = loader.discover()
            profiles = []
            for table, rows in loader.iter_table_rows(tables):
                profiles.append(
                    _profile_table(
                        business_dir,
                        table,
                        rows,
                        max_rows=max_rows_per_table,
                    )
                )
            errors: list[str] = []
        except DataLoadError as exc:
            profiles = []
            errors = [str(exc)]
        businesses.append(
            {
                "business": business_dir.name,
                "title": title,
                "category_heuristic": category,
                "files": len(
                    [
                        path
                        for path in business_dir.rglob("*")
                        if path.is_file()
                    ]
                ),
                "tables": profiles,
                "errors": errors,
            }
        )

    root_files = {
        name: (root / name).is_file()
        for name in (
            "questions.jsonl",
            "answers.jsonl",
            "calendar.csv",
            "rulebook.md",
        )
    }
    return {
        "version": 1,
        "training_root": str(root),
        "profile_mode": (
            f"sample_first_{max_rows_per_table}_rows_per_table"
            if max_rows_per_table
            else "full_stream"
        ),
        "max_rows_per_table": max_rows_per_table,
        "businesses": businesses,
        "package_files": root_files,
        "training_metadata": _profile_package_files(root),
    }


def write_profile(profile: dict[str, object], output_path: Path | str) -> None:
    """Write a profile JSON atomically without including raw column values."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(profile, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary.replace(destination)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def summarize_profile(profile: dict[str, object]) -> dict[str, object]:
    businesses = profile.get("businesses", [])
    formats: Counter[str] = Counter()
    table_count = 0
    row_count = 0
    columns: dict[str, Counter[str]] = defaultdict(Counter)
    table_errors = 0
    key_candidates = 0
    warnings: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    relationships: list[dict[str, str]] = []
    for business in businesses:
        categories[str(business.get("category_heuristic"))] += 1
        business_tables = business.get("tables", [])
        key_columns: dict[str, list[tuple[str, dict[str, object]]]] = defaultdict(list)
        for table in business_tables:
            for column in table.get("columns", []):
                if column.get("candidate_key"):
                    key_columns[_normalize_name(str(column.get("name", "")))].append(
                        (str(table.get("table")), column)
                    )
        for table in business_tables:
            table_count += 1
            row_count += int(table.get("row_count", 0))
            formats[str(table.get("format"))] += 1
            table_errors += bool(table.get("errors"))
            warnings.update(table.get("warnings", []))
            for column in table.get("columns", []):
                name = str(column.get("name", ""))
                normalized_name = _normalize_name(name)
                columns[_normalize_name(name)]["occurrences"] += 1
                columns[normalized_name][str(column.get("inferred_type"))] += 1
                columns[_normalize_name(name)][
                    "possible_key"
                ] += bool(column.get("candidate_key"))
                key_candidates += bool(column.get("candidate_key"))
                if (
                    not column.get("candidate_key")
                    and _ID_NAME.search(normalized_name)
                    and normalized_name in key_columns
                ):
                    for target_table, target_column in key_columns[normalized_name]:
                        if target_table == str(table.get("table")):
                            continue
                        relationships.append(
                            {
                                "source_table": str(table.get("table")),
                                "source_column": name,
                                "target_table": target_table,
                                "target_column": str(target_column.get("name")),
                                "evidence": "normalized_column_name_only",
                                "status": "unverified_candidate",
                            }
                        )
    recurrent_columns = []
    for normalized, counts in sorted(
        columns.items(), key=lambda item: (-item[1]["occurrences"], item[0])
    ):
        if counts["occurrences"] < 3:
            continue
        recurrent_columns.append(
            {
                "normalized_name": normalized,
                **dict(counts),
            }
        )
    return {
        "business_count": len(businesses),
        "business_categories_heuristic": dict(categories),
        "training_metadata": profile.get("training_metadata", {}),
        "table_count": table_count,
        "table_formats": dict(formats),
        "row_count": row_count,
        "table_error_count": table_errors,
        "candidate_key_column_count": key_candidates,
        "table_warning_counts": dict(warnings),
        "relationship_candidates": relationships[:500],
        "recurrent_columns": recurrent_columns,
    }


def render_schema_report(profile: dict[str, object]) -> str:
    summary = summarize_profile(profile)
    warning_counts = summary["table_warning_counts"]
    shifted_headers = warning_counts.get("leading_metadata_rows_skipped", 0)
    row_two_width_warnings = warning_counts.get("row_2_exceeds_header_width", 0)
    lines = [
        "# Báo cáo profiling schema bộ training",
        "",
        f"**Chế độ:** `{profile['profile_mode']}`.",
        "**Quyền riêng tư:** không ghi giá trị ô dữ liệu; chỉ ghi tên cột, số lượng, kiểu suy luận và nhãn heuristic.",
        "",
        "## Tóm tắt",
        "",
        f"- Doanh nghiệp: {summary['business_count']}",
        f"- Bảng/file worksheet: {summary['table_count']}",
        f"- Định dạng bảng: `{json.dumps(summary['table_formats'], ensure_ascii=False, sort_keys=True)}`",
        f"- Tổng dòng đã đọc: {summary['row_count']}",
        f"- Bảng lỗi khi đọc: {summary['table_error_count']}",
        f"- Cột được gắn ứng viên khóa (heuristic): {summary['candidate_key_column_count']}",
        f"- Phân loại ngành từ README (heuristic): `{json.dumps(summary['business_categories_heuristic'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Cột thường gặp",
        "",
        "Tên được chuẩn hóa bỏ dấu câu/chữ hoa để thống kê tần suất; không đồng nghĩa rằng các cột có cùng tên chắc chắn cùng nghĩa.",
        "",
        "| Tên chuẩn hóa | Số lần xuất hiện | Kiểu quan sát được | Cột duy nhất trong mẫu |",
        "|---|---:|---|---:|",
    ]
    for column in summary["recurrent_columns"][:60]:
        types = {
            key: value
            for key, value in column.items()
            if key not in {"normalized_name", "occurrences", "possible_key"}
        }
        lines.append(
            f"| `{column['normalized_name']}` | {column['occurrences']} | "
            f"`{json.dumps(types, ensure_ascii=False, sort_keys=True)}` | "
            f"{column['possible_key']} |"
        )

    lines.extend(["", "## Chất lượng và cảnh báo", ""])
    lines.append(
        f"- Cảnh báo header/kết cấu bảng: `{json.dumps(summary['table_warning_counts'], ensure_ascii=False, sort_keys=True)}`"
    )
    lines.append(
        f"- Lỗi đọc bảng: {summary['table_error_count']}; xem `data/train/schema_profile.json` để biết bảng cụ thể."
    )
    lines.extend(
        [
            "",
            "## Diễn giải khóa và quan hệ",
            "",
            "- Khóa chính SQLite được đọc từ metadata SQLite.",
            "- Khóa ứng viên CSV/XLSX chỉ được gợi ý nếu giá trị không null trong tối đa 10.000 dòng đầu không lặp; đây không phải xác nhận khóa chính.",
            "- Khóa ngoại chưa được xác nhận tự động. So khớp tên cột hoặc bản ghi trùng nhau có thể suy ra quan hệ sai; cần kiểm tra README, mã nghiệp vụ và tính duy nhất trước khi join.",
            f"- Quan hệ gợi ý chỉ dựa trên tên cột trùng khớp: {len(summary['relationship_candidates'])} (tối đa 500 được ghi trong JSON profile); mọi quan hệ cần xác nhận bằng bản ghi khớp, tính duy nhất và rulebook.",
            "- Cột số điện thoại/mã số có thể trông như số; không đổi mã định danh sang kiểu số để tránh mất số 0 đầu.",
            "",
            "## Câu hỏi, đáp án và tài liệu nghiệp vụ",
            "",
            f"Thống kê JSON an toàn: `{json.dumps(profile.get('training_metadata', {}), ensure_ascii=False, sort_keys=True)}`",
            "",
            "Cặp câu hỏi/đáp án được nối bằng `id`; câu hỏi có `business_id` và `as_of_date`. Các tín hiệu từ khóa có thể chồng lắp nhau, không phải nhãn phân loại của ban tổ chức. Kiểu đáp án được thống kê từ nhãn training, không phải đánh giá PA chính thức.",
            "",
            "Các nhóm nghiệp vụ và từ khóa cần tổng quát hóa: bán hàng/doanh thu, công nợ, tồn kho, nhập hàng, xếp hạng, so sánh thời gian, hàng/hạn dùng và yêu cầu thiếu dữ liệu. Phân loại ngành trong báo cáo là heuristic từ README, không phải nhãn của ban tổ chức.",
            "",
            "## Giới hạn và hành động",
            "",
            "- CSV thường cung cấp giá trị dạng chuỗi; kiểu nghiệp vụ được suy ra từ tên/giá trị bằng thống kê, không chuyển đổi dữ liệu nguồn.",
            f"- Bộ đọc đã nhận diện và bỏ qua các dòng metadata/tiêu đề đầu trang có cảnh báo ở {shifted_headers} worksheet; cần xác nhận `header_row` trước khi sử dụng vì đây là heuristic.",
            f"- Có {row_two_width_warnings} worksheet mà dòng 2 rộng hơn dòng header; các cột header trống vẫn được giữ với tên `__unnamed_N` và cảnh báo.",
            "- Profiling nhận diện null/kiểu/tên cột và khóa ứng viên heuristic; khóa ngoại, đơn vị tiền và ý nghĩa nghiệp vụ của cột vẫn cần kiểm tra với README/rulebook từng business.",
            "- Query planner chỉ nên chạy trên bảng/cột đã phân biệt rõ và quan hệ có căn cứ. Chính sách abstain/unanswerable cần chốt sau khi xác minh evaluator chính thức.",
            "- Profile đầy đủ nằm ở `data/train/schema_profile.json` (artifact cục bộ, không đưa dataset/profile vào commit). Có thể truyền `--max-rows-per-table` để chạy mẫu giới hạn khi kiểm tra nhanh.",
            "",
        ]
    )
    return "\n".join(lines)


def profile_to_json(profile: dict[str, object]) -> str:
    """Serialize a profile with stable JSON formatting."""
    return json.dumps(profile, ensure_ascii=False, indent=2) + "\n"
