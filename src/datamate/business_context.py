"""Đọc chỉ-đọc tài liệu tham chiếu doanh nghiệp cho bước lập kế hoạch."""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

MAX_CONTEXT_FILE_BYTES = 64 * 1024
MAX_CONTEXT_TOTAL_BYTES = 96 * 1024


class BusinessContextError(ValueError):
    """Lỗi khi tài liệu tham chiếu không thể được dùng an toàn làm ngữ cảnh."""


@dataclass(frozen=True)
class BusinessContextDocuments:
    rulebook: str | None
    business_readme: str | None
    calendar_columns: tuple[str, ...]
    calendar_rows: tuple[dict[str, str], ...]
    source_metadata: dict[str, dict[str, str]]


def _find_file(directory: Path, name: str) -> Path | None:
    direct = directory / name
    if direct.is_file():
        return direct
    try:
        matches = [
            item
            for item in directory.iterdir()
            if item.is_file() and item.name.casefold() == name.casefold()
        ]
    except OSError as exc:
        raise BusinessContextError(
            f"Không thể đọc danh sách tài liệu tham chiếu tại {directory}: {exc}"
        ) from exc
    if len(matches) > 1:
        raise BusinessContextError(
            f"Có nhiều tệp cùng tên không phân biệt hoa thường cho {name} tại {directory}"
        )
    return matches[0] if matches else None


def _select_shared_file(
    explicit_path: Path | str | None,
    roots: tuple[Path, ...],
    name: str,
) -> Path | None:
    if explicit_path is not None:
        path = Path(explicit_path).resolve()
        if not path.is_file():
            raise BusinessContextError(f"Không tìm thấy tài liệu tham chiếu: {path}")
        return path
    for root in roots:
        path = _find_file(root, name)
        if path is not None:
            resolved = path.resolve()
            if not resolved.is_relative_to(root.resolve()):
                raise BusinessContextError(
                    f"Tài liệu {name} trỏ ra ngoài thư mục tham chiếu"
                )
            return resolved
    return None


def _read_text(
    path: Path,
    label: str,
    *,
    max_bytes: int,
) -> tuple[str, dict[str, str]]:
    try:
        with path.open("rb") as stream:
            raw = stream.read(max_bytes + 1)
    except OSError as exc:
        raise BusinessContextError(f"Không thể đọc {label} {path}: {exc}") from exc
    if len(raw) > max_bytes:
        raise BusinessContextError(
            f"{label} vượt giới hạn ngữ cảnh {max_bytes} byte"
        )
    try:
        text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    except UnicodeDecodeError as exc:
        raise BusinessContextError(f"{label} phải là văn bản UTF-8: {path}") from exc
    if not text.strip():
        raise BusinessContextError(f"{label} rỗng: {path}")
    return text, {
        "status": "loaded",
        "file_name": path.name,
        "size_bytes": str(len(raw)),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _parse_calendar(text: str, path: Path) -> tuple[tuple[str, ...], tuple[dict[str, str], ...]]:
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""))
        columns = reader.fieldnames
        if (
            not columns
            or any(not column or not column.strip() for column in columns)
            or len(set(columns)) != len(columns)
        ):
            raise BusinessContextError(
                f"Calendar phải có tiêu đề duy nhất và không rỗng: {path}"
            )
        rows: list[dict[str, str]] = []
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise BusinessContextError(
                    f"Dòng calendar thừa ô tại {path}:{line_number}"
                )
            rows.append(
                {
                    column: row.get(column) or ""
                    for column in columns
                }
            )
    except csv.Error as exc:
        raise BusinessContextError(f"Calendar CSV sai định dạng {path}: {exc}") from exc
    return tuple(columns), tuple(rows)


def load_business_context(
    business_dir: Path | str,
    business_data_root: Path | str,
    *,
    rulebook_path: Path | str | None = None,
    calendar_path: Path | str | None = None,
    max_file_bytes: int = MAX_CONTEXT_FILE_BYTES,
) -> BusinessContextDocuments:
    """Nạp README/rulebook/calendar có giới hạn mà không sửa dữ liệu nguồn."""
    root = Path(business_data_root).resolve()
    business = Path(business_dir).resolve()
    if not root.is_dir() or not business.is_dir():
        raise BusinessContextError("Thư mục doanh nghiệp và gốc dữ liệu phải tồn tại")
    if not business.is_relative_to(root):
        raise BusinessContextError("Thư mục doanh nghiệp phải nằm dưới gốc dữ liệu")
    if max_file_bytes < 1:
        raise ValueError("max_file_bytes phải là số dương")

    source_metadata: dict[str, dict[str, str]] = {}
    readme_path = _find_file(business, "README.md")
    if readme_path is not None and not readme_path.resolve().is_relative_to(business):
        raise BusinessContextError("README doanh nghiệp trỏ ra ngoài thư mục của nó")
    if readme_path is None:
        readme = None
        source_metadata["business_readme"] = {"status": "missing"}
    else:
        readme, source_metadata["business_readme"] = _read_text(
            readme_path,
            "business README",
            max_bytes=max_file_bytes,
        )

    shared_roots = (root.parent, root)
    selected_rulebook = _select_shared_file(
        rulebook_path,
        shared_roots,
        "rulebook.md",
    )
    if selected_rulebook is None:
        rulebook = None
        source_metadata["rulebook"] = {"status": "missing"}
    else:
        rulebook, source_metadata["rulebook"] = _read_text(
            selected_rulebook,
            "rulebook",
            max_bytes=max_file_bytes,
        )

    selected_calendar = _select_shared_file(
        calendar_path,
        shared_roots,
        "calendar.csv",
    )
    if selected_calendar is None:
        calendar_columns: tuple[str, ...] = ()
        calendar_rows: tuple[dict[str, str], ...] = ()
        source_metadata["calendar"] = {"status": "missing"}
    else:
        calendar_text, source_metadata["calendar"] = _read_text(
            selected_calendar,
            "calendar",
            max_bytes=max_file_bytes,
        )
        calendar_columns, calendar_rows = _parse_calendar(
            calendar_text,
            selected_calendar,
        )

    total_bytes = sum(
        int(metadata.get("size_bytes", "0"))
        for metadata in source_metadata.values()
    )
    if total_bytes > MAX_CONTEXT_TOTAL_BYTES:
        raise BusinessContextError(
            f"Tài liệu tham chiếu có tổng {total_bytes} byte, vượt giới hạn "
            f"ngữ cảnh {MAX_CONTEXT_TOTAL_BYTES} byte"
        )

    return BusinessContextDocuments(
        rulebook=rulebook,
        business_readme=readme,
        calendar_columns=calendar_columns,
        calendar_rows=calendar_rows,
        source_metadata=source_metadata,
    )
