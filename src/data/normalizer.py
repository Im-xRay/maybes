"""
XRAY Universal Data Normalizer & Auto-Schema Profiler - Module 2
Ultra-fast tabular data ingestion into In-Memory DuckDB with dirty data cleaning,
header offset detection, date/currency normalization, and RapidFuzz entity indexing.
"""

from __future__ import annotations

import csv
import datetime
import os
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import duckdb
import openpyxl
from rapidfuzz import fuzz, process


def remove_accents(text: str) -> str:
    """Strip Vietnamese diacritics for robust matching and SQL column naming."""
    if not text:
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D")
    return text


def to_sql_identifier(col_name: str) -> str:
    """Convert raw messy column name into a clean, deterministic SQL identifier."""
    # Strip BOM and whitespace
    clean = col_name.replace("\ufeff", "").strip()
    # Remove accents
    no_accents = remove_accents(clean)
    # Replace non-alphanumeric with underscore
    ident = re.sub(r"[^\w]+", "_", no_accents).strip("_").lower()
    if not ident:
        ident = "col"
    if ident[0].isdigit():
        ident = "c_" + ident
    return ident


# Pre-compiled Regex patterns for dirty currency and date cleaning
RE_VND_SUFFIX = re.compile(r"^\s*([0-9.,]+)\s*(?:đ|vnd|vnđ|d)\s*$", re.IGNORECASE)
RE_K_SUFFIX = re.compile(r"^\s*([0-9.,]+)\s*k\s*$", re.IGNORECASE)
RE_TR_SUFFIX = re.compile(r"^\s*([0-9.,]+)\s*(?:tr|trieu|triệu)\s*$", re.IGNORECASE)
RE_VN_DATE = re.compile(r"^\s*(\d{1,2})\s+thg\s+(\d{1,2})\s+(\d{4})\s*$", re.IGNORECASE)
RE_SLASH_DATE = re.compile(r"^\s*(\d{1,2})/(\d{1,2})/(\d{4})(?:\s+(\d{1,2}:\d{2}(?::\d{2})?))?\s*$")


def clean_dirty_value(val: Any) -> Any:
    """Clean numbers with suffixes ('79k', '569.000đ', '1,2tr') and Vietnamese dates."""
    if val is None:
        return None

    if isinstance(val, (datetime.datetime, datetime.date)):
        return val.strftime("%Y-%m-%d %H:%M:%S" if isinstance(val, datetime.datetime) else "%Y-%m-%d")

    if isinstance(val, (int, float)):
        return val

    s = str(val).strip()
    if not s:
        return None

    # Date pattern: '1 thg 6 2025' -> '2025-06-01'
    m_vn_date = RE_VN_DATE.match(s)
    if m_vn_date:
        d, m, y = m_vn_date.groups()
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"

    # Date pattern: '15/08/2026' or '15/08/2026 14:30:00'
    m_slash_date = RE_SLASH_DATE.match(s)
    if m_slash_date:
        d, m, y, tm = m_slash_date.groups()
        res = f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
        if tm:
            res += f" {tm}"
        return res

    # Currency '569.000đ' or '50.000 đ'
    m_vnd = RE_VND_SUFFIX.match(s)
    if m_vnd:
        num_str = m_vnd.group(1).replace(".", "").replace(",", "")
        try:
            return int(num_str)
        except ValueError:
            pass

    # K suffix: '79k' -> 79000
    m_k = RE_K_SUFFIX.match(s)
    if m_k:
        num_str = m_k.group(1).replace(",", ".")
        try:
            return int(float(num_str) * 1000)
        except ValueError:
            pass

    # Triệu suffix: '1,2tr' -> 1200000
    m_tr = RE_TR_SUFFIX.match(s)
    if m_tr:
        num_str = m_tr.group(1).replace(",", ".")
        try:
            return int(float(num_str) * 1000000)
        except ValueError:
            pass

    # Standard comma-separated thousand numbers '560,000'
    if re.match(r"^-?\d{1,3}(,\d{3})+$", s):
        try:
            return int(s.replace(",", ""))
        except ValueError:
            pass

    # Standard dot-separated thousand numbers '560.000'
    if re.match(r"^-?\d{1,3}(\.\d{3})+$", s):
        try:
            return int(s.replace(".", ""))
        except ValueError:
            pass

    return s


@dataclass
class ColumnInfo:
    raw_name: str
    sql_name: str
    inferred_type: str
    sample_values: List[Any] = field(default_factory=list)


@dataclass
class TableInfo:
    table_name: str
    source_file: str
    row_count: int
    columns: List[ColumnInfo]
    date_columns: List[str]
    min_date: Optional[str] = None
    max_date: Optional[str] = None


class BusinessDatabase:
    """In-memory DuckDB container for a single business with schema metadata and fuzzy index."""

    def __init__(self, business_id: str):
        self.business_id = business_id
        self.con = duckdb.connect(":memory:")
        self.tables: Dict[str, TableInfo] = {}
        self.min_date: Optional[str] = None
        self.max_date: Optional[str] = None
        self.readme_text: str = ""

        # Entity index: (canonical_name, code, table_name)
        self.products: List[Tuple[str, str, str]] = []
        self.customers: List[Tuple[str, str, str]] = []

    def execute(self, sql: str) -> List[Tuple]:
        """Execute SQL query safely and return rows."""
        return self.con.execute(sql).fetchall()

    def query_scalar(self, sql: str) -> Any:
        """Execute SQL query and return a single scalar value."""
        res = self.con.execute(sql).fetchone()
        return res[0] if res else None

    def get_schema_prompt(self) -> str:
        """Generate a concise, LLM-optimized schema representation."""
        lines = [f"### CƠ SỞ DỮ LIỆU DOANH NGHIỆP: {self.business_id}"]
        if self.min_date and self.max_date:
            lines.append(f"- Phạm vi dữ liệu có sẵn: từ {self.min_date} đến hết {self.max_date}")

        for t_name, t_info in self.tables.items():
            lines.append(f"\nBẢNG `{t_name}` ({t_info.row_count} dòng, từ '{t_info.source_file}'):")
            cols_desc = []
            for c in t_info.columns:
                samples = ", ".join(repr(x) for x in c.sample_values[:2])
                cols_desc.append(f"  - `{c.sql_name}` ({c.inferred_type}) [Gốc: '{c.raw_name}'] Ví dụ: {samples}")
            lines.extend(cols_desc)

        return "\n".join(lines)

    def fuzzy_match_product(self, query: str) -> Optional[Tuple[str, str, float]]:
        """Find best matching product name and code using RapidFuzz."""
        if not self.products:
            return None
        names = [p[0] for p in self.products]
        match = process.extractOne(query, names, scorer=fuzz.token_sort_ratio)
        if match and match[1] >= 65.0:
            idx = match[2]
            return self.products[idx][0], self.products[idx][1], match[1]
        return None

    def fuzzy_match_customer(self, query: str) -> Optional[Tuple[str, str, float]]:
        """Find best matching customer name and code using RapidFuzz."""
        if not self.customers:
            return None
        names = [c[0] for c in self.customers]
        match = process.extractOne(query, names, scorer=fuzz.token_sort_ratio)
        if match and match[1] >= 65.0:
            idx = match[2]
            return self.customers[idx][0], self.customers[idx][1], match[1]
        return None


class UniversalNormalizer:
    """High-performance parser for CSV, XLSX, SQLite into In-Memory DuckDB."""

    def __init__(self, business_data_root: str):
        self.data_root = Path(business_data_root)

    def load_business(self, business_id: str) -> BusinessDatabase:
        """Loads and normalizes all data for a specific business_id."""
        b_dir = self.data_root / business_id
        if not b_dir.exists():
            raise FileNotFoundError(f"Business directory not found: {b_dir}")

        db = BusinessDatabase(business_id)

        # 1. Load README if present
        readme_path = b_dir / "README.md"
        if readme_path.exists():
            with open(readme_path, "r", encoding="utf-8") as f:
                db.readme_text = f.read()

        # 2. Ingest all data files
        for f_path in sorted(b_dir.iterdir()):
            if f_path.name == "README.md" or f_path.is_dir():
                continue
            ext = f_path.suffix.lower()
            if ext == ".csv":
                self._load_csv(f_path, db)
            elif ext == ".xlsx":
                self._load_xlsx(f_path, db)
            elif ext in (".sqlite", ".db"):
                self._load_sqlite(f_path, db)

        # 3. Compute overall date bounds from operational tables
        op_tables = [t for name, t in db.tables.items() if not any(k in name for k in ("khach_hang", "nhan_vien", "khachhang", "nhanvien"))]
        if not op_tables:
            op_tables = list(db.tables.values())

        all_mins = [t.min_date for t in op_tables if t.min_date and t.min_date >= "2024-01-01"]
        all_maxs = [t.max_date for t in op_tables if t.max_date and t.max_date <= "2027-12-31"]
        if all_mins:
            db.min_date = min(all_mins)
        if all_maxs:
            db.max_date = max(all_maxs)

        # 4. Build entity index
        self._build_entity_index(db)

        return db

    def _load_csv(self, file_path: Path, db: BusinessDatabase) -> None:
        """Load and clean CSV file."""
        # Detect encoding
        encoding = "utf-8"
        for enc in ("utf-8-sig", "utf-8", "cp1258", "latin1"):
            try:
                with open(file_path, "r", encoding=enc) as f:
                    f.read(4096)
                encoding = enc
                break
            except UnicodeDecodeError:
                continue

        with open(file_path, "r", encoding=encoding, newline="") as f:
            reader = csv.reader(f)
            raw_header = next(reader, None)
            if not raw_header:
                return

            col_names = [col.replace("\ufeff", "").strip() for col in raw_header]
            sql_cols = self._deduplicate_identifiers([to_sql_identifier(c) for c in col_names])

            rows = []
            for row in reader:
                cleaned_row = [clean_dirty_value(v) for v in row]
                # Pad or truncate row to header length
                if len(cleaned_row) < len(sql_cols):
                    cleaned_row.extend([None] * (len(sql_cols) - len(cleaned_row)))
                rows.append(cleaned_row[:len(sql_cols)])

        table_name = to_sql_identifier(file_path.stem)
        self._register_table(db, table_name, file_path.name, col_names, sql_cols, rows)

    def _load_xlsx(self, file_path: Path, db: BusinessDatabase) -> None:
        """Load and clean Excel file using ultra-fast Rust-based Calamine (fallback to openpyxl)."""
        try:
            from python_calamine import CalamineWorkbook
            wb = CalamineWorkbook.from_path(str(file_path))
            sheet_names = wb.sheet_names

            for s_name in sheet_names:
                sheet = wb.get_sheet_by_name(s_name)
                all_rows = sheet.to_python()
                if not all_rows:
                    continue

                # Header detection in first 6 rows
                preview = all_rows[:6]
                header_idx = 0
                max_str_count = 0
                for idx, r in enumerate(preview):
                    str_count = sum(1 for c in r if c is not None and str(c).strip() != "")
                    if str_count >= 2 and str_count > max_str_count:
                        max_str_count = str_count
                        header_idx = idx

                raw_header = [str(c).strip() if c is not None else f"col_{i+1}" for i, c in enumerate(all_rows[header_idx])]
                sql_cols = self._deduplicate_identifiers([to_sql_identifier(c) for c in raw_header])

                rows = []
                for r in all_rows[header_idx + 1:]:
                    cleaned = [clean_dirty_value(c) for c in r]
                    if any(v is not None for v in cleaned):
                        if len(cleaned) < len(sql_cols):
                            cleaned.extend([None] * (len(sql_cols) - len(cleaned)))
                        rows.append(cleaned[:len(sql_cols)])

                if len(sheet_names) == 1:
                    t_name = to_sql_identifier(file_path.stem)
                else:
                    t_name = to_sql_identifier(f"{file_path.stem}_{s_name}")

                self._register_table(db, t_name, f"{file_path.name}:{s_name}", raw_header, sql_cols, rows)
            return
        except Exception:
            pass

        # Fallback to openpyxl if calamine fails
        wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        try:
            for sheet in wb.worksheets:
                rows_gen = sheet.iter_rows(values_only=True)
                preview = []
                for _ in range(6):
                    r = next(rows_gen, None)
                    if r is not None:
                        preview.append(list(r))
                    else:
                        break

                if not preview:
                    continue

                # Header detection: find row with maximum non-empty string headers
                header_idx = 0
                max_str_count = 0
                for idx, r in enumerate(preview):
                    str_count = sum(1 for c in r if c is not None and str(c).strip() != "")
                    if str_count >= 2 and str_count > max_str_count:
                        max_str_count = str_count
                        header_idx = idx

                raw_header = [str(c).strip() if c is not None else f"col_{i+1}" for i, c in enumerate(preview[header_idx])]
                sql_cols = self._deduplicate_identifiers([to_sql_identifier(c) for c in raw_header])

                # Collect remaining rows
                rows = []
                for r in preview[header_idx + 1:]:
                    cleaned = [clean_dirty_value(c) for c in r]
                    if any(v is not None for v in cleaned):
                        if len(cleaned) < len(sql_cols):
                            cleaned.extend([None] * (len(sql_cols) - len(cleaned)))
                        rows.append(cleaned[:len(sql_cols)])

                for r in rows_gen:
                    cleaned = [clean_dirty_value(c) for c in r]
                    if any(v is not None for v in cleaned):
                        if len(cleaned) < len(sql_cols):
                            cleaned.extend([None] * (len(sql_cols) - len(cleaned)))
                        rows.append(cleaned[:len(sql_cols)])

                if len(wb.worksheets) == 1:
                    t_name = to_sql_identifier(file_path.stem)
                else:
                    t_name = to_sql_identifier(f"{file_path.stem}_{sheet.title}")

                self._register_table(db, t_name, f"{file_path.name}:{sheet.title}", raw_header, sql_cols, rows)
        finally:
            wb.close()

    def _load_sqlite(self, file_path: Path, db: BusinessDatabase) -> None:
        """Load tables from SQLite database into DuckDB."""
        conn = sqlite3.connect(file_path)
        try:
            cur = conn.cursor()
            tables = [row[0] for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            for t in tables:
                col_defs = cur.execute(f"PRAGMA table_info(\"{t}\")").fetchall()
                raw_header = [c[1] for c in col_defs]
                sql_cols = self._deduplicate_identifiers([to_sql_identifier(c) for c in raw_header])

                rows_raw = cur.execute(f"SELECT * FROM \"{t}\"").fetchall()
                rows = [[clean_dirty_value(v) for v in r] for r in rows_raw]

                t_name = to_sql_identifier(t)
                self._register_table(db, t_name, f"{file_path.name}:{t}", raw_header, sql_cols, rows)
        finally:
            conn.close()

    def _register_table(
        self,
        db: BusinessDatabase,
        table_name: str,
        source_name: str,
        raw_cols: List[str],
        sql_cols: List[str],
        rows: List[List[Any]],
    ) -> None:
        """Register table in DuckDB and compute column stats."""
        # Ensure unique table name in DuckDB
        base_name = table_name
        counter = 1
        while table_name in db.tables:
            table_name = f"{base_name}_{counter}"
            counter += 1

        # Build column info and determine types
        columns: List[ColumnInfo] = []
        date_cols: List[str] = []

        for i, sql_col in enumerate(sql_cols):
            raw_c = raw_cols[i] if i < len(raw_cols) else f"col_{i+1}"
            col_vals = [r[i] for r in rows if i < len(r) and r[i] is not None]
            sample_vals = col_vals[:3]

            # Inferred type check
            inferred = "VARCHAR"
            if col_vals:
                if all(isinstance(v, int) for v in col_vals):
                    inferred = "BIGINT"
                elif all(isinstance(v, (int, float)) for v in col_vals):
                    inferred = "DOUBLE"
                elif any("ngay" in sql_col or "date" in sql_col or "time" in sql_col for _ in [0]):
                    # Check if looks like date
                    if any(isinstance(v, str) and re.match(r"^\d{4}-\d{2}-\d{2}", v) for v in col_vals[:10]):
                        inferred = "TIMESTAMP"
                        date_cols.append(sql_col)

            columns.append(ColumnInfo(raw_name=raw_c, sql_name=sql_col, inferred_type=inferred, sample_values=sample_vals))

        # Create DuckDB table
        if rows:
            import pyarrow as pa
            data_dict = {col: [r[i] for r in rows] for i, col in enumerate(sql_cols)}
            pa_arrays = []
            for col in sql_cols:
                raw_col_vals = data_dict[col]
                try:
                    pa_arrays.append(pa.array(raw_col_vals))
                except (pa.ArrowInvalid, pa.ArrowTypeError):
                    # Fallback to string array for mixed-type columns
                    pa_arrays.append(pa.array([str(x) if x is not None else None for x in raw_col_vals], type=pa.string()))
            pa_table = pa.Table.from_arrays(pa_arrays, names=sql_cols)
            
            # Create permanent table in DuckDB via cursor
            cur = db.con.cursor()
            cur.register("tmp_reg", pa_table)
            cur.execute(f"CREATE OR REPLACE TABLE \"{table_name}\" AS SELECT * FROM tmp_reg")
            cur.unregister("tmp_reg")
            cur.close()
        else:
            col_specs = ", ".join(f"\"{c}\" VARCHAR" for c in sql_cols)
            db.con.execute(f"CREATE TABLE \"{table_name}\" ({col_specs})")

        # Compute date range if date columns exist
        min_date, max_date = None, None
        for d_col in date_cols:
            try:
                cur = db.con.cursor()
                res = cur.execute(f"SELECT MIN(SUBSTRING(CAST(\"{d_col}\" AS VARCHAR), 1, 10)), MAX(SUBSTRING(CAST(\"{d_col}\" AS VARCHAR), 1, 10)) FROM \"{table_name}\" WHERE \"{d_col}\" IS NOT NULL").fetchone()
                cur.close()
                if res and res[0] and res[1]:
                    if min_date is None or res[0] < min_date:
                        min_date = res[0]
                    if max_date is None or res[1] > max_date:
                        max_date = res[1]
            except Exception:
                pass

        t_info = TableInfo(
            table_name=table_name,
            source_file=source_name,
            row_count=len(rows),
            columns=columns,
            date_columns=date_cols,
            min_date=min_date,
            max_date=max_date,
        )
        db.tables[table_name] = t_info

    def _deduplicate_identifiers(self, identifiers: List[str]) -> List[str]:
        """Ensure all SQL column names are unique."""
        seen: Dict[str, int] = {}
        unique: List[str] = []
        for ident in identifiers:
            if ident not in seen:
                seen[ident] = 1
                unique.append(ident)
            else:
                seen[ident] += 1
                unique.append(f"{ident}_{seen[ident]}")
        return unique

    def _build_entity_index(self, db: BusinessDatabase) -> None:
        """Scan tables for product and customer entities and store in index."""
        for t_name, t_info in db.tables.items():
            col_names = [c.sql_name for c in t_info.columns]

            # Product names & codes
            name_col = next((c for c in col_names if c in ("ten_san_pham", "ten_hang", "ten_hang_hoa", "mat_hang", "noi_dung")), None)
            code_col = next((c for c in col_names if c in ("ma_san_pham", "ma_hang", "ma_hang_hoa", "ma_sp")), None)

            if name_col:
                query = f"SELECT DISTINCT \"{name_col}\", {f'\"{code_col}\"' if code_col else 'NULL'} FROM \"{t_name}\" WHERE \"{name_col}\" IS NOT NULL"
                try:
                    rows = db.con.execute(query).fetchall()
                    for r in rows:
                        p_name = str(r[0]).strip()
                        p_code = str(r[1]).strip() if r[1] else ""
                        if p_name:
                            db.products.append((p_name, p_code, t_name))
                except Exception:
                    pass

            # Customer names & codes
            cust_name_col = next((c for c in col_names if c in ("ten_khach_hang", "khach_hang", "doi_tac")), None)
            cust_code_col = next((c for c in col_names if c in ("ma_khach_hang", "ma_kh", "so_kh")), None)

            if cust_name_col:
                query = f"SELECT DISTINCT \"{cust_name_col}\", {f'\"{cust_code_col}\"' if cust_code_col else 'NULL'} FROM \"{t_name}\" WHERE \"{cust_name_col}\" IS NOT NULL"
                try:
                    rows = db.con.execute(query).fetchall()
                    for r in rows:
                        c_name = str(r[0]).strip()
                        c_code = str(r[1]).strip() if r[1] else ""
                        if c_name:
                            db.customers.append((c_name, c_code, t_name))
                except Exception:
                    pass
