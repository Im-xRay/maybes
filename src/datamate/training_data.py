"""Prepare leakage-resistant, manually annotated QueryPlan training data."""

from __future__ import annotations

import json
import random
import sqlite3
import tempfile
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import TextIO

from .business_context import BusinessContextDocuments, load_business_context
from .pipeline import discover_business_schema
from .question_understanding import (
    QuestionContext,
    QuestionUnderstandingError,
    build_planning_prompt,
    validate_abstention_plan,
)
from .query_plan import PlanValidationError, QueryPlan, validate_query_plan
from .plan_serialization import query_plan_from_dict


class TrainingDataError(ValueError):
    """Raised when source records, annotations, or split configuration is invalid."""


@dataclass(frozen=True)
class TrainingDataReport:
    question_count: int
    annotation_count: int
    unannotated_question_count: int
    train_count: int
    validation_count: int
    train_business_ids: tuple[str, ...]
    validation_business_ids: tuple[str, ...]
    seed: int


def _read_jsonl(stream: TextIO, source: Path):
    for line_number, line in enumerate(stream, start=1):
        if not line.strip():
            raise TrainingDataError(f"Blank JSONL line at {source}:{line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrainingDataError(
                f"Invalid JSON at {source}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise TrainingDataError(
                f"Expected a JSON object at {source}:{line_number}"
            )
        yield line_number, value


def _required_string(record: dict[str, object], key: str, context: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TrainingDataError(f"{context}: {key} must be a non-empty string")
    return value


def _safe_business_path(root: Path, business_id: str) -> Path:
    if Path(business_id).name != business_id or business_id in {".", ".."}:
        raise TrainingDataError(f"Unsafe business_id: {business_id!r}")
    path = (root / business_id).resolve()
    if not path.is_relative_to(root) or not path.is_dir():
        raise TrainingDataError(f"Business data directory not found: {business_id}")
    return path


def _plan_payload(plan: QueryPlan) -> dict[str, object]:
    return asdict(plan)


def _validate_output_location(
    output_dir: Path,
    *,
    business_root: Path,
    question_file: Path,
    annotation_file: Path,
) -> None:
    if output_dir.is_relative_to(business_root):
        raise TrainingDataError("Output directory must be outside the business data root")
    for source in (question_file, annotation_file):
        if output_dir == source.parent or output_dir.is_relative_to(source.parent):
            raise TrainingDataError(
                f"Output directory must not be inside an input directory: {source.parent}"
            )
    if output_dir.exists():
        raise TrainingDataError(
            f"Output directory already exists; choose a new path: {output_dir}"
        )


def prepare_plan_training_data(
    questions_path: Path | str,
    annotations_path: Path | str,
    business_data_root: Path | str,
    output_dir: Path | str,
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> TrainingDataReport:
    """Join human-authored plan labels to training questions and split by business.

    An annotation JSONL record has ``question_id``, ``business_id``, and
    ``query_plan`` fields. No plan labels are inferred from answer values.
    """
    if not 0 < validation_fraction < 1:
        raise TrainingDataError("validation_fraction must be between 0 and 1")
    questions_file = Path(questions_path).resolve()
    annotations_file = Path(annotations_path).resolve()
    business_root = Path(business_data_root).resolve()
    destination = Path(output_dir).resolve()
    if not questions_file.is_file() or not annotations_file.is_file():
        raise TrainingDataError("Question and annotation files must exist")
    if not business_root.is_dir():
        raise TrainingDataError("Business data root must be a directory")
    _validate_output_location(
        destination,
        business_root=business_root,
        question_file=questions_file,
        annotation_file=annotations_file,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.staging-",
        dir=destination.parent,
    ) as temporary:
        stage = Path(temporary)
        database_path = stage / "records.sqlite"
        connection = sqlite3.connect(database_path)
        try:
            connection.execute(
                "CREATE TABLE questions ("
                "question_id TEXT PRIMARY KEY, business_id TEXT NOT NULL, "
                "question TEXT NOT NULL, as_of_date TEXT)"
            )
            connection.execute(
                "CREATE TABLE examples ("
                "question_id TEXT PRIMARY KEY, business_id TEXT NOT NULL, "
                "example_json TEXT NOT NULL)"
            )
            question_count = _load_questions(connection, questions_file)
            annotation_count = _load_annotations(
                connection,
                annotations_file,
                business_root,
            )
            if annotation_count == 0:
                raise TrainingDataError(
                    "No manually annotated QueryPlans were found; no labels were fabricated"
                )
            business_ids = [
                row[0]
                for row in connection.execute(
                    "SELECT DISTINCT business_id FROM examples ORDER BY business_id"
                )
            ]
            if len(business_ids) < 2:
                raise TrainingDataError(
                    "At least two annotated businesses are required for a group holdout"
                )
            train_business_ids, validation_business_ids = _split_businesses(
                business_ids,
                validation_fraction,
                seed,
            )
            train_count = _write_split(
                connection,
                stage / "train.jsonl",
                train_business_ids,
            )
            validation_count = _write_split(
                connection,
                stage / "validation.jsonl",
                validation_business_ids,
            )
            report = TrainingDataReport(
                question_count=question_count,
                annotation_count=annotation_count,
                unannotated_question_count=question_count - annotation_count,
                train_count=train_count,
                validation_count=validation_count,
                train_business_ids=tuple(train_business_ids),
                validation_business_ids=tuple(validation_business_ids),
                seed=seed,
            )
            (stage / "preparation_report.json").write_text(
                json.dumps(
                    asdict(report),
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            connection.commit()
        finally:
            connection.close()

        stage_files = [
            path
            for path in stage.iterdir()
            if path.name != "records.sqlite"
        ]
        published = stage / "published"
        published.mkdir()
        for path in stage_files:
            path.replace(published / path.name)
        published.replace(destination)

    return report


def _load_questions(connection: sqlite3.Connection, source: Path) -> int:
    count = 0
    forbidden_label_fields = {
        "answer",
        "answers",
        "expected_answer",
        "gold",
        "label",
        "query_plan",
        "target",
    }
    with source.open("r", encoding="utf-8-sig") as stream:
        for line_number, record in _read_jsonl(stream, source):
            context = f"{source}:{line_number}"
            leaked_fields = forbidden_label_fields & record.keys()
            if leaked_fields:
                raise TrainingDataError(
                    f"{context}: question records must not include label fields: "
                    + ", ".join(sorted(leaked_fields))
                )
            question_id = _required_string(record, "id", context)
            business_id = _required_string(record, "business_id", context)
            question = _required_string(record, "question", context)
            as_of_date = record.get("as_of_date")
            if as_of_date is not None:
                if not isinstance(as_of_date, str):
                    raise TrainingDataError(f"{context}: as_of_date must be a string")
                try:
                    date.fromisoformat(as_of_date)
                except ValueError as exc:
                    raise TrainingDataError(
                        f"{context}: as_of_date must be an ISO-8601 date"
                    ) from exc
            try:
                connection.execute(
                    "INSERT INTO questions VALUES (?, ?, ?, ?)",
                    (question_id, business_id, question, as_of_date),
                )
            except sqlite3.IntegrityError as exc:
                raise TrainingDataError(
                    f"{context}: duplicate question id {question_id!r}"
                ) from exc
            count += 1
    connection.commit()
    return count


def _load_annotations(
    connection: sqlite3.Connection,
    source: Path,
    business_root: Path,
) -> int:
    count = 0
    schema_cache: dict[str, dict[str, tuple[str, ...]]] = {}
    context_cache: dict[str, BusinessContextDocuments] = {}
    annotation_fields = {"question_id", "business_id", "query_plan"}
    with source.open("r", encoding="utf-8-sig") as stream:
        for line_number, record in _read_jsonl(stream, source):
            context = f"{source}:{line_number}"
            unexpected_fields = record.keys() - annotation_fields
            if unexpected_fields:
                raise TrainingDataError(
                    f"{context}: unsupported annotation field(s): "
                    + ", ".join(sorted(unexpected_fields))
                )
            question_id = _required_string(record, "question_id", context)
            business_id = _required_string(record, "business_id", context)
            question_record = connection.execute(
                "SELECT business_id, question, as_of_date FROM questions "
                "WHERE question_id = ?",
                (question_id,),
            ).fetchone()
            if question_record is None:
                raise TrainingDataError(
                    f"{context}: unknown question_id {question_id!r}"
                )
            if question_record[0] != business_id:
                raise TrainingDataError(
                    f"{context}: business_id does not match question {question_id!r}"
                )
            if "query_plan" not in record:
                raise TrainingDataError(f"{context}: query_plan is required")
            plan = query_plan_from_dict(record["query_plan"])
            if business_id not in schema_cache:
                business_path = _safe_business_path(business_root, business_id)
                schema_cache[business_id] = discover_business_schema(
                    business_path
                ).columns
                context_cache[business_id] = load_business_context(
                    business_path,
                    business_root,
                )
            schema = schema_cache[business_id]
            documents = context_cache[business_id]
            if plan.output_type in {"unanswerable", "abstain"}:
                try:
                    validate_abstention_plan(plan, schema)
                except QuestionUnderstandingError as exc:
                    raise TrainingDataError(
                        f"{context}: invalid abstention annotation: {exc}"
                    ) from exc
            elif plan.output_type is None:
                raise TrainingDataError(
                    f"{context}: annotations must declare an output type"
                )
            else:
                try:
                    validate_query_plan(
                        plan,
                        {table: set(columns) for table, columns in schema.items()},
                    )
                except PlanValidationError as exc:
                    raise TrainingDataError(
                        f"{context}: invalid QueryPlan: {exc}"
                    ) from exc
            prompt = build_planning_prompt(
                question_record[1],
                QuestionContext(
                    schema=schema,
                    as_of_date=question_record[2],
                    rulebook=documents.rulebook,
                    business_readme=documents.business_readme,
                    calendar_columns=documents.calendar_columns,
                    calendar_rows=documents.calendar_rows,
                ),
            )
            example = {
                "question_id": question_id,
                "business_id": business_id,
                "prompt": prompt,
                "completion": json.dumps(
                    _plan_payload(plan),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            }
            try:
                connection.execute(
                    "INSERT INTO examples VALUES (?, ?, ?)",
                    (
                        question_id,
                        business_id,
                        json.dumps(
                            example,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            allow_nan=False,
                        ),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise TrainingDataError(
                    f"{context}: duplicate annotation for {question_id!r}"
                ) from exc
            count += 1
    connection.commit()
    return count


def _split_businesses(
    business_ids: list[str],
    validation_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    shuffled = list(business_ids)
    random.Random(seed).shuffle(shuffled)
    validation_count = min(
        len(shuffled) - 1,
        max(1, round(len(shuffled) * validation_fraction)),
    )
    validation = sorted(shuffled[:validation_count])
    train = sorted(shuffled[validation_count:])
    return train, validation


def _write_split(
    connection: sqlite3.Connection,
    destination: Path,
    business_ids: list[str],
) -> int:
    placeholders = ",".join("?" for _ in business_ids)
    query = (
        "SELECT example_json FROM examples "
        f"WHERE business_id IN ({placeholders}) ORDER BY question_id"
    )
    count = 0
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for (example_json,) in connection.execute(query, business_ids):
            stream.write(example_json)
            stream.write("\n")
            count += 1
    return count
