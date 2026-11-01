"""Strict local evaluation; this is not the competition's Penalized Accuracy."""

from __future__ import annotations

import json
import math
import sqlite3
import tempfile
import unicodedata
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TextIO


class EvaluationError(ValueError):
    """Raised when an evaluation input or output cannot be validated."""


def answers_match(
    expected_type: str,
    expected_value: object,
    predicted_type: str,
    predicted_value: object,
) -> bool:
    """Compare one answer pair using the local evaluator's value semantics."""
    if expected_type != predicted_type:
        return False
    try:
        expected = _canonical_value(expected_type, expected_value)
        predicted = _canonical_value(predicted_type, predicted_value)
    except EvaluationError:
        return False
    return _same_answer_value(
        expected_type,
        expected,
        predicted,
        allow_numeric_strings=False,
        gold_aliases="[]",
    )


def validate_answer_value(answer_type: str, value: object) -> None:
    """Validate one answer value using the same schema as local evaluation."""
    if answer_type == "table" and not _is_submission_table(value):
        raise EvaluationError("table answer has an invalid submission shape")
    _canonical_value(answer_type, value)


def _validate_nested_value(value: object) -> bool:
    if value is None or isinstance(value, (str, bool, int, date, datetime)):
        return True
    if isinstance(value, Decimal):
        return value.is_finite()
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_validate_nested_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _validate_nested_value(item)
            for key, item in value.items()
        )
    return False


def _json_default(value: object) -> object:
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("Non-finite decimal")
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def _jsonl_records(stream: TextIO, source: Path):
    for line_number, line in enumerate(stream, start=1):
        if not line.strip():
            raise EvaluationError(f"Blank JSONL line at {source}:{line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationError(
                f"Invalid JSON at {source}:{line_number}: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise EvaluationError(
                f"Expected a JSON object at {source}:{line_number}"
            )
        yield line_number, record


def _canonical_value(
    answer_type: object,
    value: object,
    *,
    allow_numeric_string: bool = False,
) -> str:
    if not isinstance(answer_type, str):
        raise EvaluationError("answer type must be a string")
    if answer_type in {"number", "percent"}:
        if isinstance(value, bool) or value is None or (
            isinstance(value, str) and not allow_numeric_string
        ):
            raise EvaluationError(f"{answer_type} answer must be numeric")
        try:
            number = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise EvaluationError(f"{answer_type} answer is not numeric") from exc
        if not number.is_finite():
            raise EvaluationError(f"{answer_type} answer must be finite")
        return str(number.normalize())
    if answer_type == "date":
        if not isinstance(value, str):
            raise EvaluationError("date answer must be an ISO-8601 string")
        try:
            if len(value) == 7:
                if (
                    value[4] != "-"
                    or not value[:4].isdigit()
                    or not value[5:].isdigit()
                ):
                    raise ValueError
                year, month = (int(part) for part in value.split("-"))
                date(year, month, 1)
                return value
            if (
                len(value) != 10
                or value[4] != "-"
                or value[7] != "-"
            ):
                raise ValueError
            return date.fromisoformat(value).isoformat()
        except ValueError as exc:
            raise EvaluationError("date answer must be an ISO-8601 string") from exc
    if answer_type == "string":
        if not isinstance(value, str):
            raise EvaluationError("string answer must be text")
        return json.dumps(value, ensure_ascii=False)
    if answer_type == "boolean":
        if not isinstance(value, bool):
            raise EvaluationError("boolean answer must be true or false")
    elif answer_type == "unanswerable":
        if value is not None:
            raise EvaluationError("unanswerable gold answer must be null")
    elif answer_type == "table":
        if not isinstance(value, (dict, list)):
            raise EvaluationError("table answer must be an object or array")
    elif answer_type == "list":
        if not isinstance(value, list):
            raise EvaluationError("list answer must be an array")
    elif answer_type not in {"boolean", "unanswerable", "table", "list"}:
        raise EvaluationError(f"unsupported answer type: {answer_type}")
    try:
        if not _validate_nested_value(value):
            raise EvaluationError(f"{answer_type} answer contains invalid JSON values")
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=_json_default,
        )
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"{answer_type} answer is not valid JSON data") from exc


def _required_id(record: dict[str, object], key: str, context: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise EvaluationError(f"{context}: {key} must be a non-empty string")
    return value


def _normalize_string(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value).casefold()
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )


def _is_submission_table(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {"columns", "rows"}:
        return False
    columns = value["columns"]
    rows = value["rows"]
    return (
        isinstance(columns, list)
        and all(isinstance(column, str) and column for column in columns)
        and len(set(columns)) == len(columns)
        and isinstance(rows, list)
        and all(
            isinstance(row, list)
            and len(row) == len(columns)
            and _validate_nested_value(row)
            for row in rows
        )
    )


def _load_questions(
    connection: sqlite3.Connection,
    source: Path,
) -> None:
    with source.open("r", encoding="utf-8-sig") as stream:
        for line_number, record in _jsonl_records(stream, source):
            context = f"{source}:{line_number}"
            question_id = _required_id(record, "id", context)
            business_id = _required_id(record, "business_id", context)
            try:
                connection.execute(
                    "INSERT INTO question_business VALUES (?, ?)",
                    (question_id, business_id),
                )
            except sqlite3.IntegrityError as exc:
                raise EvaluationError(
                    f"{context}: duplicate question id {question_id!r}"
                ) from exc


def _load_gold(connection: sqlite3.Connection, source: Path) -> tuple[int, int, int]:
    count = 0
    string_alias_count = 0
    string_alias_questions = 0
    with source.open("r", encoding="utf-8-sig") as stream:
        for line_number, record in _jsonl_records(stream, source):
            context = f"{source}:{line_number}"
            question_id = _required_id(record, "id", context)
            answer_type = record.get("type")
            if answer_type == "table" and not _is_submission_table(
                record.get("answer")
            ):
                raise EvaluationError(f"{context}: table answer has an invalid shape")
            answer_value = _canonical_value(answer_type, record.get("answer"))
            aliases = record.get("alias", [])
            if not isinstance(aliases, list):
                raise EvaluationError(f"{context}: alias must be an array")
            if answer_type == "string":
                if any(not isinstance(alias, str) for alias in aliases):
                    raise EvaluationError(
                        f"{context}: string answer aliases must be strings"
                    )
                string_alias_count += len(aliases)
                string_alias_questions += bool(aliases)
            elif answer_type in {"list", "table"}:
                values = (
                    record.get("answer")
                    if answer_type == "list"
                    else record["answer"]["rows"]
                )
                if aliases and len(aliases) != len(values):
                    raise EvaluationError(
                        f"{context}: aliases must align with answer items"
                    )
                if any(
                    not isinstance(item_aliases, list)
                    or any(not isinstance(alias, str) for alias in item_aliases)
                    for item_aliases in aliases
                ):
                    raise EvaluationError(
                        f"{context}: item aliases must be arrays of strings"
                    )
            else:
                if aliases:
                    raise EvaluationError(
                        f"{context}: aliases are unsupported for {answer_type}"
                    )
            answer_aliases = json.dumps(aliases, ensure_ascii=False)
            try:
                connection.execute(
                    "INSERT INTO gold VALUES (?, ?, ?, ?)",
                    (question_id, answer_type, answer_value, answer_aliases),
                )
            except sqlite3.IntegrityError as exc:
                raise EvaluationError(
                    f"{context}: duplicate gold question id {question_id!r}"
                ) from exc
            count += 1
    if count == 0:
        raise EvaluationError("Gold answer file is empty")
    return count, string_alias_count, string_alias_questions


def _plan_table_columns(plan: object) -> list[str]:
    if not isinstance(plan, dict):
        return []
    aggregations = plan.get("aggregations", [])
    calculations = plan.get("calculations", [])
    time_buckets = plan.get("time_buckets", [])
    group_by = plan.get("group_by", [])
    selected = plan.get("selected_columns", [])
    if not all(
        isinstance(items, list)
        for items in (aggregations, calculations, time_buckets, group_by, selected)
    ):
        return []
    references = selected or group_by
    columns = [
        f"{item['table']}.{item['column']}"
        for item in references
        if isinstance(item, dict)
        and isinstance(item.get("table"), str)
        and isinstance(item.get("column"), str)
    ]
    columns.extend(
        item["alias"]
        for item in [*time_buckets, *aggregations, *calculations]
        if isinstance(item, dict) and isinstance(item.get("alias"), str)
    )
    return columns


def _normalize_logged_table(value: object, plan: object) -> object:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        return value
    if value:
        columns = list(value[0])
        if any(set(row) != set(columns) for row in value):
            return value
    else:
        columns = _plan_table_columns(plan)
    return {
        "columns": columns,
        "rows": [[row[column] for column in columns] for row in value],
    }


def _table_equal(
    gold: object,
    predicted: object,
    *,
    allow_numeric_strings: bool,
    aliases: object,
) -> bool:
    if not _is_submission_table(gold) or not _is_submission_table(predicted):
        return False
    gold_columns = gold["columns"]
    predicted_columns = predicted["columns"]
    if len(gold_columns) != len(predicted_columns):
        return False
    if set(gold_columns) == set(predicted_columns):
        predicted_indexes = {
            column: index for index, column in enumerate(predicted_columns)
        }
        column_indexes = [predicted_indexes[column] for column in gold_columns]
    else:
        column_indexes = list(range(len(gold_columns)))

    predicted_rows = list(predicted["rows"])
    if len(gold["rows"]) != len(predicted_rows):
        return False
    table_aliases = aliases if isinstance(aliases, list) else []
    if table_aliases and len(table_aliases) != len(gold["rows"]):
        return False
    candidates_by_gold: list[list[int]] = []
    for row_index, gold_row in enumerate(gold["rows"]):
        row_aliases = (
            table_aliases[row_index]
            if table_aliases and isinstance(table_aliases[row_index], list)
            else []
        )
        candidates_by_gold.append(
            [
                predicted_index
                for predicted_index, predicted_row in enumerate(predicted_rows)
                if all(
                    (
                        _string_matches(
                            predicted_row[column_indexes[position]],
                            gold_value,
                            row_aliases if position == 0 else [],
                        )
                        if isinstance(gold_value, str)
                        else _structured_equal(
                            gold_value,
                            predicted_row[column_indexes[position]],
                            allow_numeric_strings=allow_numeric_strings,
                        )
                    )
                    for position, gold_value in enumerate(gold_row)
                )
            ]
        )
    matched_gold_by_prediction = [-1] * len(predicted_rows)
    matched_prediction_by_gold = [-1] * len(gold["rows"])
    for start_gold in range(len(gold["rows"])):
        queue = [start_gold]
        visited_gold = {start_gold}
        parent_gold_by_prediction: dict[int, int] = {}
        free_prediction: int | None = None
        cursor = 0
        while cursor < len(queue) and free_prediction is None:
            gold_index = queue[cursor]
            cursor += 1
            for prediction_index in candidates_by_gold[gold_index]:
                if prediction_index in parent_gold_by_prediction:
                    continue
                parent_gold_by_prediction[prediction_index] = gold_index
                matched_gold = matched_gold_by_prediction[prediction_index]
                if matched_gold == -1:
                    free_prediction = prediction_index
                    break
                if matched_gold not in visited_gold:
                    visited_gold.add(matched_gold)
                    queue.append(matched_gold)
        if free_prediction is None:
            return False
        prediction_index = free_prediction
        while True:
            gold_index = parent_gold_by_prediction[prediction_index]
            previous_prediction = matched_prediction_by_gold[gold_index]
            matched_gold_by_prediction[prediction_index] = gold_index
            matched_prediction_by_gold[gold_index] = prediction_index
            if previous_prediction == -1:
                break
            prediction_index = previous_prediction
    return True


def _load_predictions(connection: sqlite3.Connection, source: Path) -> int:
    count = 0
    with source.open("r", encoding="utf-8-sig") as stream:
        for line_number, record in _jsonl_records(stream, source):
            context = f"{source}:{line_number}"
            question_id = _required_id(
                record,
                "question_id" if "question_id" in record else "id",
                context,
            )
            nested = record.get("answer")
            internal_log = isinstance(nested, dict) and (
                "answer_type" in nested or "status" in nested
            )
            answer_record = nested if internal_log else {}
            answer_type = (
                record.get("type")
                or answer_record.get("answer_type")
                or record.get("answer_type")
            )
            if internal_log:
                value = nested.get("value")
                if answer_type == "table":
                    value = _normalize_logged_table(
                        value,
                        record.get("query_plan"),
                    )
            else:
                value = nested
            valid = 1
            if answer_type == "abstain":
                canonical = "null"
                if not internal_log and "answer" in record:
                    valid = 0
            elif answer_type == "unanswerable" and not internal_log:
                if (
                    isinstance(value, dict)
                    and value.get("type") == "unanswerable"
                    and set(value) in ({"type"}, {"type", "reason"})
                    and (
                        "reason" not in value
                        or isinstance(value["reason"], str)
                    )
                ):
                    canonical = "null"
                else:
                    canonical = "null"
                    valid = 0
            elif (
                answer_type == "table"
                and not internal_log
                and not _is_submission_table(value)
            ):
                canonical = "null"
                valid = 0
            else:
                try:
                    canonical = _canonical_value(
                        answer_type,
                        value,
                        allow_numeric_string=internal_log,
                    )
                except EvaluationError:
                    canonical = "null"
                    valid = 0
            status = answer_record.get("status", record.get("status"))
            business_id = record.get("business_id")
            if not isinstance(business_id, str):
                business_id = None
            elapsed_ms = record.get("elapsed_ms")
            if (
                isinstance(elapsed_ms, bool)
                or not isinstance(elapsed_ms, (int, float))
                or not math.isfinite(elapsed_ms)
                or elapsed_ms < 0
            ):
                elapsed_ms = None
            try:
                connection.execute(
                    "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        question_id,
                        answer_type if isinstance(answer_type, str) else None,
                        canonical,
                        status if isinstance(status, str) else None,
                        business_id,
                        elapsed_ms,
                        valid,
                        int(internal_log),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise EvaluationError(
                    f"{context}: duplicate prediction question id {question_id!r}"
                ) from exc
            count += 1
    return count


def evaluate_predictions(
    answers_path: Path | str,
    predictions_path: Path | str,
    report_path: Path | str,
    *,
    questions_path: Path | str | None = None,
) -> dict[str, object]:
    """Compute transparent exact-match diagnostics without applying official PA."""
    answers = Path(answers_path).resolve()
    predictions = Path(predictions_path).resolve()
    destination = Path(report_path).resolve()
    questions = Path(questions_path).resolve() if questions_path else None
    if not answers.is_file() or not predictions.is_file():
        raise EvaluationError("Gold answers and predictions must exist")
    if questions is not None and not questions.is_file():
        raise EvaluationError("Questions file does not exist")
    sources = (answers, predictions) + ((questions,) if questions else ())
    if destination.exists():
        raise EvaluationError(f"Report already exists: {destination}")
    if any(
        destination == source.parent or destination.is_relative_to(source.parent)
        for source in sources
    ):
        raise EvaluationError("Report must be outside input directories")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="datamate-eval-") as temporary:
        connection = sqlite3.connect(Path(temporary) / "evaluation.sqlite")
        try:
            connection.execute(
                "CREATE TABLE gold (question_id TEXT PRIMARY KEY, "
                "answer_type TEXT NOT NULL, answer_value TEXT NOT NULL, "
                "answer_aliases TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE predictions (question_id TEXT PRIMARY KEY, "
                "answer_type TEXT, answer_value TEXT NOT NULL, status TEXT, "
                "business_id TEXT, elapsed_ms REAL, valid INTEGER NOT NULL, "
                "allow_numeric_strings INTEGER NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE question_business ("
                "question_id TEXT PRIMARY KEY, business_id TEXT NOT NULL)"
            )
            if questions is not None:
                _load_questions(connection, questions)
            (
                gold_count,
                string_alias_count,
                string_alias_questions,
            ) = _load_gold(connection, answers)
            prediction_count = _load_predictions(connection, predictions)
            connection.commit()
            report = _aggregate_report(
                connection,
                gold_count=gold_count,
                string_alias_count=string_alias_count,
                string_alias_questions=string_alias_questions,
                prediction_count=prediction_count,
            )
        finally:
            connection.close()

    destination.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def _aggregate_report(
    connection: sqlite3.Connection,
    *,
    gold_count: int,
    string_alias_count: int,
    string_alias_questions: int,
    prediction_count: int,
) -> dict[str, object]:
    exact_matches = 0
    type_matches = 0
    missing_predictions = 0
    invalid_predictions = 0
    business_mismatches = 0
    unanswerable_total = 0
    unanswerable_correct = 0
    abstentions = 0
    incorrect_predictions = 0
    latency_total = 0.0
    latency_count = 0
    max_latency = 0.0
    by_type: dict[str, dict[str, int]] = {}
    by_business: dict[str, dict[str, int]] = {}
    query = """
        SELECT g.question_id, p.question_id, g.answer_type, g.answer_value,
               g.answer_aliases,
               p.answer_type, p.answer_value, p.business_id, p.elapsed_ms,
               p.valid, p.allow_numeric_strings, q.business_id
        FROM gold AS g
        LEFT JOIN predictions AS p ON p.question_id = g.question_id
        LEFT JOIN question_business AS q ON q.question_id = g.question_id
        ORDER BY g.question_id
    """
    for (
        question_id,
        predicted_question_id,
        gold_type,
        gold_value,
        gold_aliases,
        predicted_type,
        predicted_value,
        predicted_business,
        elapsed_ms,
        valid,
        allow_numeric_strings,
        gold_business,
    ) in connection.execute(query):
        type_bucket = by_type.setdefault(
            gold_type,
            {"count": 0, "type_correct": 0, "exact_match": 0},
        )
        business_id = gold_business or predicted_business or "unknown"
        business_bucket = by_business.setdefault(
            business_id,
            {"count": 0, "exact_match": 0},
        )
        type_bucket["count"] += 1
        business_bucket["count"] += 1
        if predicted_type is None:
            missing_predictions += 1
        if valid == 0:
            invalid_predictions += 1
        type_matches_this = valid == 1 and predicted_type == gold_type
        exact_this = type_matches_this and _same_answer_value(
            gold_type,
            gold_value,
            predicted_value,
            allow_numeric_strings=bool(allow_numeric_strings),
            gold_aliases=gold_aliases,
        )
        if type_matches_this:
            type_matches += 1
            type_bucket["type_correct"] += 1
        if exact_this:
            exact_matches += 1
            type_bucket["exact_match"] += 1
            business_bucket["exact_match"] += 1
        is_abstention = (
            predicted_question_id is None
            or (predicted_type == "abstain" and valid == 1)
        )
        if is_abstention:
            abstentions += 1
        elif not exact_this:
            incorrect_predictions += 1
        if gold_type == "unanswerable":
            unanswerable_total += 1
            if exact_this:
                unanswerable_correct += 1
        if (
            gold_business is not None
            and predicted_business is not None
            and gold_business != predicted_business
        ):
            business_mismatches += 1
        if elapsed_ms is not None:
            latency_total += elapsed_ms
            latency_count += 1
            max_latency = max(max_latency, elapsed_ms)

    extra_predictions = connection.execute(
        "SELECT COUNT(*) FROM predictions AS p "
        "LEFT JOIN gold AS g ON g.question_id = p.question_id "
        "WHERE g.question_id IS NULL"
    ).fetchone()[0]
    return {
        "metric": "local_strict_exact_match_not_official_PA",
        "official_pa_computed": False,
        "aliases_compared_for_strings": True,
        "aliases_compared_for_lists": True,
        "aliases_compared_for_table_rows": True,
        "string_alias_count": string_alias_count,
        "string_alias_questions": string_alias_questions,
        "gold_count": gold_count,
        "prediction_count": prediction_count,
        "exact_matches": exact_matches,
        "exact_match_fraction": exact_matches / gold_count if gold_count else 0.0,
        "incorrect_predictions": incorrect_predictions,
        "abstentions": abstentions,
        "strict_local_reward_sum": exact_matches - incorrect_predictions,
        "strict_local_reward_scope": (
            "Applies +1/-1/0 to local outcomes with aliases for strings, list "
            "items and table row names. Case/whitespace/punctuation are normalized; "
            "some ranking/order rules remain incomplete. This is not the official "
            "reward or aggregate PA."
        ),
        "answer_type_matches": type_matches,
        "answer_type_accuracy": type_matches / gold_count if gold_count else 0.0,
        "missing_predictions": missing_predictions,
        "extra_predictions": extra_predictions,
        "invalid_predictions": invalid_predictions,
        "business_id_mismatches": business_mismatches,
        "unanswerable_count": unanswerable_total,
        "unanswerable_exact_matches": unanswerable_correct,
        "latency_ms": {
            "count": latency_count,
            "average": latency_total / latency_count if latency_count else None,
            "maximum": max_latency if latency_count else None,
        },
        "by_answer_type": by_type,
        "by_business": by_business,
    }


def _string_matches(
    predicted: object,
    gold: object,
    aliases: object,
) -> bool:
    if not isinstance(predicted, str) or not isinstance(gold, str):
        return False
    acceptable = [gold]
    if isinstance(aliases, list):
        acceptable.extend(alias for alias in aliases if isinstance(alias, str))
    normalized_prediction = _normalize_string(predicted)
    return normalized_prediction in {
        _normalize_string(value) for value in acceptable
    }


def _numeric_equal(gold: object, predicted: object) -> bool:
    if isinstance(gold, bool) or isinstance(predicted, bool):
        return False
    try:
        expected = Decimal(str(gold))
        actual = Decimal(str(predicted))
    except (InvalidOperation, ValueError):
        return False
    if not expected.is_finite() or not actual.is_finite():
        return False
    if expected == actual:
        return True
    if (
        expected == expected.to_integral_value()
        and actual == actual.to_integral_value()
    ):
        return False
    try:
        expected_float = float(expected)
        actual_float = float(actual)
    except (OverflowError, ValueError):
        return False
    if (
        not math.isfinite(expected_float)
        or not math.isfinite(actual_float)
    ):
        return False
    if expected_float == actual_float:
        return True
    tolerance = max(math.ulp(expected_float), math.ulp(actual_float))
    return abs(expected_float - actual_float) <= tolerance


def _list_equal(
    gold: object,
    predicted: object,
    *,
    allow_numeric_strings: bool,
    aliases: object,
) -> bool:
    if not isinstance(gold, list) or not isinstance(predicted, list):
        return False
    if len(gold) != len(predicted):
        return False
    item_aliases = aliases if isinstance(aliases, list) else []
    if item_aliases and len(item_aliases) != len(gold):
        return False
    for index, (gold_item, predicted_item) in enumerate(zip(gold, predicted)):
        aliases_for_item = (
            item_aliases[index]
            if item_aliases and isinstance(item_aliases[index], list)
            else []
        )
        if isinstance(gold_item, str):
            if not _string_matches(predicted_item, gold_item, aliases_for_item):
                return False
        elif not _structured_equal(
            gold_item,
            predicted_item,
            allow_numeric_strings=allow_numeric_strings,
        ):
            return False
    return True


def _same_answer_value(
    answer_type: str,
    gold_value: str,
    predicted_value: str,
    *,
    allow_numeric_strings: bool,
    gold_aliases: str,
) -> bool:
    if answer_type in {"number", "percent"}:
        return _numeric_equal(gold_value, predicted_value)
    if answer_type == "string":
        try:
            gold = json.loads(gold_value)
            predicted = json.loads(predicted_value)
            aliases = json.loads(gold_aliases)
        except json.JSONDecodeError:
            return False
        if not isinstance(gold, str) or not isinstance(predicted, str):
            return False
        return _string_matches(predicted, gold, aliases)
    if answer_type == "table":
        try:
            gold = json.loads(gold_value)
            predicted = json.loads(predicted_value)
            aliases = json.loads(gold_aliases)
        except json.JSONDecodeError:
            return False
        return _table_equal(
            gold,
            predicted,
            allow_numeric_strings=allow_numeric_strings,
            aliases=aliases,
        )
    if answer_type != "list":
        return gold_value == predicted_value
    try:
        gold = json.loads(gold_value)
        predicted = json.loads(predicted_value)
        aliases = json.loads(gold_aliases)
    except json.JSONDecodeError:
        return False
    return _list_equal(
        gold,
        predicted,
        allow_numeric_strings=allow_numeric_strings,
        aliases=aliases,
    )


def _structured_equal(
    gold: object,
    predicted: object,
    *,
    allow_numeric_strings: bool,
) -> bool:
    if isinstance(gold, bool):
        return isinstance(predicted, bool) and gold == predicted
    if isinstance(gold, (int, float)):
        if isinstance(predicted, bool) or (
            isinstance(predicted, str) and not allow_numeric_strings
        ):
            return False
        return _numeric_equal(gold, predicted)
    if gold is None:
        return predicted is None
    if isinstance(gold, str):
        return isinstance(predicted, str) and gold == predicted
    if isinstance(gold, list):
        return (
            isinstance(predicted, list)
            and len(gold) == len(predicted)
            and all(
                _structured_equal(
                    left,
                    right,
                    allow_numeric_strings=allow_numeric_strings,
                )
                for left, right in zip(gold, predicted)
            )
        )
    if isinstance(gold, dict):
        return (
            isinstance(predicted, dict)
            and gold.keys() == predicted.keys()
            and all(
                _structured_equal(
                    gold[key],
                    predicted[key],
                    allow_numeric_strings=allow_numeric_strings,
                )
                for key in gold
            )
        )
    return False
