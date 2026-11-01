from decimal import Decimal

import pytest

from datamate.answer_validation import (
    AnswerValidationError,
    Evidence,
    ValidatedAnswer,
    validate_answer,
)


EVIDENCE = (Evidence("orders.csv::orders", ("amount",), "sum", 3),)


@pytest.mark.parametrize(
    ("answer_type", "value"),
    [
        ("number", 0),
        ("number", Decimal("12.50")),
        ("percent", -12.5),
        ("string", "Không có dữ liệu"),
        ("date", "2026-01-31"),
        ("date", "2026-01"),
        ("boolean", False),
        ("list", [1, "x"]),
        ("list", [Decimal("1.25")]),
        ("table", [{"month": "01", "revenue": Decimal("12.50")}]),
    ],
)
def test_answerable_types_accept_valid_values(answer_type: str, value: object) -> None:
    answer = ValidatedAnswer(answer_type, value, "answered", EVIDENCE)  # type: ignore[arg-type]

    validate_answer(answer)


@pytest.mark.parametrize(
    ("answer_type", "value"),
    [
        ("number", True),
        ("number", float("nan")),
        ("percent", float("inf")),
        ("string", 3),
        ("date", "31/01/2026"),
        ("boolean", 1),
        ("list", {"not": "an array"}),
        ("table", ["not an object row"]),
    ],
)
def test_answerable_types_reject_invalid_values(answer_type: str, value: object) -> None:
    answer = ValidatedAnswer(answer_type, value, "answered", EVIDENCE)  # type: ignore[arg-type]

    with pytest.raises(AnswerValidationError):
        validate_answer(answer)


@pytest.mark.parametrize(
    ("answer_type", "status"),
    [
        ("unanswerable", "no_data"),
        ("unanswerable", "ambiguous"),
        ("unanswerable", "insufficient_data"),
        ("abstain", "abstained"),
    ],
)
def test_non_answer_states_are_distinguished(
    answer_type: str, status: str
) -> None:
    validate_answer(
        ValidatedAnswer(answer_type, None, status)  # type: ignore[arg-type]
    )


def test_answerable_result_requires_evidence() -> None:
    with pytest.raises(AnswerValidationError, match="requires evidence"):
        validate_answer(ValidatedAnswer("number", 7, "answered"))


def test_internal_errors_are_not_publishable_as_answers() -> None:
    with pytest.raises(AnswerValidationError, match="unsupported answer status"):
        validate_answer(
            ValidatedAnswer("unanswerable", None, "internal_error")  # type: ignore[arg-type]
        )


def test_abstain_and_unanswerable_statuses_cannot_be_confused() -> None:
    with pytest.raises(AnswerValidationError, match="requires abstained status"):
        validate_answer(ValidatedAnswer("abstain", None, "no_data"))  # type: ignore[arg-type]
    with pytest.raises(AnswerValidationError, match="cannot use abstained status"):
        validate_answer(
            ValidatedAnswer("unanswerable", None, "abstained")  # type: ignore[arg-type]
        )
