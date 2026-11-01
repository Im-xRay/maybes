import json

import pytest

from datamate.evaluation import answers_match
from datamate.manual_testing import (
    ManualTestError,
    _read_input,
    _summary,
    _validate_expected_answer,
)


def test_expected_answer_uses_submission_and_evaluator_value_shapes() -> None:
    assert _validate_expected_answer({"type": "number", "answer": 12}) == (
        "number",
        12,
    )
    assert _validate_expected_answer(
        {"type": "table", "answer": {"columns": ["name"], "rows": [["A"]]}}
    ) == ("table", {"columns": ["name"], "rows": [["A"]]})
    assert _validate_expected_answer(
        {"type": "unanswerable", "answer": None}
    ) == ("unanswerable", None)
    assert _validate_expected_answer(None) is None


@pytest.mark.parametrize(
    "answer",
    [
        {"type": "abstain", "answer": None},
        {"type": "number", "answer": "12"},
        {
            "type": "table",
            "answer": {"columns": ["name"], "rows": [["A", "extra"]]},
        },
        {"type": "unanswerable", "answer": "không rõ"},
        {"type": "number", "answer": 12, "reason": "extra"},
    ],
)
def test_invalid_manual_expected_answer_is_rejected(answer: object) -> None:
    with pytest.raises(ManualTestError):
        _validate_expected_answer(answer)


def test_manual_input_reader_retains_invalid_lines_instead_of_skipping(
    tmp_path,
) -> None:
    source = tmp_path / "questions.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "q1",
                        "business_id": "biz",
                        "question": "Tính tổng",
                        "expected_answer": None,
                    }
                ),
                "{invalid json",
                json.dumps(
                    {"id": "q1", "business_id": "biz", "question": "Lặp id"}
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    entries = _read_input(source)

    assert len(entries) == 3
    assert entries[0][2] is None
    assert isinstance(entries[1][2], json.JSONDecodeError)
    assert isinstance(entries[2][2], ManualTestError)


def test_manual_answer_comparison_reuses_evaluator_semantics() -> None:
    assert answers_match("number", 1.1, "number", 1.1)
    assert not answers_match("number", 1, "number", 2)
    assert answers_match(
        "table",
        {"columns": ["name", "total"], "rows": [["A", 10], ["B", 20]]},
        "table",
        {"columns": ["name", "total"], "rows": [["B", 20], ["A", 10]]},
    )
    assert answers_match("unanswerable", None, "unanswerable", None)
    assert not answers_match("unanswerable", None, "abstain", None)


def test_abstention_is_not_correct_and_uses_neutral_evaluator_accounting() -> None:
    report = _summary(
        [
            {
                "expected_answer_valid": True,
                "expected_answer": {"type": "number", "answer": 10},
                "generated_answer": {"type": "abstain"},
                "prediction_status": "abstained",
                "correctness": False,
                "error": None,
            }
        ]
    )

    assert report["accuracy"] == 0
    assert report["correct_answers"] == 0
    assert report["wrong_answers"] == 0
    assert report["abstentions"] == 1
