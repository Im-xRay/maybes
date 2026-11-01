import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from datamate.training_data import TrainingDataError
from datamate.training_runner import (
    _JsonlPlanDataset,
    train_query_planner,
    validate_training_splits,
)


def _write_split(
    path: Path,
    question_id: str,
    business_id: str,
    *,
    completion: dict[str, object] | None = None,
) -> None:
    record = {
        "question_id": question_id,
        "business_id": business_id,
        "prompt": "Prompt:",
        "completion": json.dumps(
            completion
            or {"base_table": "sales::data", "output_type": "number"}
        ),
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_training_runner_refuses_cpu_before_model_or_output_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    _write_split(train_path, "q-train", "biz-train")
    _write_split(validation_path, "q-valid", "biz-valid")
    output_path = tmp_path / "training-run"
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")

    with pytest.raises(TrainingDataError, match="GPU-only"):
        train_query_planner(
            model_path=model_path,
            train_path=train_path,
            validation_path=validation_path,
            output_dir=output_path,
            mode="full",
        )

    assert not output_path.exists()
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_training_dataset_masks_prompt_and_keeps_completion_targets(
    tmp_path: Path,
) -> None:
    path = tmp_path / "train.jsonl"
    _write_split(path, "q-train", "biz-train")

    class ByteTokenizer:
        chat_template = None
        eos_token = "<eos>"

        def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
            return {"input_ids": list(text.encode("utf-8"))}

    class FakeTorch:
        long = "long"

        @staticmethod
        def tensor(values: list[int], *, dtype: str) -> list[int]:
            assert dtype == "long"
            return values

        @staticmethod
        def ones(size: int, *, dtype: str) -> list[int]:
            assert dtype == "long"
            return [1] * size

    dataset = _JsonlPlanDataset(path, [0], ByteTokenizer(), FakeTorch(), 1024)
    item = dataset[0]
    prompt_size = len("Prompt:".encode("utf-8"))

    assert item["labels"][:prompt_size] == [-100] * prompt_size
    assert item["labels"][prompt_size:] == item["input_ids"][prompt_size:]
    assert all(item["attention_mask"])


def test_training_dataset_rejects_prompt_over_sequence_limit(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_split(path, "q-train", "biz-train")

    class ByteTokenizer:
        chat_template = None
        eos_token = "<eos>"

        def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
            return {"input_ids": list(text.encode("utf-8"))}

    class FakeTorch:
        long = "long"

        @staticmethod
        def tensor(values: list[int], *, dtype: str) -> list[int]:
            return values

        @staticmethod
        def ones(size: int, *, dtype: str) -> list[int]:
            return [1] * size

    dataset = _JsonlPlanDataset(path, [0], ByteTokenizer(), FakeTorch(), 4)

    with pytest.raises(TrainingDataError, match="Prompt exceeds max sequence length"):
        dataset[0]


def test_training_split_validator_accepts_structurally_valid_abstention(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    abstention = {
        "base_table": "sales::data",
        "output_type": "unanswerable",
        "metadata": {
            "status": "insufficient_data",
            "reason": "Thiếu dữ liệu kỳ trước để so sánh.",
        },
    }
    _write_split(
        train_path,
        "q-train",
        "biz-train",
        completion=abstention,
    )
    _write_split(validation_path, "q-valid", "biz-valid")

    summary = validate_training_splits(train_path, validation_path)

    assert summary.train_examples == summary.validation_examples == 1
