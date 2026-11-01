"""GPU-only, offline fine-tuning runner for reviewed QueryPlan JSONL labels."""

from __future__ import annotations

import json
import os
import inspect
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .plan_serialization import query_plan_from_json
from .question_understanding import (
    QuestionUnderstandingError,
    validate_abstention_plan,
)
from .training_data import TrainingDataError


@dataclass(frozen=True)
class TrainingSplitSummary:
    train_examples: int
    validation_examples: int
    train_businesses: int
    validation_businesses: int


def _inspect_split(path: Path) -> tuple[list[int], set[str], set[str]]:
    if not path.is_file():
        raise TrainingDataError(f"Training split does not exist: {path}")
    offsets: list[int] = []
    question_ids: set[str] = set()
    business_ids: set[str] = set()
    with path.open("rb") as stream:
        while True:
            offset = stream.tell()
            raw_line = stream.readline()
            if not raw_line:
                break
            if not raw_line.strip():
                raise TrainingDataError(f"Blank JSONL line at {path}:{len(offsets) + 1}")
            try:
                record = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise TrainingDataError(
                    f"Invalid UTF-8 JSONL at {path}:{len(offsets) + 1}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise TrainingDataError(
                    f"Expected object at {path}:{len(offsets) + 1}"
                )
            question_id = record.get("question_id")
            business_id = record.get("business_id")
            prompt = record.get("prompt")
            completion = record.get("completion")
            if not all(
                isinstance(item, str) and item.strip()
                for item in (question_id, business_id, prompt, completion)
            ):
                raise TrainingDataError(
                    f"Missing required training fields at {path}:{len(offsets) + 1}"
                )
            if question_id in question_ids:
                raise TrainingDataError(
                    f"Duplicate question_id {question_id!r} in {path}"
                )
            try:
                plan = query_plan_from_json(completion)
            except ValueError as exc:
                raise TrainingDataError(
                    f"Invalid target QueryPlan at {path}:{len(offsets) + 1}: {exc}"
                ) from exc
            if plan.output_type is None:
                raise TrainingDataError(
                    f"Target must declare an output type at {path}:{len(offsets) + 1}"
                )
            if plan.output_type in {"unanswerable", "abstain"}:
                try:
                    validate_abstention_plan(plan, {plan.base_table: ()})
                except QuestionUnderstandingError as exc:
                    raise TrainingDataError(
                        f"Invalid abstention target at {path}:{len(offsets) + 1}: {exc}"
                    ) from exc
            offsets.append(offset)
            question_ids.add(question_id)
            business_ids.add(business_id)
    if not offsets:
        raise TrainingDataError(f"Training split is empty: {path}")
    return offsets, question_ids, business_ids


def validate_training_splits(
    train_path: Path | str,
    validation_path: Path | str,
) -> TrainingSplitSummary:
    """Validate labels, duplicates, and business-level leakage before training."""
    _, train_questions, train_businesses = _inspect_split(Path(train_path))
    _, validation_questions, validation_businesses = _inspect_split(
        Path(validation_path)
    )
    overlap = train_businesses & validation_businesses
    if overlap:
        raise TrainingDataError(
            "Business leakage across train/validation: "
            + ", ".join(sorted(overlap))
        )
    duplicate_questions = train_questions & validation_questions
    if duplicate_questions:
        raise TrainingDataError(
            "Question leakage across train/validation: "
            + ", ".join(sorted(duplicate_questions))
        )
    return TrainingSplitSummary(
        train_examples=len(train_questions),
        validation_examples=len(validation_questions),
        train_businesses=len(train_businesses),
        validation_businesses=len(validation_businesses),
    )


class _JsonlPlanDataset:
    """Random-access JSONL dataset that tokenizes one example at a time."""

    def __init__(
        self,
        path: Path,
        offsets: list[int],
        tokenizer: Any,
        torch: Any,
        max_sequence_length: int,
    ) -> None:
        self.path = path
        self.offsets = offsets
        self.tokenizer = tokenizer
        self.torch = torch
        self.max_sequence_length = max_sequence_length

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[index])
            record = json.loads(stream.readline().decode("utf-8"))
        prompt = record["prompt"]
        completion = record["completion"]
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            user_prompt = self.tokenizer.apply_chat_template(
                messages[:1],
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            user_prompt = prompt
            full_text = prompt + completion + (self.tokenizer.eos_token or "")
        prompt_ids = self.tokenizer(
            user_prompt,
            add_special_tokens=not bool(getattr(self.tokenizer, "chat_template", None)),
            truncation=False,
        )["input_ids"]
        full_ids = self.tokenizer(
            full_text,
            add_special_tokens=not bool(getattr(self.tokenizer, "chat_template", None)),
            truncation=False,
        )["input_ids"]
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise TrainingDataError(
                "Tokenizer prompt is not a prefix of full chat; "
                "this model's chat template needs a model-specific training adapter"
            )
        if len(prompt_ids) >= self.max_sequence_length:
            raise TrainingDataError(
                f"Prompt exceeds max sequence length ({self.max_sequence_length})"
            )
        if len(full_ids) > self.max_sequence_length:
            raise TrainingDataError(
                f"Prompt plus target exceeds max sequence length "
                f"({self.max_sequence_length})"
            )
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        if not any(label != -100 for label in labels):
            raise TrainingDataError("No target tokens remain after prompt masking")
        return {
            "input_ids": self.torch.tensor(full_ids, dtype=self.torch.long),
            "attention_mask": self.torch.ones(len(full_ids), dtype=self.torch.long),
            "labels": self.torch.tensor(labels, dtype=self.torch.long),
        }


def _collator(tokenizer: Any, torch: Any):
    def collate(examples: list[dict[str, Any]]) -> dict[str, Any]:
        width = max(len(example["input_ids"]) for example in examples)
        input_ids = []
        attention_masks = []
        labels = []
        for example in examples:
            padding = width - len(example["input_ids"])
            input_ids.append(
                torch.cat(
                    [
                        example["input_ids"],
                        torch.full(
                            (padding,),
                            tokenizer.pad_token_id,
                            dtype=torch.long,
                        ),
                    ]
                )
            )
            attention_masks.append(
                torch.cat(
                    [
                        example["attention_mask"],
                        torch.zeros(padding, dtype=torch.long),
                    ]
                )
            )
            labels.append(
                torch.cat(
                    [
                        example["labels"],
                        torch.full((padding,), -100, dtype=torch.long),
                    ]
                )
            )
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels),
        }

    return collate


def train_query_planner(
    *,
    model_path: Path | str,
    train_path: Path | str,
    validation_path: Path | str,
    output_dir: Path | str,
    mode: str = "lora",
    lora_target_modules: tuple[str, ...] = (),
    parameter_cap: int = 15_000_000_000,
    max_sequence_length: int = 2048,
    epochs: float = 3.0,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float | None = None,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    precision: str = "bf16",
    seed: int = 42,
) -> dict[str, object]:
    """Fine-tune a local checkpoint on a CUDA host; CPU and online loading are refused."""
    if mode not in {"lora", "full"}:
        raise TrainingDataError("mode must be 'lora' or 'full'; QLoRA is model-specific")
    if mode == "lora" and not lora_target_modules:
        raise TrainingDataError(
            "LoRA requires explicit architecture-specific target module names"
        )
    if precision not in {"fp32", "fp16", "bf16"}:
        raise TrainingDataError("precision must be fp32, fp16, or bf16")
    if parameter_cap < 1 or max_sequence_length < 2:
        raise TrainingDataError("parameter cap and sequence length must be positive")
    if epochs <= 0 or batch_size < 1 or gradient_accumulation_steps < 1:
        raise TrainingDataError("training sizes must be positive")
    if learning_rate is not None and learning_rate <= 0:
        raise TrainingDataError("learning_rate must be positive")
    if lora_rank < 1 or lora_alpha < 1 or not 0 <= lora_dropout < 1:
        raise TrainingDataError("LoRA rank, alpha, or dropout is invalid")
    resolved_learning_rate = learning_rate or (2e-4 if mode == "lora" else 2e-5)

    model_directory = Path(model_path).resolve()
    output_directory = Path(output_dir).resolve()
    if not model_directory.is_dir():
        raise TrainingDataError("model_path must be a local checkpoint directory")
    if output_directory.exists():
        raise TrainingDataError(
            f"Output directory already exists; choose a new path: {output_directory}"
        )
    train_file = Path(train_path).resolve()
    validation_file = Path(validation_path).resolve()
    split_summary = validate_training_splits(train_file, validation_file)
    train_offsets, _, _ = _inspect_split(train_file)
    validation_offsets, _, _ = _inspect_split(validation_file)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import torch
    except ImportError as exc:
        raise TrainingDataError(
            "PyTorch is missing; install a CUDA-matched build on the dedicated "
            "training host"
        ) from exc

    if not torch.cuda.is_available():
        raise TrainingDataError(
            "Training is GPU-only; CUDA is unavailable, no CPU fallback is permitted"
        )
    try:
        import peft
        import transformers
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise TrainingDataError(
            "GPU training dependencies are missing; install requirements-training.txt "
            "on the dedicated training host"
        ) from exc
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise TrainingDataError("Selected GPU does not support bf16")
    torch_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[precision]
    device_name = torch.cuda.get_device_name(0)
    free_memory, total_memory = torch.cuda.mem_get_info(0)
    if free_memory == 0:
        raise TrainingDataError("CUDA reports no free device memory")
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_directory),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise TrainingDataError("Tokenizer must define a pad token or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        str(model_directory),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch_dtype,
    )
    base_parameters = sum(parameter.numel() for parameter in model.parameters())
    if base_parameters > parameter_cap:
        raise TrainingDataError(
            f"Base model has {base_parameters:,} parameters, above cap "
            f"{parameter_cap:,}"
        )
    model.to(torch.device("cuda:0"))

    if mode == "lora":
        available_modules = {
            name.rsplit(".", 1)[-1] for name, _ in model.named_modules()
        }
        missing_targets = set(lora_target_modules) - available_modules
        if missing_targets:
            raise TrainingDataError(
                "LoRA target modules not present in checkpoint: "
                + ", ".join(sorted(missing_targets))
            )
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=list(lora_target_modules),
                bias="none",
            ),
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if total_parameters > parameter_cap:
        raise TrainingDataError(
            f"Model plus trainable adapters has {total_parameters:,} parameters, "
            f"above cap {parameter_cap:,}"
        )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    train_dataset = _JsonlPlanDataset(
        train_file,
        train_offsets,
        tokenizer,
        torch,
        max_sequence_length,
    )
    validation_dataset = _JsonlPlanDataset(
        validation_file,
        validation_offsets,
        tokenizer,
        torch,
        max_sequence_length,
    )
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir()
    training_argument_names = inspect.signature(
        TrainingArguments.__init__
    ).parameters
    evaluation_argument = (
        "eval_strategy"
        if "eval_strategy" in training_argument_names
        else "evaluation_strategy"
    )
    training_arguments = {
        "output_dir": str(output_directory),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": resolved_learning_rate,
        "logging_strategy": "steps",
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "warmup_ratio": 0.03,
        "weight_decay": 0.01,
        "report_to": [],
        "dataloader_num_workers": 0,
        "gradient_checkpointing": True,
        "fp16": precision == "fp16",
        "bf16": precision == "bf16",
        "seed": seed,
        "data_seed": seed,
    }
    training_arguments[evaluation_argument] = "epoch"
    args = TrainingArguments(**training_arguments)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "data_collator": _collator(tokenizer, torch),
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
    }
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    final_directory = output_directory / "final"
    trainer.save_model(str(final_directory))
    tokenizer.save_pretrained(str(final_directory))
    summary: dict[str, object] = {
        "model_path": str(model_directory),
        "mode": mode,
        "precision": precision,
        "base_parameters": base_parameters,
        "total_inference_parameters": total_parameters,
        "parameter_cap": parameter_cap,
        "device": device_name,
        "free_vram_bytes_at_start": free_memory,
        "total_vram_bytes": total_memory,
        "seed": seed,
        "split": asdict(split_summary),
        "training_config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": resolved_learning_rate,
            "max_sequence_length": max_sequence_length,
            "lora_target_modules": list(lora_target_modules),
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
        },
        "software_versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "metrics": trainer.state.log_history,
    }
    (output_directory / "training_summary.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary
