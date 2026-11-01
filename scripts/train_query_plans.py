"""GPU-only offline LoRA/full fine-tuning for reviewed QueryPlan annotations."""

from __future__ import annotations

import argparse
import json

from datamate.training_runner import train_query_planner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Local model checkpoint")
    parser.add_argument("--train", required=True, help="Prepared train.jsonl")
    parser.add_argument("--validation", required=True, help="Prepared validation.jsonl")
    parser.add_argument("--output", required=True, help="New run output directory")
    parser.add_argument("--mode", choices=("lora", "full"), default="lora")
    parser.add_argument(
        "--lora-target-modules",
        help="Comma-separated exact module names, required for LoRA",
    )
    parser.add_argument("--parameter-cap", type=int, default=15_000_000_000)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Defaults to 2e-4 for LoRA or 2e-5 for full fine-tuning",
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="bf16",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    targets = tuple(
        value.strip()
        for value in (args.lora_target_modules or "").split(",")
        if value.strip()
    )
    summary = train_query_planner(
        model_path=args.model_path,
        train_path=args.train,
        validation_path=args.validation,
        output_dir=args.output,
        mode=args.mode,
        lora_target_modules=targets,
        parameter_cap=args.parameter_cap,
        max_sequence_length=args.max_sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        precision=args.precision,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
