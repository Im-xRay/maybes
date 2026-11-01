"""
XRAY Model Training Pipeline - Module 4
Two-Stage SFT and DPO Training Script for Qwen2.5-Coder with LoRA / QLoRA using TRL & PEFT.
Includes automated adapter merging for 100% offline deployment.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional


def run_sft_training(
    model_name_or_path: str,
    train_file: str,
    output_dir: str,
    max_seq_length: int = 4096,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
    lora_r: int = 16,
    lora_alpha: int = 32,
    use_bf16: bool = True,
):
    """Stage 1: Supervised Fine-Tuning (SFT) on Valid Trajectories."""
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer

    print(f"=== STARTING STAGE 1: SFT ON {model_name_or_path} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=train_file, split="train")

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=False,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="messages",
        max_seq_length=max_seq_length,
        args=training_args,
    )

    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"SFT Adapter saved to: {output_dir}")


def run_dpo_training(
    model_name_or_path: str,
    train_file: str,
    output_dir: str,
    max_prompt_length: int = 3072,
    max_length: int = 4096,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    learning_rate: float = 5e-5,
    beta: float = 0.1,
    lora_r: int = 16,
    lora_alpha: int = 32,
    use_bf16: bool = True,
):
    """Stage 2: Direct Preference Optimization (DPO) for Penalized Accuracy Alignment."""
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import DPOTrainer

    print(f"=== STARTING STAGE 2: DPO ALIGNMENT ON {model_name_or_path} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=train_file, split="train")

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=False,
    )

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        save_strategy="epoch",
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        report_to="none",
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # peft reference model is managed automatically
        args=training_args,
        beta=beta,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
        max_length=max_length,
        peft_config=peft_config,
    )

    dpo_trainer.train()
    dpo_trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"DPO Adapter saved to: {output_dir}")


def merge_and_export_adapter(base_model_path: str, adapter_path: str, output_path: str):
    """Merge LoRA adapter into base model weights and save full model for offline inference."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"=== MERGING ADAPTER {adapter_path} INTO {base_model_path} ===")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=False)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=False,
    )

    merged_model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = merged_model.merge_and_unload()

    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    print(f"Full merged standalone model successfully exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train SFT and DPO models for DataMate.")
    parser.add_argument("--stage", choices=["sft", "dpo", "merge"], required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-14B-Instruct")
    parser.add_argument("--adapter-path", default="models/sft_adapter")
    parser.add_argument("--train-data", default="XRAY/trajectories_sft.jsonl")
    parser.add_argument("--output-dir", default="models/final_model")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    if args.stage == "sft":
        run_sft_training(
            model_name_or_path=args.base_model,
            train_file=args.train_data,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_epochs=args.epochs,
        )
    elif args.stage == "dpo":
        run_dpo_training(
            model_name_or_path=args.base_model,
            train_file=args.train_data,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
    elif args.stage == "merge":
        merge_and_export_adapter(
            base_model_path=args.base_model,
            adapter_path=args.adapter_path,
            output_path=args.output_dir,
        )


if __name__ == "__main__":
    main()
