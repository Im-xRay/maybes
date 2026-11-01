"""Merge a trained local LoRA adapter into its local base checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Local base checkpoint")
    parser.add_argument("--adapter-path", required=True, help="Trained LoRA adapter")
    parser.add_argument("--output", required=True, help="New merged model directory")
    parser.add_argument("--parameter-cap", type=int, default=15_000_000_000)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    adapter_path = Path(args.adapter_path).resolve()
    output_path = Path(args.output).resolve()
    if not model_path.is_dir() or not adapter_path.is_dir():
        parser.error("model-path and adapter-path must be local directories")
    if output_path.exists():
        parser.error(f"output directory already exists: {output_path}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        parser.error(
            f"GPU export dependencies are missing; install requirements-training.txt: {exc}"
        )
    if not torch.cuda.is_available():
        parser.error("LoRA merge is GPU-only; CUDA is unavailable")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        parser.error("Selected GPU does not support bf16")

    torch_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.precision]
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch_dtype,
    )
    base_parameters = sum(parameter.numel() for parameter in base_model.parameters())
    if base_parameters > args.parameter_cap:
        parser.error(
            f"Base model has {base_parameters:,} parameters, above cap "
            f"{args.parameter_cap:,}"
        )
    base_model.to(torch.device("cuda:0"))
    adapted_model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        is_trainable=False,
        local_files_only=True,
    )
    merged_model = adapted_model.merge_and_unload(safe_merge=True)
    total_parameters = sum(parameter.numel() for parameter in merged_model.parameters())
    if total_parameters > args.parameter_cap:
        parser.error(
            f"Merged inference model has {total_parameters:,} parameters, above cap "
            f"{args.parameter_cap:,}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.mkdir()
    merged_model.save_pretrained(
        str(output_path),
        safe_serialization=True,
    )
    tokenizer.save_pretrained(str(output_path))
    (output_path / "export_manifest.json").write_text(
        json.dumps(
            {
                "base_checkpoint_name": model_path.name,
                "adapter_name": adapter_path.name,
                "precision": args.precision,
                "base_parameters": base_parameters,
                "merged_inference_parameters": total_parameters,
                "parameter_cap": args.parameter_cap,
                "quantization_reduces_parameter_count": False,
                "remote_code_enabled": False,
                "local_files_only": True,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
