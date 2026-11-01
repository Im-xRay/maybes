"""CUDA-only local Transformers backend with offline, deterministic generation."""

from __future__ import annotations

import os
import math
import json
import threading
import time
from pathlib import Path
from typing import Any

from .question_understanding import QuestionUnderstandingError


class LocalModelError(QuestionUnderstandingError):
    """Raised when a local inference checkpoint cannot be loaded or used safely."""


def _validate_checkpoint_files(model_path: Path) -> None:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise LocalModelError("Local checkpoint is missing config.json")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalModelError(f"Local checkpoint config.json is invalid: {exc}") from exc
    if not isinstance(config, dict):
        raise LocalModelError("Local checkpoint config.json must contain an object")
    if not isinstance(config.get("model_type"), str) or not config["model_type"]:
        raise LocalModelError("Local checkpoint config.json must declare model_type")

    weight_index = next(
        (
            model_path / name
            for name in (
                "model.safetensors.index.json",
                "pytorch_model.bin.index.json",
            )
            if (model_path / name).is_file()
        ),
        None,
    )
    if weight_index is not None:
        try:
            index = json.loads(weight_index.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LocalModelError(
                f"Local checkpoint weight index is invalid: {weight_index.name}"
            ) from exc
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise LocalModelError(
                f"Local checkpoint weight index has no weight_map: {weight_index.name}"
            )
        shard_values = list(weight_map.values())
        if any(
            not isinstance(shard, str) or not shard or Path(shard).name != shard
            for shard in shard_values
        ):
            raise LocalModelError("Local checkpoint weight index has an unsafe shard path")
        shards = set(shard_values)
        missing_shards = sorted(
            shard
            for shard in shards
            if not (model_path / shard).is_file() or (model_path / shard).stat().st_size == 0
        )
        if missing_shards:
            raise LocalModelError(
                "Local checkpoint is missing weight shard(s): "
                + ", ".join(missing_shards)
            )
    elif not any(
        (model_path / name).is_file() and (model_path / name).stat().st_size > 0
        for name in ("model.safetensors", "pytorch_model.bin")
    ):
        raise LocalModelError(
            "Local checkpoint is missing model.safetensors or pytorch_model.bin weights"
        )

    tokenizer_files = (
        "tokenizer.json",
        "tokenizer.model",
        "spiece.model",
        "sentencepiece.bpe.model",
        "vocab.json",
        "vocab.txt",
    )
    if not any(
        (model_path / name).is_file() and (model_path / name).stat().st_size > 0
        for name in tokenizer_files
    ):
        raise LocalModelError("Local checkpoint is missing tokenizer vocabulary files")
    tokenizer_json_path = model_path / "tokenizer.json"
    if tokenizer_json_path.is_file():
        try:
            tokenizer_json = json.loads(
                tokenizer_json_path.read_text(encoding="utf-8-sig")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise LocalModelError(
                "Local checkpoint tokenizer.json is invalid"
            ) from exc
        if not isinstance(tokenizer_json, dict):
            raise LocalModelError(
                "Local checkpoint tokenizer.json must contain an object"
            )
    tokenizer_config_path = model_path / "tokenizer_config.json"
    if tokenizer_config_path.is_file():
        try:
            tokenizer_config = json.loads(
                tokenizer_config_path.read_text(encoding="utf-8-sig")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise LocalModelError(
                f"Local checkpoint tokenizer_config.json is invalid: {exc}"
            ) from exc
        if not isinstance(tokenizer_config, dict):
            raise LocalModelError(
                "Local checkpoint tokenizer_config.json must contain an object"
            )


def _estimate_kv_cache_bytes(
    config: Any,
    *,
    sequence_tokens: int,
    bytes_per_value: int,
) -> int:
    layers = getattr(config, "num_hidden_layers", None)
    attention_heads = getattr(config, "num_attention_heads", None)
    key_value_heads = getattr(config, "num_key_value_heads", None)
    hidden_size = getattr(config, "hidden_size", None)
    head_dim = getattr(config, "head_dim", None)
    if key_value_heads is None:
        key_value_heads = attention_heads
    if head_dim is None and (
        isinstance(hidden_size, int)
        and not isinstance(hidden_size, bool)
        and isinstance(attention_heads, int)
        and not isinstance(attention_heads, bool)
        and attention_heads > 0
        and hidden_size % attention_heads == 0
    ):
        head_dim = hidden_size // attention_heads
    dimensions = (layers, key_value_heads, head_dim)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in dimensions
    ):
        raise LocalModelError(
            "Cannot estimate KV-cache memory from the local model configuration"
        )
    return 2 * layers * key_value_heads * head_dim * sequence_tokens * bytes_per_value


class LocalTransformersBackend:
    """A lazy-loaded, single-model backend for the model-independent planner adapter."""

    def __init__(
        self,
        model_path: Path | str,
        *,
        precision: str = "bf16",
        max_new_tokens: int = 1024,
        max_prompt_tokens: int = 16_384,
        parameter_cap: int = 15_000_000_000,
        cpu_threads: int = 2,
        device: str = "cuda",
        generation_timeout_seconds: float | None = None,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_dir():
            raise LocalModelError("model_path must be a local checkpoint directory")
        if precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or not 1 <= max_new_tokens <= 4096
        ):
            raise ValueError("max_new_tokens must be between 1 and 4096")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (max_prompt_tokens, parameter_cap, cpu_threads)
        ):
            raise ValueError("token limits, parameter cap, and threads must be positive")
        if generation_timeout_seconds is not None:
            try:
                timeout_is_valid = (
                    not isinstance(generation_timeout_seconds, bool)
                    and isinstance(generation_timeout_seconds, (int, float))
                    and math.isfinite(float(generation_timeout_seconds))
                    and generation_timeout_seconds > 0
                )
            except OverflowError:
                timeout_is_valid = False
            if not timeout_is_valid:
                raise ValueError(
                    "generation_timeout_seconds must be finite and positive"
                )
        self.precision = precision
        self.max_new_tokens = max_new_tokens
        self.max_prompt_tokens = max_prompt_tokens
        self.parameter_cap = parameter_cap
        self.cpu_threads = cpu_threads
        self.device = device
        self.generation_timeout_seconds = generation_timeout_seconds
        self.model_name = self.model_path.name
        self.parameter_count: int | None = None
        self.model_storage_bytes: int | None = None
        self.estimated_kv_cache_bytes: int | None = None
        self.cuda_free_bytes_before_transfer: int | None = None
        self.cuda_total_bytes: int | None = None
        self.cuda_peak_allocated_bytes: int | None = None
        self.cuda_peak_reserved_bytes: int | None = None
        self.load_seconds: float | None = None
        self.last_generation_seconds: float | None = None
        self.total_generation_seconds = 0.0
        self.generated_tokens = 0
        self.generation_count = 0
        self.process_rss_bytes: int | None = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None:
            return
        load_started = time.perf_counter()
        _validate_checkpoint_files(self.model_path)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            import torch
        except ImportError as exc:
            raise LocalModelError(
                "PyTorch is required for local inference; no remote API fallback exists"
            ) from exc
        if self.device == "cuda" and not torch.cuda.is_available():
            raise LocalModelError(
                "Local Transformers inference is configured for CUDA; select device=cpu "
                "explicitly to use CPU inference"
            )
        if self.device == "cpu" and self.precision != "fp32":
            raise LocalModelError(
                "CPU inference requires fp32 precision; select it explicitly"
            )
        if (
            self.device == "cuda"
            and self.precision == "bf16"
            and not torch.cuda.is_bf16_supported(including_emulation=False)
        ):
            raise LocalModelError("Selected GPU does not support bf16")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise LocalModelError(
                "Transformers is required; install the selected model's runtime dependencies"
            ) from exc
        try:
            import psutil
        except ImportError as exc:
            raise LocalModelError(
                "psutil is required to record local inference memory usage"
            ) from exc

        torch.set_num_threads(self.cpu_threads)
        dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.precision]
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            trust_remote_code=False,
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise LocalModelError("Tokenizer must define a pad token or EOS token")
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=dtype,
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count > self.parameter_cap:
            raise LocalModelError(
                f"Model has {parameter_count:,} parameters, above cap "
                f"{self.parameter_cap:,}"
            )
        model_storage_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        ) + sum(
            buffer.numel() * buffer.element_size()
            for buffer in model.buffers()
        )
        estimated_kv_cache_bytes = _estimate_kv_cache_bytes(
            model.config,
            sequence_tokens=self.max_prompt_tokens + self.max_new_tokens,
            bytes_per_value={"fp32": 4, "fp16": 2, "bf16": 2}[self.precision],
        )
        free_bytes = None
        total_bytes = None
        if self.device == "cuda":
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            except RuntimeError as exc:
                raise LocalModelError(
                    f"Cannot inspect CUDA device memory before loading {self.model_name}"
                ) from exc
            required_bytes = model_storage_bytes + estimated_kv_cache_bytes
            if required_bytes > free_bytes:
                raise LocalModelError(
                    f"Model weights, buffers, and estimated KV cache need at least "
                    f"{required_bytes:,} CUDA bytes ({model_storage_bytes:,} weights/buffers "
                    f"+ {estimated_kv_cache_bytes:,} KV cache), but only {free_bytes:,} "
                    "are free. The estimate excludes runtime workspace; select "
                    "device=cpu explicitly to try CPU inference."
                )
        device = torch.device("cuda:0" if self.device == "cuda" else "cpu")
        try:
            model.to(device)
        except RuntimeError as exc:
            if self.device == "cuda" and "out of memory" in str(exc).casefold():
                raise LocalModelError(
                    "CUDA ran out of memory while loading the model; no automatic "
                    "device fallback was attempted. Select device=cpu explicitly."
                ) from exc
            raise
        model.eval()
        self._model = model
        self._tokenizer = tokenizer
        self._torch = torch
        self.parameter_count = parameter_count
        self.model_storage_bytes = model_storage_bytes
        self.estimated_kv_cache_bytes = estimated_kv_cache_bytes
        self.cuda_free_bytes_before_transfer = free_bytes
        self.cuda_total_bytes = total_bytes
        self.process_rss_bytes = psutil.Process(os.getpid()).memory_info().rss
        self.load_seconds = time.perf_counter() - load_started

    def metrics(self) -> dict[str, int | float | str | None]:
        """Return observed load, generation, memory, and throughput measurements."""
        tokens_per_second = (
            self.generated_tokens / self.total_generation_seconds
            if self.total_generation_seconds > 0
            else None
        )
        return {
            "device": self.device,
            "precision": self.precision,
            "parameter_count": self.parameter_count,
            "model_storage_bytes": self.model_storage_bytes,
            "estimated_kv_cache_bytes": self.estimated_kv_cache_bytes,
            "load_seconds": self.load_seconds,
            "process_rss_bytes": self.process_rss_bytes,
            "cuda_total_bytes": self.cuda_total_bytes,
            "cuda_free_bytes_before_transfer": self.cuda_free_bytes_before_transfer,
            "cuda_peak_allocated_bytes": self.cuda_peak_allocated_bytes,
            "cuda_peak_reserved_bytes": self.cuda_peak_reserved_bytes,
            "generation_count": self.generation_count,
            "generated_tokens": self.generated_tokens,
            "last_generation_seconds": self.last_generation_seconds,
            "total_generation_seconds": self.total_generation_seconds,
            "tokens_per_second": tokens_per_second,
        }

    def load(self) -> None:
        """Load the selected checkpoint once; subsequent calls are idempotent."""
        with self._lock:
            self._load()

    def count_prompt_tokens(self, prompt: str) -> int:
        """Count tokens with the same chat template used for generation."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be non-empty text")
        with self._lock:
            self._load()
            tokenizer = self._tokenizer
            if getattr(tokenizer, "chat_template", None):
                encoded = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
            else:
                encoded = tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=True,
                    truncation=False,
                )
            return int(encoded["input_ids"].shape[-1])

    def generate(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be non-empty text")
        with self._lock:
            self._load()
            tokenizer = self._tokenizer
            torch = self._torch
            if getattr(tokenizer, "chat_template", None):
                encoded = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
            else:
                encoded = tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=True,
                    truncation=False,
                )
            input_ids = encoded["input_ids"]
            if input_ids.shape[-1] > self.max_prompt_tokens:
                raise LocalModelError(
                    f"Prompt has {input_ids.shape[-1]} tokens; limit is "
                    f"{self.max_prompt_tokens}"
                )
            inputs = {
                key: value.to(torch.device("cuda:0" if self.device == "cuda" else "cpu"))
                for key, value in encoded.items()
                if key in {"input_ids", "attention_mask"}
            }
            generation_options: dict[str, object] = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
                "num_beams": 1,
                "use_cache": True,
                "pad_token_id": tokenizer.pad_token_id,
                "temperature": None,
                "top_p": None,
                "top_k": None,
            }
            if self.generation_timeout_seconds is not None:
                generation_options["max_time"] = self.generation_timeout_seconds
            if tokenizer.eos_token_id is not None:
                generation_options["eos_token_id"] = tokenizer.eos_token_id
            generation_started = time.monotonic()
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats(0)
            with torch.inference_mode():
                generated = self._model.generate(**inputs, **generation_options)
            if self.device == "cuda":
                torch.cuda.synchronize(0)
            generation_elapsed = time.monotonic() - generation_started
            if (
                self.generation_timeout_seconds is not None
                and generation_elapsed > self.generation_timeout_seconds
            ):
                raise LocalModelError(
                    "Model generation exceeded the configured timeout of "
                    f"{self.generation_timeout_seconds:g} seconds"
                )
            continuation = generated[0, input_ids.shape[-1] :]
            if continuation.numel() == 0:
                raise LocalModelError("Model returned an empty completion")
            generated_tokens = continuation.numel()
            self.last_generation_seconds = generation_elapsed
            self.total_generation_seconds += generation_elapsed
            self.generated_tokens += generated_tokens
            self.generation_count += 1
            if self.device == "cuda":
                self.cuda_peak_allocated_bytes = torch.cuda.max_memory_allocated(0)
                self.cuda_peak_reserved_bytes = torch.cuda.max_memory_reserved(0)
            import psutil

            self.process_rss_bytes = psutil.Process(os.getpid()).memory_info().rss
            return tokenizer.decode(
                continuation,
                skip_special_tokens=True,
            ).strip()
