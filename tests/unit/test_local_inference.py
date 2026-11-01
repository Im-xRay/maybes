import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import datamate.local_inference as local_inference
from datamate.local_inference import LocalModelError, LocalTransformersBackend


def _write_checkpoint_files(checkpoint) -> None:
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "config.json").write_text(
        '{"model_type":"test"}',
        encoding="utf-8",
    )
    (checkpoint / "tokenizer.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"test weights")


def test_local_backend_requires_checkpoint_and_has_no_remote_fallback(
    tmp_path,
    monkeypatch,
) -> None:
    with pytest.raises(LocalModelError, match="local checkpoint"):
        LocalTransformersBackend(tmp_path / "missing")

    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint_files(checkpoint)
    monkeypatch.setitem(sys.modules, "torch", None)
    backend = LocalTransformersBackend(checkpoint)

    with pytest.raises(LocalModelError, match="no remote API fallback"):
        backend.generate("Tổng doanh thu?")


def test_local_backend_rejects_checkpoint_with_missing_indexed_weight_shard(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint_files(checkpoint)
    (checkpoint / "model.safetensors").unlink()
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.weight": "model-00001-of-00002.safetensors"}}),
        encoding="utf-8",
    )
    backend = LocalTransformersBackend(checkpoint)

    with pytest.raises(LocalModelError, match="missing weight shard"):
        backend.load()


def test_local_backend_rejects_invalid_config_before_loading_runtime(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint_files(checkpoint)
    (checkpoint / "config.json").write_text("{", encoding="utf-8")

    with pytest.raises(LocalModelError, match="config.json is invalid"):
        LocalTransformersBackend(checkpoint).load()


def _fake_cuda_backend(
    tmp_path,
    monkeypatch,
    *,
    free_bytes: int,
    device: str = "cuda",
    precision: str = "fp16",
    bf16_supported: bool = True,
    generation_timeout_seconds: float | None = None,
):
    class FakeTensor:
        def __init__(self, size: int, element_size: int) -> None:
            self.size = size
            self.bytes_per_element = element_size
            self.shape = (1, size)

        def numel(self) -> int:
            return self.size

        def element_size(self) -> int:
            return self.bytes_per_element

        def to(self, _device):
            return self

        def __getitem__(self, _index):
            return FakeTensor(1, self.bytes_per_element)

    class FakeModel:
        moved_to = None
        config = SimpleNamespace(
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            hidden_size=8,
        )

        def parameters(self):
            return [FakeTensor(4, 2)]

        def buffers(self):
            return [FakeTensor(2, 4)]

        def to(self, device):
            self.moved_to = device
            return self

        def eval(self):
            return self

        def generate(self, **options):
            self.generation_options = options
            return FakeTensor(2, 2)

    class FakeTokenizer:
        chat_template = None
        pad_token_id = 0
        eos_token_id = 0

        def __call__(self, _prompt, **_options):
            return {"input_ids": FakeTensor(2, 2)}

        def decode(self, _continuation, *, skip_special_tokens):
            assert skip_special_tokens is True
            return "{}"

    model = FakeModel()
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            is_bf16_supported=lambda including_emulation=True: bf16_supported,
            mem_get_info=lambda device: (free_bytes, 4096),
            reset_peak_memory_stats=lambda _device: None,
            synchronize=lambda _device: None,
            max_memory_allocated=lambda _device: 128,
            max_memory_reserved=lambda _device: 256,
        ),
        set_num_threads=lambda _: None,
        float32="float32",
        float16="float16",
        bfloat16="bfloat16",
        device=lambda name: name,
        inference_mode=nullcontext,
    )
    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: FakeTokenizer()
        ),
        AutoModelForCausalLM=SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: model
        ),
    )
    fake_psutil = SimpleNamespace(
        Process=lambda _pid: SimpleNamespace(
            memory_info=lambda: SimpleNamespace(rss=1024)
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint_files(checkpoint)
    return LocalTransformersBackend(
        checkpoint,
        device=device,
        precision=precision,
        generation_timeout_seconds=generation_timeout_seconds,
    ), model


def test_local_backend_rejects_insufficient_cuda_memory_before_device_transfer(
    tmp_path,
    monkeypatch,
) -> None:
    backend, model = _fake_cuda_backend(tmp_path, monkeypatch, free_bytes=15)

    with pytest.raises(LocalModelError, match="only 15 are free"):
        backend.load()

    assert model.moved_to is None


def test_local_backend_records_cuda_memory_and_model_storage(
    tmp_path,
    monkeypatch,
) -> None:
    estimated_cache_bytes = 2 * 2 * 1 * 4 * (16_384 + 1_024) * 2
    backend, model = _fake_cuda_backend(
        tmp_path,
        monkeypatch,
        free_bytes=16 + estimated_cache_bytes,
    )

    backend.load()

    assert model.moved_to == "cuda:0"
    assert backend.parameter_count == 4
    assert backend.model_storage_bytes == 16
    assert backend.estimated_kv_cache_bytes == estimated_cache_bytes
    assert backend.cuda_free_bytes_before_transfer == 16 + estimated_cache_bytes
    assert backend.cuda_total_bytes == 4096


def test_local_backend_uses_cpu_only_when_explicitly_configured(
    tmp_path,
    monkeypatch,
) -> None:
    backend, model = _fake_cuda_backend(
        tmp_path,
        monkeypatch,
        free_bytes=0,
        device="cpu",
        precision="fp32",
    )

    backend.load()
    output = backend.generate("Tổng doanh thu?")

    assert output == "{}"
    assert model.moved_to == "cpu"
    assert backend.cuda_free_bytes_before_transfer is None
    assert backend.metrics()["device"] == "cpu"
    assert backend.metrics()["load_seconds"] is not None
    assert backend.metrics()["generated_tokens"] == 1
    assert backend.metrics()["process_rss_bytes"] == 1024


def test_local_backend_counts_tokens_with_generation_tokenizer(
    tmp_path,
    monkeypatch,
) -> None:
    backend, _ = _fake_cuda_backend(
        tmp_path,
        monkeypatch,
        free_bytes=1_000_000,
    )

    assert backend.count_prompt_tokens("prompt") == 2


def test_local_backend_requires_explicit_fp32_for_cpu(
    tmp_path,
    monkeypatch,
) -> None:
    backend, _ = _fake_cuda_backend(
        tmp_path,
        monkeypatch,
        free_bytes=0,
        device="cpu",
        precision="fp16",
    )

    with pytest.raises(LocalModelError, match="CPU inference requires fp32"):
        backend.load()


def test_local_backend_rejects_bf16_emulation_on_unsupported_gpu(
    tmp_path,
    monkeypatch,
) -> None:
    backend, model = _fake_cuda_backend(
        tmp_path,
        monkeypatch,
        free_bytes=1_000_000,
        precision="bf16",
        bf16_supported=False,
    )

    with pytest.raises(LocalModelError, match="does not support bf16"):
        backend.load()

    assert model.moved_to is None


def test_local_backend_does_not_fallback_after_cuda_memory_failure(
    tmp_path,
    monkeypatch,
) -> None:
    backend, model = _fake_cuda_backend(tmp_path, monkeypatch, free_bytes=15)

    with pytest.raises(LocalModelError, match="select device=cpu explicitly"):
        backend.load()

    assert model.moved_to is None


def test_local_backend_rejects_cuda_memory_without_room_for_estimated_cache(
    tmp_path,
    monkeypatch,
) -> None:
    estimated_cache_bytes = 2 * 2 * 1 * 4 * (16_384 + 1_024) * 2
    backend, model = _fake_cuda_backend(
        tmp_path,
        monkeypatch,
        free_bytes=15 + estimated_cache_bytes,
    )

    with pytest.raises(LocalModelError, match="estimated KV cache"):
        backend.load()

    assert model.moved_to is None


def test_local_backend_requires_supported_cache_dimensions_in_config(
    tmp_path,
    monkeypatch,
) -> None:
    backend, model = _fake_cuda_backend(tmp_path, monkeypatch, free_bytes=1_000_000)
    model.config = SimpleNamespace()

    with pytest.raises(LocalModelError, match="Cannot estimate KV-cache"):
        backend.load()

    assert model.moved_to is None


def test_local_backend_enforces_configured_generation_timeout(
    tmp_path,
    monkeypatch,
) -> None:
    backend, model = _fake_cuda_backend(
        tmp_path,
        monkeypatch,
        free_bytes=1_000_000,
        generation_timeout_seconds=1,
    )
    backend.load()
    ticks = iter((10.0, 12.0))
    monkeypatch.setattr(local_inference.time, "monotonic", lambda: next(ticks))

    with pytest.raises(LocalModelError, match="exceeded the configured timeout"):
        backend.generate("prompt")

    assert model.generation_options["max_time"] == 1
