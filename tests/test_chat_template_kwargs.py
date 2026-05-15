# SPDX-License-Identifier: Apache-2.0
"""Tests for chat template kwargs forwarding."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import vllm_mlx.server as srv
from vllm_mlx.engine.base import GenerationOutput


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _serve_args(**overrides):
    values = dict(
        model="mlx-community/Qwen3-0.6B-8bit",
        models_config=None,
        host="127.0.0.1",
        port=8000,
        max_num_seqs=256,
        prefill_batch_size=8,
        completion_batch_size=32,
        enable_prefix_cache=True,
        disable_prefix_cache=False,
        prefix_cache_size=100,
        cache_memory_mb=None,
        cache_memory_percent=0.20,
        no_memory_aware_cache=False,
        kv_cache_quantization=False,
        kv_cache_quantization_bits=8,
        kv_cache_quantization_group_size=64,
        kv_cache_min_quantize_tokens=256,
        stream_interval=1,
        max_tokens=32768,
        max_request_tokens=32768,
        continuous_batching=False,
        use_paged_cache=False,
        paged_cache_block_size=64,
        max_cache_blocks=1000,
        chunked_prefill_tokens=0,
        enable_mtp=False,
        mtp_num_draft_tokens=1,
        mtp_optimistic=False,
        prefill_step_size=2048,
        specprefill=False,
        specprefill_threshold=8192,
        specprefill_keep_pct=0.3,
        specprefill_draft_model=None,
        mcp_config=None,
        api_key=None,
        rate_limit=0,
        timeout=300.0,
        enable_auto_tool_choice=False,
        tool_call_parser=None,
        reasoning_parser=None,
        mllm=False,
        default_temperature=None,
        default_top_p=None,
        default_thinking_token_budget=None,
        default_chat_template_kwargs=None,
        served_model_name=None,
        embedding_model=None,
        rerank_model=None,
        gpu_memory_utilization=0.90,
        enable_metrics=False,
        download_timeout=120,
        download_retries=3,
        offline=False,
        mllm_prefill_step_size=0,
        lazy_load_model=True,
        auto_unload_idle_seconds=0.0,
        max_kv_size=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_chat_completion_request_preserves_chat_template_kwargs():
    request = srv.ChatCompletionRequest(
        model="test-model",
        messages=[srv.Message(role="user", content="Hello")],
        chat_template_kwargs={"enable_thinking": False},
    )

    assert request.chat_template_kwargs == {"enable_thinking": False}


def test_batched_engine_applies_chat_template_kwargs():
    with patch("vllm_mlx.engine.batched.is_mllm_model", return_value=False):
        from vllm_mlx.engine.batched import BatchedEngine

        engine = BatchedEngine("test-model")
        engine._tokenizer = MagicMock()
        engine._tokenizer.apply_chat_template.return_value = "prompt"

        prompt = engine._apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            chat_template_kwargs={"enable_thinking": False},
        )

        assert prompt == "prompt"
        engine._tokenizer.apply_chat_template.assert_called_once()
        assert (
            engine._tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"]
            is False
        )


def test_batched_engine_mllm_falls_back_to_tokenizer_when_processor_has_no_template():
    with patch("vllm_mlx.engine.batched.is_mllm_model", return_value=True):
        from vllm_mlx.engine.batched import BatchedEngine

        engine = BatchedEngine("test-mllm-model")
        engine._is_mllm = True

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "prompt-from-tokenizer"

        processor = MagicMock()
        processor.tokenizer = tokenizer
        processor.apply_chat_template.side_effect = ValueError(
            "Cannot use apply_chat_template because this processor does not have a chat template."
        )
        engine._processor = processor

        prompt = engine._apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            chat_template_kwargs={"enable_thinking": False},
        )

        assert prompt == "prompt-from-tokenizer"
        processor.apply_chat_template.assert_called_once()
        tokenizer.apply_chat_template.assert_called_once()


def test_chat_completion_endpoint_forwards_chat_template_kwargs():
    captured = {}

    class FakeEngine:
        model_name = "test-model"
        is_mllm = False
        preserve_native_tool_format = False

        async def chat(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return GenerationOutput(
                text="ORBIT",
                prompt_tokens=4,
                completion_tokens=1,
                finish_reason="stop",
            )

    client = TestClient(srv.app)
    original_engine = srv._engine
    original_model_name = srv._model_name
    srv._engine = FakeEngine()
    srv._model_name = "test-model"
    try:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Reply with ORBIT."}],
                "max_tokens": 8,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
    finally:
        srv._engine = original_engine
        srv._model_name = original_model_name

    assert response.status_code == 200
    assert captured["kwargs"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert response.json()["choices"][0]["message"]["content"] == "ORBIT"


def test_chat_completion_endpoint_applies_server_default_chat_template_kwargs():
    captured = {}

    class FakeEngine:
        model_name = "test-model"
        is_mllm = False
        preserve_native_tool_format = False

        async def chat(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return GenerationOutput(
                text="ORBIT",
                prompt_tokens=4,
                completion_tokens=1,
                finish_reason="stop",
            )

    client = TestClient(srv.app)
    original_engine = srv._engine
    original_model_name = srv._model_name
    original_defaults = getattr(srv, "_default_chat_template_kwargs", None)
    srv._engine = FakeEngine()
    srv._model_name = "test-model"
    srv._default_chat_template_kwargs = {"enable_thinking": False}
    try:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Reply with ORBIT."}],
                "max_tokens": 8,
            },
        )
    finally:
        srv._engine = original_engine
        srv._model_name = original_model_name
        srv._default_chat_template_kwargs = original_defaults

    assert response.status_code == 200
    assert captured["kwargs"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert response.json()["choices"][0]["message"]["content"] == "ORBIT"


def test_serve_command_defaults_reasoning_parser_to_non_thinking_chat(monkeypatch):
    import uvicorn

    import vllm_mlx.cli as cli
    import vllm_mlx.server as server
    from vllm_mlx.utils import download

    original_defaults = getattr(server, "_default_chat_template_kwargs", None)
    monkeypatch.setattr(server, "load_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(download, "ensure_model_downloaded", lambda *args, **kwargs: None)
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)

    try:
        cli.serve_command(_serve_args(reasoning_parser="qwen3"))
        assert server._default_chat_template_kwargs == {"enable_thinking": False}
    finally:
        server._default_chat_template_kwargs = original_defaults


def test_serve_command_preserves_explicit_chat_template_kwargs(monkeypatch):
    import uvicorn

    import vllm_mlx.cli as cli
    import vllm_mlx.server as server
    from vllm_mlx.utils import download

    original_defaults = getattr(server, "_default_chat_template_kwargs", None)
    monkeypatch.setattr(server, "load_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(download, "ensure_model_downloaded", lambda *args, **kwargs: None)
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)

    explicit = {"enable_thinking": True}
    try:
        cli.serve_command(
            _serve_args(
                reasoning_parser="qwen3",
                default_chat_template_kwargs=explicit,
            )
        )
        assert server._default_chat_template_kwargs == explicit
    finally:
        server._default_chat_template_kwargs = original_defaults


def test_chat_completion_endpoint_request_kwargs_override_server_defaults():
    captured = {}

    class FakeEngine:
        model_name = "test-model"
        is_mllm = False
        preserve_native_tool_format = False

        async def chat(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return GenerationOutput(
                text="ORBIT",
                prompt_tokens=4,
                completion_tokens=1,
                finish_reason="stop",
            )

    client = TestClient(srv.app)
    original_engine = srv._engine
    original_model_name = srv._model_name
    original_defaults = getattr(srv, "_default_chat_template_kwargs", None)
    srv._engine = FakeEngine()
    srv._model_name = "test-model"
    srv._default_chat_template_kwargs = {
        "enable_thinking": False,
        "server_default_only": "yes",
    }
    try:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Reply with ORBIT."}],
                "max_tokens": 8,
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "request_only": 1,
                },
            },
        )
    finally:
        srv._engine = original_engine
        srv._model_name = original_model_name
        srv._default_chat_template_kwargs = original_defaults

    assert response.status_code == 200
    assert captured["kwargs"]["chat_template_kwargs"] == {
        "enable_thinking": True,
        "server_default_only": "yes",
        "request_only": 1,
    }
    assert response.json()["choices"][0]["message"]["content"] == "ORBIT"


def test_llm_chat_applies_chat_template_kwargs_before_generate():
    from vllm_mlx.models.llm import MLXLanguageModel

    model = MLXLanguageModel.__new__(MLXLanguageModel)
    model._loaded = True
    model.tokenizer = MagicMock()
    model.tokenizer.apply_chat_template.return_value = "prompt"
    model.generate = MagicMock(return_value="ok")

    result = model.chat(
        [{"role": "user", "content": "Hello"}],
        chat_template_kwargs={"enable_thinking": False},
    )

    assert result == "ok"
    model.tokenizer.apply_chat_template.assert_called_once()
    assert (
        model.tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"] is False
    )
    model.generate.assert_called_once()


@pytest.mark.anyio
async def test_simple_engine_mllm_chat_forwards_chat_template_kwargs():
    from vllm_mlx.engine.simple import SimpleEngine

    with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=True):
        engine = SimpleEngine("test-model")
        engine._loaded = True
        engine._is_mllm = True
        engine._text_model = None
        engine._model = MagicMock()
        engine._model.stream_chat = MagicMock(
            return_value=iter(
                [
                    SimpleNamespace(
                        text="OK",
                        prompt_tokens=5,
                        finish_reason="stop",
                    )
                ]
            )
        )

        output = await engine.chat(
            [{"role": "user", "content": "Hello"}],
            chat_template_kwargs={"enable_thinking": False},
        )

        assert output.text == "OK"
        assert engine._model.stream_chat.call_args.kwargs["chat_template_kwargs"] == {
            "enable_thinking": False
        }
        # Text-only MLLM non-stream chat now aggregates the streaming path.
        assert engine._model.chat.call_count == 0


@pytest.mark.anyio
async def test_simple_engine_stream_generate_text_applies_chat_template_kwargs():
    from vllm_mlx.engine.simple import SimpleEngine

    with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=True):
        engine = SimpleEngine("test-model")
        engine._loaded = True
        engine._is_mllm = True
        engine._text_tokenizer = MagicMock()
        engine._text_tokenizer.apply_chat_template.return_value = "prompt"
        engine._text_model = MagicMock()
        engine._text_model.model = MagicMock()

        with (
            patch("mlx_lm.stream_generate", return_value=iter(())),
            patch("mlx_lm.models.cache.make_prompt_cache", return_value=[]),
            patch("mlx_lm.sample_utils.make_sampler", return_value=object()),
        ):
            chunks = [
                chunk
                async for chunk in engine._stream_generate_text(
                    [{"role": "user", "content": "Hello"}],
                    max_tokens=8,
                    temperature=0.7,
                    top_p=0.9,
                    chat_template_kwargs={"enable_thinking": False},
                )
            ]

        assert chunks
        engine._text_tokenizer.apply_chat_template.assert_called_once()
        assert (
            engine._text_tokenizer.apply_chat_template.call_args.kwargs[
                "enable_thinking"
            ]
            is False
        )
