"""Exercise the real HTTP adapters against local simulated protocol endpoints."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Literal, cast
from unittest.mock import patch

import aiohttp
import pytest
from aiohttp import web
from image_support import image_bytes

from jouzetsu.backend_models import (
    BackendConfig,
    BackendConfiguration,
    BackendKind,
    GenerationProfile,
    LMStudioTransport,
    ModelRef,
)
from jouzetsu.backend_protocol import (
    BackendError,
    BackendErrorCategory,
    BackendReachability,
    GenerationRequest,
    InferenceMessage,
)
from jouzetsu.backends import ChatCompletionsBackend, LMStudioBackend
from jouzetsu.chat_completions import (
    ChatCompletionsTransport,
    _encode_request,  # pyright: ignore[reportPrivateUsage]
    iter_sse_data,
    json_object,
)
from jouzetsu.config import ConfigStore
from jouzetsu.controllers import PersistenceController
from jouzetsu.image_storage import ImageStorage
from jouzetsu.inference import GenerationMode, InferenceRuntime, JobPhase
from jouzetsu.models import Chat, ImageAttachment, ImageMediaType
from jouzetsu.ollama_backend import OllamaBackend
from jouzetsu.responses_transport import encode_request as encode_responses_request
from jouzetsu.runtime import (
    ChatStreamEvent,
    ModelCapabilities,
    ModelReady,
    PredictionComplete,
    PredictionFragment,
)
from jouzetsu.sampling import CHAT_COMPLETIONS_PARAMETERS, ParameterSupport
from jouzetsu.state_registry import ChatRegistry
from jouzetsu.storage import ChatStorage


@asynccontextmanager
async def server(
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> AsyncGenerator[str]:
    app = web.Application()
    _ = app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    address = cast(tuple[str, int], runner.addresses[0])
    try:
        yield f"http://127.0.0.1:{address[1]}/v1"
    finally:
        await runner.cleanup()


def config(url: str, *, native: bool = False) -> BackendConfig:
    return BackendConfig.create(
        BackendKind.LM_STUDIO if native else BackendKind.OPENAI_COMPATIBLE, "Test", url
    )


@pytest.mark.parametrize(
    "failure", ["overflow", "error", "interrupted", "tool_call", "wrong_role"]
)
def test_ollama_generation_failures_are_safe(failure: str) -> None:
    async def exercise() -> None:
        async def handler(req: web.Request) -> web.Response:
            if req.path.endswith("/api/version"):
                return web.json_response({"version": "0.40.16"})
            if failure == "overflow":
                return web.json_response({"error": "SECRET"}, status=400)
            if failure == "error":
                return web.Response(text=json.dumps({"error": "SECRET"}) + "\n")
            if failure in ("tool_call", "wrong_role"):
                message = (
                    {"role": "user"}
                    if failure == "wrong_role"
                    else {"tool_calls": [{"function": {"name": "unknown"}}]}
                )
                return web.Response(
                    text=json.dumps(
                        {"model": "catalog-model", "message": message, "done": True}
                    )
                    + "\n"
                )
            return web.Response(
                text=json.dumps(
                    {
                        "model": "catalog-model",
                        "message": {"content": "partial"},
                        "done": False,
                    }
                )
                + "\n"
            )

        async with server(handler) as url:
            backend = BackendConfig.create(BackendKind.OLLAMA, "Ollama", url)
            adapter = OllamaBackend(backend)
            with pytest.raises(BackendError) as exc:
                _ = [item async for item in adapter.stream(request(backend))]
            assert "SECRET" not in str(exc.value)
            assert exc.value.category is (
                BackendErrorCategory.INCOMPATIBLE
                if failure in ("overflow", "tool_call")
                else BackendErrorCategory.PROTOCOL
            )
            await adapter.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("version", ["0.10.0", "0.13.9", "unknown", "0.40.0-rc1", None])
def test_ollama_blocks_unverified_context_controls(version: str | None) -> None:
    async def exercise() -> None:
        calls: list[str] = []

        async def handler(req: web.Request) -> web.Response:
            calls.append(req.path)
            return web.json_response({"version": version})

        async with server(handler) as url:
            backend = BackendConfig.create(BackendKind.OLLAMA, "Ollama", url)
            adapter = OllamaBackend(backend)
            with pytest.raises(BackendError, match="0.14.0"):
                _ = [event async for event in adapter.stream(request(backend))]
            assert len(calls) == 1 and calls[0].endswith("/api/version")
            await adapter.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("idle_minutes", [None, 7])
def test_ollama_discovery_and_stream_preserve_history_and_images(
    tmp_path: Path,
    idle_minutes: int | None,
) -> None:
    async def exercise() -> None:
        payloads: list[dict[str, object]] = []

        async def handler(req: web.Request) -> web.Response:
            if req.path.endswith("/api/version"):
                return web.json_response({"version": "0.40.16"})
            if req.path.endswith("/api/tags"):
                return web.json_response(
                    {"models": [{"name": "catalog-model", "size": 42}]}
                )
            if req.path.endswith("/api/ps"):
                return web.json_response({"models": []})
            if req.path.endswith("/api/show"):
                return web.json_response(
                    {
                        "capabilities": ["completion", "vision", "thinking"],
                        "thinking": {"values": [False, True, "max"], "default": True},
                    }
                )
            payloads.append(json_object(cast(object, await req.json())))
            chunks = [
                {
                    "model": "catalog-model",
                    "message": {"thinking": "Think", "content": ""},
                    "done": False,
                },
                {
                    "model": "catalog-model",
                    "message": {"content": "Answer"},
                    "done": False,
                },
                {
                    "model": "catalog-model",
                    "message": {"content": ""},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 12,
                    "eval_count": 3,
                    "eval_duration": 1_000_000_000,
                },
            ]
            return web.Response(
                text="".join(json.dumps(chunk) + "\n" for chunk in chunks),
                content_type="application/x-ndjson",
            )

        async with server(handler) as url:
            backend = replace(
                BackendConfig.create(BackendKind.OLLAMA, "Ollama", url),
                ollama_idle_unload_minutes=idle_minutes,
            )
            images = ImageStorage(tmp_path / "images")
            image = images.store(image_bytes(), "test.png")
            adapter = OllamaBackend(backend, images=images)
            discovery = await adapter.discover_models()
            caps = discovery.models[0].capabilities
            assert caps.vision is True and caps.reasoning_options == (
                "off",
                "on",
                "max",
            )
            assert caps.reasoning_default == "on"
            prepared = replace(
                request(backend),
                capabilities=caps,
                reasoning="on",
                messages=(
                    InferenceMessage("user", "Look", (image,)),
                    InferenceMessage("assistant", "Old"),
                    InferenceMessage("user", "Next"),
                ),
            )
            events = [item async for item in adapter.stream(prepared)]
            assert [
                (item.content, item.is_reasoning)
                for item in events
                if isinstance(item, PredictionFragment)
            ] == [("Think", True), ("Answer", False)]
            assert (
                next(
                    item for item in events if isinstance(item, PredictionComplete)
                ).metrics.prompt_tokens
                == 12
            )
            payload = payloads[0]
            if idle_minutes is None:
                assert payload["keep_alive"] == -1
            else:
                assert payload["keep_alive"] == "7m"
            assert (
                payload["truncate"] is False
                and payload["shift"] is False
                and payload["think"] is True
            )
            messages = cast(list[dict[str, object]], payload["messages"])
            assert [message["role"] for message in messages] == [
                "user",
                "assistant",
                "user",
            ]
            assert len(cast(list[str], messages[0]["images"])) == 1
            assert (
                cast(dict[str, object], payload["options"])["num_predict"]
                == prepared.parameters.max_tokens
            )
            await adapter.close()

    asyncio.run(exercise())


def request(backend: BackendConfig) -> GenerationRequest:
    return GenerationRequest(
        ModelRef(backend.id, "catalog-model"),
        (InferenceMessage("user", "first"), InferenceMessage("user", "second")),
        ModelCapabilities(),
        CHAT_COMPLETIONS_PARAMETERS,
    )


@pytest.mark.parametrize(
    ("selection", "wire_value"),
    [("off", "none"), ("on", "medium"), ("low", "low"), ("high", "high")],
)
def test_lmstudio_reasoning_request_mapping(selection: str, wire_value: str) -> None:
    backend = config("http://localhost/v1", native=True)
    generation = replace(
        request(backend), reasoning=cast(Literal["off", "on", "low", "high"], selection)
    )
    ready = ModelReady("catalog-model", "catalog-model", 0)
    native_payload = json.loads(_encode_request(generation, ready, None, native=True))
    assert native_payload["reasoning_effort"] == wire_value
    with pytest.raises(BackendError, match="Reasoning controls are unavailable"):
        _encode_request(generation, ready, None)


def test_reported_reasoning_level_is_retained() -> None:
    async def exercise() -> None:
        async def handler(req: web.Request) -> web.Response:
            payload = chunk("reply")
            payload["reasoning_effort"] = "high"
            return web.Response(
                text=event(payload) + "data: [DONE]\n\n",
                content_type="text/event-stream",
            )

        async with server(handler) as url:
            backend = config(url)
            adapter = ChatCompletionsBackend(backend)
            events = [event async for event in adapter.stream(request(backend))]
            completion = next(
                event for event in events if isinstance(event, PredictionComplete)
            )
            assert completion.metrics.reasoning_level == "high"
            await adapter.close()

    asyncio.run(exercise())


def event(payload: object) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\r\n\r\n"


def chunk(
    content: object, *, model: str = "catalog-model", reasoning: str | None = None
) -> dict[str, object]:
    return {
        "model": model,
        "choices": [
            {"index": 0, "delta": {"content": content, "reasoning_content": reasoning}}
        ],
    }


def test_generic_stream_preserves_roles_reasoning_usage_and_actual_identity() -> None:
    async def exercise() -> None:
        payloads: list[dict[str, object]] = []

        async def handler(req: web.Request) -> web.Response:
            payloads.append(json_object(cast(object, await req.json())))
            body = (
                event(
                    chunk(
                        "café <think>ordinary</think>",
                        model="actual-model",
                        reasoning="trace",
                    )
                )
                + event(
                    {
                        "model": "actual-model",
                        "choices": [
                            {"delta": dict[str, object](), "finish_reason": "stop"}
                        ],
                        "usage": {"prompt_tokens": 8, "completion_tokens": 3},
                    }
                )
                + "data: [DONE]\n\n"
            )
            return web.Response(text=body, content_type="text/event-stream")

        async with server(handler) as url:
            backend = config(url)
            adapter = ChatCompletionsBackend(backend)
            assert adapter.model_management is None
            events = [item async for item in adapter.stream(request(backend))]
            await adapter.close()
        ready = next(item for item in events if isinstance(item, ModelReady))
        assert ready.model_key == "actual-model"
        fragments = [item for item in events if isinstance(item, PredictionFragment)]
        assert [(item.content, item.is_reasoning) for item in fragments] == [
            ("trace", True),
            ("café <think>ordinary</think>", False),
        ]
        complete = next(item for item in events if isinstance(item, PredictionComplete))
        assert (
            complete.metrics.prompt_tokens == 8 and complete.metrics.output_tokens == 3
        )
        assert payloads[0]["model"] == "catalog-model"
        assert payloads[0]["messages"] == [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
        assert "ttl" not in payloads[0]
        assert len(payloads) == 1

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "body",
    [
        "data: {bad}\n\n",
        event({"error": {"message": "SECRET echoed prompt"}}),
        event(chunk(["not text"])),
        event({"choices": [{"index": 1, "delta": {"content": "wrong choice"}}]}),
        event(chunk("partial")),
        'data: {"choices": []}',
        event(chunk("first", model="a")) + event(chunk("second", model="b")),
    ],
)
def test_protocol_errors_are_visible_safe_and_never_retried(body: str) -> None:
    async def exercise() -> None:
        calls = 0

        async def handler(req: web.Request) -> web.Response:
            nonlocal calls
            calls += 1
            return web.Response(text=body, content_type="text/event-stream")

        async with server(handler) as url:
            backend = config(url)
            adapter = ChatCompletionsBackend(backend)
            try:
                with pytest.raises(BackendError) as error:
                    _ = [item async for item in adapter.stream(request(backend))]
                assert error.value.category is BackendErrorCategory.PROTOCOL
                assert "SECRET" not in str(error.value)
                assert calls == 1
            finally:
                await adapter.close()

    asyncio.run(exercise())


def test_partial_text_is_yielded_before_truncated_stream_error() -> None:
    async def exercise() -> None:
        async def handler(req: web.Request) -> web.Response:
            return web.Response(
                text=event(chunk("partial")), content_type="text/event-stream"
            )

        async with server(handler) as url:
            backend = config(url)
            adapter = ChatCompletionsBackend(backend)
            fragments: list[str] = []
            with pytest.raises(BackendError):
                async for item in adapter.stream(request(backend)):
                    if isinstance(item, PredictionFragment):
                        fragments.append(item.content)
            assert fragments == ["partial"]
            await adapter.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("status", [400, 404, 422, 500, 302])
def test_http_error_bodies_are_not_exposed_or_retried(status: int) -> None:
    async def exercise() -> None:
        calls = 0

        async def handler(req: web.Request) -> web.Response:
            nonlocal calls
            calls += 1
            return web.Response(
                status=status,
                text="SECRET prompt and reasoning",
                headers={"Location": "/other"},
            )

        async with server(handler) as url:
            backend = config(url)
            adapter = ChatCompletionsBackend(backend)
            with pytest.raises(BackendError) as error:
                _ = [item async for item in adapter.stream(request(backend))]
            assert "SECRET" not in str(error.value)
            assert calls == 1
            assert error.value.category is (
                BackendErrorCategory.INCOMPATIBLE
                if status in {400, 404, 422}
                else BackendErrorCategory.PROTOCOL
            )
            await adapter.close()

    asyncio.run(exercise())


def test_discovery_unsupported_remains_online_with_manual_models() -> None:
    async def exercise() -> None:
        async def handler(req: web.Request) -> web.Response:
            return web.Response(status=404)

        async with server(handler) as url:
            adapter = ChatCompletionsBackend(config(url))
            assert not (await adapter.discover_models()).supported
            assert (await adapter.probe()).state is BackendReachability.ONLINE
            await adapter.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "payload",
    [{"data": [{"id": "a"}, {"id": "a"}]}, {"data": [{"id": ""}]}, {"other": []}],
)
def test_invalid_discovery_is_not_a_successful_empty_inventory(payload: object) -> None:
    async def exercise() -> None:
        async def handler(req: web.Request) -> web.Response:
            return web.json_response(payload)

        async with server(handler) as url:
            adapter = ChatCompletionsBackend(config(url))
            with pytest.raises(BackendError):
                _ = await adapter.discover_models()
            await adapter.close()

    asyncio.run(exercise())


def test_native_management_uses_exact_catalog_and_instance_ids() -> None:
    async def exercise() -> None:
        loaded = False
        requests: list[tuple[str, dict[str, object]]] = []

        async def handler(req: web.Request) -> web.Response:
            nonlocal loaded
            if req.method == "GET":
                assert req.path == "/api/v1/models"
                return web.json_response(
                    {
                        "models": [
                            {
                                "type": "llm",
                                "key": "catalog-model",
                                "display_name": "Catalog",
                                "loaded_instances": [
                                    {
                                        "id": "instance-1",
                                        "config": {"context_length": 4096},
                                    }
                                ]
                                if loaded
                                else [],
                            }
                        ]
                    }
                )
            body = json_object(cast(object, await req.json()))
            requests.append((req.path, body))
            if req.path == "/api/v1/models/load":
                loaded = True
                return web.json_response({"instance_id": "instance-1"})
            if req.path == "/api/v1/models/unload":
                loaded = False
                return web.json_response({})
            assert req.path == "/v1/chat/completions"
            return web.Response(
                text=event(chunk("reply", model="instance-1")) + "data: [DONE]\n\n",
                content_type="text/event-stream",
            )

        async with server(handler) as url:
            backend = config(url, native=True)
            adapter = LMStudioBackend(backend)
            _ = [item async for item in adapter.load_model("catalog-model")]
            events = [item async for item in adapter.stream(request(backend))]
            ready = next(item for item in events if isinstance(item, ModelReady))
            assert (
                ready.model_key == "catalog-model" and ready.instance_id == "instance-1"
            )
            await adapter.unload_model("instance-1")
            assert requests[0] == (
                "/api/v1/models/load",
                {"model": "catalog-model", "echo_load_config": True},
            )
            assert requests[1][1]["model"] == "instance-1"
            assert requests[2] == (
                "/api/v1/models/unload",
                {"instance_id": "instance-1"},
            )
            assert all("ttl" not in body for _, body in requests)
            with pytest.raises(BackendError, match="unavailable"):
                _ = [item async for item in adapter.load_model("different-model")]
            assert len(requests) == 3
            await adapter.close()

    asyncio.run(exercise())


def test_responses_transport_preserves_history_images_and_streamed_reasoning(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        calls: list[dict[str, object]] = []

        async def handler(req: web.Request) -> web.Response:
            if req.method == "GET":
                return web.json_response(
                    {
                        "models": [
                            {
                                "type": "llm",
                                "key": "catalog-model",
                                "display_name": "Catalog",
                                "loaded_instances": [{"id": "instance-1"}],
                            }
                        ]
                    }
                )
            assert req.path == "/v1/responses"
            calls.append(json_object(cast(object, await req.json())))
            chunks = [
                {
                    "type": "response.created",
                    "response": {
                        "model": "instance-1",
                        "id": "response-1",
                        "truncation": "disabled",
                    },
                },
                {"type": "response.output_item.added", "item": {"type": "reasoning"}},
                {"type": "response.reasoning_text.delta", "delta": "Thinking"},
                {"type": "response.output_item.added", "item": {"type": "message"}},
                {"type": "response.output_text.delta", "delta": "Answer"},
                {
                    "type": "response.completed",
                    "response": {
                        "model": "instance-1",
                        "id": "response-1",
                        "truncation": "disabled",
                        "status": "completed",
                        "output": [
                            {"type": "reasoning"},
                            {
                                "type": "message",
                                "content": [{"type": "output_text", "text": "Answer"}],
                            },
                        ],
                        "reasoning": {"effort": "medium"},
                        "usage": {"input_tokens": 12, "output_tokens": 8},
                    },
                },
            ]
            return web.Response(
                text="".join(event(chunk) for chunk in chunks),
                content_type="text/event-stream",
            )

        async with server(handler) as url:
            backend = replace(
                config(url, native=True), lmstudio_transport=LMStudioTransport.RESPONSES
            )
            images = ImageStorage(tmp_path / "images")
            image = images.store(image_bytes(), "test.png")
            prepared = replace(
                request(backend),
                messages=(
                    InferenceMessage("system", "System"),
                    InferenceMessage("user", "Look", (image,)),
                    InferenceMessage("assistant", "Prior answer"),
                    InferenceMessage("user", "Continue"),
                ),
                capabilities=ModelCapabilities(vision=True),
                reasoning="on",
            )
            adapter = LMStudioBackend(backend, images=images)
            events = [item async for item in adapter.stream(prepared)]
            assert [
                (item.content, item.is_reasoning)
                for item in events
                if isinstance(item, PredictionFragment)
            ] == [("Thinking", True), ("Answer", False)]
            complete = next(
                item for item in events if isinstance(item, PredictionComplete)
            )
            assert complete.metrics.reasoning_level == "on"
            assert (complete.metrics.prompt_tokens, complete.metrics.output_tokens) == (
                12,
                8,
            )
            payload = calls[0]
            assert payload["store"] is False and payload["stream"] is True
            assert payload["truncation"] == "disabled"
            assert payload["reasoning"] == {"effort": "medium"}
            assert payload["max_output_tokens"] == prepared.parameters.max_tokens
            messages = cast(list[dict[str, object]], payload["input"])
            assert [message["role"] for message in messages] == [
                "system",
                "user",
                "assistant",
                "user",
            ]
            parts = cast(list[dict[str, object]], messages[1]["content"])
            assert parts[0] == {"type": "input_text", "text": "Look"}
            assert str(parts[1]["image_url"]).startswith("data:image/png;base64,")
            await adapter.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("selection", "effort"),
    [
        (None, None),
        ("off", "none"),
        ("on", "medium"),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("max", "max"),
    ],
)
def test_responses_encodes_only_supported_parameters(
    selection: str | None, effort: str | None
) -> None:
    backend = config("http://localhost/v1", native=True)
    prepared = replace(
        request(backend),
        reasoning=cast(
            Literal["off", "on", "low", "medium", "high", "max"] | None, selection
        ),
        parameter_support=ParameterSupport(
            temperature=True, top_p=False, max_tokens=False
        ),
    )
    payload = json.loads(
        encode_responses_request(
            prepared, ModelReady("catalog-model", "Catalog", 0), None
        )
    )
    assert payload.get("reasoning") == ({"effort": effort} if effort else None)
    assert payload["temperature"] == prepared.parameters.temperature
    assert "top_p" not in payload and "max_output_tokens" not in payload


@pytest.mark.parametrize(
    ("tail", "category"),
    [
        ([], BackendErrorCategory.PROTOCOL),
        (
            [{"type": "response.failed", "error": "SECRET echoed body"}],
            BackendErrorCategory.PROTOCOL,
        ),
        (
            [{"type": "response.output_item.added", "item": {"type": "function_call"}}],
            BackendErrorCategory.INCOMPATIBLE,
        ),
        (
            [{"type": "response.output_text.delta", "delta": ["invalid"]}],
            BackendErrorCategory.PROTOCOL,
        ),
        (
            [
                {
                    "type": "response.completed",
                    "response": {
                        "id": "response-1",
                        "model": "catalog-model",
                        "truncation": "disabled",
                        "status": "completed",
                    },
                }
            ],
            BackendErrorCategory.PROTOCOL,
        ),
        (
            [
                {
                    "type": "response.completed",
                    "response": {
                        "id": "other",
                        "model": "catalog-model",
                        "truncation": "disabled",
                        "status": "completed",
                    },
                }
            ],
            BackendErrorCategory.PROTOCOL,
        ),
        (
            [{"type": "response.output_item.added", "item": {"type": ["invalid"]}}],
            BackendErrorCategory.INCOMPATIBLE,
        ),
        (
            [
                {
                    "type": "response.output_item.done",
                    "item": {"type": "message", "role": "user"},
                }
            ],
            BackendErrorCategory.PROTOCOL,
        ),
        (
            [
                {
                    "type": "response.output_item.done",
                    "item": {"type": "message", "content": [{"type": "refusal"}]},
                }
            ],
            BackendErrorCategory.INCOMPATIBLE,
        ),
        (
            [
                {
                    "type": "response.completed",
                    "response": {
                        "id": "response-1",
                        "model": "catalog-model",
                        "truncation": "disabled",
                        "status": "completed",
                        "output": [{"type": "function_call"}],
                    },
                }
            ],
            BackendErrorCategory.INCOMPATIBLE,
        ),
    ],
)
def test_responses_transport_rejects_invalid_or_incomplete_streams(
    tail: list[dict[str, object]], category: BackendErrorCategory
) -> None:
    async def exercise() -> None:
        async def handler(req: web.Request) -> web.Response:
            chunks = [
                {
                    "type": "response.created",
                    "response": {
                        "id": "response-1",
                        "model": "catalog-model",
                        "truncation": "disabled",
                    },
                },
                *tail,
            ]
            return web.Response(
                text="".join(event(chunk) for chunk in chunks),
                content_type="text/event-stream",
            )

        async with server(handler) as url:
            backend = config(url, native=True)
            transport = ChatCompletionsTransport(url)
            with pytest.raises(BackendError) as exc:
                _ = [
                    item
                    async for item in transport.stream(
                        request(backend),
                        ModelReady("catalog-model", "Catalog", 0),
                        native=True,
                        responses=True,
                    )
                ]
            assert exc.value.category is category
            assert "SECRET" not in str(exc.value)
            await transport.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("truncation", [None, "auto", "unknown"])
def test_responses_requires_truncation_confirmation(truncation: str | None) -> None:
    async def exercise() -> None:
        async def handler(req: web.Request) -> web.Response:
            response: dict[str, object] = {"id": "response-1", "model": "catalog-model"}
            if truncation is not None:
                response["truncation"] = truncation
            return web.Response(
                text=event({"type": "response.created", "response": response})
                + event({"type": "response.output_text.delta", "delta": "unsafe"}),
                content_type="text/event-stream",
            )

        async with server(handler) as url:
            transport = ChatCompletionsTransport(url)
            received: list[ChatStreamEvent] = []
            with pytest.raises(BackendError) as exc:
                async for item in transport.stream(
                    request(config(url, native=True)),
                    ModelReady("catalog-model", "Catalog", 0),
                    native=True,
                    responses=True,
                ):
                    received.append(item)
            assert exc.value.category is BackendErrorCategory.INCOMPATIBLE
            assert "truncation" in str(exc.value)
            assert received == []
            await transport.close()

    asyncio.run(exercise())


def test_responses_incomplete_stream_keeps_partial_text_and_stop_reason() -> None:
    async def exercise() -> None:
        async def handler(req: web.Request) -> web.Response:
            response = {
                "id": "response-1",
                "model": "catalog-model",
                "truncation": "disabled",
            }
            chunks: list[dict[str, object]] = [
                {"type": "response.created", "response": response},
                {"type": "response.output_text.delta", "delta": "Partial"},
                {
                    "type": "response.incomplete",
                    "response": {
                        **response,
                        "status": "incomplete",
                        "output": [
                            {
                                "type": "message",
                                "content": [{"type": "output_text", "text": "Partial"}],
                            }
                        ],
                        "incomplete_details": {"reason": "max_output_tokens"},
                    },
                },
            ]
            return web.Response(
                text="".join(event(chunk) for chunk in chunks),
                content_type="text/event-stream",
            )

        async with server(handler) as url:
            backend = config(url, native=True)
            transport = ChatCompletionsTransport(url)
            events = [
                item
                async for item in transport.stream(
                    request(backend),
                    ModelReady("catalog-model", "Catalog", 0),
                    native=True,
                    responses=True,
                )
            ]
            assert [
                item.content for item in events if isinstance(item, PredictionFragment)
            ] == ["Partial"]
            assert (
                next(
                    item for item in events if isinstance(item, PredictionComplete)
                ).metrics.stop_reason
                == "max_output_tokens"
            )
            await transport.close()

    asyncio.run(exercise())


def test_known_unsupported_parameters_are_preserved_but_not_forwarded() -> None:
    async def exercise() -> None:
        body: dict[str, object] = {}

        async def handler(req: web.Request) -> web.Response:
            body.update(json_object(cast(object, await req.json())))
            return web.Response(
                text=event(chunk("reply")) + "data: [DONE]\n\n",
                content_type="text/event-stream",
            )

        async with server(handler) as url:
            backend = config(url)
            prepared = replace(
                request(backend),
                parameter_support=ParameterSupport(
                    temperature=False, top_p=None, max_tokens=True
                ),
            )
            adapter = ChatCompletionsBackend(backend)
            _ = [item async for item in adapter.stream(prepared)]
            await adapter.close()
            assert "temperature" not in body and "top_p" not in body
            assert body["max_tokens"] == prepared.parameters.max_tokens

    asyncio.run(exercise())


def test_timeouts_and_connection_errors_have_safe_explicit_categories() -> None:
    async def exercise() -> None:
        backend = config("http://localhost:1234/v1")
        for exception, category in (
            (TimeoutError("SECRET"), BackendErrorCategory.TIMEOUT),
            (aiohttp.ClientConnectionError("SECRET"), BackendErrorCategory.OFFLINE),
        ):
            adapter = ChatCompletionsBackend(backend)
            with (
                patch.object(aiohttp.ClientSession, "post", side_effect=exception),
                pytest.raises(BackendError) as error,
            ):
                _ = [item async for item in adapter.stream(request(backend))]
            assert error.value.category is category
            assert "SECRET" not in str(error.value)
            await adapter.close()

    asyncio.run(exercise())


def test_sse_utf8_crlf_comments_and_multiple_data_lines_across_every_boundary() -> None:
    async def exercise() -> None:
        payload = ": heartbeat\r\nevent: message\r\ndata: café\r\ndata: second\r\n\r\ndata: [DONE]\n\n".encode()
        for boundary in range(1, len(payload)):

            async def chunks(split: int) -> AsyncIterator[bytes]:
                yield payload[:split]
                yield payload[split:]

            assert [item async for item in iter_sse_data(chunks(boundary))] == [
                "café\nsecond",
                "[DONE]",
            ]

    asyncio.run(exercise())


@pytest.mark.parametrize("newline", [b"\n", b"\r\n", b"\r"])
def test_sse_bom_and_all_line_endings_at_every_byte_boundary(newline: bytes) -> None:
    async def exercise() -> None:
        payload = b"\xef\xbb\xbf" + newline.join(
            ["data: café".encode(), b"", b"data: [DONE]", b"", b""]
        )
        for boundary in range(1, len(payload)):

            async def chunks(split: int) -> AsyncIterator[bytes]:
                yield payload[:split]
                yield payload[split:]

            assert [data async for data in iter_sse_data(chunks(boundary))] == [
                "café",
                "[DONE]",
            ]

    asyncio.run(exercise())


def test_sse_size_limit_applies_to_events_not_aggregate_network_chunks() -> None:
    async def exercise() -> None:
        async def grouped() -> AsyncIterator[bytes]:
            yield b"data: small\n\n" * 20

        with patch("jouzetsu.chat_completions._MAX_EVENT_BYTES", 32):
            assert [data async for data in iter_sse_data(grouped())] == ["small"] * 20

            async def oversized() -> AsyncIterator[bytes]:
                yield b"data: " + b"x" * 40 + b"\n\n"

            with pytest.raises(BackendError, match="size limit"):
                _ = [data async for data in iter_sse_data(oversized())]

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "choice",
    [
        {"delta": {}, "finish_reason": ""},
        {"delta": {"role": "user", "content": "wrong role"}},
        {"delta": {"role": []}},
        {"message": {"content": "not streaming"}},
        {"delta": {}, "finish_reason": "tool_calls"},
    ],
)
def test_invalid_or_unsupported_terminal_choices_fail_visibly(
    choice: dict[str, object],
) -> None:
    async def exercise() -> None:
        async def handler(req: web.Request) -> web.Response:
            return web.Response(
                text=event({"choices": [choice]}) + "data: [DONE]\n\n",
                content_type="text/event-stream",
            )

        async with server(handler) as url:
            backend = config(url)
            adapter = ChatCompletionsBackend(backend)
            with pytest.raises(BackendError):
                _ = [item async for item in adapter.stream(request(backend))]
            await adapter.close()

    asyncio.run(exercise())


def test_stream_rejects_content_after_a_terminal_choice() -> None:
    async def exercise() -> None:
        async def handler(req: web.Request) -> web.Response:
            body = (
                event(
                    {
                        "choices": [
                            {"delta": dict[str, object](), "finish_reason": "stop"}
                        ]
                    }
                )
                + event(chunk("late content"))
                + "data: [DONE]\n\n"
            )
            return web.Response(text=body, content_type="text/event-stream")

        async with server(handler) as url:
            backend = config(url)
            adapter = ChatCompletionsBackend(backend)
            with pytest.raises(BackendError, match="after it had finished"):
                _ = [item async for item in adapter.stream(request(backend))]
            await adapter.close()

    asyncio.run(exercise())


def test_closing_adapter_stream_closes_its_transport_iterator_immediately() -> None:
    async def exercise() -> None:
        closed = asyncio.Event()
        retained: list[AsyncGenerator[ChatStreamEvent]] = []

        async def stream() -> AsyncGenerator[ChatStreamEvent]:
            try:
                yield ModelReady("catalog-model", "Model", 0)
                await asyncio.Event().wait()
            finally:
                closed.set()

        def transport_stream(
            self: ChatCompletionsTransport,
            prepared: GenerationRequest,
            ready: ModelReady,
            *,
            native: bool = False,
        ) -> AsyncGenerator[ChatStreamEvent]:
            iterator = stream()
            retained.append(iterator)
            return iterator

        backend = config("http://localhost:1234/v1")
        adapter = ChatCompletionsBackend(backend)
        with patch.object(ChatCompletionsTransport, "stream", new=transport_stream):
            iterator = adapter.stream(request(backend))
            _ = await anext(iterator)
            await iterator.aclose()
            assert closed.is_set()
        await adapter.close()

    asyncio.run(exercise())


def test_slow_request_encoding_leaves_the_event_loop_available() -> None:
    async def exercise() -> None:
        entered, release = threading.Event(), threading.Event()
        calls = 0

        async def handler(req: web.Request) -> web.Response:
            nonlocal calls
            calls += 1
            return web.Response(
                text=event(chunk("reply")) + "data: [DONE]\n\n",
                content_type="text/event-stream",
            )

        async with server(handler) as url:
            backend = config(url)
            adapter = ChatCompletionsBackend(backend)
            prepared = request(backend)
            encoded = _encode_request(
                prepared, ModelReady("catalog-model", "Model", 0), None
            )

            def slow_encode(
                request: GenerationRequest,
                ready: ModelReady,
                images: ImageStorage | None,
                *,
                native: bool = False,
            ) -> bytes:
                entered.set()
                assert release.wait(5)
                return encoded

            async def consume() -> None:
                _ = [item async for item in adapter.stream(prepared)]

            with patch("jouzetsu.chat_completions._encode_request", new=slow_encode):
                streaming = asyncio.create_task(consume())
                while not entered.is_set():
                    await asyncio.sleep(0)
                assert not streaming.done() and calls == 0
                # Reaching here proves the loop is running while the worker is held.
                release.set()
                await streaming
            assert calls == 1
            await adapter.close()

    asyncio.run(exercise())


def runtime_for_backend(home: Path, backend: BackendConfig) -> InferenceRuntime:
    store = ConfigStore.for_home(home)
    settings = store.load()
    settings.backends = BackendConfiguration((backend,), backend.id)
    store.save(settings)
    persistence = PersistenceController(
        settings, store, ChatStorage(settings.chats_file)
    )
    return InferenceRuntime(ChatRegistry(lambda: ""), persistence)


@pytest.mark.parametrize("idle_minutes, keep_alive", [(None, -1), (3, "3m")])
def test_ollama_model_management_tracks_real_state(
    tmp_path: Path, idle_minutes: int | None, keep_alive: int | str
) -> None:
    async def exercise() -> None:
        loaded = False
        actions: list[dict[str, object]] = []

        async def handler(req: web.Request) -> web.Response:
            nonlocal loaded
            if req.path.endswith("/api/tags"):
                return web.json_response({"models": [{"name": "catalog-model"}]})
            if req.path.endswith("/api/ps"):
                return web.json_response(
                    {"models": [{"name": "catalog-model"}] if loaded else []}
                )
            if req.path.endswith("/api/show"):
                return web.json_response({"capabilities": ["completion"]})
            assert req.path.endswith("/api/generate")
            payload = json_object(cast(object, await req.json()))
            actions.append(payload)
            loaded = payload["keep_alive"] != 0
            return web.json_response(
                {
                    "model": "catalog-model",
                    "done": True,
                    "done_reason": "load" if loaded else "unload",
                }
            )

        async with server(handler) as url:
            backend = replace(
                BackendConfig.create(BackendKind.OLLAMA, "Ollama", url),
                ollama_idle_unload_minutes=idle_minutes,
            ).register_model("catalog-model")
            runtime = runtime_for_backend(tmp_path, backend)
            await runtime.refresh_backend(backend.id)
            assert not runtime.inventory(backend.id)[0].is_loaded
            model = ModelRef(backend.id, "catalog-model")
            await runtime.load_model(model)
            assert actions[-1]["keep_alive"] == keep_alive
            await runtime.refresh_backend(backend.id)
            assert runtime.inventory(backend.id)[0].loaded_instance_ids == (
                "catalog-model",
            )
            await runtime.unload_model(model)
            assert actions[-1]["keep_alive"] == 0
            await runtime.refresh_backend(backend.id)
            assert not runtime.inventory(backend.id)[0].is_loaded
            assert await runtime.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("operation", ["load", "unload"])
def test_ollama_rejects_unconfirmed_model_management(operation: str) -> None:
    async def exercise() -> None:
        async def handler(req: web.Request) -> web.Response:
            return web.json_response(
                {"model": "catalog-model", "done": True, "done_reason": "unexpected"}
            )

        async with server(handler) as url:
            adapter = OllamaBackend(
                BackendConfig.create(BackendKind.OLLAMA, "Ollama", url)
            )
            with pytest.raises(BackendError, match="did not confirm"):
                if operation == "load":
                    _ = [event async for event in adapter.load_model("catalog-model")]
                else:
                    await adapter.unload_model("catalog-model")
            await adapter.close()

    asyncio.run(exercise())


def test_responses_runtime_replays_edits_regeneration_continuation_and_forks(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        payloads: list[dict[str, object]] = []

        async def handler(req: web.Request) -> web.Response:
            if req.method == "GET":
                return web.json_response(
                    {
                        "models": [
                            {
                                "type": "llm",
                                "key": "catalog-model",
                                "display_name": "Catalog",
                                "loaded_instances": [{"id": "instance-1"}],
                            }
                        ]
                    }
                )
            assert req.path == "/v1/responses"
            payloads.append(json_object(cast(object, await req.json())))
            response = {
                "id": "response-1",
                "model": "instance-1",
                "truncation": "disabled",
            }
            count = len(payloads)
            if count == 8:
                response["truncation"] = "auto"
            chunks: list[dict[str, object]] = [
                {"type": "response.created", "response": response},
                {
                    "type": "response.reasoning_text.delta"
                    if count == 6
                    else "response.output_text.delta",
                    "delta": "Thinking only"
                    if count == 6
                    else "Partial"
                    if count == 7
                    else f"Reply {count}",
                },
            ]
            if count != 7:
                chunks.append(
                    {
                        "type": "response.completed",
                        "response": {
                            **response,
                            "truncation": "auto" if count == 9 else "disabled",
                            "status": "completed",
                            "output": [{"type": "reasoning"}]
                            if count == 6
                            else [
                                {
                                    "type": "message",
                                    "content": [
                                        {
                                            "type": "output_text",
                                            "text": f"Reply {count}",
                                        }
                                    ],
                                }
                            ],
                        },
                    }
                )
            return web.Response(
                text="".join(event(chunk) for chunk in chunks),
                content_type="text/event-stream",
            )

        async with server(handler) as url:
            backend = replace(
                config(url, native=True),
                defaults=GenerationProfile("catalog-model"),
                lmstudio_transport=LMStudioTransport.RESPONSES,
            ).register_model("catalog-model")
            runtime = runtime_for_backend(tmp_path, backend)
            chat = await runtime.create_chat()
            first = await runtime.send_message(
                chat.id, chat.revision, "first-id", "First"
            )
            assert first.job
            await first.job.wait()
            second = await runtime.send_message(
                chat.id, chat.revision, "second-id", "Second"
            )
            assert second.job
            await second.job.wait()
            regeneration = await runtime.generate(
                chat.id, chat.revision, mode=GenerationMode.REGENERATE
            )
            assert regeneration
            await regeneration.wait()
            assert chat.messages[-1].content == "Reply 3"
            messages = cast(list[dict[str, object]], payloads[-1]["input"])
            assert [
                (item["role"], item["content"])
                for item in messages
                if item["role"] != "system"
            ] == [("user", "First"), ("assistant", "Reply 1"), ("user", "Second")]
            continuation = await runtime.generate(
                chat.id, chat.revision, mode=GenerationMode.CONTINUE
            )
            assert continuation
            await continuation.wait()
            messages = cast(list[dict[str, object]], payloads[-1]["input"])
            assert messages[-2]["content"] == "Reply 3"
            assert messages[-1]["role"] == "user"
            first_id = chat.messages[0].id

            def edit_first(candidate: Chat) -> None:
                _ = candidate.update_message(first_id, "Edited")

            await runtime.edit_chat(
                chat.id,
                chat.revision,
                edit_first,
            )
            fork = await runtime.fork_chat(chat.id, chat.revision, chat.messages[2].id)
            forked = await runtime.generate(fork.id, fork.revision)
            assert forked
            await forked.wait()
            messages = cast(list[dict[str, object]], payloads[-1]["input"])
            assert [
                (item["role"], item["content"])
                for item in messages
                if item["role"] != "system"
            ] == [("user", "Edited"), ("assistant", "Reply 1"), ("user", "Second")]
            assert fork.messages[-1].content == "Reply 5"
            reasoning_only = await runtime.generate(
                fork.id, fork.revision, mode=GenerationMode.REGENERATE
            )
            assert reasoning_only
            await reasoning_only.wait()
            assert reasoning_only.phase is JobPhase.FAILED
            assert fork.messages[-1].content == "Reply 5"
            interrupted = await runtime.generate(
                fork.id, fork.revision, mode=GenerationMode.REGENERATE
            )
            assert interrupted
            await interrupted.wait()
            assert interrupted.phase is JobPhase.FAILED
            assert fork.messages[-1].content == "Partial"
            for _ in range(2):
                unsafe = await runtime.generate(
                    fork.id, fork.revision, mode=GenerationMode.REGENERATE
                )
                assert unsafe
                await unsafe.wait()
                assert unsafe.phase is JobPhase.FAILED
                assert (
                    unsafe.error
                    and unsafe.error.category is BackendErrorCategory.INCOMPATIBLE
                )
                assert fork.messages[-1].content == "Partial"
            assert all(payload["store"] is False for payload in payloads)
            assert await runtime.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("discovery", ["unsupported", "malformed", "unavailable"])
def test_manual_model_generation_is_independent_of_inventory_endpoint(
    tmp_path: Path, discovery: Literal["unsupported", "malformed", "unavailable"]
) -> None:
    async def exercise() -> None:
        calls: list[str] = []

        async def handler(req: web.Request) -> web.Response:
            calls.append(req.path)
            if req.path == "/v1/models":
                if discovery == "unsupported":
                    return web.Response(status=404)
                if discovery == "malformed":
                    return web.json_response({"unexpected": "schema"})
                return web.Response(status=503, text="SECRET echoed response")
            assert req.path == "/v1/chat/completions"
            payload = json_object(cast(object, await req.json()))
            assert payload["model"] == "catalog-model"
            return web.Response(
                text=event(chunk("Manual model reply")) + "data: [DONE]\n\n",
                content_type="text/event-stream",
            )

        async with server(handler) as url:
            backend = replace(
                config(url), defaults=GenerationProfile("catalog-model")
            ).register_model("catalog-model")
            runtime = runtime_for_backend(tmp_path, backend)
            chat = await runtime.create_chat()
            sent = await runtime.send_message(chat.id, 0, "manual", "Hello")
            assert sent.job
            await sent.job.wait()
            assert sent.job.phase is JobPhase.FINISHED
            assert chat.messages[-1].content == "Manual model reply"
            assert runtime.health(backend.id).state is BackendReachability.ONLINE
            error = runtime.discovery_error(backend.id)
            if discovery == "unsupported":
                assert error is None
            else:
                assert error and error.category is BackendErrorCategory.PROTOCOL
                assert "SECRET" not in str(error)
            assert calls == ["/v1/models", "/v1/chat/completions"]
            assert (
                runtime.configuration.backend(backend.id)
                .model("catalog-model")
                .manually_registered
            )
            saved = runtime.persistence.storage.load_all()
            assert len(saved) == 1 and saved[0].to_dict() == chat.to_dict()
            assert await runtime.shutdown()

    asyncio.run(exercise())


def test_unreachable_discovery_marks_backend_offline_without_attempting_inference(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        with socket.socket() as bound:
            bound.bind(("127.0.0.1", 0))
            # A bound socket without listen deterministically refuses connections.
            port = bound.getsockname()[1]
            backend = replace(
                config(f"http://127.0.0.1:{port}/v1"),
                defaults=GenerationProfile("catalog-model"),
            ).register_model("catalog-model")
            runtime = runtime_for_backend(tmp_path, backend)
            chat = await runtime.create_chat()
            with patch.object(
                ChatCompletionsBackend,
                "stream",
                side_effect=AssertionError("Must not infer offline"),
            ) as streaming:
                sent = await runtime.send_message(
                    chat.id, 0, "offline", "Saved offline"
                )
                assert sent.job
                await sent.job.wait()
                streaming.assert_not_called()
            assert sent.job.phase is JobPhase.FAILED
            assert (
                sent.job.error
                and sent.job.error.category is BackendErrorCategory.OFFLINE
            )
            assert runtime.health(backend.id).state is BackendReachability.OFFLINE
            error = runtime.discovery_error(backend.id)
            assert error and error.category is BackendErrorCategory.OFFLINE
            assert len(runtime.persistence.storage.load_all()[0].messages) == 1
            assert await runtime.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("vision", [None, False])
def test_discovery_failure_does_not_bypass_fail_closed_vision(
    tmp_path: Path, vision: bool | None
) -> None:
    async def exercise() -> None:
        posts = 0

        async def handler(req: web.Request) -> web.Response:
            nonlocal posts
            if req.method == "POST":
                posts += 1
            return web.json_response({"malformed": "inventory"})

        async with server(handler) as url:
            backend = replace(
                config(url), defaults=GenerationProfile("catalog-model")
            ).register_model("catalog-model")
            registered = backend.model("catalog-model")
            backend = replace(
                backend,
                models=(
                    replace(
                        registered,
                        descriptor=replace(
                            registered.descriptor,
                            capabilities=ModelCapabilities(vision=vision),
                        ),
                    ),
                ),
            )
            runtime = runtime_for_backend(tmp_path, backend)
            with pytest.raises(BackendError):
                await runtime.refresh_backend(backend.id)
            chat = await runtime.create_chat()
            image = ImageAttachment(
                "a" * 64, "original.png", ImageMediaType.PNG, 1, 1, 1
            )
            sent = await runtime.send_message(
                chat.id, 0, "vision", "Look", images=(image,)
            )
            assert sent.job is None
            assert posts == 0
            assert chat.messages[0].images == (image,)
            error = runtime.registry.get(chat.id).latest_error
            assert error and error.category is BackendErrorCategory.INCOMPATIBLE
            assert runtime.discovery_error(backend.id) is not None
            assert await runtime.shutdown()

    asyncio.run(exercise())


def test_discovery_timeout_remains_visible_while_manual_inference_can_succeed(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        async def handler(req: web.Request) -> web.Response:
            assert req.path == "/v1/chat/completions"
            return web.Response(
                text=event(chunk("Reply")) + "data: [DONE]\n\n",
                content_type="text/event-stream",
            )

        async with server(handler) as url:
            backend = replace(
                config(url), defaults=GenerationProfile("catalog-model")
            ).register_model("catalog-model")
            runtime = runtime_for_backend(tmp_path, backend)
            chat = await runtime.create_chat()
            with patch.object(
                aiohttp.ClientSession,
                "request",
                side_effect=TimeoutError("inventory timeout"),
            ):
                with pytest.raises(BackendError):
                    await runtime.refresh_backend(backend.id)
                assert runtime.health(backend.id).state is BackendReachability.STALE
                job = await runtime.send_message(chat.id, 0, "timeout", "Hello")
                assert job.job
                await job.job.wait()
            assert job.job.phase is JobPhase.FINISHED
            assert chat.messages[-1].content == "Reply"
            error = runtime.discovery_error(backend.id)
            assert error and error.category is BackendErrorCategory.TIMEOUT
            assert await runtime.shutdown()

    asyncio.run(exercise())
