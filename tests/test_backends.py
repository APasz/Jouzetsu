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

from jouzetsu.backend_models import (
    BackendConfig,
    BackendConfiguration,
    BackendKind,
    GenerationProfile,
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
from jouzetsu.inference import InferenceRuntime, JobPhase
from jouzetsu.models import ImageAttachment, ImageMediaType
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


def request(backend: BackendConfig) -> GenerationRequest:
    return GenerationRequest(
        ModelRef(backend.id, "catalog-model"),
        (InferenceMessage("user", "first"), InferenceMessage("user", "second")),
        ModelCapabilities(),
        CHAT_COMPLETIONS_PARAMETERS,
    )


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
