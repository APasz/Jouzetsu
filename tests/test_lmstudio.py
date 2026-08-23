from __future__ import annotations

import json
import unittest
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Self, cast, override

import aiohttp

from jouzetsu.config import GenerationSettings, ServerSettings
from jouzetsu.lmstudio import BRITISH_ENGLISH_INSTRUCTION, LMStudioClient
from jouzetsu.models import Chat
from jouzetsu.runtime import PredictionComplete, PredictionFragment


class RecordingResponse:
    def __init__(self, status: int, body: str) -> None:
        self.status: int = status
        self._body: str = body

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        _ = (exc_type, exc, traceback)

    async def text(self) -> str:
        return self._body


class _SseEventContent:
    def __init__(self, data: bytes) -> None:
        self._data: bytes = data

    async def iter_chunked(self, _chunk_size: int) -> AsyncIterator[bytes]:
        yield self._data


class StreamingRecordingResponse(RecordingResponse):
    def __init__(self, stream_data: bytes) -> None:
        super().__init__(200, "")
        self.content: _SseEventContent = _SseEventContent(stream_data)


class RecordingSession:
    def __init__(self) -> None:
        self.get_body: str = '{"models": []}'
        self.get_urls: list[str] = []
        self.posts: list[dict[str, object]] = []
        self.post_urls: list[str] = []
        self.get_timeouts: list[aiohttp.ClientTimeout] = []
        self.post_timeouts: list[aiohttp.ClientTimeout] = []
        self.stream_data: bytes = b"data: [DONE]\n\n"

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        timeout: aiohttp.ClientTimeout,
    ) -> RecordingResponse:
        _ = headers
        self.get_urls.append(url)
        self.get_timeouts.append(timeout)
        return RecordingResponse(200, self.get_body)

    def post(
        self,
        url: str,
        *,
        json: dict[str, object],
        headers: dict[str, str],
        timeout: aiohttp.ClientTimeout,
    ) -> RecordingResponse:
        _ = headers
        self.post_urls.append(url)
        self.posts.append(dict(json))
        self.post_timeouts.append(timeout)
        if json.get("stream") is True:
            return StreamingRecordingResponse(self.stream_data)
        return RecordingResponse(200, "{}")


class RecordingLMStudioClient(LMStudioClient):
    def __init__(self, server: ServerSettings) -> None:
        super().__init__(server)
        self.recording_session: RecordingSession = RecordingSession()

    @override
    async def _get_session(self) -> aiohttp.ClientSession:
        return cast(aiohttp.ClientSession, cast(object, self.recording_session))

    async def load_demo_model_for_test(self) -> None:
        await self._load_model("demo-model")


class LMStudioClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_load_model_omits_ttl_even_when_auto_unload_is_configured(
        self,
    ) -> None:
        """The native endpoint does not accept an idle-TTL load option."""
        client = RecordingLMStudioClient(ServerSettings(auto_unload_minutes=5))

        await client.load_demo_model_for_test()

        self.assertEqual(
            client.recording_session.posts,
            [{"model": "demo-model", "echo_load_config": True}],
        )

    async def test_configured_auto_unload_uses_jit_loading_with_ttl(self) -> None:
        client = RecordingLMStudioClient(ServerSettings(auto_unload_minutes=5))
        client.recording_session.get_body = json.dumps(
            {
                "models": [
                    {
                        "type": "llm",
                        "key": "demo-model",
                        "display_name": "Demo Model",
                        "loaded_instances": [],
                    }
                ]
            }
        )
        chat = Chat()
        _ = chat.add_message("user", "Hello")

        _ = [
            event
            async for event in client.stream_chat(
                chat, "demo-model", GenerationSettings()
            )
        ]

        self.assertEqual(len(client.recording_session.posts), 1)
        self.assertEqual(client.recording_session.posts[0]["model"], "demo-model")
        self.assertEqual(client.recording_session.posts[0]["ttl"], 300)
        self.assertTrue(client.recording_session.posts[0]["stream"])

    async def test_unload_model_posts_instance_id(self) -> None:
        client = RecordingLMStudioClient(ServerSettings())

        await client.unload_model_instance("demo-instance")

        self.assertEqual(
            client.recording_session.posts, [{"instance_id": "demo-instance"}]
        )

    async def test_native_api_root_tracks_the_shared_server_settings(self) -> None:
        server = ServerSettings(base_url="http://first.example:1234/v1")
        client = RecordingLMStudioClient(server)

        _ = await client.list_model_inventory()
        server.base_url = "http://second.example:4321/v1"
        await client.unload_model_instance("demo-instance")

        self.assertEqual(
            client.recording_session.get_urls,
            ["http://first.example:1234/api/v1/models"],
        )
        self.assertEqual(
            client.recording_session.post_urls,
            ["http://second.example:4321/api/v1/models/unload"],
        )

    async def test_management_requests_use_bounded_timeouts(self) -> None:
        client = RecordingLMStudioClient(ServerSettings())

        _ = await client.list_model_inventory()
        await client.unload_model_instance("demo-instance")

        self.assertEqual(client.recording_session.get_timeouts[0].total, 15.0)
        self.assertEqual(client.recording_session.get_timeouts[0].sock_connect, 10.0)
        self.assertEqual(client.recording_session.post_timeouts[0].total, 60.0)
        self.assertEqual(client.recording_session.post_timeouts[0].sock_connect, 10.0)

    async def test_streaming_request_uses_idle_and_connection_timeouts(self) -> None:
        client = RecordingLMStudioClient(ServerSettings())
        chat = Chat()
        _ = chat.add_message("user", "Hello")

        _ = [
            event
            async for event in client._stream_completion(  # pyright: ignore[reportPrivateUsage]
                chat, "demo-model", GenerationSettings()
            )
        ]

        timeout = client.recording_session.post_timeouts[0]
        self.assertIsNone(timeout.total)
        self.assertEqual(timeout.sock_connect, 10.0)
        self.assertEqual(timeout.sock_read, 120.0)

    async def test_streaming_preserves_visible_content_when_a_chunk_also_has_reasoning(
        self,
    ) -> None:
        client = RecordingLMStudioClient(ServerSettings())
        client.recording_session.stream_data = (
            b'data: {"choices":[{"delta":{"reasoning_content":"private thought","content":"Visible answer"}}]}\n\n'
            b"data: [DONE]\n\n"
        )
        chat = Chat()
        _ = chat.add_message("user", "Hello")

        events = [
            event
            async for event in client._stream_completion(  # pyright: ignore[reportPrivateUsage]
                chat, "demo-model", GenerationSettings()
            )
        ]
        fragments = [event for event in events if isinstance(event, PredictionFragment)]

        self.assertEqual(
            [(fragment.content, fragment.is_reasoning) for fragment in fragments],
            [("private thought", True), ("Visible answer", False)],
        )
        self.assertTrue(all(fragment.token_count is None for fragment in fragments))

    async def test_streaming_uses_lm_studio_usage_for_exact_token_counts(self) -> None:
        client = RecordingLMStudioClient(ServerSettings())
        client.recording_session.stream_data = (
            b'data: {"choices":[{"delta":{"content":"One fragment"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":" and another"}}]}\n\n'
            b'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3}}\n\n'
            b"data: [DONE]\n\n"
        )
        chat = Chat()
        _ = chat.add_message("user", "Hello")

        events = [
            event
            async for event in client._stream_completion(  # pyright: ignore[reportPrivateUsage]
                chat, "demo-model", GenerationSettings()
            )
        ]
        fragments = [event for event in events if isinstance(event, PredictionFragment)]
        completed = next(
            event for event in events if isinstance(event, PredictionComplete)
        )

        self.assertEqual([fragment.token_count for fragment in fragments], [None, None])
        self.assertEqual(completed.metrics.prompt_tokens, 7)
        self.assertEqual(completed.metrics.output_tokens, 3)

    async def test_build_messages_appends_british_english_instruction(self) -> None:
        chat = Chat()
        _ = chat.add_message("user", "Hello")
        generation = GenerationSettings(
            system_prompt="Answer tersely.", british_english=True
        )

        messages = LMStudioClient._build_messages(chat, generation)  # pyright: ignore[reportPrivateUsage]

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(
            messages[0]["content"], f"Answer tersely.\n\n{BRITISH_ENGLISH_INSTRUCTION}"
        )
        self.assertEqual(messages[1], {"role": "user", "content": "Hello"})

    async def test_list_model_inventory_parses_instance_and_capabilities(self) -> None:
        client = RecordingLMStudioClient(ServerSettings())
        client.recording_session.get_body = json.dumps(
            {
                "models": [
                    {
                        "type": "llm",
                        "publisher": "google",
                        "key": "google/gemma",
                        "display_name": "Gemma",
                        "architecture": "gemma",
                        "quantization": {"name": "Q4_K_M", "bits_per_weight": 4},
                        "size_bytes": 123456789,
                        "params_string": "4B",
                        "loaded_instances": [
                            {
                                "id": "gemma-instance",
                                "config": {
                                    "context_length": 4096,
                                    "eval_batch_size": 512,
                                    "parallel": 2,
                                    "flash_attention": True,
                                    "num_experts": 8,
                                    "offload_kv_cache_to_gpu": False,
                                },
                            }
                        ],
                        "max_context_length": 8192,
                        "format": "gguf",
                        "capabilities": {
                            "vision": True,
                            "trained_for_tool_use": True,
                            "reasoning": {
                                "allowed_options": ["off", "on"],
                                "default": "on",
                            },
                        },
                        "description": "Demo description",
                        "variants": ["google/gemma@q4_k_m"],
                        "selected_variant": "google/gemma@q4_k_m",
                    }
                ]
            }
        )

        inventory = await client.list_model_inventory()

        self.assertEqual(len(inventory), 1)
        model = inventory[0]
        self.assertEqual(model.publisher, "google")
        self.assertEqual(model.quantization_name, "Q4_K_M")
        self.assertEqual(model.bits_per_weight, 4.0)
        self.assertEqual(model.loaded_instance_ids, ("gemma-instance",))
        self.assertEqual(model.loaded_instances[0].context_length, 4096)
        self.assertTrue(model.loaded_instances[0].flash_attention)
        self.assertTrue(model.capabilities.vision)
        self.assertTrue(model.capabilities.trained_for_tool_use)
        self.assertTrue(model.capabilities.reasoning)
        self.assertEqual(model.capabilities.reasoning_default, "on")
