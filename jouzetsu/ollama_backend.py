"""Ollama native discovery and streaming chat transport."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncGenerator
from dataclasses import replace
from typing import cast, override

import aiohttp

from .async_workers import run_in_worker
from .backend_models import BackendConfig, validate_model_id
from .backend_protocol import (
    BackendError,
    BackendErrorCategory,
    GenerationRequest,
    ModelDiscovery,
    ModelManagement,
)
from .backends import ChatCompletionsBackend
from .chat_completions import (
    CONNECT_TIMEOUT,
    LOAD_TIMEOUT,
    STREAM_IDLE_TIMEOUT,
    UNLOAD_TIMEOUT,
    build_messages,
    encode_payload,
    json_object,
    protocol_error,
    token_count,
)
from .image_storage import ImageStorage
from .runtime import (
    REASONING_OPTIONS,
    ChatStreamEvent,
    FirstToken,
    GenerationMetrics,
    ModelCapabilities,
    ModelDescriptor,
    ModelLoadProgress,
    ModelReady,
    PredictionComplete,
    PredictionFragment,
    ReasoningOption,
)

_MINIMUM_OLLAMA_VERSION = (0, 14, 0)


def _validate_version(payload: dict[str, object]) -> None:
    """Require a release known to implement both context-preservation flags."""
    version = payload.get("version")
    match = (
        re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version)
        if isinstance(version, str)
        else None
    )
    if match is None or tuple(map(int, match.groups())) < _MINIMUM_OLLAMA_VERSION:
        raise BackendError(
            BackendErrorCategory.INCOMPATIBLE,
            "Ollama 0.14.0 or later is required to preserve conversation history.",
        )


def _reasoning(value: object) -> ReasoningOption | None:
    if value is False:
        return "off"
    if value is True:
        return "on"
    if isinstance(value, str) and value in REASONING_OPTIONS[2:]:
        return value
    return None


def _descriptor(name: str, details: dict[str, object], size: object) -> ModelDescriptor:
    raw_caps = details.get("capabilities")
    if not isinstance(raw_caps, list) or any(
        not isinstance(v, str) for v in cast(list[object], raw_caps)
    ):
        raise protocol_error("Ollama returned invalid model capabilities.")
    caps = cast(list[str], raw_caps)
    thinking = details.get("thinking")
    options: tuple[ReasoningOption, ...] = ()
    default: ReasoningOption | None = None
    if thinking is not None:
        info = json_object(thinking)
        values = info.get("values")
        if not isinstance(values, list):
            raise protocol_error("Ollama returned invalid reasoning options.")
        options = tuple(
            dict.fromkeys(
                option
                for value in cast(list[object], values)
                if (option := _reasoning(value)) is not None
            )
        )
        default = _reasoning(info.get("default"))
        if default not in options:
            default = None
    has_reasoning = "thinking" in caps
    if not has_reasoning:
        options = ()
        default = None
    return ModelDescriptor(
        name,
        name,
        size_bytes=size if type(size) is int and size > 0 else None,
        capabilities=ModelCapabilities(
            vision="vision" in caps,
            trained_for_tool_use="tools" in caps,
            reasoning=has_reasoning,
            reasoning_options=options,
            reasoning_default=default,
        ),
    )


def _encode_request(
    request: GenerationRequest,
    images: ImageStorage | None,
    idle_unload_minutes: int | None,
) -> bytes:
    messages: list[dict[str, object]] = []
    for message in build_messages(request.messages, images):
        content = message["content"]
        if isinstance(content, str):
            messages.append({"role": message["role"], "content": content})
        else:
            text = "".join(part["text"] for part in content if part["type"] == "text")
            image_data = [
                part["image_url"]["url"].split(",", 1)[1]
                for part in content
                if part["type"] == "image_url"
            ]
            messages.append(
                {"role": message["role"], "content": text, "images": image_data}
            )
    parameters: dict[str, object] = {
        name: value for name, value in request.wire_parameters.items()
    }
    if "max_tokens" in parameters:
        parameters["num_predict"] = parameters.pop("max_tokens")
    payload: dict[str, object] = {
        "model": request.model.model_id,
        "messages": messages,
        "options": parameters,
        "stream": True,
        "truncate": False,
        "shift": False,
    }
    if request.reasoning is not None:
        payload["think"] = {"off": False, "on": True}.get(
            request.reasoning, request.reasoning
        )
    payload["keep_alive"] = _keep_alive(idle_unload_minutes)
    return encode_payload(payload)


def _keep_alive(minutes: int | None) -> str | int:
    return f"{minutes}m" if minutes is not None else -1


class OllamaBackend(ChatCompletionsBackend):
    """Use Ollama's native API for capabilities, reasoning and full history."""

    def __init__(
        self, config: BackendConfig, *, images: ImageStorage | None = None
    ) -> None:
        super().__init__(config, images=images)
        self._root = config.base_url.rstrip("/")
        self._idle_unload_minutes = config.ollama_idle_unload_minutes

    @property
    @override
    def model_management(self) -> ModelManagement:
        return self

    @override
    async def discover_models(self) -> ModelDiscovery:
        payload = await self._transport.json_request("GET", f"{self._root}/api/tags")
        assert payload is not None
        raw_models = payload.get("models")
        if not isinstance(raw_models, list):
            raise protocol_error("Ollama returned an invalid model inventory.")
        running = await self._transport.json_request("GET", f"{self._root}/api/ps")
        assert running is not None
        raw_running = running.get("models")
        if not isinstance(raw_running, list):
            raise protocol_error("Ollama returned invalid running models.")
        running_names: set[str] = set()
        for raw in cast(list[object], raw_running):
            name = json_object(raw).get("name")
            if not isinstance(name, str) or not name.strip() or name != name.strip():
                raise protocol_error("Ollama returned an invalid running model id.")
            running_names.add(name)
        models: list[ModelDescriptor] = []
        for raw in cast(list[object], raw_models):
            item = json_object(raw)
            name = item.get("name")
            if not isinstance(name, str):
                raise protocol_error("Ollama returned an invalid model id.")
            try:
                validate_model_id(name)
            except ValueError as exc:
                raise protocol_error("Ollama returned an invalid model id.") from exc
            details = await self._transport.json_request(
                "POST", f"{self._root}/api/show", payload={"model": name}
            )
            assert details is not None
            descriptor = _descriptor(name, details, item.get("size"))
            if name in running_names:
                descriptor = replace(descriptor, loaded_instance_ids=(name,))
            models.append(descriptor)
        try:
            return ModelDiscovery(True, tuple(models))
        except ValueError as exc:
            raise protocol_error("Ollama returned duplicate model ids.") from exc

    async def load_model(
        self, model_id: str
    ) -> AsyncGenerator[ModelLoadProgress | ModelReady]:
        validate_model_id(model_id)
        yield ModelLoadProgress(None)
        response = await self._transport.json_request(
            "POST",
            f"{self._root}/api/generate",
            payload={
                "model": model_id,
                "stream": False,
                "keep_alive": _keep_alive(self._idle_unload_minutes),
            },
            timeout=LOAD_TIMEOUT,
            failure=BackendErrorCategory.LOAD_FAILURE,
        )
        if (
            response is None
            or response.get("done") is not True
            or response.get("model") != model_id
            or response.get("done_reason") != "load"
        ):
            raise BackendError(
                BackendErrorCategory.LOAD_FAILURE,
                "Ollama did not confirm the model was loaded.",
            )
        yield ModelReady(model_id, model_id, 0)

    async def unload_model(self, instance_id: str) -> None:
        validate_model_id(instance_id)
        response = await self._transport.json_request(
            "POST",
            f"{self._root}/api/generate",
            payload={"model": instance_id, "stream": False, "keep_alive": 0},
            timeout=UNLOAD_TIMEOUT,
        )
        if (
            response is None
            or response.get("done") is not True
            or response.get("model") != instance_id
            or response.get("done_reason") != "unload"
        ):
            raise protocol_error("Ollama did not confirm the model was unloaded.")

    @override
    async def stream(
        self, request: GenerationRequest
    ) -> AsyncGenerator[ChatStreamEvent]:
        request.validate_backend(self.id)
        version = await self._transport.json_request("GET", f"{self._root}/api/version")
        assert version is not None
        _validate_version(version)
        payload = await run_in_worker(
            _encode_request, request, self._transport.images, self._idle_unload_minutes
        )
        session = await self._transport.session()
        started = asyncio.get_running_loop().time()
        first: float | None = None
        completed = False
        try:
            async with session.post(
                f"{self._root}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                allow_redirects=False,
                timeout=aiohttp.ClientTimeout(
                    total=None,
                    sock_connect=CONNECT_TIMEOUT,
                    sock_read=STREAM_IDLE_TIMEOUT,
                ),
            ) as response:
                if response.status != 200:
                    raise BackendError(
                        BackendErrorCategory.INCOMPATIBLE
                        if response.status in (400, 404, 422)
                        else BackendErrorCategory.PROTOCOL,
                        f"Ollama rejected generation (HTTP {response.status}); check the model and context limit.",
                    )
                yield ModelReady(request.model.model_id, request.model.model_id, 0)
                async for line in response.content:
                    try:
                        chunk = json_object(cast(object, json.loads(line)))
                    except (ValueError, TypeError) as exc:
                        raise protocol_error() from exc
                    if "error" in chunk:
                        raise protocol_error("Ollama reported a streaming error.")
                    if chunk.get("model") != request.model.model_id:
                        raise protocol_error(
                            "Ollama changed model identity during generation."
                        )
                    message = json_object(chunk.get("message"))
                    if message.get("role") not in (None, "assistant"):
                        raise protocol_error(
                            "Ollama returned an invalid completion role."
                        )
                    if message.get("tool_calls") or message.get("images"):
                        raise BackendError(
                            BackendErrorCategory.INCOMPATIBLE,
                            "Ollama returned unsupported non-text output.",
                        )
                    for field, is_reasoning in (("thinking", True), ("content", False)):
                        value = message.get(field, "")
                        if not isinstance(value, str):
                            raise protocol_error()
                        if value:
                            if first is None:
                                first = asyncio.get_running_loop().time()
                                yield FirstToken()
                            yield PredictionFragment(value, None, is_reasoning)
                    done = chunk.get("done")
                    if type(done) is not bool:
                        raise protocol_error()
                    if done:
                        reason = chunk.get("done_reason")
                        if reason is not None and not isinstance(reason, str):
                            raise protocol_error()
                        count = token_count(chunk, "eval_count")
                        duration = token_count(chunk, "eval_duration")
                        yield PredictionComplete(
                            GenerationMetrics(
                                output_tokens=count,
                                prompt_tokens=token_count(chunk, "prompt_eval_count"),
                                tokens_per_second=count * 1_000_000_000 / duration
                                if count is not None and duration
                                else None,
                                time_to_first_token_seconds=first - started
                                if first is not None
                                else None,
                                stop_reason=reason,
                            )
                        )
                        completed = True
                        break
            if not completed:
                raise protocol_error(
                    "Ollama disconnected before completing its response."
                )
        except TimeoutError as exc:
            raise BackendError(
                BackendErrorCategory.TIMEOUT, "Ollama stopped responding."
            ) from exc
        except aiohttp.ClientConnectionError as exc:
            raise BackendError(
                BackendErrorCategory.OFFLINE, "The Ollama connection was lost."
            ) from exc
        except (aiohttp.ClientError, ValueError) as exc:
            raise protocol_error() from exc
