"""Typed async adapter for LM Studio's native and OpenAI-compatible APIs."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import AsyncIterator
from logging import Logger
from typing import Protocol, cast
from urllib.parse import SplitResult, urlsplit, urlunsplit

import aiohttp
from aiohttp.client import ClientSession

from .async_workers import run_in_worker
from .backend_protocol import BackendError, InferenceMessage
from .chat_completions import (
    CONNECT_TIMEOUT,
    DISCOVERY_TIMEOUT,
    LOAD_TIMEOUT,
    STREAM_IDLE_TIMEOUT,
    UNLOAD_TIMEOUT,
    WireMessage,
    build_messages,
    completion_events,
    iter_sse_data,
)
from .config import GenerationSettings, ServerSettings
from .image_storage import ImageStorage
from .models import Chat
from .runtime import (
    MODEL_FORMATS,
    REASONING_OPTIONS,
    ChatStreamEvent,
    ModelCapabilities,
    ModelDescriptor,
    ModelFormat,
    ModelInstanceDescriptor,
    ModelLoadProgress,
    ModelReady,
    PromptProcessingProgress,
    ReasoningOption,
    vision_model_error,
)

log: Logger = logging.getLogger(__name__)
BRITISH_ENGLISH_INSTRUCTION: str = (
    "Use British English spelling and conventions in assistant responses, "
    "unless the user explicitly asks for another dialect."
)
_CONNECT_TIMEOUT_SECONDS: float = CONNECT_TIMEOUT
_INVENTORY_TIMEOUT_SECONDS: float = DISCOVERY_TIMEOUT
_MODEL_UNLOAD_TIMEOUT_SECONDS: float = UNLOAD_TIMEOUT
_MODEL_LOAD_TIMEOUT_SECONDS: float = LOAD_TIMEOUT
_STREAM_IDLE_TIMEOUT_SECONDS: float = STREAM_IDLE_TIMEOUT


class LMStudioError(RuntimeError):
    """Raised when LM Studio rejects a request or is unreachable."""


class LMStudioClientProtocol(Protocol):
    """Operations required by application state and model selection UI."""

    async def list_model_inventory(self) -> tuple[ModelDescriptor, ...]: ...

    async def list_models(self) -> list[str]: ...

    async def unload_model_instance(self, instance_id: str) -> None: ...

    def stream_chat(
        self,
        chat: Chat,
        model: str,
        generation: GenerationSettings,
    ) -> AsyncIterator[ChatStreamEvent]: ...

    async def close(self) -> None: ...


def _parse_json(raw: str) -> object:
    """Contain the untyped standard-library JSON boundary."""
    return cast(object, json.loads(raw))


def _as_object_mapping(raw: object) -> dict[str, object] | None:
    """Return a string-keyed object after validating an external JSON value."""
    if not isinstance(raw, dict):
        return None
    mapping: dict[object, object] = cast(dict[object, object], raw)
    if not all(isinstance(key, str) for key in mapping):
        return None
    return {cast(str, key): value for key, value in mapping.items()}


def _optional_int(mapping: dict[str, object] | None, key: str) -> int | None:
    if mapping is None:
        return None
    value: object | None = mapping.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_bool(mapping: dict[str, object] | None, key: str) -> bool | None:
    if mapping is None:
        return None
    value: object | None = mapping.get(key)
    return value if isinstance(value, bool) else None


def lmstudio_server_root(base_url: str) -> str:
    """Remove a configured compatibility API suffix from a server URL."""
    parsed: SplitResult = urlsplit(base_url)
    path: str = parsed.path.rstrip("/")
    for suffix in ("/api/v1", "/v1"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", "")).rstrip("/")


class LMStudioClient:
    """LM Studio model management and full-history chat streaming."""

    def __init__(
        self, server: ServerSettings, *, image_storage: ImageStorage | None = None
    ) -> None:
        self._server: ServerSettings = server
        self._images: ImageStorage | None = image_storage
        self._session: aiohttp.ClientSession | None = None

    @property
    def _server_root(self) -> str:
        """Resolve the native API root from the current shared server settings."""

        return lmstudio_server_root(self._server.base_url)

    async def list_model_inventory(self) -> tuple[ModelDescriptor, ...]:
        """Return downloaded LLMs and their currently loaded instances."""
        url: str = f"{self._server_root}/api/v1/models"
        try:
            session: ClientSession = await self._get_session()
            async with session.get(
                url,
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(
                    total=_INVENTORY_TIMEOUT_SECONDS,
                    sock_connect=_CONNECT_TIMEOUT_SECONDS,
                ),
            ) as response:
                body: str = await response.text()
                if response.status != 200:
                    raise LMStudioError(f"Model query returned HTTP {response.status}")
                payload: dict[str, object] | None = _as_object_mapping(
                    _parse_json(body)
                )
        except LMStudioError as exc:
            log.warning("LM Studio model inventory request failed error=%s", exc)
            raise
        except Exception as exc:
            log.warning("LM Studio model inventory request failed error=%s", exc)
            raise LMStudioError(f"Could not query models: {exc}") from exc

        return self.parse_model_inventory(payload)

    @classmethod
    def parse_model_inventory(
        cls, payload: dict[str, object] | None
    ) -> tuple[ModelDescriptor, ...]:
        """Decode native inventory metadata independently of HTTP transport."""

        if payload is None:
            raise LMStudioError("Model query returned an invalid response")
        raw_models: object | None = payload.get("models")
        if not isinstance(raw_models, list):
            raise LMStudioError("Model query returned an invalid response")

        inventory: list[ModelDescriptor] = []
        for raw_model in cast(list[object], raw_models):
            model: dict[str, object] | None = _as_object_mapping(raw_model)
            if model is None or not isinstance(model.get("type"), str):
                raise LMStudioError("Model inventory contains an invalid entry")
            if model.get("type") != "llm":
                continue
            key: object | None = model.get("key")
            display_name: object | None = model.get("display_name")
            if (
                not isinstance(key, str)
                or not key.strip()
                or key != key.strip()
                or not isinstance(display_name, str)
            ):
                raise LMStudioError(
                    "Model inventory contains an invalid model identity"
                )
            raw_instances: object | None = model.get("loaded_instances")
            if raw_instances is not None and not isinstance(raw_instances, list):
                raise LMStudioError("Model inventory contains invalid instances")
            instances: list[object] = (
                cast(list[object], raw_instances)
                if isinstance(raw_instances, list)
                else []
            )
            loaded_instances: list[ModelInstanceDescriptor] = []
            for raw_instance in instances:
                instance: dict[str, object] | None = _as_object_mapping(raw_instance)
                if instance is None:
                    raise LMStudioError("Model inventory contains an invalid instance")
                instance_id: object | None = instance.get("id")
                if (
                    not isinstance(instance_id, str)
                    or not instance_id.strip()
                    or instance_id != instance_id.strip()
                ):
                    raise LMStudioError(
                        "Model inventory contains an invalid instance identity"
                    )
                loaded_instances.append(
                    cls._model_instance_descriptor(instance_id, instance)
                )
            context_length: int | None = cls._first_context_length(instances)
            max_context_length: object | None = model.get("max_context_length")
            publisher: object | None = model.get("publisher")
            architecture: object | None = model.get("architecture")
            quantization: dict[str, object] | None = _as_object_mapping(
                model.get("quantization")
            )
            quantization_name: object | None = (
                quantization.get("name") if quantization is not None else None
            )
            bits_per_weight: object | None = (
                quantization.get("bits_per_weight")
                if quantization is not None
                else None
            )
            size_bytes: object | None = model.get("size_bytes")
            params_string: object | None = model.get("params_string")
            model_format: object | None = model.get("format")
            description: object | None = model.get("description")
            raw_variants: object | None = model.get("variants")
            variants: list[str] = (
                [
                    item
                    for item in cast(list[object], raw_variants)
                    if isinstance(item, str)
                ]
                if isinstance(raw_variants, list)
                else []
            )
            selected_variant: object | None = model.get("selected_variant")
            inventory.append(
                ModelDescriptor(
                    key=key,
                    display_name=display_name,
                    loaded_instance_ids=tuple(
                        instance.id for instance in loaded_instances
                    ),
                    loaded_instances=tuple[ModelInstanceDescriptor, ...](
                        loaded_instances
                    ),
                    context_length=context_length,
                    max_context_length=max_context_length
                    if isinstance(max_context_length, int)
                    else None,
                    auto_unload_minutes=cls._first_auto_unload_minutes(
                        model, instances
                    ),
                    publisher=publisher if isinstance(publisher, str) else "",
                    architecture=architecture if isinstance(architecture, str) else "",
                    quantization_name=quantization_name
                    if isinstance(quantization_name, str)
                    else "",
                    bits_per_weight=float(bits_per_weight)
                    if isinstance(bits_per_weight, int | float)
                    and not isinstance(bits_per_weight, bool)
                    else None,
                    size_bytes=size_bytes if isinstance(size_bytes, int) else None,
                    params_string=params_string
                    if isinstance(params_string, str)
                    else "",
                    format=cls._model_format(model_format),
                    capabilities=cls._model_capabilities(model),
                    description=description if isinstance(description, str) else "",
                    variants=tuple[str, ...](variants),
                    selected_variant=selected_variant
                    if isinstance(selected_variant, str)
                    else "",
                )
            )
        log.debug(
            "retrieved LM Studio model inventory model_count=%d loaded_instance_count=%d",
            len(inventory),
            sum(len(model.loaded_instance_ids) for model in inventory),
        )
        return tuple[ModelDescriptor, ...](inventory)

    async def list_models(self) -> list[str]:
        """Return downloaded LLM keys suitable for the model selector."""
        return [model.key for model in await self.list_model_inventory()]

    async def unload_model_instance(self, instance_id: str) -> None:
        """Unload one loaded LM Studio model instance from memory."""
        url: str = f"{self._server_root}/api/v1/models/unload"
        payload: dict[str, object] = {"instance_id": instance_id}
        try:
            session: ClientSession = await self._get_session()
            async with session.post(
                url,
                json=payload,
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(
                    total=_MODEL_UNLOAD_TIMEOUT_SECONDS,
                    sock_connect=_CONNECT_TIMEOUT_SECONDS,
                ),
            ) as response:
                if response.status != 200:
                    raise LMStudioError(f"Model unload returned HTTP {response.status}")
        except LMStudioError as exc:
            log.warning(
                "LM Studio model unload failed instance_id=%s error=%s",
                instance_id,
                exc,
            )
            raise
        except Exception as exc:
            log.warning(
                "LM Studio model unload failed instance_id=%s error=%s",
                instance_id,
                exc,
            )
            raise LMStudioError(f"Could not unload model: {exc}") from exc
        log.info("LM Studio model instance unloaded instance_id=%s", instance_id)

    async def stream_chat(
        self,
        chat: Chat,
        model: str,
        generation: GenerationSettings,
    ) -> AsyncIterator[ChatStreamEvent]:
        """Yield loading, prompt, text/reasoning, and completion events.

        When an idle TTL is configured, loading is deferred to the completion
        request. LM Studio applies a ``ttl`` only when that request JIT-loads a
        new model; its native load endpoint cannot configure the timer.
        """
        try:
            descriptor: ModelDescriptor = await self._resolve_model(model)
            if chat.has_images:
                reason: str = vision_model_error(descriptor)
                if reason:
                    raise LMStudioError(reason)
            ttl_seconds: int | None = self._ttl_seconds()
            if not descriptor.is_loaded:
                yield ModelLoadProgress(None)
                if ttl_seconds is None:
                    await self._load_model(descriptor.key)
                    descriptor = await self._require_loaded_model(descriptor.key)
                else:
                    log.info(
                        "deferring model load to LM Studio JIT so idle TTL can apply model_key=%s ttl_seconds=%d",
                        descriptor.key,
                        ttl_seconds,
                    )

            context_length: int = (
                descriptor.context_length or descriptor.max_context_length or 0
            )
            yield ModelReady(descriptor.key, descriptor.display_name, context_length)
            yield PromptProcessingProgress(None)
            async for event in self._stream_completion(
                chat, descriptor.key, generation
            ):
                yield event
        except asyncio.CancelledError:
            raise
        except LMStudioError as exc:
            log.warning("LM Studio chat stream failed model=%s error=%s", model, exc)
            raise
        except Exception as exc:
            log.warning("LM Studio chat stream failed model=%s error=%s", model, exc)
            raise LMStudioError(f"Prediction failed: {exc}") from exc

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _resolve_model(self, requested_model: str) -> ModelDescriptor:
        inventory: tuple[ModelDescriptor, ...] = await self.list_model_inventory()
        if requested_model:
            descriptor: ModelDescriptor | None = next(
                (
                    model
                    for model in inventory
                    if model.key == requested_model
                    or requested_model in model.loaded_instance_ids
                ),
                None,
            )
            if descriptor is None:
                raise LMStudioError(f"Model is not available: {requested_model}")
            return descriptor

        descriptor = next((model for model in inventory if model.is_loaded), None)
        if descriptor is None:
            raise LMStudioError("No model is selected or currently loaded")
        return descriptor

    async def _require_loaded_model(self, model_key: str) -> ModelDescriptor:
        inventory: tuple[ModelDescriptor, ...] = await self.list_model_inventory()
        descriptor: ModelDescriptor | None = next(
            (model for model in inventory if model.key == model_key), None
        )
        if descriptor is None or not descriptor.is_loaded:
            raise LMStudioError(f"LM Studio did not load model: {model_key}")
        return descriptor

    async def _load_model(self, model_key: str) -> None:
        url: str = f"{self._server_root}/api/v1/models/load"
        payload: dict[str, object] = {"model": model_key, "echo_load_config": True}
        log.info("requesting LM Studio model load model_key=%s", model_key)
        session: ClientSession = await self._get_session()
        async with session.post(
            url,
            json=payload,
            headers=self._headers(),
            timeout=aiohttp.ClientTimeout(
                total=_MODEL_LOAD_TIMEOUT_SECONDS, sock_connect=_CONNECT_TIMEOUT_SECONDS
            ),
        ) as response:
            if response.status != 200:
                inventory: tuple[
                    ModelDescriptor, ...
                ] = await self.list_model_inventory()
                if any(
                    model.key == model_key and model.is_loaded for model in inventory
                ):
                    log.info(
                        "LM Studio model was already loaded model_key=%s", model_key
                    )
                    return
                raise LMStudioError(f"Model load returned HTTP {response.status}")
        log.info("LM Studio model load accepted model_key=%s", model_key)

    async def _stream_completion(
        self,
        chat: Chat,
        model_key: str,
        generation: GenerationSettings,
    ) -> AsyncIterator[ChatStreamEvent]:
        messages: list[WireMessage] = (
            await run_in_worker(self._build_messages, chat, generation, self._images)
            if chat.has_images
            else self._build_messages(chat, generation)
        )
        payload: dict[str, object] = {
            "model": model_key,
            "messages": messages,
            "temperature": generation.temperature,
            "top_p": generation.top_p,
            "max_tokens": generation.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        ttl_seconds: int | None = self._ttl_seconds()
        if ttl_seconds is not None:
            payload["ttl"] = ttl_seconds
        session: ClientSession = await self._get_session()
        request_started_at: float = asyncio.get_running_loop().time()

        url: str = f"{self._server.base_url.rstrip('/')}/chat/completions"
        log.debug(
            "requesting LM Studio chat completion model=%s message_count=%d",
            model_key,
            len(messages),
        )
        async with session.post(
            url,
            json=payload,
            headers=self._headers(accept="text/event-stream"),
            timeout=aiohttp.ClientTimeout(
                total=None,
                sock_connect=_CONNECT_TIMEOUT_SECONDS,
                sock_read=_STREAM_IDLE_TIMEOUT_SECONDS,
            ),
        ) as response:
            if response.status != 200:
                raise LMStudioError(f"Chat completion returned HTTP {response.status}")

            try:
                async for event in completion_events(
                    response,
                    ready=ModelReady(model_key, model_key, 0),
                    started_at=request_started_at,
                    wire_model=model_key,
                ):
                    if not isinstance(event, ModelReady):
                        yield event
            except BackendError as exc:
                raise LMStudioError(str(exc)) from exc

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            log.debug("created LM Studio HTTP session")
        return self._session

    def _headers(self, *, accept: str = "application/json") -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json", "Accept": accept}
        if self._server.api_key:
            headers["Authorization"] = f"Bearer {self._server.api_key}"
        return headers

    @staticmethod
    async def _iter_sse_data(response: aiohttp.ClientResponse) -> AsyncIterator[str]:
        async for data in iter_sse_data(response.content.iter_chunked(4096)):
            yield data

    @staticmethod
    def _model_instance_descriptor(
        instance_id: str, instance: dict[str, object]
    ) -> ModelInstanceDescriptor:
        config: dict[str, object] | None = _as_object_mapping(instance.get("config"))
        return ModelInstanceDescriptor(
            id=instance_id,
            context_length=_optional_int(config, "context_length"),
            eval_batch_size=_optional_int(config, "eval_batch_size"),
            parallel=_optional_int(config, "parallel"),
            flash_attention=_optional_bool(config, "flash_attention"),
            num_experts=_optional_int(config, "num_experts"),
            offload_kv_cache_to_gpu=_optional_bool(config, "offload_kv_cache_to_gpu"),
            auto_unload_minutes=LMStudioClient._first_auto_unload_minutes(
                {}, [instance]
            ),
        )

    @staticmethod
    def _model_format(raw: object) -> ModelFormat | None:
        if raw in MODEL_FORMATS:
            return raw
        return None

    @staticmethod
    def _model_capabilities(model: dict[str, object]) -> ModelCapabilities:
        capabilities: dict[str, object] | None = _as_object_mapping(
            model.get("capabilities")
        )
        if capabilities is None:
            return ModelCapabilities()
        reasoning: dict[str, object] | None = _as_object_mapping(
            capabilities.get("reasoning")
        )
        reasoning_options: tuple[ReasoningOption, ...] = ()
        reasoning_default: object | None = None
        if reasoning is not None:
            raw_allowed_options: object | None = reasoning.get("allowed_options")
            if isinstance(raw_allowed_options, list):
                reasoning_options = tuple(
                    option
                    for option in cast(list[object], raw_allowed_options)
                    if option in REASONING_OPTIONS
                )
            reasoning_default = reasoning.get("default")
        return ModelCapabilities(
            vision=_optional_bool(capabilities, "vision"),
            trained_for_tool_use=_optional_bool(capabilities, "trained_for_tool_use"),
            reasoning=True
            if reasoning is not None
            else _optional_bool(capabilities, "reasoning"),
            reasoning_options=reasoning_options,
            reasoning_default=reasoning_default
            if reasoning_default in REASONING_OPTIONS
            else None,
        )

    @staticmethod
    def _first_context_length(instances: list[object]) -> int | None:
        for raw_instance in instances:
            instance: dict[str, object] | None = _as_object_mapping(raw_instance)
            if instance is None:
                continue
            config: dict[str, object] | None = _as_object_mapping(
                instance.get("config")
            )
            if config is None:
                continue
            context_length: object | None = config.get("context_length")
            if isinstance(context_length, int):
                return context_length
        return None

    @classmethod
    def _first_auto_unload_minutes(
        cls, model: dict[str, object], instances: list[object]
    ) -> int | None:
        for raw_value in (
            model.get("ttl"),
            model.get("idle_ttl_seconds"),
            model.get("auto_unload_seconds"),
        ):
            minutes = cls._ttl_minutes(raw_value)
            if minutes is not None:
                return minutes
        for raw_instance in instances:
            instance: dict[str, object] | None = _as_object_mapping(raw_instance)
            if instance is None:
                continue
            for raw_value in (
                instance.get("ttl"),
                instance.get("idle_ttl_seconds"),
                instance.get("auto_unload_seconds"),
            ):
                minutes = cls._ttl_minutes(raw_value)
                if minutes is not None:
                    return minutes
            config: dict[str, object] | None = _as_object_mapping(
                instance.get("config")
            )
            if config is None:
                continue
            for raw_value in (
                config.get("ttl"),
                config.get("idle_ttl_seconds"),
                config.get("auto_unload_seconds"),
            ):
                minutes = cls._ttl_minutes(raw_value)
                if minutes is not None:
                    return minutes
        return None

    @staticmethod
    def _ttl_minutes(value: object) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
            return None
        return max(1, math.ceil(float(value) / 60.0))

    def _ttl_seconds(self) -> int | None:
        minutes: int | None = self._server.auto_unload_minutes
        if minutes is None:
            return None
        return minutes * 60

    @staticmethod
    def _optional_int(value: object) -> int | None:
        return value if isinstance(value, int) else None

    @staticmethod
    def _build_messages(
        chat: Chat, generation: GenerationSettings, images: ImageStorage | None = None
    ) -> list[WireMessage]:
        system_prompt: str = (
            chat.system_prompt.strip() or generation.system_prompt.strip()
        )
        if generation.british_english:
            system_prompt = "\n\n".join(
                part for part in (system_prompt, BRITISH_ENGLISH_INSTRUCTION) if part
            )
        messages: list[InferenceMessage] = []
        if system_prompt:
            messages.append(InferenceMessage("system", system_prompt))
        messages.extend(
            InferenceMessage(message.role, message.content, message.images)
            for message in chat.messages
            if message.role in {"user", "assistant"}
            and (message.content or message.images)
        )
        return build_messages(messages, images)
