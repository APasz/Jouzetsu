"""Concrete adapters for native LM Studio and generic streaming Chat Completions."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import aclosing
from datetime import UTC, datetime
from typing import cast, override

from .backend_models import (
    BackendConfig,
    BackendId,
    BackendKind,
    LMStudioTransport,
    validate_model_id,
)
from .backend_protocol import (
    BackendError,
    BackendErrorCategory,
    BackendHealth,
    BackendReachability,
    GenerationRequest,
    ModelDiscovery,
    ModelManagement,
)
from .chat_completions import (
    LOAD_TIMEOUT,
    UNLOAD_TIMEOUT,
    ChatCompletionsTransport,
    json_object,
    protocol_error,
)
from .image_storage import ImageStorage
from .lmstudio import LMStudioClient, LMStudioError, lmstudio_server_root
from .runtime import ChatStreamEvent, ModelDescriptor, ModelLoadProgress, ModelReady


class ChatCompletionsBackend:
    """Generic servers have discovery where available and no native management."""

    def __init__(
        self, config: BackendConfig, *, images: ImageStorage | None = None
    ) -> None:
        self._id: BackendId = config.id
        self._transport: ChatCompletionsTransport = ChatCompletionsTransport(
            config.base_url, images
        )

    @property
    def id(self) -> BackendId:
        return self._id

    @property
    def model_management(self) -> ModelManagement | None:
        return None

    async def probe(self) -> BackendHealth:
        try:
            _ = await self.discover_models()
        except BackendError as exc:
            if exc.category is not BackendErrorCategory.OFFLINE:
                raise
            return BackendHealth(BackendReachability.OFFLINE, datetime.now(UTC))
        return BackendHealth(BackendReachability.ONLINE, datetime.now(UTC))

    async def discover_models(self) -> ModelDiscovery:
        payload = await self._transport.json_request(
            "GET",
            f"{self._transport.base_url}/models",
            unsupported=frozenset({404, 405, 501}),
        )
        if payload is None:
            return ModelDiscovery(False)
        raw_models = payload.get("data")
        if not isinstance(raw_models, list):
            raise protocol_error("The backend returned an invalid model inventory.")
        models: list[ModelDescriptor] = []
        for raw in cast(list[object], raw_models):
            model_id = json_object(raw).get("id")
            if not isinstance(model_id, str):
                raise protocol_error("The backend returned an invalid model id.")
            try:
                validate_model_id(model_id)
            except ValueError as exc:
                raise protocol_error(
                    "The backend returned an invalid model id."
                ) from exc
            models.append(ModelDescriptor(model_id, model_id))
        try:
            return ModelDiscovery(True, tuple(models))
        except ValueError as exc:
            raise protocol_error("The backend returned duplicate model ids.") from exc

    async def stream(
        self, request: GenerationRequest
    ) -> AsyncGenerator[ChatStreamEvent]:
        request.validate_backend(self.id)
        async with aclosing(
            self._transport.stream(
                request, ModelReady(request.model.model_id, request.model.model_id, 0)
            )
        ) as events:
            async for event in events:
                yield event

    async def close(self) -> None:
        await self._transport.close()


class LMStudioBackend(ChatCompletionsBackend):
    """Native management plus the shared Chat Completions wire transport."""

    def __init__(
        self, config: BackendConfig, *, images: ImageStorage | None = None
    ) -> None:
        super().__init__(config, images=images)
        self._responses: bool = config.lmstudio_transport is LMStudioTransport.RESPONSES
        self._native_root: str = (
            f"{lmstudio_server_root(config.base_url)}/api/v1/models"
        )

    @property
    @override
    def model_management(self) -> ModelManagement:
        return self

    @override
    async def discover_models(self) -> ModelDiscovery:
        payload = await self._transport.json_request("GET", self._native_root)
        try:
            return ModelDiscovery(True, LMStudioClient.parse_model_inventory(payload))
        except (LMStudioError, ValueError, TypeError) as exc:
            raise protocol_error(
                "LM Studio returned an invalid model inventory."
            ) from exc

    async def _model(self, model_id: str) -> ModelDescriptor:
        discovery = await self.discover_models()
        for model in discovery.models:
            if model.key == model_id:
                return model
        raise BackendError(
            BackendErrorCategory.INCOMPATIBLE,
            "The selected model is unavailable on this backend.",
        )

    async def load_model(
        self, model_id: str
    ) -> AsyncGenerator[ModelLoadProgress | ModelReady]:
        validate_model_id(model_id)
        model = await self._model(model_id)
        if not model.is_loaded:
            yield ModelLoadProgress(None)
            _ = await self._transport.json_request(
                "POST",
                f"{self._native_root}/load",
                payload={"model": model_id, "echo_load_config": True},
                timeout=LOAD_TIMEOUT,
                failure=BackendErrorCategory.LOAD_FAILURE,
            )
            model = await self._model(model_id)
        if not model.is_loaded:
            raise BackendError(
                BackendErrorCategory.LOAD_FAILURE,
                "LM Studio did not confirm the requested model was loaded.",
            )
        yield self._ready(model)

    async def unload_model(self, instance_id: str) -> None:
        validate_model_id(instance_id)
        _ = await self._transport.json_request(
            "POST",
            f"{self._native_root}/unload",
            payload={"instance_id": instance_id},
            timeout=UNLOAD_TIMEOUT,
        )

    @staticmethod
    def _ready(model: ModelDescriptor) -> ModelReady:
        return ModelReady(
            model.key,
            model.display_name,
            model.context_length or model.max_context_length or 0,
            model.loaded_instance_ids[0],
        )

    @override
    async def stream(
        self, request: GenerationRequest
    ) -> AsyncGenerator[ChatStreamEvent]:
        request.validate_backend(self.id)
        model = await self._model(request.model.model_id)
        if not model.is_loaded:
            raise BackendError(
                BackendErrorCategory.INCOMPATIBLE, "The selected model is not loaded."
            )
        async with aclosing(
            self._transport.stream(
                request, self._ready(model), native=True, responses=self._responses
            )
        ) as events:
            async for event in events:
                yield event


def create_backend(
    config: BackendConfig, *, images: ImageStorage | None = None
) -> ChatCompletionsBackend:
    match config.kind:
        case BackendKind.LM_STUDIO:
            return LMStudioBackend(config, images=images)
        case BackendKind.OLLAMA:
            from .ollama_backend import OllamaBackend

            return OllamaBackend(config, images=images)
        case BackendKind.OPENAI_COMPATIBLE:
            return ChatCompletionsBackend(config, images=images)
