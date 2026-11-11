"""Narrow transport contracts; job ownership and prompt assembly stay in core."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol, cast

from .backend_models import BackendId, ModelRef
from .models import ImageAttachment, Role
from .runtime import (
    REASONING_OPTIONS,
    ChatStreamEvent,
    ModelCapabilities,
    ModelDescriptor,
    ModelLoadProgress,
    ModelReady,
    ReasoningOption,
)
from .sampling import (
    DEFAULT_SAMPLING,
    ParameterName,
    ParameterSupport,
    SamplingParameters,
)


class BackendReachability(str, Enum):
    UNKNOWN = "unknown"
    ONLINE = "online"
    OFFLINE = "offline"
    STALE = "stale"


@dataclass(frozen=True, slots=True)
class BackendHealth:
    state: BackendReachability = BackendReachability.UNKNOWN
    checked_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.state), BackendReachability):
            raise TypeError("health state must be a BackendReachability")
        if self.checked_at is not None and self.checked_at.utcoffset() is None:
            raise ValueError("health timestamps must include a timezone")
        if self.state is not BackendReachability.UNKNOWN and self.checked_at is None:
            raise ValueError("observed or stale health requires a check timestamp")


class BackendErrorCategory(str, Enum):
    OFFLINE = "offline"
    LOAD_FAILURE = "load_failure"
    TIMEOUT = "timeout"
    INCOMPATIBLE = "incompatible"
    PROTOCOL = "protocol"
    OTHER = "other"


class BackendError(RuntimeError):
    """An adapter error with a safe explanation, never an HTTP response body."""

    def __init__(self, category: BackendErrorCategory, message: str) -> None:
        if not isinstance(cast(object, category), BackendErrorCategory):
            raise TypeError("backend error category must be a BackendErrorCategory")
        if not message.strip():
            raise ValueError("backend errors require an explanation")
        self.category: BackendErrorCategory = category
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class InferenceMessage:
    """One ordered prompt entry, including original image references."""

    role: Role
    content: str
    images: tuple[ImageAttachment, ...] = ()

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError("invalid inference message role")
        if not self.content and not self.images:
            raise ValueError("inference messages cannot be empty")
        if not isinstance(cast(object, self.images), tuple) or not all(
            isinstance(cast(object, image), ImageAttachment) for image in self.images
        ):
            raise TypeError("inference images must be an immutable tuple of references")
        if self.images and self.role != "user":
            raise ValueError("only user messages can contain images")


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """Immutable prepared input; adapters never receive a mutable chat or config."""

    model: ModelRef
    messages: tuple[InferenceMessage, ...]
    capabilities: ModelCapabilities
    parameter_support: ParameterSupport
    parameters: SamplingParameters = DEFAULT_SAMPLING
    reasoning: ReasoningOption | None = None

    def __post_init__(self) -> None:
        if self.reasoning is not None and self.reasoning not in REASONING_OPTIONS:
            raise ValueError("unknown reasoning option")
        if not isinstance(cast(object, self.messages), tuple) or not all(
            isinstance(cast(object, message), InferenceMessage)
            for message in self.messages
        ):
            raise TypeError(
                "request messages must be an immutable tuple of InferenceMessage values"
            )
        if not self.messages:
            raise ValueError("generation requires at least one prompt message")
        if (
            any(message.images for message in self.messages)
            and self.capabilities.vision is not True
        ):
            raise BackendError(
                BackendErrorCategory.INCOMPATIBLE,
                "The selected model is not known to support images.",
            )

    def validate_backend(self, backend_id: BackendId) -> None:
        """Adapters must reject cross-backend requests before transport begins."""

        if self.model.backend_id != backend_id:
            raise BackendError(
                BackendErrorCategory.INCOMPATIBLE,
                "The request belongs to a different backend.",
            )

    @property
    def wire_parameters(self) -> dict[ParameterName, float | int]:
        return self.parameter_support.restrict_by(
            self.capabilities.parameters
        ).wire_values(self.parameters)


@dataclass(frozen=True, slots=True)
class ModelDiscovery:
    """An unsupported discovery endpoint differs from a supported empty inventory."""

    supported: bool
    models: tuple[ModelDescriptor, ...] = ()

    def __post_init__(self) -> None:
        if type(self.supported) is not bool:
            raise TypeError("discovery support must be boolean")
        if not isinstance(cast(object, self.models), tuple) or not all(
            isinstance(cast(object, model), ModelDescriptor) for model in self.models
        ):
            raise TypeError(
                "discovery models must be an immutable tuple of descriptors"
            )
        if not self.supported and self.models:
            raise ValueError("unsupported discovery cannot return model metadata")
        ids = [model.key for model in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("discovery returned duplicate model ids")


class ModelManagement(Protocol):
    """Optional native operations, scoped to the adapter's backend identity."""

    def load_model(
        self, model_id: str
    ) -> AsyncGenerator[ModelLoadProgress | ModelReady]: ...

    async def unload_model(self, instance_id: str) -> None: ...


class InferenceBackend(Protocol):
    @property
    def id(self) -> BackendId: ...

    @property
    def model_management(self) -> ModelManagement | None: ...

    async def probe(self) -> BackendHealth: ...

    async def discover_models(self) -> ModelDiscovery: ...

    def stream(self, request: GenerationRequest) -> AsyncGenerator[ChatStreamEvent]: ...

    async def close(self) -> None: ...
