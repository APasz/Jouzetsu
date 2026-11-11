"""Backend configuration and chat profile values, independent of live clients."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Literal, NewType, cast
from urllib.parse import urlsplit
from uuid import uuid4

from .runtime import REASONING_OPTIONS, ModelDescriptor, ReasoningOption
from .sampling import (
    CHAT_COMPLETIONS_PARAMETERS,
    DEFAULT_SAMPLING,
    ParameterSupport,
    SamplingParameters,
)

BackendId = NewType("BackendId", str)
type ReasoningSelection = Literal["model_default", "automatic"] | ReasoningOption
_BACKEND_ID_PATTERN = re.compile(r"be_[0-9a-f]{32}")


def new_backend_id() -> BackendId:
    """Create identity independently of endpoint, display name, and adapter type."""

    return BackendId(f"be_{uuid4().hex}")


def validate_backend_id(backend_id: BackendId) -> None:
    if not isinstance(
        cast(object, backend_id), str
    ) or not _BACKEND_ID_PATTERN.fullmatch(backend_id):
        raise ValueError(
            "backend id must be be_ followed by 32 lowercase hexadecimal digits"
        )


def validate_model_id(model_id: str) -> None:
    if (
        not isinstance(cast(object, model_id), str)
        or not model_id
        or model_id != model_id.strip()
    ):
        raise ValueError(
            "model id must be non-empty text without surrounding whitespace"
        )


class BackendKind(str, Enum):
    LM_STUDIO = "lm_studio"
    OPENAI_COMPATIBLE = "openai_compatible"


@dataclass(frozen=True, slots=True)
class ModelRef:
    """A model identifier has meaning only inside its owning backend."""

    backend_id: BackendId
    model_id: str

    def __post_init__(self) -> None:
        validate_backend_id(self.backend_id)
        validate_model_id(self.model_id)


@dataclass(frozen=True, slots=True)
class GenerationProfile:
    """The same snapshot shape serves backend defaults and per-chat profiles."""

    model_id: str | None = None
    parameters: SamplingParameters = DEFAULT_SAMPLING
    reasoning: ReasoningSelection = "model_default"

    def __post_init__(self) -> None:
        if self.model_id is not None:
            validate_model_id(self.model_id)
        if not isinstance(cast(object, self.parameters), SamplingParameters):
            raise TypeError("profile parameters must be SamplingParameters")
        if self.reasoning not in ("model_default", "automatic", *REASONING_OPTIONS):
            raise ValueError("unknown reasoning selection")


@dataclass(frozen=True, slots=True)
class ModelManagementSettings:
    automatic_loading: bool = True
    idle_unload_minutes: int | None = None

    def __post_init__(self) -> None:
        if type(self.automatic_loading) is not bool:
            raise TypeError("automatic_loading must be a boolean")
        if self.idle_unload_minutes is not None and (
            type(self.idle_unload_minutes) is not int or self.idle_unload_minutes < 1
        ):
            raise ValueError("idle_unload_minutes must be a positive integer or null")


@dataclass(frozen=True, slots=True)
class RegisteredModel:
    """Persistent metadata and alias; loaded instances remain runtime information."""

    descriptor: ModelDescriptor
    alias: str = ""
    manually_registered: bool = False
    reasoning_default: ReasoningOption | None = None
    reasoning_options_override: tuple[ReasoningOption, ...] = ()
    reasoning_default_unavailable: bool = False

    def __post_init__(self) -> None:
        if type(self.reasoning_default_unavailable) is not bool:
            raise TypeError("reasoning default unavailable must be a boolean")
        if self.reasoning_default_unavailable and self.reasoning_default is not None:
            raise ValueError("an unavailable reasoning default must use Automatic")
        if (
            self.reasoning_default is not None
            and self.reasoning_default not in REASONING_OPTIONS
        ):
            raise ValueError("unknown model reasoning default")
        if not isinstance(cast(object, self.reasoning_options_override), tuple) or any(
            option not in REASONING_OPTIONS
            for option in self.reasoning_options_override
        ):
            raise ValueError("invalid reasoning options override")
        if not isinstance(cast(object, self.descriptor), ModelDescriptor):
            raise TypeError("registered metadata must be a ModelDescriptor")
        validate_model_id(self.descriptor.key)
        if not self.descriptor.display_name.strip():
            raise ValueError("model display name cannot be blank")
        if self.descriptor.loaded_instance_ids or self.descriptor.loaded_instances:
            raise ValueError(
                "registered metadata must not contain live model instances"
            )
        if (
            not isinstance(cast(object, self.alias), str)
            or self.alias != self.alias.strip()
        ):
            raise ValueError("model alias must be text without surrounding whitespace")
        if type(self.manually_registered) is not bool:
            raise TypeError("manually_registered must be a boolean")
        for name, value in (
            ("context_length", self.descriptor.context_length),
            ("max_context_length", self.descriptor.max_context_length),
            ("auto_unload_minutes", self.descriptor.auto_unload_minutes),
            ("size_bytes", self.descriptor.size_bytes),
        ):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer or null")
        bits = self.descriptor.bits_per_weight
        if bits is not None and (
            isinstance(bits, bool) or not math.isfinite(bits) or bits <= 0
        ):
            raise ValueError("bits_per_weight must be positive and finite or null")

    @classmethod
    def discovered(cls, descriptor: ModelDescriptor) -> RegisteredModel:
        return cls(replace(descriptor, loaded_instance_ids=(), loaded_instances=()))

    @property
    def reasoning_options(self) -> tuple[ReasoningOption, ...]:
        return (
            self.reasoning_options_override
            or self.descriptor.capabilities.reasoning_options
        )

    def resolve_reasoning(
        self, selection: ReasoningSelection
    ) -> tuple[ReasoningOption | None, bool]:
        """Return the request option and whether the selection remains available."""
        chosen = (
            self.reasoning_default
            if selection == "model_default"
            else None
            if selection == "automatic"
            else selection
        )
        if chosen is None:
            return None, True
        return (chosen, True) if chosen in self.reasoning_options else (None, False)


@dataclass(frozen=True, slots=True)
class BackendIdentity:
    """Retained after deletion so identity cannot be reassociated by name or URL."""

    id: BackendId
    kind: BackendKind
    name: str

    def __post_init__(self) -> None:
        validate_backend_id(self.id)
        if not isinstance(cast(object, self.kind), BackendKind):
            raise TypeError("backend kind must be a BackendKind")
        if not isinstance(cast(object, self.name), str) or not self.name.strip():
            raise ValueError("backend name cannot be blank")


@dataclass(frozen=True, slots=True)
class BackendConfig(BackendIdentity):
    base_url: str
    enabled: bool = True
    concurrency_limit: int = 1
    defaults: GenerationProfile = GenerationProfile()
    parameter_support: ParameterSupport = CHAT_COMPLETIONS_PARAMETERS
    model_management: ModelManagementSettings | None = None
    models: tuple[RegisteredModel, ...] = ()

    def __post_init__(self) -> None:
        super(BackendConfig, self).__post_init__()
        if not isinstance(cast(object, self.base_url), str) or any(
            character.isspace() for character in self.base_url
        ):
            raise ValueError("base_url must be an HTTP API URL without whitespace")
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("base_url must be an absolute HTTP or HTTPS URL")
        if (
            parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "base_url must not contain credentials, a query, or a fragment"
            )
        _ = parsed.port
        if type(self.enabled) is not bool:
            raise TypeError("backend enabled must be a boolean")
        if type(self.concurrency_limit) is not int or self.concurrency_limit < 1:
            raise ValueError("concurrency_limit must be a positive integer")
        if not isinstance(
            cast(object, self.defaults), GenerationProfile
        ) or not isinstance(cast(object, self.parameter_support), ParameterSupport):
            raise TypeError(
                "backend defaults and parameter support must be typed values"
            )
        if self.kind is BackendKind.LM_STUDIO:
            if not isinstance(self.model_management, ModelManagementSettings):
                raise ValueError(
                    "LM Studio requires explicit model management settings"
                )
        elif self.model_management is not None:
            raise ValueError("generic backends do not support model management")
        if not isinstance(cast(object, self.models), tuple) or not all(
            isinstance(cast(object, model), RegisteredModel) for model in self.models
        ):
            raise TypeError(
                "registered models must be a tuple of RegisteredModel values"
            )
        ids = [model.descriptor.key for model in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate registered model id")

    @classmethod
    def create(cls, kind: BackendKind, name: str, base_url: str) -> BackendConfig:
        return cls(
            id=new_backend_id(),
            kind=kind,
            name=name,
            base_url=base_url,
            model_management=ModelManagementSettings()
            if kind is BackendKind.LM_STUDIO
            else None,
        )

    def model(self, model_id: str) -> RegisteredModel:
        for model in self.models:
            if model.descriptor.key == model_id:
                return model
        raise KeyError(f"unregistered model: {model_id}")

    def support_for_model(self, model_id: str | None) -> ParameterSupport:
        model = next(
            (model for model in self.models if model.descriptor.key == model_id), None
        )
        return (
            self.parameter_support
            if model is None
            else self.parameter_support.restrict_by(
                model.descriptor.capabilities.parameters
            )
        )

    def register_model(self, model_id: str, *, alias: str = "") -> BackendConfig:
        validate_model_id(model_id)
        if any(model.descriptor.key == model_id for model in self.models):
            raise ValueError(f"model is already registered: {model_id}")
        model = RegisteredModel(
            descriptor=ModelDescriptor(key=model_id, display_name=model_id),
            alias=alias,
            manually_registered=True,
        )
        return replace(self, models=(*self.models, model))

    def merge_discovery(
        self, descriptors: tuple[ModelDescriptor, ...]
    ) -> BackendConfig:
        """Discovery updates metadata without removing omitted models or aliases."""

        models = {model.descriptor.key: model for model in self.models}
        seen: set[str] = set()
        for descriptor in descriptors:
            if descriptor.key in seen:
                raise ValueError(f"duplicate discovered model: {descriptor.key}")
            seen.add(descriptor.key)
            discovered = RegisteredModel.discovered(descriptor)
            previous = models.get(descriptor.key)
            models[descriptor.key] = (
                replace(previous, descriptor=discovered.descriptor)
                if previous is not None
                else discovered
            )
            model = models[descriptor.key]
            if (
                model.reasoning_default is not None
                and model.reasoning_default not in model.reasoning_options
                and (
                    model.descriptor.capabilities.reasoning is False
                    or model.descriptor.capabilities.reasoning_options
                )
            ):
                models[descriptor.key] = replace(
                    model, reasoning_default=None, reasoning_default_unavailable=True
                )
        updated = tuple(models.values())
        return self if updated == self.models else replace(self, models=updated)

    def set_model_alias(self, model_id: str, alias: str) -> BackendConfig:
        model = self.model(model_id)
        updated = replace(model, alias=alias)
        return replace(
            self,
            models=tuple(updated if item is model else item for item in self.models),
        )


@dataclass(frozen=True, slots=True)
class BackendConfiguration:
    """Backend configuration component for the new application schema."""

    backends: tuple[BackendConfig, ...] = ()
    default_backend_id: BackendId | None = None
    retired_backends: tuple[BackendIdentity, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.backends), tuple) or not all(
            isinstance(cast(object, backend), BackendConfig)
            for backend in self.backends
        ):
            raise TypeError("backends must be a tuple of BackendConfig values")
        if not isinstance(cast(object, self.retired_backends), tuple) or not all(
            type(backend) is BackendIdentity for backend in self.retired_backends
        ):
            raise TypeError(
                "retired backends must be a tuple of BackendIdentity values"
            )
        active_ids = [backend.id for backend in self.backends]
        all_ids = [*active_ids, *(backend.id for backend in self.retired_backends)]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError(
                "backend identity must be unique across active and retired entries"
            )
        if (
            self.default_backend_id is not None
            and self.default_backend_id not in active_ids
        ):
            raise ValueError("default backend must identify a configured backend")

    def backend(self, backend_id: BackendId) -> BackendConfig:
        for backend in self.backends:
            if backend.id == backend_id:
                return backend
        raise KeyError(f"backend is missing or deleted: {backend_id}")

    def with_backend(self, backend: BackendConfig) -> BackendConfiguration:
        if any(identity.id == backend.id for identity in self.retired_backends):
            raise ValueError("retired backend identity cannot be reused")
        updated = tuple(
            backend if item.id == backend.id else item for item in self.backends
        )
        if not any(item.id == backend.id for item in self.backends):
            updated = (*updated, backend)
        return replace(self, backends=updated)

    def remove_backend(self, backend_id: BackendId) -> BackendConfiguration:
        backend = self.backend(backend_id)
        if self.default_backend_id == backend_id:
            raise ValueError("clear or change the default backend before removing it")
        return replace(
            self,
            backends=tuple(item for item in self.backends if item.id != backend_id),
            retired_backends=(
                *self.retired_backends,
                BackendIdentity(backend.id, backend.kind, backend.name),
            ),
        )


@dataclass(frozen=True, slots=True)
class ChatInferenceSettings:
    """Independent profiles; global defaults are copied only on explicit actions."""

    selected_backend_id: BackendId | None = None
    profiles: Mapping[BackendId, GenerationProfile] = field(
        default_factory=dict[BackendId, GenerationProfile]
    )

    def __post_init__(self) -> None:
        for backend_id, profile in self.profiles.items():
            validate_backend_id(backend_id)
            if not isinstance(cast(object, profile), GenerationProfile):
                raise TypeError("chat profiles must contain GenerationProfile values")
        if self.selected_backend_id is not None:
            validate_backend_id(self.selected_backend_id)
            if self.selected_backend_id not in self.profiles:
                raise ValueError("selected backend must have a stored chat profile")
        object.__setattr__(self, "profiles", MappingProxyType(dict(self.profiles)))

    @classmethod
    def new_chat(cls, config: BackendConfiguration) -> ChatInferenceSettings:
        backend_id = config.default_backend_id
        if backend_id is None:
            return cls()
        return cls(backend_id, {backend_id: config.backend(backend_id).defaults})

    @property
    def selected_profile(self) -> GenerationProfile | None:
        return (
            self.profiles.get(self.selected_backend_id)
            if self.selected_backend_id is not None
            else None
        )

    @property
    def selected_model(self) -> ModelRef | None:
        profile = self.selected_profile
        if (
            self.selected_backend_id is None
            or profile is None
            or profile.model_id is None
        ):
            return None
        return ModelRef(self.selected_backend_id, profile.model_id)

    def select_backend(
        self, config: BackendConfiguration, backend_id: BackendId
    ) -> ChatInferenceSettings:
        if backend_id == self.selected_backend_id:
            return self
        backend = config.backend(backend_id)
        profiles = dict(self.profiles)
        if backend_id not in profiles:
            defaults = backend.defaults
            previous = self.selected_profile
            profiles[backend_id] = (
                replace(
                    defaults,
                    parameters=defaults.parameters.copy_compatible_from(
                        previous.parameters,
                        backend.support_for_model(defaults.model_id),
                    ),
                )
                if previous is not None
                else defaults
            )
        profiles[backend_id] = replace(profiles[backend_id], reasoning="model_default")
        return ChatInferenceSettings(backend_id, profiles)

    def set_profile(self, profile: GenerationProfile) -> ChatInferenceSettings:
        if self.selected_backend_id is None:
            raise ValueError("select a backend before changing its profile")
        return replace(
            self, profiles={**self.profiles, self.selected_backend_id: profile}
        )

    def reset_to_defaults(self, config: BackendConfiguration) -> ChatInferenceSettings:
        if self.selected_backend_id is None:
            raise ValueError("select a backend before resetting its profile")
        return self.set_profile(config.backend(self.selected_backend_id).defaults)


@dataclass(frozen=True, slots=True)
class GenerationAttribution:
    """Actual generating identity plus labels retained across renames and deletion."""

    model: ModelRef
    backend_name: str
    model_name: str
    instance_id: str | None = None
    reasoning: ReasoningOption | Literal["automatic"] | None = None

    def __post_init__(self) -> None:
        if not self.backend_name.strip() or not self.model_name.strip():
            raise ValueError("generation attribution labels cannot be blank")
        if self.instance_id is not None:
            validate_model_id(self.instance_id)
        if self.reasoning is not None and self.reasoning not in (
            "automatic",
            *REASONING_OPTIONS,
        ):
            raise ValueError("unknown attributed reasoning option")
