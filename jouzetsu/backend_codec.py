"""Strict JSON components for backend configuration and chat inference settings.

These codecs do not migrate or rewrite the current application's documents.
"""

from __future__ import annotations

from typing import Final

from .backend_models import (
    BackendConfig,
    BackendConfiguration,
    BackendId,
    BackendIdentity,
    BackendKind,
    ChatInferenceSettings,
    GenerationAttribution,
    GenerationProfile,
    ModelManagementSettings,
    ModelRef,
    RegisteredModel,
)
from .json_codec import JsonDecoder
from .runtime import (
    MODEL_FORMATS,
    REASONING_OPTIONS,
    ModelCapabilities,
    ModelDescriptor,
    ModelFormat,
    ReasoningOption,
)
from .sampling import DEFAULT_SAMPLING, ParameterSupport, SamplingParameters

BACKEND_SCHEMA_VERSION: Final[int] = 1
INFERENCE_SCHEMA_VERSION: Final[int] = 1
type JsonObject = dict[str, object]


def encode_backends(config: BackendConfiguration) -> JsonObject:
    return {
        "schema_version": BACKEND_SCHEMA_VERSION,
        "backends": [_encode_backend(backend) for backend in config.backends],
        "default_backend_id": config.default_backend_id,
        "retired_backends": [
            _encode_identity(identity) for identity in config.retired_backends
        ],
    }


def decode_backends(raw: object) -> BackendConfiguration:
    decoder = JsonDecoder()
    path = "backends"
    root = _versioned(
        decoder,
        raw,
        path,
        BACKEND_SCHEMA_VERSION,
        {"backends", "default_backend_id", "retired_backends"},
    )
    backends = tuple(
        _decode_backend(decoder, item, f"{path}.backends[{index}]")
        for index, item in enumerate(
            decoder.list(root.get("backends"), path=f"{path}.backends")
        )
    )
    default_id = decoder.optional_string(
        root.get("default_backend_id"), path=f"{path}.default_backend_id"
    )
    retired = tuple(
        _decode_identity(decoder, item, f"{path}.retired_backends[{index}]")
        for index, item in enumerate(
            decoder.list(root.get("retired_backends"), path=f"{path}.retired_backends")
        )
    )
    return decoder.construct(
        lambda: BackendConfiguration(
            backends, BackendId(default_id) if default_id is not None else None, retired
        ),
        path=path,
    )


def encode_inference(settings: ChatInferenceSettings) -> JsonObject:
    return {
        "schema_version": INFERENCE_SCHEMA_VERSION,
        "selected_backend_id": settings.selected_backend_id,
        "profiles": {
            backend_id: _encode_profile(profile)
            for backend_id, profile in settings.profiles.items()
        },
    }


def decode_inference(raw: object) -> ChatInferenceSettings:
    decoder = JsonDecoder()
    path = "inference"
    root = _versioned(
        decoder,
        raw,
        path,
        INFERENCE_SCHEMA_VERSION,
        {"selected_backend_id", "profiles"},
    )
    selected = decoder.optional_string(
        root.get("selected_backend_id"), path=f"{path}.selected_backend_id"
    )
    profiles = {
        BackendId(backend_id): _decode_profile(
            decoder, profile, f"{path}.profiles.{backend_id}"
        )
        for backend_id, profile in decoder.mapping(
            root.get("profiles"), path=f"{path}.profiles"
        ).items()
    }
    return decoder.construct(
        lambda: ChatInferenceSettings(
            BackendId(selected) if selected is not None else None, profiles
        ),
        path=path,
    )


def encode_attribution(attribution: GenerationAttribution) -> JsonObject:
    return {
        "backend_id": attribution.model.backend_id,
        "model_id": attribution.model.model_id,
        "backend_name": attribution.backend_name,
        "model_name": attribution.model_name,
        "instance_id": attribution.instance_id,
    }


def decode_attribution(raw: object) -> GenerationAttribution:
    decoder = JsonDecoder()
    path = "attribution"
    values = decoder.object(
        raw,
        path=path,
        required=frozenset(
            {"backend_id", "model_id", "backend_name", "model_name", "instance_id"}
        ),
    )
    backend_id = decoder.string(
        values.get("backend_id"), path=f"{path}.backend_id", default=""
    )
    model_id = decoder.string(
        values.get("model_id"), path=f"{path}.model_id", default=""
    )
    backend_name = decoder.string(
        values.get("backend_name"), path=f"{path}.backend_name", default=""
    )
    model_name = decoder.string(
        values.get("model_name"), path=f"{path}.model_name", default=""
    )
    instance = decoder.optional_string(
        values.get("instance_id"), path=f"{path}.instance_id"
    )
    return decoder.construct(
        lambda: GenerationAttribution(
            ModelRef(BackendId(backend_id), model_id),
            backend_name,
            model_name,
            instance,
        ),
        path=path,
    )


def _versioned(
    decoder: JsonDecoder, raw: object, path: str, version: int, fields: set[str]
) -> JsonObject:
    values = decoder.object(
        raw, path=path, required=frozenset({"schema_version", *fields})
    )
    if (
        decoder.integer(
            values.get("schema_version"), path=f"{path}.schema_version", default=version
        )
        != version
    ):
        decoder.issue(f"{path}.schema_version", f"must be {version}")
    return values


def _encode_identity(identity: BackendIdentity) -> JsonObject:
    return {"id": identity.id, "kind": identity.kind.value, "name": identity.name}


def _identity_fields(
    decoder: JsonDecoder, values: JsonObject, path: str
) -> tuple[BackendId, BackendKind, str]:
    backend_id = BackendId(
        decoder.string(values.get("id"), path=f"{path}.id", default="")
    )
    kind = BackendKind(
        decoder.choice(
            values.get("kind"),
            path=f"{path}.kind",
            choices=tuple(kind.value for kind in BackendKind),
        )
    )
    name = decoder.string(values.get("name"), path=f"{path}.name", default="")
    return backend_id, kind, name


def _decode_identity(decoder: JsonDecoder, raw: object, path: str) -> BackendIdentity:
    values = decoder.object(raw, path=path, required=frozenset({"id", "kind", "name"}))
    backend_id, kind, name = _identity_fields(decoder, values, path)
    return decoder.construct(lambda: BackendIdentity(backend_id, kind, name), path=path)


def _encode_backend(backend: BackendConfig) -> JsonObject:
    return {
        **_encode_identity(backend),
        "base_url": backend.base_url,
        "enabled": backend.enabled,
        "concurrency_limit": backend.concurrency_limit,
        "defaults": _encode_profile(backend.defaults),
        "parameter_support": _encode_support(backend.parameter_support),
        "model_management": {
            "automatic_loading": backend.model_management.automatic_loading,
            "idle_unload_minutes": backend.model_management.idle_unload_minutes,
        }
        if backend.model_management is not None
        else None,
        "models": [_encode_model(model) for model in backend.models],
    }


def _decode_backend(decoder: JsonDecoder, raw: object, path: str) -> BackendConfig:
    values = decoder.object(
        raw,
        path=path,
        required=frozenset(
            {
                "id",
                "kind",
                "name",
                "base_url",
                "enabled",
                "concurrency_limit",
                "defaults",
                "parameter_support",
                "model_management",
                "models",
            }
        ),
    )
    backend_id, kind, name = _identity_fields(decoder, values, path)
    url = decoder.string(values.get("base_url"), path=f"{path}.base_url", default="")
    enabled = decoder.boolean(
        values.get("enabled"), path=f"{path}.enabled", default=True
    )
    concurrency = decoder.integer(
        values.get("concurrency_limit"), path=f"{path}.concurrency_limit", default=1
    )
    defaults = _decode_profile(decoder, values.get("defaults"), f"{path}.defaults")
    support = _decode_support(
        decoder, values.get("parameter_support"), f"{path}.parameter_support"
    )
    management = _decode_management(
        decoder, values.get("model_management"), f"{path}.model_management"
    )
    models = tuple(
        _decode_model(decoder, item, f"{path}.models[{index}]")
        for index, item in enumerate(
            decoder.list(values.get("models"), path=f"{path}.models")
        )
    )
    return decoder.construct(
        lambda: BackendConfig(
            backend_id,
            kind,
            name,
            url,
            enabled,
            concurrency,
            defaults,
            support,
            management,
            models,
        ),
        path=path,
    )


def _decode_management(
    decoder: JsonDecoder, raw: object, path: str
) -> ModelManagementSettings | None:
    if raw is None:
        return None
    values = decoder.object(
        raw, path=path, required=frozenset({"automatic_loading", "idle_unload_minutes"})
    )
    automatic = decoder.boolean(
        values.get("automatic_loading"), path=f"{path}.automatic_loading", default=True
    )
    minutes = decoder.optional_integer(
        values.get("idle_unload_minutes"),
        path=f"{path}.idle_unload_minutes",
        default=None,
    )
    return decoder.construct(
        lambda: ModelManagementSettings(automatic, minutes), path=path
    )


def _encode_support(support: ParameterSupport) -> JsonObject:
    return {
        "temperature": support.temperature,
        "top_p": support.top_p,
        "max_tokens": support.max_tokens,
    }


def _decode_support(decoder: JsonDecoder, raw: object, path: str) -> ParameterSupport:
    values = decoder.object(
        raw, path=path, required=frozenset({"temperature", "top_p", "max_tokens"})
    )
    return ParameterSupport(
        temperature=decoder.optional_boolean(
            values.get("temperature"), path=f"{path}.temperature"
        ),
        top_p=decoder.optional_boolean(values.get("top_p"), path=f"{path}.top_p"),
        max_tokens=decoder.optional_boolean(
            values.get("max_tokens"), path=f"{path}.max_tokens"
        ),
    )


def _encode_profile(profile: GenerationProfile) -> JsonObject:
    return {
        "model_id": profile.model_id,
        "parameters": {
            "temperature": profile.parameters.temperature,
            "top_p": profile.parameters.top_p,
            "max_tokens": profile.parameters.max_tokens,
        },
    }


def _decode_profile(decoder: JsonDecoder, raw: object, path: str) -> GenerationProfile:
    values = decoder.object(
        raw, path=path, required=frozenset({"model_id", "parameters"})
    )
    model = decoder.optional_string(values.get("model_id"), path=f"{path}.model_id")
    sampling = decoder.object(
        values.get("parameters"),
        path=f"{path}.parameters",
        required=frozenset({"temperature", "top_p", "max_tokens"}),
    )
    temperature = decoder.number(
        sampling.get("temperature"),
        path=f"{path}.parameters.temperature",
        default=DEFAULT_SAMPLING.temperature,
    )
    top_p = decoder.number(
        sampling.get("top_p"),
        path=f"{path}.parameters.top_p",
        default=DEFAULT_SAMPLING.top_p,
    )
    max_tokens = decoder.integer(
        sampling.get("max_tokens"),
        path=f"{path}.parameters.max_tokens",
        default=DEFAULT_SAMPLING.max_tokens,
    )
    return decoder.construct(
        lambda: GenerationProfile(
            model, SamplingParameters(temperature, top_p, max_tokens)
        ),
        path=path,
    )


def _encode_capabilities(capabilities: ModelCapabilities) -> JsonObject:
    return {
        "vision": capabilities.vision,
        "trained_for_tool_use": capabilities.trained_for_tool_use,
        "reasoning": capabilities.reasoning,
        "reasoning_options": list(capabilities.reasoning_options),
        "reasoning_default": capabilities.reasoning_default,
        "parameters": _encode_support(capabilities.parameters),
    }


def _decode_capabilities(
    decoder: JsonDecoder, raw: object, path: str
) -> ModelCapabilities:
    values = decoder.object(
        raw,
        path=path,
        required=frozenset(
            {
                "vision",
                "trained_for_tool_use",
                "reasoning",
                "reasoning_options",
                "reasoning_default",
                "parameters",
            }
        ),
    )
    vision = decoder.optional_boolean(values.get("vision"), path=f"{path}.vision")
    tools = decoder.optional_boolean(
        values.get("trained_for_tool_use"), path=f"{path}.trained_for_tool_use"
    )
    reasoning = decoder.optional_boolean(
        values.get("reasoning"), path=f"{path}.reasoning"
    )
    options: tuple[ReasoningOption, ...] = tuple(
        decoder.choice(
            value, path=f"{path}.reasoning_options[{index}]", choices=REASONING_OPTIONS
        )
        for index, value in enumerate(
            decoder.list(
                values.get("reasoning_options"), path=f"{path}.reasoning_options"
            )
        )
    )
    default: ReasoningOption | None = (
        decoder.choice(
            values.get("reasoning_default"),
            path=f"{path}.reasoning_default",
            choices=REASONING_OPTIONS,
        )
        if values.get("reasoning_default") is not None
        else None
    )
    support = _decode_support(decoder, values.get("parameters"), f"{path}.parameters")
    return decoder.construct(
        lambda: ModelCapabilities(vision, tools, reasoning, options, default, support),
        path=path,
    )


def _encode_model(model: RegisteredModel) -> JsonObject:
    metadata = model.descriptor
    return {
        "id": metadata.key,
        "display_name": metadata.display_name,
        "alias": model.alias,
        "manually_registered": model.manually_registered,
        "context_length": metadata.context_length,
        "max_context_length": metadata.max_context_length,
        "auto_unload_minutes": metadata.auto_unload_minutes,
        "publisher": metadata.publisher,
        "architecture": metadata.architecture,
        "quantization_name": metadata.quantization_name,
        "bits_per_weight": metadata.bits_per_weight,
        "size_bytes": metadata.size_bytes,
        "params_string": metadata.params_string,
        "format": metadata.format,
        "capabilities": _encode_capabilities(metadata.capabilities),
        "description": metadata.description,
        "variants": list(metadata.variants),
        "selected_variant": metadata.selected_variant,
    }


def _decode_model(decoder: JsonDecoder, raw: object, path: str) -> RegisteredModel:
    values = decoder.object(
        raw,
        path=path,
        required=frozenset(
            {
                "id",
                "display_name",
                "alias",
                "manually_registered",
                "context_length",
                "max_context_length",
                "auto_unload_minutes",
                "publisher",
                "architecture",
                "quantization_name",
                "bits_per_weight",
                "size_bytes",
                "params_string",
                "format",
                "capabilities",
                "description",
                "variants",
                "selected_variant",
            }
        ),
    )
    model_id = decoder.string(values.get("id"), path=f"{path}.id", default="")
    name = decoder.string(
        values.get("display_name"), path=f"{path}.display_name", default=""
    )
    alias = decoder.string(values.get("alias"), path=f"{path}.alias", default="")
    manual = decoder.boolean(
        values.get("manually_registered"),
        path=f"{path}.manually_registered",
        default=False,
    )
    context = decoder.optional_integer(
        values.get("context_length"), path=f"{path}.context_length", default=None
    )
    maximum = decoder.optional_integer(
        values.get("max_context_length"),
        path=f"{path}.max_context_length",
        default=None,
    )
    ttl = decoder.optional_integer(
        values.get("auto_unload_minutes"),
        path=f"{path}.auto_unload_minutes",
        default=None,
    )
    publisher = decoder.string(
        values.get("publisher"), path=f"{path}.publisher", default=""
    )
    architecture = decoder.string(
        values.get("architecture"), path=f"{path}.architecture", default=""
    )
    quantization = decoder.string(
        values.get("quantization_name"), path=f"{path}.quantization_name", default=""
    )
    bits = (
        decoder.number(
            values.get("bits_per_weight"), path=f"{path}.bits_per_weight", default=0.0
        )
        if values.get("bits_per_weight") is not None
        else None
    )
    size = decoder.optional_integer(
        values.get("size_bytes"), path=f"{path}.size_bytes", default=None
    )
    params = decoder.string(
        values.get("params_string"), path=f"{path}.params_string", default=""
    )
    model_format: ModelFormat | None = (
        decoder.choice(
            values.get("format"), path=f"{path}.format", choices=MODEL_FORMATS
        )
        if values.get("format") is not None
        else None
    )
    capabilities = _decode_capabilities(
        decoder, values.get("capabilities"), f"{path}.capabilities"
    )
    description = decoder.string(
        values.get("description"), path=f"{path}.description", default=""
    )
    variants = tuple(
        decoder.string(value, path=f"{path}.variants[{index}]", default="")
        for index, value in enumerate(
            decoder.list(values.get("variants"), path=f"{path}.variants")
        )
    )
    selected = decoder.string(
        values.get("selected_variant"), path=f"{path}.selected_variant", default=""
    )
    descriptor = ModelDescriptor(
        key=model_id,
        display_name=name,
        context_length=context,
        max_context_length=maximum,
        auto_unload_minutes=ttl,
        publisher=publisher,
        architecture=architecture,
        quantization_name=quantization,
        bits_per_weight=bits,
        size_bytes=size,
        params_string=params,
        format=model_format,
        capabilities=capabilities,
        description=description,
        variants=variants,
        selected_variant=selected,
    )
    return decoder.construct(
        lambda: RegisteredModel(descriptor, alias, manual), path=path
    )
