from __future__ import annotations

import json
from dataclasses import replace
from typing import cast

import pytest

from jouzetsu.backend_codec import (
    decode_attribution,
    decode_backends,
    decode_inference,
    encode_attribution,
    encode_backends,
    encode_inference,
)
from jouzetsu.backend_models import (
    BackendConfig,
    BackendConfiguration,
    BackendKind,
    ChatInferenceSettings,
    GenerationAttribution,
    GenerationProfile,
    LMStudioTransport,
    ModelRef,
)
from jouzetsu.json_codec import JsonValidationError
from jouzetsu.runtime import ModelCapabilities, ModelDescriptor
from jouzetsu.sampling import ParameterSupport, SamplingParameters


def json_round_trip(raw: object) -> object:
    return cast(object, json.loads(json.dumps(raw, allow_nan=False)))


def test_reasoning_settings_and_attribution_round_trip() -> None:
    backend = configured_backend()
    model = replace(
        backend.models[0],
        reasoning_default="on",
        reasoning_options_override=("off", "on"),
    )
    backend = replace(backend, models=(model, *backend.models[1:]))
    configuration = BackendConfiguration((backend,), backend.id)
    assert (
        decode_backends(json_round_trip(encode_backends(configuration)))
        == configuration
    )
    unavailable = replace(
        model, reasoning_default=None, reasoning_default_unavailable=True
    )
    fallback_config = replace(
        configuration,
        backends=(replace(backend, models=(unavailable, *backend.models[1:])),),
    )
    assert (
        decode_backends(json_round_trip(encode_backends(fallback_config)))
        == fallback_config
    )
    inference = ChatInferenceSettings(
        backend.id,
        {backend.id: GenerationProfile(model.descriptor.key, reasoning="off")},
    )
    assert decode_inference(json_round_trip(encode_inference(inference))) == inference
    attribution = GenerationAttribution(
        ModelRef(backend.id, model.descriptor.key), "Backend", "Model", reasoning="off"
    )
    assert (
        decode_attribution(json_round_trip(encode_attribution(attribution)))
        == attribution
    )


def test_attribution_rejects_invalid_reasoning_level() -> None:
    backend = configured_backend()
    attribution = GenerationAttribution(
        ModelRef(backend.id, "model"), "Backend", "Model"
    )
    raw = encode_attribution(attribution)
    raw["reasoning"] = "invalid"
    with pytest.raises(JsonValidationError, match="attribution.reasoning"):
        decode_attribution(raw)


def test_responses_transport_round_trips_and_old_config_defaults_to_chat_completions() -> (
    None
):
    backend = replace(
        configured_backend(), lmstudio_transport=LMStudioTransport.RESPONSES
    )
    configuration = BackendConfiguration((backend,), backend.id)
    encoded = encode_backends(configuration)
    assert decode_backends(json_round_trip(encoded)) == configuration
    item = cast(list[dict[str, object]], encoded["backends"])[0]
    del item["lmstudio_transport"]
    assert (
        decode_backends(json_round_trip(encoded)).backend(backend.id).lmstudio_transport
        is LMStudioTransport.CHAT_COMPLETIONS
    )
    item["lmstudio_transport"] = "invalid"
    with pytest.raises(JsonValidationError, match="lmstudio_transport"):
        decode_backends(json_round_trip(encoded))


def test_ollama_idle_unload_round_trip_and_legacy_default() -> None:
    backend = replace(
        BackendConfig.create(BackendKind.OLLAMA, "Ollama", "http://localhost:11434"),
        ollama_idle_unload_minutes=7,
    )
    configuration = BackendConfiguration((backend,), backend.id)
    encoded = encode_backends(configuration)
    assert decode_backends(json_round_trip(encoded)) == configuration
    item = cast(list[dict[str, object]], encoded["backends"])[0]
    del item["ollama_idle_unload_minutes"]
    assert (
        decode_backends(json_round_trip(encoded))
        .backend(backend.id)
        .ollama_idle_unload_minutes
        is None
    )
    item["ollama_idle_unload_minutes"] = 0
    with pytest.raises(JsonValidationError, match="idle unload"):
        decode_backends(json_round_trip(encoded))


def configured_backend() -> BackendConfig:
    descriptor = ModelDescriptor(
        key="publisher/model",
        display_name="A model",
        context_length=4096,
        max_context_length=32768,
        auto_unload_minutes=5,
        publisher="Publisher",
        architecture="arch",
        quantization_name="Q4",
        bits_per_weight=4.5,
        size_bytes=100000,
        params_string="8B",
        format="gguf",
        description="Description",
        variants=("small", "large"),
        selected_variant="small",
        capabilities=ModelCapabilities(
            vision=True,
            trained_for_tool_use=False,
            reasoning=True,
            reasoning_options=("off", "on"),
            reasoning_default="on",
            parameters=ParameterSupport(True, False, None),
        ),
    )
    return (
        replace(
            BackendConfig.create(
                BackendKind.LM_STUDIO, "Local", "http://localhost:1234/v1"
            ),
            enabled=False,
            concurrency_limit=2,
            defaults=GenerationProfile(
                descriptor.key, SamplingParameters(0.5, 0.8, 1000)
            ),
        )
        .merge_discovery((descriptor,))
        .set_model_alias(descriptor.key, "My model")
        .register_model("manual")
    )


def set_json_path(
    raw: dict[str, object], path: tuple[str | int, ...], value: object
) -> None:
    def lookup(node: object, key: str | int) -> object:
        return (
            cast(list[object], node)[key]
            if isinstance(key, int)
            else cast(dict[str, object], node)[key]
        )

    node: object = raw
    for part in path[:-1]:
        node = lookup(node, part)
    final = path[-1]
    if isinstance(final, int):
        cast(list[object], node)[final] = value
    else:
        cast(dict[str, object], node)[final] = value


def test_backend_configuration_round_trip_preserves_cached_metadata_and_tombstones() -> (
    None
):
    active = configured_backend()
    retired = BackendConfig.create(
        BackendKind.OPENAI_COMPATIBLE, "Old", active.base_url
    )
    config = replace(
        BackendConfiguration((active, retired)).remove_backend(retired.id),
        default_backend_id=active.id,
    )
    encoded = encode_backends(config)
    assert decode_backends(json_round_trip(encoded)) == config
    text = json.dumps(encoded)
    assert "loaded_instance" not in text
    assert "api_key" not in text
    assert (
        decode_backends(encoded)
        .backend(active.id)
        .model("manual")
        .descriptor.capabilities.vision
        is None
    )


def test_fresh_backend_and_unassigned_chat_components_round_trip() -> None:
    assert (
        decode_backends(json_round_trip(encode_backends(BackendConfiguration())))
        == BackendConfiguration()
    )
    assert (
        decode_inference(json_round_trip(encode_inference(ChatInferenceSettings())))
        == ChatInferenceSettings()
    )


def test_chat_profiles_and_generation_labels_survive_backend_deletion_and_rename() -> (
    None
):
    first = configured_backend()
    second = BackendConfig.create(
        BackendKind.OPENAI_COMPATIBLE, "Second", first.base_url
    )
    config = BackendConfiguration((first, second), first.id)
    chat = ChatInferenceSettings.new_chat(config).select_backend(config, second.id)
    attribution = GenerationAttribution(
        ModelRef(first.id, "publisher/model"), first.name, "My model", "actual-instance"
    )
    removed = replace(config, default_backend_id=None).remove_backend(first.id)
    renamed = removed.with_backend(replace(second, name="Renamed"))
    decoded = decode_inference(json_round_trip(encode_inference(chat)))
    assert decoded == chat
    assert first.id in decoded.profiles
    assert renamed.backend(second.id).name == "Renamed"
    assert (
        decode_attribution(json_round_trip(encode_attribution(attribution)))
        == attribution
    )
    assert attribution.backend_name == "Local"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), True),
        (("schema_version",), 2),
        (("unexpected",), "setting"),
        (("backends",), {}),
        (("default_backend_id",), "missing"),
        (("backends", 0, "kind"), "unknown"),
        (("backends", 0, "id"), "Local"),
        (("backends", 0, "enabled"), 1),
        (("backends", 0, "concurrency_limit"), True),
        (("backends", 0, "concurrency_limit"), 0),
        (("backends", 0, "base_url"), "localhost"),
        (("backends", 0, "parameter_support", "temperature"), "yes"),
        (("backends", 0, "model_management"), None),
        (("backends", 0, "model_management", "automatic_loading"), "true"),
        (("backends", 0, "model_management", "idle_unload_minutes"), False),
        (("backends", 0, "defaults", "model_id"), ""),
        (("backends", 0, "defaults", "parameters", "temperature"), float("nan")),
        (("backends", 0, "defaults", "parameters", "temperature"), 10**400),
        (("backends", 0, "defaults", "parameters", "top_p"), 1.1),
        (("backends", 0, "defaults", "parameters", "max_tokens"), 1.5),
        (("backends", 0, "models", 0, "capabilities", "vision"), 1),
        (
            ("backends", 0, "models", 0, "capabilities", "reasoning_options"),
            ["arbitrary"],
        ),
        (("backends", 0, "models", 0, "capabilities", "reasoning_default"), "high"),
        (("backends", 0, "models", 0, "format"), "other"),
        (("backends", 0, "models", 0, "context_length"), -1),
        (("backends", 0, "models", 0, "size_bytes"), True),
        (("backends", 0, "models", 0, "bits_per_weight"), 0),
        (("backends", 0, "models", 0, "bits_per_weight"), float("inf")),
        (("backends", 0, "models", 0, "loaded_instance_ids"), ["live"]),
    ],
)
def test_invalid_backend_documents_fail_with_a_location(
    path: tuple[str | int, ...], value: object
) -> None:
    config = BackendConfiguration((configured_backend(),))
    raw = encode_backends(config)
    set_json_path(raw, path, value)
    with pytest.raises(JsonValidationError) as exc:
        decode_backends(raw)
    assert exc.value.issues
    assert all(issue.path.startswith("backends") for issue in exc.value.issues)


def test_missing_fields_are_rejected_instead_of_using_defaults() -> None:
    raw = encode_backends(BackendConfiguration())
    del raw["default_backend_id"]
    with pytest.raises(JsonValidationError, match="default_backend_id: is required"):
        decode_backends(raw)


def test_duplicate_active_retired_and_model_ids_are_rejected() -> None:
    backend = configured_backend()
    raw = encode_backends(BackendConfiguration((backend,)))
    active = cast(list[object], raw["backends"])
    active.append(active[0])
    with pytest.raises(JsonValidationError, match="unique"):
        decode_backends(raw)
    raw = encode_backends(BackendConfiguration((backend,)))
    raw["retired_backends"] = [
        {"id": backend.id, "kind": backend.kind.value, "name": backend.name}
    ]
    with pytest.raises(JsonValidationError, match="unique"):
        decode_backends(raw)
    raw = encode_backends(BackendConfiguration((backend,)))
    model_list = cast(
        list[object],
        cast(dict[str, object], cast(list[object], raw["backends"])[0])["models"],
    )
    model_list.append(model_list[0])
    with pytest.raises(JsonValidationError, match="duplicate registered model"):
        decode_backends(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 0),
        ("profiles", []),
        ("selected_backend_id", "missing"),
        ("extra", False),
    ],
)
def test_invalid_inference_document_is_rejected(field: str, value: object) -> None:
    raw = encode_inference(ChatInferenceSettings())
    raw[field] = value
    with pytest.raises(JsonValidationError):
        decode_inference(raw)


def test_profile_requires_valid_identity_and_selected_profile() -> None:
    backend = configured_backend()
    raw = encode_inference(
        ChatInferenceSettings(backend.id, {backend.id: backend.defaults})
    )
    raw["profiles"] = {}
    with pytest.raises(JsonValidationError, match="stored chat profile"):
        decode_inference(raw)
    raw["selected_backend_id"] = None
    raw["profiles"] = {
        "named-backend": {
            "model_id": None,
            "parameters": {"temperature": 1.0, "top_p": 1.0, "max_tokens": 1},
        }
    }
    with pytest.raises(JsonValidationError, match="backend id"):
        decode_inference(raw)


def test_attribution_rejects_missing_or_blank_actual_identity() -> None:
    backend = configured_backend()
    raw = encode_attribution(
        GenerationAttribution(ModelRef(backend.id, "model"), "Local", "Model")
    )
    raw["backend_id"] = ""
    with pytest.raises(JsonValidationError, match="backend id"):
        decode_attribution(raw)
