from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from jouzetsu.backend_models import (
    BackendConfig,
    BackendConfiguration,
    BackendId,
    BackendKind,
    ChatInferenceSettings,
    GenerationAttribution,
    GenerationProfile,
    LMStudioTransport,
    ModelManagementSettings,
    ModelRef,
    RegisteredModel,
)
from jouzetsu.runtime import (
    ModelCapabilities,
    ModelDescriptor,
    ModelInstanceDescriptor,
    vision_model_error,
)
from jouzetsu.sampling import ParameterSupport, SamplingParameters


def backend(name: str = "Local", *, model: str = "model-a") -> BackendConfig:
    return replace(
        BackendConfig.create(
            BackendKind.OPENAI_COMPATIBLE, name, "http://localhost:1234/v1"
        ),
        defaults=GenerationProfile(model),
    )


def test_identity_survives_edits_and_same_endpoint_gets_a_new_identity() -> None:
    original = backend()
    renamed = replace(
        original, name="Renamed", base_url="https://other.example/prefix/v1/"
    )
    recreated = backend()
    assert original.id == renamed.id
    assert original.id != recreated.id
    assert ModelRef(original.id, "same") != ModelRef(recreated.id, "same")


def test_responses_transport_is_scoped_to_lm_studio() -> None:
    with pytest.raises(ValueError, match="only available for LM Studio"):
        replace(backend(), lmstudio_transport=LMStudioTransport.RESPONSES)


def test_ollama_idle_unload_requires_positive_minutes_and_ollama() -> None:
    ollama = BackendConfig.create(
        BackendKind.OLLAMA, "Ollama", "http://localhost:11434"
    )
    for value in (0, -1, True):
        with pytest.raises(ValueError, match="positive integer"):
            replace(ollama, ollama_idle_unload_minutes=value)
    with pytest.raises(ValueError, match="only available for Ollama"):
        replace(backend(), ollama_idle_unload_minutes=1)


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:1234",
        "https://example.test/custom/api",
        "https://example.test/v1/",
    ],
)
def test_custom_api_roots_are_preserved(url: str) -> None:
    assert replace(backend(), base_url=url).base_url == url


@pytest.mark.parametrize(
    "url",
    [
        "localhost:1234/v1",
        "ftp://localhost/v1",
        "http://user:secret@localhost/v1",
        "http://localhost/v1?token=secret",
        "http://localhost/v1#fragment",
        "http://local host/v1",
        "http://localhost:70000/v1",
    ],
)
def test_invalid_backend_urls_are_rejected(url: str) -> None:
    with pytest.raises(ValueError):
        replace(backend(), base_url=url)


@pytest.mark.parametrize(
    "identity", ["", "Local", "http://localhost/v1", "be_ABCD", "be_" + "g" * 32]
)
def test_backend_ids_are_opaque_and_validated(identity: str) -> None:
    with pytest.raises(ValueError, match="backend id"):
        replace(backend(), id=BackendId(identity))


@pytest.mark.parametrize("limit", [0, -1, True])
def test_concurrency_requires_a_positive_integer(limit: int) -> None:
    with pytest.raises(ValueError, match="concurrency"):
        replace(backend(), concurrency_limit=limit)


def test_management_is_optional_and_lm_studio_defaults_to_automatic_loading() -> None:
    generic = backend()
    native = BackendConfig.create(BackendKind.LM_STUDIO, "Native", generic.base_url)
    assert generic.model_management is None
    assert native.model_management == ModelManagementSettings(automatic_loading=True)
    with pytest.raises(ValueError, match="generic"):
        replace(generic, model_management=ModelManagementSettings())
    with pytest.raises(ValueError, match="explicit"):
        replace(native, model_management=None)


def test_new_chats_snapshot_defaults_without_later_inheritance() -> None:
    initial = backend()
    config = BackendConfiguration((initial,), initial.id)
    chat = ChatInferenceSettings.new_chat(config)
    changed = replace(
        initial,
        defaults=GenerationProfile("model-b", SamplingParameters(1.2, 0.5, 512)),
    )
    updated_config = config.with_backend(changed)
    assert chat.selected_profile == initial.defaults
    assert (
        ChatInferenceSettings.new_chat(updated_config).selected_profile
        == changed.defaults
    )
    assert chat.reset_to_defaults(updated_config).selected_profile == changed.defaults
    assert chat.selected_profile == initial.defaults


def test_chats_created_without_a_default_remain_unassigned() -> None:
    empty = BackendConfiguration()
    chat = ChatInferenceSettings.new_chat(empty)
    added = backend()
    configured = replace(empty.with_backend(added), default_backend_id=added.id)
    assert chat == ChatInferenceSettings()
    assert chat.selected_model is None
    assert ChatInferenceSettings.new_chat(configured).selected_model == ModelRef(
        added.id, "model-a"
    )
    with pytest.raises(ValueError, match="select a backend"):
        chat.set_profile(GenerationProfile())


def test_first_switch_copies_only_compatible_values_and_preserves_target_model() -> (
    None
):
    source = replace(
        backend("Source"),
        defaults=GenerationProfile("source", SamplingParameters(1.4, 0.4, 100)),
    )
    target = replace(
        backend("Target", model="target"),
        parameter_support=ParameterSupport(True, False, None),
    )
    config = BackendConfiguration((source, target), source.id)
    original = ChatInferenceSettings.new_chat(config)
    switched = original.select_backend(config, target.id)
    expected = GenerationProfile(
        "target",
        SamplingParameters(
            1.4, target.defaults.parameters.top_p, target.defaults.parameters.max_tokens
        ),
    )
    assert switched.selected_profile == expected
    changed = switched.set_profile(
        GenerationProfile("second-target", SamplingParameters(0.1, 0.2, 200))
    )
    returned = changed.select_backend(config, source.id)
    assert returned.selected_profile == source.defaults
    assert (
        returned.select_backend(config, target.id).selected_profile
        == changed.selected_profile
    )
    assert original.profiles == {source.id: source.defaults}


def test_model_parameter_restrictions_apply_to_first_profile_copy() -> None:
    source = replace(
        backend("Source"), defaults=GenerationProfile("source", SamplingParameters(1.4))
    )
    target = backend("Target", model="restricted")
    descriptor = ModelDescriptor(
        "restricted",
        "Restricted",
        capabilities=ModelCapabilities(parameters=ParameterSupport(temperature=False)),
    )
    target = target.merge_discovery((descriptor,))
    config = BackendConfiguration((source, target), source.id)
    chat = ChatInferenceSettings.new_chat(config).select_backend(config, target.id)
    assert chat.selected_profile == target.defaults


def test_profile_maps_and_nested_values_are_immutable_snapshots() -> None:
    selected = backend()
    profiles = {selected.id: selected.defaults}
    chat = ChatInferenceSettings(selected.id, profiles)
    profiles.clear()
    assert chat.selected_profile == selected.defaults
    with pytest.raises(TypeError):
        cast(dict[BackendId, GenerationProfile], chat.profiles)[selected.id] = (
            GenerationProfile()
        )


def test_discovery_retains_omitted_models_manual_registration_and_aliases() -> None:
    initial = backend().register_model("manual", alias="My alias")
    discovered = ModelDescriptor(
        "manual",
        "Downloaded name",
        loaded_instance_ids=("live-instance",),
        loaded_instances=(ModelInstanceDescriptor("live-instance"),),
        capabilities=ModelCapabilities(vision=True),
        max_context_length=32768,
    )
    refreshed = initial.merge_discovery((discovered, ModelDescriptor("other", "Other")))
    manual = refreshed.model("manual")
    assert manual.alias == "My alias"
    assert manual.manually_registered
    assert manual.descriptor.capabilities.vision is True
    assert manual.descriptor.max_context_length == 32768
    assert not manual.descriptor.is_loaded
    assert not manual.descriptor.loaded_instances
    assert refreshed.merge_discovery(()) is refreshed
    assert (
        refreshed.merge_discovery((ModelDescriptor("other", "Updated"),)).model(
            "manual"
        )
        == manual
    )
    assert (
        refreshed.set_model_alias("manual", "Renamed").model("manual").alias
        == "Renamed"
    )
    with pytest.raises(ValueError, match="already registered"):
        refreshed.register_model("manual")
    with pytest.raises(ValueError, match="duplicate"):
        refreshed.merge_discovery((discovered, discovered))


def test_discovery_resets_disappearing_reasoning_default_permanently() -> None:
    descriptor = ModelDescriptor(
        "model-a",
        "Model",
        capabilities=ModelCapabilities(reasoning=True, reasoning_options=("off", "on")),
    )
    original = backend().merge_discovery((descriptor,))
    configured = replace(original.model("model-a"), reasoning_default="on")
    original = replace(original, models=(configured,))
    unknown = original.merge_discovery(
        (replace(descriptor, capabilities=ModelCapabilities()),)
    )
    assert unknown.model("model-a").reasoning_default == "on"
    assert not unknown.model("model-a").reasoning_default_unavailable
    changed = replace(
        descriptor,
        capabilities=ModelCapabilities(reasoning=True, reasoning_options=("off",)),
    )
    refreshed = original.merge_discovery((changed,))
    assert refreshed.model("model-a").reasoning_default is None
    assert refreshed.model("model-a").reasoning_default_unavailable
    restored = refreshed.merge_discovery((descriptor,))
    assert restored.model("model-a").reasoning_default is None
    assert restored.model("model-a").reasoning_default_unavailable


def test_unknown_capabilities_are_distinct_and_never_imply_vision_support() -> None:
    manual = backend().register_model("manual").model("manual").descriptor
    unsupported = replace(manual, capabilities=ModelCapabilities(vision=False))
    supported = replace(manual, capabilities=ModelCapabilities(vision=True))
    assert manual.capabilities.vision is None
    assert manual.capabilities != unsupported.capabilities
    assert vision_model_error(manual)
    assert vision_model_error(unsupported)
    assert not vision_model_error(supported)


def test_deleted_backend_profiles_and_attribution_cannot_be_reassociated() -> None:
    original = backend()
    config = BackendConfiguration((original,))
    chat = ChatInferenceSettings().select_backend(config, original.id)
    attribution = GenerationAttribution(
        ModelRef(original.id, "model-a"), original.name, "Original model", "instance-a"
    )
    removed = config.remove_backend(original.id)
    recreated = backend()
    updated = removed.with_backend(recreated)
    assert chat.selected_model == attribution.model
    assert original.id in chat.profiles
    assert updated.retired_backends[0].name == original.name
    with pytest.raises(KeyError, match="deleted"):
        updated.backend(original.id)
    with pytest.raises(ValueError, match="retired"):
        updated.with_backend(original)
    with pytest.raises(ValueError, match="default"):
        replace(config, default_backend_id=original.id).remove_backend(original.id)


def test_duplicate_backend_and_model_ids_are_rejected() -> None:
    selected = backend()
    with pytest.raises(ValueError, match="unique"):
        BackendConfiguration((selected, selected))
    model = RegisteredModel(ModelDescriptor("a", "A"))
    with pytest.raises(ValueError, match="duplicate"):
        replace(selected, models=(model, model))


def test_unsupported_and_unknown_parameters_keep_saved_values_without_emission() -> (
    None
):
    saved = SamplingParameters(1.2, 0.3, 256)
    support = ParameterSupport(temperature=False, top_p=None, max_tokens=True)
    assert support.wire_values(saved) == {"max_tokens": 256}
    assert saved == SamplingParameters(1.2, 0.3, 256)


@pytest.mark.parametrize(
    "parameters",
    [
        (float("nan"), 0.5, 1),
        (float("inf"), 0.5, 1),
        (-0.1, 0.5, 1),
        (2.1, 0.5, 1),
        (1.0, 1.1, 1),
        (1.0, 0.5, True),
        (1.0, 0.5, 0),
    ],
)
def test_sampling_rejects_invalid_values(parameters: tuple[float, float, int]) -> None:
    with pytest.raises(ValueError):
        SamplingParameters(*parameters)
