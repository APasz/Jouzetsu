from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import cast

import pytest

from jouzetsu.backend_models import ModelRef, new_backend_id
from jouzetsu.backend_protocol import (
    BackendError,
    BackendErrorCategory,
    BackendHealth,
    BackendReachability,
    GenerationRequest,
    InferenceMessage,
    ModelDiscovery,
)
from jouzetsu.models import ImageAttachment, ImageMediaType
from jouzetsu.runtime import ModelCapabilities, ModelDescriptor
from jouzetsu.sampling import (
    CHAT_COMPLETIONS_PARAMETERS,
    ParameterSupport,
    SamplingParameters,
)


def request(*, capabilities: ModelCapabilities | None = None) -> GenerationRequest:
    return GenerationRequest(
        model=ModelRef(new_backend_id(), "model"),
        messages=(
            InferenceMessage("system", "Instructions"),
            InferenceMessage("user", "First"),
            InferenceMessage("user", "Second"),
        ),
        capabilities=capabilities if capabilities is not None else ModelCapabilities(),
        parameter_support=CHAT_COMPLETIONS_PARAMETERS,
    )


def image() -> ImageAttachment:
    return ImageAttachment("a" * 64, "example.png", ImageMediaType.PNG, 10, 10, 100)


def test_request_is_an_immutable_ordered_snapshot_including_consecutive_roles() -> None:
    prepared = request()
    assert [message.role for message in prepared.messages] == ["system", "user", "user"]
    assert [message.content for message in prepared.messages] == [
        "Instructions",
        "First",
        "Second",
    ]
    with pytest.raises(TypeError, match="immutable tuple"):
        replace(
            prepared,
            messages=cast(tuple[InferenceMessage, ...], list(prepared.messages)),
        )


@pytest.mark.parametrize("vision", [False, None])
def test_image_requests_are_blocked_without_known_vision_support(
    vision: bool | None,
) -> None:
    prepared = request(capabilities=ModelCapabilities(vision=vision))
    with pytest.raises(BackendError) as exc:
        replace(prepared, messages=(InferenceMessage("user", "", (image(),)),))
    assert exc.value.category is BackendErrorCategory.INCOMPATIBLE


def test_vision_request_preserves_original_image_references_and_text() -> None:
    original = image()
    message = InferenceMessage("user", "Look at this", (original,))
    prepared = replace(
        request(capabilities=ModelCapabilities(vision=True)), messages=(message,)
    )
    assert prepared.messages == (message,)
    assert prepared.messages[0].images[0] is original


def test_invalid_message_and_empty_request_are_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        InferenceMessage("user", "")
    with pytest.raises(ValueError, match="only user"):
        InferenceMessage("assistant", "Text", (image(),))
    with pytest.raises(TypeError, match="immutable tuple"):
        InferenceMessage("user", "Text", cast(tuple[ImageAttachment, ...], [image()]))
    with pytest.raises(ValueError, match="at least one"):
        replace(request(), messages=())


def test_cross_backend_request_is_rejected_before_transport() -> None:
    prepared = request()
    prepared.validate_backend(prepared.model.backend_id)
    with pytest.raises(BackendError) as exc:
        prepared.validate_backend(new_backend_id())
    assert exc.value.category is BackendErrorCategory.INCOMPATIBLE


def test_model_restrictions_and_unknown_backend_support_are_omitted_from_wire_values() -> (
    None
):
    prepared = replace(
        request(
            capabilities=ModelCapabilities(
                parameters=ParameterSupport(temperature=False)
            )
        ),
        parameter_support=ParameterSupport(True, None, True),
        parameters=SamplingParameters(1.7, 0.4, 800),
    )
    assert prepared.wire_parameters == {"max_tokens": 800}
    assert prepared.parameters == SamplingParameters(1.7, 0.4, 800)
    wire = prepared.wire_parameters
    wire["max_tokens"] = 1
    assert prepared.wire_parameters == {"max_tokens": 800}


def test_discovery_unsupported_empty_and_invalid_are_distinct() -> None:
    assert ModelDiscovery(False) != ModelDiscovery(True)
    descriptor = ModelDescriptor("model", "Model")
    with pytest.raises(ValueError, match="unsupported"):
        ModelDiscovery(False, (descriptor,))
    with pytest.raises(ValueError, match="duplicate"):
        ModelDiscovery(True, (descriptor, descriptor))
    with pytest.raises(TypeError, match="boolean"):
        ModelDiscovery(cast(bool, 1))


def test_health_retains_online_offline_unknown_and_stale_states() -> None:
    now = datetime.now(UTC)
    assert BackendHealth().state is BackendReachability.UNKNOWN
    for state in (
        BackendReachability.ONLINE,
        BackendReachability.OFFLINE,
        BackendReachability.STALE,
    ):
        assert BackendHealth(state, now).state is state
        with pytest.raises(ValueError, match="timestamp"):
            BackendHealth(state)
    with pytest.raises(ValueError, match="timezone"):
        BackendHealth(BackendReachability.ONLINE, now.replace(tzinfo=None))


def test_errors_have_explicit_categories_and_explanations() -> None:
    failure = BackendError(
        BackendErrorCategory.TIMEOUT, "The backend stopped responding."
    )
    assert failure.category is BackendErrorCategory.TIMEOUT
    assert str(failure) == "The backend stopped responding."
    with pytest.raises(ValueError, match="explanation"):
        BackendError(BackendErrorCategory.OTHER, "")
