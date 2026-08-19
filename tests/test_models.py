from __future__ import annotations

import math
from unittest.mock import patch

import pytest

from jouzetsu.config import GenerationSettings
from jouzetsu.character_presets import BASE_CHARACTER_PRESETS, EXTRA_CHARACTER_PRESETS
from jouzetsu.models import ChatTitleSource, Character, CharacterChatBinding, CharacterField, Chat, ChatSamplingOverrides, Message


@pytest.mark.parametrize("timestamp", ["NaN", "Infinity", "-Infinity", -1])
def test_message_rejects_non_finite_or_negative_timestamp(timestamp: object) -> None:
    with pytest.raises(ValueError, match="non-negative finite"):
        _ = Message.from_dict({"role": "user", "created_at": timestamp})


@pytest.mark.parametrize("timestamp", ["NaN", "Infinity", "-Infinity", -1])
def test_chat_rejects_non_finite_or_negative_timestamp(timestamp: object) -> None:
    with pytest.raises(ValueError, match="non-negative finite"):
        _ = Chat.from_dict({"created_at": timestamp})


def test_message_accepts_finite_non_negative_timestamp() -> None:
    message: Message = Message.from_dict({"role": "user", "created_at": "123.5"})

    assert math.isfinite(message.created_at)
    assert message.created_at == 123.5
    assert message.updated_at == 123.5


def test_message_updated_at_round_trips_separately_from_created_at() -> None:
    message = Message(role="assistant", content="Revised", created_at=123.5, updated_at=456.0)

    reloaded = Message.from_dict(message.to_dict())

    assert reloaded.created_at == 123.5
    assert reloaded.updated_at == 456.0


def test_message_content_update_preserves_created_at() -> None:
    message = Message(role="user", content="Original", created_at=123.5, updated_at=123.5)

    with patch("jouzetsu.models.time.time", return_value=456.0):
        message.update_content("Revised")

    assert message.content == "Revised"
    assert message.created_at == 123.5
    assert message.updated_at == 456.0


def test_continuity_rewrite_round_trips_and_is_cleared_by_a_content_edit() -> None:
    message = Message(role="assistant", content="Original", created_at=123.5, updated_at=123.5)
    message.set_continuity_rewrite("Revised")

    reloaded = Message.from_dict(message.to_dict())
    reloaded.update_content("Manually edited")

    assert message.continuity_rewrite == "Revised"
    assert reloaded.continuity_rewrite == ""


def test_continuity_rewrite_rejects_non_assistant_messages() -> None:
    message = Message(role="user", content="Question")

    with pytest.raises(ValueError, match="only assistant messages"):
        message.set_continuity_rewrite("Revised")

    with pytest.raises(ValueError, match="only assistant messages"):
        _ = Message(role="user", content="Question", continuity_rewrite="Revised")


def test_chat_postprocess_british_spellings_round_trips() -> None:
    chat = Chat(postprocess_british_spellings=True)

    reloaded = Chat.from_dict(chat.to_dict())

    assert reloaded.postprocess_british_spellings is True


def test_chat_enables_british_spelling_postprocessing_by_default() -> None:
    chat = Chat.from_dict({})

    assert Chat().postprocess_british_spellings is True
    assert chat.postprocess_british_spellings is True


def test_message_reasoning_round_trips_and_is_cleared_by_a_content_edit() -> None:
    message = Message(role="assistant", content="Answer", reasoning="Reviewed the request before responding.")

    reloaded = Message.from_dict(message.to_dict())
    reloaded.update_content("Edited answer")

    assert message.reasoning == "Reviewed the request before responding."
    assert reloaded.reasoning == ""


def test_assistant_message_model_round_trips_and_survives_a_content_edit() -> None:
    message = Message(role="assistant", content="Answer", model="acme/precise-1")

    reloaded = Message.from_dict(message.to_dict())
    reloaded.update_content("Edited answer")

    assert reloaded.model == "acme/precise-1"


def test_message_rejects_invalid_model() -> None:
    with pytest.raises(ValueError, match="message.model must be text"):
        _ = Message.from_dict({"role": "assistant", "model": ["not text"]})

    with pytest.raises(ValueError, match="only assistant messages can have models"):
        _ = Message(role="user", content="Question", model="acme/precise-1")


def test_message_rejects_invalid_reasoning() -> None:
    with pytest.raises(ValueError, match="message.reasoning must be text"):
        _ = Message.from_dict({"role": "assistant", "reasoning": ["not text"]})

    with pytest.raises(ValueError, match="only assistant messages can have reasoning"):
        _ = Message(role="user", content="Question", reasoning="Not allowed")


def test_chat_enables_reasoning_persistence_by_default() -> None:
    assert Chat().save_reasoning is True
    assert Chat.from_dict({}).save_reasoning is True


def test_chat_title_source_round_trips_and_manual_title_is_retained() -> None:
    chat = Chat()
    chat.set_manual_title("New chat")

    reloaded = Chat.from_dict(chat.to_dict())
    reloaded.set_auto_title_from_first_message("A first message")

    assert reloaded.title == "New chat"
    assert reloaded.title_source is ChatTitleSource.MANUAL


def test_legacy_new_chat_placeholder_loads_as_a_pristine_chat() -> None:
    chat = Chat.from_dict(
        {
            "title": "New chat",
            "messages": [{"role": "user", "content": ""}],
        }
    )

    assert chat.messages == []
    assert chat.title_source is ChatTitleSource.AUTO


@pytest.mark.parametrize("title_source", ["generated", 1, None])
def test_chat_rejects_invalid_explicit_title_source(title_source: object) -> None:
    with pytest.raises(ValueError, match="title_source"):
        _ = Chat.from_dict({"title_source": title_source})


def test_chat_sampling_overrides_round_trip_and_resolve_against_global_defaults() -> None:
    chat = Chat(sampling_overrides=ChatSamplingOverrides(temperature=0.2, max_tokens=512))

    reloaded = Chat.from_dict(chat.to_dict())
    effective = reloaded.sampling_overrides.resolve(GenerationSettings(temperature=0.8, top_p=0.9, max_tokens=1024))

    assert reloaded.sampling_overrides == ChatSamplingOverrides(temperature=0.2, max_tokens=512)
    assert effective.temperature == 0.2
    assert effective.top_p == 0.9
    assert effective.max_tokens == 512


def test_chat_rejects_invalid_persisted_sampling_overrides() -> None:
    with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
        _ = Chat.from_dict({"sampling_overrides": {"temperature": 2.1}})


def test_chat_fork_preserves_chat_settings_and_message_reasoning() -> None:
    chat = Chat(postprocess_british_spellings=True, save_reasoning=False)
    _ = chat.add_message("user", "Hello")
    message = chat.add_message("assistant", "Answer")
    message.set_reasoning("Source trace")

    forked = chat.fork_through(message.id)

    assert forked.postprocess_british_spellings is True
    assert forked.save_reasoning is False
    assert forked.messages[-1].reasoning == "Source trace"


def test_merging_assistant_messages_with_different_models_clears_model_provenance() -> None:
    chat = Chat()
    first = chat.add_message("assistant", "First response")
    first.set_model("acme/first")
    second = chat.add_message("assistant", "Second response")
    second.set_model("acme/second")

    merged = chat.merge_message_with_previous(second.id)

    assert merged.content == "First response\n\nSecond response"
    assert merged.model == ""


def test_character_uses_its_own_field_schema_and_compiles_only_populated_values() -> None:
    character = Character(
        name="Mira",
        fields=[
            CharacterField(label="Personality", kind="long_text", value="Warm and observant."),
            CharacterField(label="Height", value=""),
        ],
    )

    prompt = character.compiled_system_prompt()
    reloaded = Character.from_dict(character.to_dict())

    assert "You are roleplaying as Mira." in prompt
    assert "Personality: Warm and observant." in prompt
    assert "Height:" not in prompt
    assert reloaded == character


def test_character_field_presets_have_one_base_layer_and_composable_extra_layer() -> None:
    assert {preset.layer for preset in BASE_CHARACTER_PRESETS} == {"base"}
    assert {preset.layer for preset in EXTRA_CHARACTER_PRESETS} == {"extra"}
    assert all(preset.fields for preset in (*BASE_CHARACTER_PRESETS, *EXTRA_CHARACTER_PRESETS))
    assert all(preset.fields_json().startswith("[") for preset in (*BASE_CHARACTER_PRESETS, *EXTRA_CHARACTER_PRESETS))


def test_character_revised_profile_only_advances_revision_when_data_changes() -> None:
    field = CharacterField(label="Role", value="Pilot")
    character = Character(name="Mira", revision=3, fields=[field])

    unchanged = character.revised_profile("Mira", [field])

    assert unchanged is character
    assert character.revision == 3

    revised = character.revised_profile("Mira", [CharacterField(id=field.id, label="Role", value="Navigator")])

    assert revised is not character
    assert character.revision == 3
    assert revised.revision == 4
    assert revised.fields[0].value == "Navigator"


def test_character_chat_binding_round_trips_with_chat_and_fork() -> None:
    binding = CharacterChatBinding(id="abc123", name="Mira", revision=2)
    chat = Chat(character=binding)
    message = chat.add_message("user", "Hello")

    reloaded = Chat.from_dict(chat.to_dict())
    forked = chat.fork_through(message.id)

    assert reloaded.character == binding
    assert forked.character == binding
