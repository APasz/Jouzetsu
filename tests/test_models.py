from __future__ import annotations

import math
from typing import cast
from unittest.mock import patch

import pytest

from jouzetsu.character_presets import BASE_CHARACTER_PRESETS, EXTRA_CHARACTER_PRESETS
from jouzetsu.config import GenerationSettings
from jouzetsu.models import (
    Character,
    CharacterChatBinding,
    CharacterField,
    CharacterName,
    CharacterPresetSelection,
    Chat,
    ChatPromptMode,
    ChatSamplingOverrides,
    ChatTitleSource,
    Message,
    character_cast_chat_title,
    chat_prompt_mode_from_text,
    compiled_character_cast_system_prompt,
    compiled_character_chat_system_prompt,
)


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
    message = Message(
        role="assistant", content="Revised", created_at=123.5, updated_at=456.0
    )

    reloaded = Message.from_dict(message.to_dict())

    assert reloaded.created_at == 123.5
    assert reloaded.updated_at == 456.0


def test_message_content_update_preserves_created_at() -> None:
    message = Message(
        role="user", content="Original", created_at=123.5, updated_at=123.5
    )

    with patch("jouzetsu.models.time.time", return_value=456.0):
        message.update_content("Revised")

    assert message.content == "Revised"
    assert message.created_at == 123.5
    assert message.updated_at == 456.0


def test_continuity_rewrite_round_trips_and_is_cleared_by_a_content_edit() -> None:
    message = Message(
        role="assistant", content="Original", created_at=123.5, updated_at=123.5
    )
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


def test_chat_document_schema_version_round_trips_and_loads_legacy_documents() -> None:
    chat = Chat(title="Versioned chat")
    document = chat.to_dict()
    legacy_document: dict[str, object] = dict(document)
    _ = legacy_document.pop("schema_version")

    assert document["schema_version"] == 1
    assert Chat.from_dict(document) == chat
    assert Chat.from_dict(legacy_document).to_dict()["schema_version"] == 1


def test_chat_rejects_a_future_document_schema_version() -> None:
    document: dict[str, object] = dict(Chat().to_dict())
    document["schema_version"] = 2

    with pytest.raises(ValueError, match="newer than supported version 1"):
        _ = Chat.from_dict(document)


def test_chat_enables_british_spelling_postprocessing_by_default() -> None:
    chat = Chat.from_dict({})

    assert Chat().postprocess_british_spellings is True
    assert chat.postprocess_british_spellings is True


def test_message_reasoning_round_trips_and_is_cleared_by_a_content_edit() -> None:
    message = Message(
        role="assistant",
        content="Answer",
        reasoning="Reviewed the request before responding.",
    )

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
    with pytest.raises(TypeError, match="message.model must be text"):
        _ = Message.from_dict({"role": "assistant", "model": ["not text"]})

    with pytest.raises(ValueError, match="only assistant messages can have models"):
        _ = Message(role="user", content="Question", model="acme/precise-1")


def test_message_rejects_invalid_reasoning() -> None:
    with pytest.raises(TypeError, match="message.reasoning must be text"):
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


def test_chat_rejects_invalid_explicit_title_source_value() -> None:
    with pytest.raises(ValueError, match="title_source"):
        _ = Chat.from_dict({"title_source": "generated"})


@pytest.mark.parametrize("title_source", [1, None])
def test_chat_rejects_non_text_explicit_title_source(title_source: object) -> None:
    with pytest.raises(TypeError, match="title_source"):
        _ = Chat.from_dict({"title_source": title_source})


def test_chat_sampling_overrides_round_trip_and_resolve_against_global_defaults() -> (
    None
):
    chat = Chat(
        sampling_overrides=ChatSamplingOverrides(temperature=0.2, max_tokens=512)
    )

    reloaded = Chat.from_dict(chat.to_dict())
    effective = reloaded.sampling_overrides.resolve(
        GenerationSettings(temperature=0.8, top_p=0.9, max_tokens=1024)
    )

    assert reloaded.sampling_overrides == ChatSamplingOverrides(
        temperature=0.2, max_tokens=512
    )
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


def test_merging_assistant_messages_with_different_models_clears_model_provenance() -> (
    None
):
    chat = Chat()
    first = chat.add_message("assistant", "First response")
    first.set_model("acme/first")
    second = chat.add_message("assistant", "Second response")
    second.set_model("acme/second")

    merged = chat.merge_message_with_previous(second.id)

    assert merged.content == "First response\n\nSecond response"
    assert merged.model == ""


def test_character_uses_its_own_field_schema_and_compiles_only_populated_values() -> (
    None
):
    character = Character(
        name_parts=CharacterName("Mira"),
        fields=[
            CharacterField(
                label="Personality", kind="long_text", value="Warm and observant."
            ),
            CharacterField(label="Height", value=""),
        ],
    )

    prompt = character.compiled_system_prompt()
    reloaded = Character.from_dict(character.to_dict())

    assert "You are roleplaying as Mira." in prompt
    assert "Personality: Warm and observant." in prompt
    assert "Height:" not in prompt
    assert reloaded == character


def test_character_name_supports_an_optional_family_name() -> None:
    full_name = CharacterName.from_display_name("Mira Ash")
    mononym = CharacterName(given_name=" Mira ", family_name=" ")
    character = Character(name_parts=mononym)

    assert full_name.given_name == "Mira"
    assert full_name.family_name == "Ash"
    assert full_name.display_name == "Mira Ash"
    assert mononym.display_name == "Mira"
    assert character.name_parts == CharacterName(given_name="Mira")
    assert character.compiled_system_prompt().startswith(
        "You are roleplaying as Mira.\n\n"
    )


def test_character_name_round_trips_multiword_parts_and_loads_legacy_display_names() -> (
    None
):
    name = CharacterName(given_name="Mary Jane", family_name="van Helsing")
    character = Character(name_parts=name)
    serialized = character.to_dict()
    reloaded = Character.from_dict(serialized)
    legacy = Character.from_dict({"id": "legacy123", "name": "Mira Ash"})

    assert serialized["schema_version"] == 1
    assert serialized["name"] == {"given": "Mary Jane", "family": "van Helsing"}
    assert serialized["is_draft"] is False
    assert reloaded.name_parts == name
    assert reloaded.name == "Mary Jane van Helsing"
    assert legacy.name_parts == CharacterName(given_name="Mira", family_name="Ash")
    assert legacy.is_draft is False


def test_character_draft_state_preserves_onboarding_and_allows_the_literal_old_name() -> (
    None
):
    draft = Character.draft()
    draft_document = draft.to_dict()
    named = draft.revised_profile(CharacterName("New", "character"), [])

    assert draft.is_draft is True
    assert draft.name_parts is None
    assert draft_document["schema_version"] == 1
    assert draft_document["is_draft"] is True
    assert draft_document["name"] is None
    assert Character.from_dict(draft_document) == draft
    assert named.is_draft is False
    assert named.name == "New character"
    assert named.compiled_system_prompt().startswith(
        "You are roleplaying as New character.\n\n"
    )


def test_legacy_default_character_name_migrates_to_an_explicit_draft() -> None:
    legacy_document: dict[str, object] = {
        "id": "legacy123",
        "name": "New character",
        "fields": [
            {
                "id": "role",
                "label": "Role",
                "kind": "short_text",
                "value": "Pilot",
            }
        ],
        "presets": {"base": "builtin:humanoid", "extras": ["builtin:voice"]},
    }

    migrated = Character.from_dict(legacy_document)
    canonical_document = migrated.to_dict()

    assert migrated.is_draft is True
    assert migrated.name_parts is None
    assert migrated.fields[0].value == "Pilot"
    assert migrated.presets == CharacterPresetSelection(
        base_id="builtin:humanoid", extra_ids=("builtin:voice",)
    )
    assert canonical_document["schema_version"] == 1
    assert canonical_document["is_draft"] is True
    assert canonical_document["name"] is None
    assert Character.from_dict(canonical_document) == migrated


def test_character_rejects_a_future_document_schema_version() -> None:
    document: dict[str, object] = dict(
        Character(name_parts=CharacterName("Mira")).to_dict()
    )
    document["schema_version"] = 2

    with pytest.raises(ValueError, match="newer than supported version 1"):
        _ = Character.from_dict(document)


def test_character_preserves_selected_profile_templates_and_loads_older_profiles() -> (
    None
):
    character = Character(
        name_parts=CharacterName("Mira"),
        presets=CharacterPresetSelection(
            base_id="builtin:humanoid",
            extra_ids=("builtin:identity", "builtin:voice"),
        ),
    )
    serialized = character.to_dict()
    reloaded = Character.from_dict(serialized)
    legacy = dict(serialized)
    _ = legacy.pop("presets")

    assert reloaded.presets == character.presets
    assert Character.from_dict(legacy).presets == CharacterPresetSelection()


def test_character_preset_selection_rejects_mutable_or_non_text_values() -> None:
    with pytest.raises(TypeError, match="base preset id must be text"):
        _ = CharacterPresetSelection(base_id=cast(str, 1))
    with pytest.raises(TypeError, match="immutable tuple"):
        _ = CharacterPresetSelection(
            extra_ids=cast(tuple[str, ...], ["builtin:identity"])
        )
    with pytest.raises(TypeError, match="preset ids must be text"):
        _ = CharacterPresetSelection(extra_ids=cast(tuple[str, ...], (1,)))


def test_character_preset_selection_rejects_invalid_persisted_value_types() -> None:
    with pytest.raises(TypeError, match="presets.base must be text"):
        _ = CharacterPresetSelection.from_dict({"base": 1})
    with pytest.raises(TypeError, match="presets.extras must be a list"):
        _ = CharacterPresetSelection.from_dict({"extras": "builtin:identity"})
    with pytest.raises(TypeError, match="presets.extras must be a list"):
        _ = CharacterPresetSelection.from_dict({"extras": ["builtin:identity", 1]})


def test_character_field_presets_have_one_base_layer_and_composable_extra_layer() -> (
    None
):
    assert {preset.layer for preset in BASE_CHARACTER_PRESETS} == {"base"}
    assert {preset.layer for preset in EXTRA_CHARACTER_PRESETS} == {"extra"}
    assert all(
        preset.fields for preset in (*BASE_CHARACTER_PRESETS, *EXTRA_CHARACTER_PRESETS)
    )
    assert all(
        preset.fields_json().startswith("[")
        for preset in (*BASE_CHARACTER_PRESETS, *EXTRA_CHARACTER_PRESETS)
    )


def test_character_revised_profile_only_advances_revision_when_data_changes() -> None:
    field = CharacterField(label="Role", value="Pilot")
    character = Character(name_parts=CharacterName("Mira"), revision=3, fields=[field])

    unchanged = character.revised_profile(CharacterName("Mira"), [field])

    assert unchanged is character
    assert character.revision == 3

    revised = character.revised_profile(
        CharacterName("Mira"),
        [CharacterField(id=field.id, label="Role", value="Navigator")],
    )

    assert revised is not character
    assert character.revision == 3
    assert revised.revision == 4
    assert revised.fields[0].value == "Navigator"


def test_character_revised_profile_advances_revision_when_template_choices_change() -> (
    None
):
    character = Character(name_parts=CharacterName("Mira"), revision=3)

    revised = character.revised_profile(
        CharacterName("Mira"),
        [],
        presets=CharacterPresetSelection(extra_ids=("builtin:identity",)),
    )

    assert revised.revision == 4
    assert revised.presets.extra_ids == ("builtin:identity",)


def test_character_chat_binding_round_trips_with_chat_and_fork() -> None:
    binding = CharacterChatBinding(id="abc123", name="Mira", revision=2)
    chat = Chat(character_cast=(binding,))
    message = chat.add_message("user", "Hello")

    reloaded = Chat.from_dict(chat.to_dict())
    forked = chat.fork_through(message.id)

    assert reloaded.character == binding
    assert forked.character == binding
    assert reloaded.character_cast == (binding,)
    assert forked.character_cast == (binding,)
    assert reloaded.prompt_mode is ChatPromptMode.ROLEPLAY
    assert forked.prompt_mode is ChatPromptMode.ROLEPLAY


def test_chat_migrates_a_legacy_single_character_binding_to_a_cast() -> None:
    binding = CharacterChatBinding(id="abc123", name="Mira", revision=2)
    chat = Chat(character_cast=(binding,))
    legacy_document: dict[str, object] = dict(chat.to_dict())
    _ = legacy_document.pop("schema_version")
    _ = legacy_document.pop("character_cast")
    _ = legacy_document.pop("prompt_mode")
    legacy_document["character"] = binding.to_dict()

    reloaded = Chat.from_dict(legacy_document)

    assert reloaded.character_cast == (binding,)
    assert reloaded.character == binding
    assert reloaded.prompt_mode is ChatPromptMode.ROLEPLAY
    assert "character" not in reloaded.to_dict()


def test_character_chat_prompt_mode_round_trips_with_its_prompt_snapshot() -> None:
    binding = CharacterChatBinding(id="abc123", name="Mira", revision=2)
    chat = Chat(
        character_cast=(binding,),
        prompt_mode=ChatPromptMode.CUSTOM,
        system_prompt="Answer in nautical metaphors.\n\nCharacter profile:\nRole: Pilot",
    )

    reloaded = Chat.from_dict(chat.to_dict())

    assert reloaded.prompt_mode is ChatPromptMode.CUSTOM
    assert reloaded.system_prompt == chat.system_prompt


def test_character_cast_compiles_one_collective_prompt_and_concise_title() -> None:
    mira = Character(
        id="mira123",
        name_parts=CharacterName("Mira"),
        fields=[CharacterField(label="Role", value="Cartographer")],
    )
    ren = Character(
        id="ren123",
        name_parts=CharacterName("Ren"),
        fields=[CharacterField(label="Role", value="Pilot")],
    )

    prompt = compiled_character_cast_system_prompt((mira, ren))

    assert "You are roleplaying as the following characters." in prompt
    assert "Mira:\nCharacter profile:\nRole: Cartographer" in prompt
    assert "Ren:\nCharacter profile:\nRole: Pilot" in prompt
    assert "Clearly attribute each speaker" in prompt
    assert character_cast_chat_title((mira, ren)) == "Mira & Ren"
    assert (
        compiled_character_cast_system_prompt((mira,)) == mira.compiled_system_prompt()
    )


def test_character_cast_rejects_duplicate_profiles() -> None:
    mira = Character(id="mira123", name_parts=CharacterName("Mira"))

    with pytest.raises(ValueError, match="duplicate characters"):
        _ = compiled_character_cast_system_prompt((mira, mira))
    with pytest.raises(ValueError, match="duplicate characters"):
        _ = Chat(character_cast=(mira.chat_binding(), mira.chat_binding()))


def test_character_prompt_modes_compile_assistant_story_and_custom_casts() -> None:
    mira = Character(
        id="mira123",
        name_parts=CharacterName("Mira"),
        fields=[CharacterField(label="Role", value="Cartographer")],
    )
    ren = Character(
        id="ren123",
        name_parts=CharacterName("Ren"),
        fields=[CharacterField(label="Role", value="Pilot")],
    )

    assistant_prompt = compiled_character_chat_system_prompt(
        (mira, ren), mode=ChatPromptMode.ASSISTANT
    )
    story_prompt = compiled_character_chat_system_prompt(
        (mira, ren),
        mode=ChatPromptMode.STORY,
        story_direction="A cozy mystery aboard an airship.",
    )
    single_story_prompt = compiled_character_chat_system_prompt(
        (mira,), mode=ChatPromptMode.STORY
    )
    custom_prompt = compiled_character_chat_system_prompt(
        (mira, ren),
        mode=ChatPromptMode.CUSTOM,
        custom_instruction="Answer in concise mission briefings.",
    )

    assert "collaborative assistant team" in assistant_prompt
    assert "Do not roleplay a scene unless the user asks." in assistant_prompt
    assert "Mira:\nCharacter profile:\nRole: Cartographer" in assistant_prompt
    assert story_prompt.startswith(
        "You are the narrator of an immersive collaborative story featuring the "
        "following characters."
    )
    assert "Story direction:\nA cozy mystery aboard an airship." in story_prompt
    assert "Ren:\nCharacter profile:\nRole: Pilot" in story_prompt
    assert "Write in third person" in story_prompt
    assert "featuring Mira." in single_story_prompt
    assert "Story direction:" not in single_story_prompt
    assert custom_prompt.startswith("Answer in concise mission briefings.")
    assert "Ren:\nCharacter profile:\nRole: Pilot" in custom_prompt
    assert "You are roleplaying" not in custom_prompt


def test_custom_prompt_mode_requires_an_instruction_and_modes_are_parsed_explicitly() -> (
    None
):
    mira = Character(id="mira123", name_parts=CharacterName("Mira"))

    with pytest.raises(ValueError, match="custom instruction cannot be empty"):
        _ = compiled_character_chat_system_prompt((mira,), mode=ChatPromptMode.CUSTOM)
    with pytest.raises(ValueError, match="unknown chat prompt mode"):
        _ = chat_prompt_mode_from_text("narrator")

    assert chat_prompt_mode_from_text(" assistant ") is ChatPromptMode.ASSISTANT
    assert chat_prompt_mode_from_text(" story ") is ChatPromptMode.STORY
