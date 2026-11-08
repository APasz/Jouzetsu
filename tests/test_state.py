from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import cast, override

from jouzetsu.character_storage import CharacterStorage
from jouzetsu.config import (
    AppConfig,
    AppPaths,
    GenerationSettings,
    ThemeColorway,
    ThemeSettings,
    UiSettings,
    default_config,
)
from jouzetsu.events import StateChangeKind
from jouzetsu.models import (
    Character,
    CharacterField,
    CharacterName,
    Chat,
    ChatPromptMode,
)
from jouzetsu.state import AppState
from jouzetsu.storage import ChatStorage


def _saved_config_section(path: Path, name: str) -> dict[str, object]:
    """Read one known object section from a persisted configuration document."""

    document = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    return cast(dict[str, object], document[name])


class FailingCharacterStorage(CharacterStorage):
    """A deterministic persistence failure used to verify state publication order."""

    @override
    def save(self, character: Character) -> None:
        _ = character
        raise OSError("character storage unavailable")


class AppStateTests(unittest.TestCase):
    # unittest initialises these through setUp before every test method.
    tmp_dir: tempfile.TemporaryDirectory[str]  # pyright: ignore[reportUninitializedInstanceVariable]
    state: AppState  # pyright: ignore[reportUninitializedInstanceVariable]

    @override
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        paths = AppPaths.for_home(Path(self.tmp_dir.name))
        chats_file = paths.chats_file
        config = AppConfig(
            paths=paths,
            ui=UiSettings(auto_open_browser=False),
        )
        self.state = AppState(config, ChatStorage(chats_file))

    @override
    def tearDown(self) -> None:
        _ = asyncio.run(self.state.shutdown())
        self.tmp_dir.cleanup()

    def test_character_profile_write_failure_does_not_publish_partial_state(
        self,
    ) -> None:
        async def scenario() -> None:
            character: Character = await self.state.new_character()
            self.state.character_storage = FailingCharacterStorage(
                Path(self.tmp_dir.name) / "failing-characters"
            )

            with self.assertRaisesRegex(OSError, "character storage unavailable"):
                _ = await self.state.update_character(
                    character.id,
                    expected_revision=character.revision,
                    name=CharacterName("Mira"),
                    fields=[CharacterField(label="Role", value="Pilot")],
                )

            self.assertIs(self.state.character(character.id), character)
            self.assertTrue(character.is_draft)
            self.assertIsNone(character.name_parts)
            self.assertEqual(character.fields, [])

            self.state.character_storage = FailingCharacterStorage(
                Path(self.tmp_dir.name) / "failing-new-characters"
            )
            with self.assertRaisesRegex(OSError, "character storage unavailable"):
                _ = await self.state.new_character()
            self.assertEqual(self.state.characters, [character])

        asyncio.run(scenario())

    def test_character_profile_revision_prevents_stale_edits_and_binds_the_saved_revision_to_a_chat(
        self,
    ) -> None:
        async def scenario() -> None:
            character: Character = await self.state.new_character()
            updated: Character = await self.state.update_character(
                character.id,
                expected_revision=character.revision,
                name=CharacterName("Mira"),
                fields=[CharacterField(label="Role", value="Pilot")],
            )

            with self.assertRaisesRegex(ValueError, "changed elsewhere"):
                _ = await self.state.update_character(
                    character.id,
                    expected_revision=character.revision,
                    name=CharacterName("Stale", "Mira"),
                    fields=[],
                )

            chat: Chat = await self.state.start_chat_from_character(character.id)
            self.assertEqual(updated.revision, 2)
            self.assertEqual(self.state.character(character.id), updated)
            self.assertIsNotNone(chat.character)
            self.assertEqual(
                chat.character.revision if chat.character is not None else 0,
                updated.revision,
            )
            self.assertIn("Role: Pilot", chat.system_prompt)
            self.assertEqual(chat.messages, [])

            _ = await self.state.inference.send_message(
                chat.id, chat.revision, "hello", "Hello Mira", generate=False
            )
            self.assertEqual(chat.title, "Mira")

        asyncio.run(scenario())

    def test_character_cast_captures_each_selected_profile_revision(self) -> None:
        async def scenario() -> None:
            mira: Character = await self.state.new_character()
            ren: Character = await self.state.new_character()
            saved_mira: Character = await self.state.update_character(
                mira.id,
                expected_revision=mira.revision,
                name=CharacterName("Mira"),
                fields=[CharacterField(label="Role", value="Cartographer")],
            )
            saved_ren: Character = await self.state.update_character(
                ren.id,
                expected_revision=ren.revision,
                name=CharacterName("Ren"),
                fields=[CharacterField(label="Role", value="Pilot")],
            )

            chat: Chat = await self.state.start_chat_from_characters(
                (saved_mira.id, saved_ren.id)
            )

            self.assertEqual(chat.title, "Mira & Ren")
            self.assertEqual(
                tuple(
                    (binding.id, binding.revision) for binding in chat.character_cast
                ),
                (
                    (saved_mira.id, saved_mira.revision),
                    (saved_ren.id, saved_ren.revision),
                ),
            )
            self.assertIsNone(chat.character)
            self.assertIn(
                "Mira:\nCharacter profile:\nRole: Cartographer", chat.system_prompt
            )
            self.assertIn("Ren:\nCharacter profile:\nRole: Pilot", chat.system_prompt)
            self.assertIn("Clearly attribute each speaker", chat.system_prompt)

            updated_mira: Character = await self.state.update_character(
                saved_mira.id,
                expected_revision=saved_mira.revision,
                name=CharacterName("Mira"),
                fields=[CharacterField(label="Role", value="Navigator")],
            )

            self.assertEqual(updated_mira.revision, saved_mira.revision + 1)
            self.assertIn("Role: Cartographer", chat.system_prompt)
            self.assertNotIn("Role: Navigator", chat.system_prompt)
            self.assertEqual(chat.character_cast[0].revision, saved_mira.revision)

        asyncio.run(scenario())

    def test_character_cast_requires_unique_known_profile_ids(self) -> None:
        async def scenario() -> None:
            character: Character = await self.state.new_character()

            with self.assertRaisesRegex(ValueError, "at least one character"):
                _ = await self.state.start_chat_from_characters(())
            with self.assertRaisesRegex(ValueError, "duplicate characters"):
                _ = await self.state.start_chat_from_characters(
                    (character.id, character.id)
                )
            with self.assertRaisesRegex(ValueError, "unknown character"):
                _ = await self.state.start_chat_from_characters(("missing",))

        asyncio.run(scenario())

    def test_character_cast_prompt_modes_snapshot_the_selected_framing(self) -> None:
        async def scenario() -> None:
            mira: Character = await self.state.new_character()
            ren: Character = await self.state.new_character()
            mira = await self.state.update_character(
                mira.id,
                expected_revision=mira.revision,
                name=CharacterName("Mira"),
                fields=[],
            )
            ren = await self.state.update_character(
                ren.id,
                expected_revision=ren.revision,
                name=CharacterName("Ren"),
                fields=[],
            )

            assistant_chat: Chat = await self.state.start_chat_from_characters(
                (mira.id, ren.id), prompt_mode=ChatPromptMode.ASSISTANT
            )
            story_chat: Chat = await self.state.start_chat_from_characters(
                (mira.id, ren.id),
                prompt_mode=ChatPromptMode.STORY,
                story_direction="A tense rescue on an ocean moon.",
            )
            custom_chat: Chat = await self.state.start_chat_from_character(
                mira.id,
                prompt_mode=ChatPromptMode.CUSTOM,
                custom_instruction="Use terse technical summaries.",
            )

            self.assertIs(assistant_chat.prompt_mode, ChatPromptMode.ASSISTANT)
            self.assertIn("collaborative assistant team", assistant_chat.system_prompt)
            self.assertIs(story_chat.prompt_mode, ChatPromptMode.STORY)
            self.assertIn("A tense rescue on an ocean moon.", story_chat.system_prompt)
            self.assertIn("Write in third person", story_chat.system_prompt)
            self.assertIs(custom_chat.prompt_mode, ChatPromptMode.CUSTOM)
            self.assertTrue(
                custom_chat.system_prompt.startswith("Use terse technical summaries.")
            )
            with self.assertRaisesRegex(
                ValueError, "custom instruction cannot be empty"
            ):
                _ = await self.state.start_chat_from_character(
                    mira.id, prompt_mode=ChatPromptMode.CUSTOM
                )

        asyncio.run(scenario())

    def test_set_global_system_prompt_persists_to_config_file(self) -> None:
        asyncio.run(self.state.set_global_system_prompt("Answer tersely."))

        self.assertEqual(self.state.global_system_prompt, "Answer tersely.")
        saved = self.state.config.config_file.read_text(encoding="utf-8")
        self.assertIn("Answer tersely.", saved)

    def test_set_generation_defaults_validates_and_persists_atomically(self) -> None:
        settings = GenerationSettings(
            temperature=0.4,
            top_p=0.8,
            max_tokens=1024,
            system_prompt="Use compact answers.",
            british_english=True,
            british_spelling_replacements=[],
        )

        asyncio.run(self.state.set_generation_defaults(settings))

        self.assertEqual(self.state.config.generation, settings)
        saved = json.loads(self.state.config.config_file.read_text(encoding="utf-8"))
        self.assertEqual(
            saved["generation"],
            {
                "temperature": 0.4,
                "top_p": 0.8,
                "max_tokens": 1024,
                "system_prompt": "Use compact answers.",
                "british_english": True,
                "british_spelling_replacements": [],
            },
        )

    def test_reset_theme_colorways_restore_only_the_selected_colourway(self) -> None:
        theme: ThemeSettings = replace(
            self.state.config.theme,
            app=replace(
                self.state.config.theme.app,
                accent="#101112",
                message_action_delete="#131415",
            ),
            user=replace(self.state.config.theme.user, accent="#131415"),
            assistant=replace(self.state.config.theme.assistant, accent="#161718"),
            system=replace(self.state.config.theme.system, accent="#191a1b"),
        )
        defaults: ThemeSettings = default_config(self.state.config.paths).theme
        original_colorways = dict(theme.colorways())
        default_colorways = dict(defaults.colorways())

        for colorway in ThemeColorway:
            self.state.config.theme = theme

            asyncio.run(self.state.reset_theme_colorway(colorway))

            colorways = dict(self.state.config.theme.colorways())
            self.assertEqual(colorways[colorway], default_colorways[colorway])
            for other in set(ThemeColorway).difference({colorway}):
                self.assertEqual(colorways[other], original_colorways[other])

        saved = _saved_config_section(self.state.config.config_file, "theme")
        self.assertEqual(saved["system"], {"accent": defaults.system.accent})

    def test_set_generation_defaults_rejects_invalid_values_without_mutating_config(
        self,
    ) -> None:
        original_settings = self.state.config.generation
        invalid_settings = GenerationSettings(temperature=2.1)

        with self.assertRaisesRegex(ValueError, "temperature must be between 0 and 2"):
            asyncio.run(self.state.set_generation_defaults(invalid_settings))

        self.assertIs(self.state.config.generation, original_settings)

    def test_notify_prunes_listener_for_deleted_client(self) -> None:
        stale_calls = 0
        healthy_calls = 0

        async def stale_listener(_kind: StateChangeKind) -> None:
            nonlocal stale_calls
            stale_calls += 1
            raise RuntimeError("The client this element belongs to has been deleted.")

        async def healthy_listener(_kind: StateChangeKind) -> None:
            nonlocal healthy_calls
            healthy_calls += 1

        _ = self.state.add_listener(stale_listener)
        _ = self.state.add_listener(healthy_listener)

        async def scenario() -> None:
            with self.assertNoLogs("jouzetsu.state", level="WARNING"):
                await self.state.set_global_system_prompt("First")
                await self.state.set_global_system_prompt("Second")

        asyncio.run(scenario())

        self.assertEqual(stale_calls, 1)
        self.assertEqual(healthy_calls, 2)
        asyncio.run(self.state.set_global_system_prompt("Third"))
        self.assertEqual(stale_calls, 1)
        self.assertEqual(healthy_calls, 3)

    def test_listener_removal_callback_is_idempotent(self) -> None:
        calls = 0

        async def listener(_kind: StateChangeKind) -> None:
            nonlocal calls
            calls += 1

        remove_listener = self.state.add_listener(listener)

        remove_listener()
        remove_listener()

        asyncio.run(self.state.set_global_system_prompt("No listener call"))
        self.assertEqual(calls, 0)
