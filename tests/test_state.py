from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from collections.abc import AsyncIterator, Iterable, Sequence
from pathlib import Path
from typing import cast, override
from unittest.mock import patch

from jouzetsu.character_storage import CharacterStorage
from jouzetsu.config import (
    AppConfig,
    AppPaths,
    ConfigStore,
    GenerationSettings,
    IconColorSettings,
    ServerSettings,
    UiSettings,
)
from jouzetsu.continuity import CONTINUITY_REVIEW_PROMPT
from jouzetsu.events import StateChangeKind
from jouzetsu.lmstudio import LMStudioClient, LMStudioError
from jouzetsu.models import (
    Character,
    CharacterField,
    CharacterName,
    Chat,
    ChatJSON,
    ChatPromptMode,
    ChatSamplingOverrides,
    Message,
)
from jouzetsu.runtime import (
    ChatStreamEvent,
    FirstToken,
    GenerationMetrics,
    ModelDescriptor,
    ModelLoadProgress,
    ModelReady,
    PredictionComplete,
    PredictionFragment,
    PromptProcessingProgress,
    RuntimePhase,
    RuntimeStatus,
)
from jouzetsu.state import AppState
from jouzetsu.state_generation import GenerationController
from jouzetsu.storage import ChatStorage


class FakeLMStudioClient(LMStudioClient):
    def __init__(self, tokens: list[str] | None = None) -> None:
        super().__init__(ServerSettings())
        self.tokens: list[str] = tokens or ["Hello", " world"]
        self.generations: list[GenerationSettings] = []
        self.closed: bool = False
        self.unloaded_instance_ids: list[str] = []

    @override
    async def list_models(self) -> list[str]:
        return ["demo-model"]

    @override
    async def list_model_inventory(self) -> tuple[ModelDescriptor, ...]:
        loaded_instance_ids: tuple[str, ...] = (
            () if "demo-model" in self.unloaded_instance_ids else ("demo-model",)
        )
        return (
            ModelDescriptor(
                key="demo-model",
                display_name="Demo Model",
                loaded_instance_ids=loaded_instance_ids,
                context_length=4096,
                max_context_length=8192,
            ),
        )

    @override
    async def unload_model_instance(self, instance_id: str) -> None:
        self.unloaded_instance_ids.append(instance_id)

    @override
    async def stream_chat(
        self,
        chat: Chat,
        model: str,
        generation: GenerationSettings,
    ) -> AsyncIterator[ChatStreamEvent]:
        _ = (chat, model)
        self.generations.append(generation)
        yield ModelReady("demo-model", "Demo Model", 4096)
        yield PromptProcessingProgress(1.0)
        yield FirstToken()
        for token in self.tokens:
            await asyncio.sleep(0)
            yield PredictionFragment(token, 1, False)
        yield PredictionComplete(
            GenerationMetrics(
                output_tokens=len(self.tokens),
                prompt_tokens=4,
                tokens_per_second=20.0,
                time_to_first_token_seconds=0.1,
                stop_reason="eosFound",
            )
        )

    @override
    async def close(self) -> None:
        self.closed = True


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
    client: FakeLMStudioClient  # pyright: ignore[reportUninitializedInstanceVariable]
    state: AppState  # pyright: ignore[reportUninitializedInstanceVariable]

    @override
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        paths = AppPaths.for_home(Path(self.tmp_dir.name))
        chats_file = paths.chats_file
        config = AppConfig(
            paths=paths,
            server=ServerSettings(default_model="demo-model"),
            ui=UiSettings(auto_open_browser=False),
        )
        self.client = FakeLMStudioClient()
        self.state = AppState(config, ChatStorage(chats_file), self.client)

    @override
    def tearDown(self) -> None:
        _ = asyncio.run(self.state.shutdown())
        self.tmp_dir.cleanup()

    def test_send_user_message_appends_first_message_and_streams_assistant_reply(
        self,
    ) -> None:
        sent_message = None

        self.assertEqual(self.state.active_chat.messages, [])

        async def scenario() -> None:
            nonlocal sent_message
            sent_message = await self.state.send_user_message("First message")
            await self._wait_for_generation()

        asyncio.run(scenario())

        messages = self.state.active_chat.messages
        self.assertEqual([message.role for message in messages], ["user", "assistant"])
        self.assertEqual(messages[0].content, "First message")
        self.assertEqual(messages[1].content, "Hello world")
        self.assertEqual(self.state.active_chat.title, "First message")
        self.assertIs(sent_message, messages[0])

    def test_explicit_new_chat_title_is_not_replaced_by_the_first_message(self) -> None:
        async def scenario() -> None:
            await self.state.rename_chat(self.state.active_chat.id, "New chat")
            _ = await self.state.send_user_message("First message")
            await self._wait_for_generation()

        asyncio.run(scenario())

        self.assertEqual(self.state.active_chat.title, "New chat")

    def test_assistant_messages_record_the_model_used_for_each_generation(self) -> None:
        self.state.config.generation.continuity_review = False

        async def model_tracking_stream(
            chat: Chat,
            model: str,
            generation: GenerationSettings,
        ) -> AsyncIterator[ChatStreamEvent]:
            _ = (chat, generation)
            resolved_model: str = f"resolved/{model}"
            yield ModelReady(resolved_model, resolved_model, 4096)
            yield FirstToken()
            yield PredictionFragment(f"Reply from {model}", 1, False)
            yield PredictionComplete(GenerationMetrics(output_tokens=1))

        self.client.stream_chat = model_tracking_stream

        async def scenario() -> list[str]:
            _ = await self.state.send_user_message("First")
            await self._wait_for_generation()
            await self.state.set_active_chat_model("second-model")
            _ = await self.state.send_user_message("Second")
            await self._wait_for_generation()
            return [
                message.model
                for message in self.state.active_chat.messages
                if message.role == "assistant"
            ]

        message_models: list[str] = asyncio.run(scenario())
        reloaded: list[Chat] = ChatStorage(self.state.config.chats_file).load_all()

        self.assertEqual(
            message_models, ["resolved/demo-model", "resolved/second-model"]
        )
        self.assertEqual(
            [
                message.model
                for message in reloaded[0].messages
                if message.role == "assistant"
            ],
            message_models,
        )

    def test_new_chat_postprocesses_streamed_british_spellings_by_default(self) -> None:
        self.client.tokens = ["Mo", "m likes color and FAVORITE in my_color"]

        async def scenario() -> None:
            _ = await self.state.send_user_message("First message")
            await self._wait_for_generation()

        asyncio.run(scenario())

        self.assertEqual(
            self.state.active_chat.messages[1].content,
            "Mum likes colour and FAVOURITE in my_color",
        )

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
            self.assertEqual(character.name, "New character")
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

            _ = await self.state.send_user_message("Hello Mira")
            await self._wait_for_generation()
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

    def test_continuity_review_proposes_a_rewrite_without_changing_the_visible_reply(
        self,
    ) -> None:
        review_generations: list[GenerationSettings] = []
        phases: list[RuntimePhase] = []

        async def reviewed_stream(
            chat: Chat,
            model: str,
            generation: GenerationSettings,
        ) -> AsyncIterator[ChatStreamEvent]:
            _ = model
            review_generations.append(generation)
            if chat.messages[-1].content == CONTINUITY_REVIEW_PROMPT:
                yield FirstToken()
                yield PredictionFragment("REPLACE\nA revised color answer", 5, False)
                yield PredictionComplete(GenerationMetrics(output_tokens=5))
                return
            yield ModelReady("demo-model", "Demo Model", 4096)
            yield FirstToken()
            yield PredictionFragment("The original color answer", 4, False)
            yield PredictionComplete(GenerationMetrics(output_tokens=4))

        self.client.stream_chat = reviewed_stream

        async def listener(kind: StateChangeKind) -> None:
            if kind is StateChangeKind.STATUS:
                phases.append(self.state.runtime_status.phase)

        _ = self.state.add_listener(listener)

        async def scenario() -> str:
            _ = await self.state.send_user_message("Question")
            await self._wait_for_generation()
            return self.state.active_chat.messages[-1].id

        assistant_id = asyncio.run(scenario())
        assistant = self.state.active_chat.find_message(assistant_id)
        assert assistant is not None

        self.assertEqual(assistant.content, "The original colour answer")
        self.assertEqual(assistant.continuity_rewrite, "A revised colour answer")
        self.assertEqual(
            [generation.temperature for generation in review_generations], [0.75, 0.0]
        )
        self.assertIn(RuntimePhase.REVIEWING, phases)

        asyncio.run(self.state.apply_continuity_rewrite(assistant_id))

        self.assertEqual(assistant.content, "A revised colour answer")
        self.assertEqual(assistant.continuity_rewrite, "")

    def test_runtime_refresh_preserves_the_reviewing_phase(self) -> None:
        review_started = asyncio.Event()
        finish_review = asyncio.Event()

        async def reviewed_stream(
            chat: Chat,
            model: str,
            generation: GenerationSettings,
        ) -> AsyncIterator[ChatStreamEvent]:
            _ = (model, generation)
            if chat.messages[-1].content == CONTINUITY_REVIEW_PROMPT:
                _ = review_started.set()
                _ = await finish_review.wait()
                yield PredictionFragment("KEEP", 1, False)
                return
            yield FirstToken()
            yield PredictionFragment("Original reply", 2, False)
            yield PredictionComplete(GenerationMetrics(output_tokens=2))

        self.client.stream_chat = reviewed_stream

        async def scenario() -> tuple[RuntimePhase, RuntimePhase]:
            _ = await self.state.send_user_message("Question")
            _ = await review_started.wait()
            phase_before_refresh = self.state.runtime_status.phase
            await self.state.refresh_runtime_status()
            phase_after_refresh = self.state.runtime_status.phase
            _ = finish_review.set()
            await self._wait_for_generation()
            return phase_before_refresh, phase_after_refresh

        phase_before_refresh, phase_after_refresh = asyncio.run(scenario())

        self.assertIs(phase_before_refresh, RuntimePhase.REVIEWING)
        self.assertIs(phase_after_refresh, RuntimePhase.REVIEWING)

    def test_continuity_review_failure_keeps_the_completed_reply(self) -> None:
        async def review_failure_stream(
            chat: Chat,
            model: str,
            generation: GenerationSettings,
        ) -> AsyncIterator[ChatStreamEvent]:
            _ = (model, generation)
            if chat.messages[-1].content == CONTINUITY_REVIEW_PROMPT:
                raise LMStudioError("review unavailable")
            yield FirstToken()
            yield PredictionFragment("Original reply", 2, False)
            yield PredictionComplete(GenerationMetrics(output_tokens=2))

        self.client.stream_chat = review_failure_stream

        async def scenario() -> None:
            _ = await self.state.send_user_message("Question")
            await self._wait_for_generation()

        asyncio.run(scenario())

        assistant = self.state.active_chat.messages[-1]
        self.assertEqual(assistant.content, "Original reply")
        self.assertEqual(assistant.continuity_rewrite, "")
        self.assertIs(self.state.runtime_status.phase, RuntimePhase.READY)

    def test_discard_continuity_rewrite_keeps_the_original_reply(self) -> None:
        assistant = self.state.active_chat.add_message("assistant", "Original reply")
        assistant.set_continuity_rewrite("Proposed rewrite")

        asyncio.run(self.state.discard_continuity_rewrite(assistant.id))

        self.assertEqual(assistant.content, "Original reply")
        self.assertEqual(assistant.continuity_rewrite, "")

    def test_continuity_rewrite_rejects_an_earlier_assistant_message(self) -> None:
        assistant = self.state.active_chat.add_message("assistant", "Original reply")
        assistant.set_continuity_rewrite("Proposed rewrite")
        _ = self.state.active_chat.add_message("user", "Follow-up question")

        with self.assertRaisesRegex(ValueError, "only for the last assistant message"):
            asyncio.run(self.state.apply_continuity_rewrite(assistant.id))

    def test_disabled_continuity_review_uses_only_the_primary_generation(self) -> None:
        self.state.config.generation.continuity_review = False

        async def scenario() -> None:
            _ = await self.state.send_user_message("Question")
            await self._wait_for_generation()

        asyncio.run(scenario())

        self.assertEqual(len(self.client.generations), 1)

    def test_chat_sampling_overrides_apply_to_generation_and_persist(self) -> None:
        overrides = ChatSamplingOverrides(temperature=0.2, top_p=0.8, max_tokens=512)

        async def scenario() -> None:
            await self.state.set_active_chat_sampling_overrides(overrides)
            _ = await self.state.send_user_message("First message")
            await self._wait_for_generation()

        asyncio.run(scenario())

        generation: GenerationSettings = self.client.generations[0]
        self.assertEqual(generation.temperature, 0.2)
        self.assertEqual(generation.top_p, 0.8)
        self.assertEqual(generation.max_tokens, 512)
        self.assertEqual(self.state.active_chat.sampling_overrides, overrides)
        reloaded = ChatStorage(self.state.config.chats_file).load_all()
        self.assertEqual(
            reloaded[0].sampling_overrides,
            ChatSamplingOverrides(temperature=0.2, top_p=0.8, max_tokens=512),
        )

    def test_invalid_chat_sampling_overrides_do_not_mutate_the_active_chat(
        self,
    ) -> None:
        original: ChatSamplingOverrides = self.state.active_chat.sampling_overrides

        async def scenario() -> None:
            with self.assertRaisesRegex(
                ValueError, "temperature must be between 0 and 2"
            ):
                await self.state.set_active_chat_sampling_overrides(
                    ChatSamplingOverrides(temperature=2.1)
                )

        asyncio.run(scenario())

        self.assertEqual(self.state.active_chat.sampling_overrides, original)

    def test_stream_notifications_are_bounded_and_typed(self) -> None:
        self.client.tokens = ["x"] * 100
        change_kinds: list[StateChangeKind] = []

        async def listener(kind: StateChangeKind) -> None:
            change_kinds.append(kind)

        _ = self.state.add_listener(listener)

        async def scenario() -> None:
            _ = await self.state.send_user_message("Generate quickly")
            await self._wait_for_generation()

        asyncio.run(scenario())

        stream_updates = change_kinds.count(StateChangeKind.STREAM)
        self.assertGreaterEqual(stream_updates, 1)
        self.assertLess(stream_updates, len(self.client.tokens))
        self.assertEqual(change_kinds.count(StateChangeKind.FULL), 2)

    def test_continue_last_response_creates_new_assistant_message_without_a_user_message(
        self,
    ) -> None:
        captured_history: list[tuple[str, str]] = []

        async def continuation_stream(
            chat: Chat,
            model: str,
            generation: GenerationSettings,
        ) -> AsyncIterator[ChatStreamEvent]:
            _ = generation
            captured_history.extend(
                (message.role, message.content) for message in chat.messages
            )
            yield ModelReady(model, "Demo Model", 4096)
            yield PromptProcessingProgress(1.0)
            yield FirstToken()
            yield PredictionFragment(" continued", 1, False)
            yield PredictionComplete(GenerationMetrics(output_tokens=1))

        async def scenario() -> str:
            _ = await self.state.send_user_message("Start a response")
            await self._wait_for_generation()
            assistant_message_id = self.state.active_chat.messages[-1].id
            self.client.stream_chat = continuation_stream
            await self.state.continue_last_response()
            await self._wait_for_generation()
            return assistant_message_id

        assistant_message_id = asyncio.run(scenario())

        messages = self.state.active_chat.messages
        self.assertEqual(
            [message.role for message in messages], ["user", "assistant", "assistant"]
        )
        self.assertEqual(messages[1].id, assistant_message_id)
        self.assertEqual(messages[1].content, "Hello world")
        self.assertNotEqual(messages[-1].id, assistant_message_id)
        self.assertEqual(messages[-1].content, " continued")
        self.assertEqual(
            captured_history[:2],
            [("user", "Start a response"), ("assistant", "Hello world")],
        )
        self.assertEqual(captured_history[2][0], "user")
        self.assertIn(
            "Continue the previous assistant response", captured_history[2][1]
        )

    def test_resend_user_message_generates_a_reply_without_duplicating_the_user_turn(
        self,
    ) -> None:
        async def scenario() -> tuple[str, str]:
            source: Message = await self.state.send_user_message("Ask again")
            await self._wait_for_generation()
            removed_assistant: Message | None = self.state.active_chat.remove_message(
                self.state.active_chat.messages[-1].id
            )
            self.assertIsNotNone(removed_assistant)
            resent: Message = await self.state.resend_user_message(source.id)
            await self._wait_for_generation()
            return source.id, resent.id

        source_message_id, resent_message_id = asyncio.run(scenario())

        messages = self.state.active_chat.messages
        self.assertEqual([message.role for message in messages], ["user", "assistant"])
        self.assertEqual(messages[0].content, "Ask again")
        self.assertEqual(messages[0].id, source_message_id)
        self.assertEqual(messages[0].id, resent_message_id)

    def test_regenerate_replaces_the_final_reply_without_sending_it_back_to_the_model(
        self,
    ) -> None:
        captured_history: list[tuple[str, str]] = []
        change_kinds: list[StateChangeKind] = []
        self.state.config.generation.continuity_review = False

        async def regenerated_stream(
            chat: Chat,
            model: str,
            generation: GenerationSettings,
        ) -> AsyncIterator[ChatStreamEvent]:
            _ = generation
            captured_history.extend(
                (message.role, message.content) for message in chat.messages
            )
            yield ModelReady(model, "Demo Model", 4096)
            yield FirstToken()
            yield PredictionFragment("Fresh answer", 1, False)
            yield PredictionComplete(GenerationMetrics(output_tokens=1))

        async def listener(kind: StateChangeKind) -> None:
            change_kinds.append(kind)

        _ = self.state.add_listener(listener)

        async def scenario() -> str:
            _ = await self.state.send_user_message("Ask again")
            await self._wait_for_generation()
            previous_reply_id: str = self.state.active_chat.messages[-1].id
            self.client.stream_chat = regenerated_stream
            change_kinds.clear()
            await self.state.regenerate_last()
            await self._wait_for_generation()
            return previous_reply_id

        previous_reply_id = asyncio.run(scenario())

        messages = self.state.active_chat.messages
        self.assertEqual(captured_history, [("user", "Ask again")])
        self.assertEqual([message.role for message in messages], ["user", "assistant"])
        self.assertNotEqual(messages[-1].id, previous_reply_id)
        self.assertEqual(messages[-1].content, "Fresh answer")
        self.assertEqual(change_kinds.count(StateChangeKind.FULL), 2)

    def test_regenerate_last_continuation_reuses_the_continuation_prompt(
        self,
    ) -> None:
        captured_history: list[tuple[str, str]] = []
        self.state.config.generation.continuity_review = False

        async def regenerated_stream(
            chat: Chat,
            model: str,
            generation: GenerationSettings,
        ) -> AsyncIterator[ChatStreamEvent]:
            _ = generation
            captured_history.extend(
                (message.role, message.content) for message in chat.messages
            )
            yield ModelReady(model, "Demo Model", 4096)
            yield FirstToken()
            yield PredictionFragment("Fresh continuation", 1, False)
            yield PredictionComplete(GenerationMetrics(output_tokens=1))

        async def scenario() -> str:
            _ = await self.state.send_user_message("Ask again")
            await self._wait_for_generation()
            previous_continuation: Message = self.state.active_chat.add_message(
                "assistant", "Previous continuation"
            )
            self.client.stream_chat = regenerated_stream
            await self.state.regenerate_last()
            await self._wait_for_generation()
            return previous_continuation.id

        previous_continuation_id = asyncio.run(scenario())

        messages = self.state.active_chat.messages
        self.assertEqual(
            [message.role for message in messages], ["user", "assistant", "assistant"]
        )
        self.assertNotEqual(messages[-1].id, previous_continuation_id)
        self.assertEqual(messages[-1].content, "Fresh continuation")
        self.assertEqual(
            captured_history[:2],
            [("user", "Ask again"), ("assistant", "Hello world")],
        )
        self.assertEqual(captured_history[2][0], "user")
        self.assertIn(
            "Continue the previous assistant response", captured_history[2][1]
        )

    def test_continue_last_response_requires_a_completed_assistant_message(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError, "last message must be a completed assistant response"
        ):
            asyncio.run(self.state.continue_last_response())

    def test_merge_assistant_message_with_previous_same_role_message(self) -> None:
        async def scenario() -> tuple[str, str]:
            _ = await self.state.send_user_message("Start a response")
            await self._wait_for_generation()
            first_assistant_id = self.state.active_chat.messages[1].id
            second_assistant = self.state.active_chat.add_message(
                "assistant", "Continued response"
            )
            _ = await self.state.merge_message_with_previous(second_assistant.id)
            return first_assistant_id, second_assistant.id

        first_assistant_id, second_assistant_id = asyncio.run(scenario())

        messages = self.state.active_chat.messages
        self.assertEqual([message.role for message in messages], ["user", "assistant"])
        self.assertEqual(messages[1].id, first_assistant_id)
        self.assertNotEqual(messages[1].id, second_assistant_id)
        self.assertEqual(messages[1].content, "Hello world\n\nContinued response")

    def test_merge_user_message_with_previous_same_role_message(self) -> None:
        async def scenario() -> tuple[str, str]:
            first_user = self.state.active_chat.add_message("user", "First user note")
            second_user = self.state.active_chat.add_message("user", "Second user note")
            _ = await self.state.merge_message_with_previous(second_user.id)
            return first_user.id, second_user.id

        first_user_id, second_user_id = asyncio.run(scenario())

        messages = self.state.active_chat.messages
        self.assertEqual([message.role for message in messages], ["user"])
        self.assertEqual(messages[0].id, first_user_id)
        self.assertNotEqual(messages[0].id, second_user_id)
        self.assertEqual(messages[0].content, "First user note\n\nSecond user note")

    def test_merge_message_requires_previous_same_role_message(self) -> None:
        async def scenario() -> None:
            _ = await self.state.send_user_message("Start a response")
            await self._wait_for_generation()
            _ = await self.state.merge_message_with_previous(
                self.state.active_chat.messages[1].id
            )

        with self.assertRaisesRegex(
            ValueError, "previous message must have the same role"
        ):
            asyncio.run(scenario())

    def test_set_active_chat_model_persists_value(self) -> None:
        asyncio.run(self.state.set_active_chat_model("custom-model"))

        reloaded = ChatStorage(self.state.config.chats_file).load_all()
        self.assertEqual(reloaded[0].model, "custom-model")

    def test_generation_sensitive_chat_settings_reject_mutation_while_generating(
        self,
    ) -> None:
        async def slow_stream(
            chat: Chat,
            model: str,
            generation: GenerationSettings,
        ) -> AsyncIterator[ChatStreamEvent]:
            _ = (chat, generation)
            yield ModelReady(model, "Demo Model", 4096)
            yield FirstToken()
            while True:
                await asyncio.sleep(0.001)
                yield PredictionFragment("x", 1, False)

        self.client.stream_chat = slow_stream

        async def scenario() -> None:
            _ = await self.state.send_user_message("Generate while changing settings")
            await asyncio.sleep(0.01)
            with self.assertRaisesRegex(
                ValueError, "cannot set model while generation is in progress"
            ):
                await self.state.set_active_chat_model("other-model")
            with self.assertRaisesRegex(
                ValueError, "cannot set system prompt while generation is in progress"
            ):
                await self.state.set_active_chat_system_prompt("A changed prompt")
            with self.assertRaisesRegex(
                ValueError,
                "cannot set sampling overrides while generation is in progress",
            ):
                await self.state.set_active_chat_sampling_overrides(
                    ChatSamplingOverrides(temperature=0.2)
                )
            with self.assertRaisesRegex(
                ValueError,
                "cannot set British spelling post-processing while generation is in progress",
            ):
                await self.state.set_active_chat_postprocess_british_spellings(False)
            with self.assertRaisesRegex(
                ValueError,
                "cannot set reasoning persistence while generation is in progress",
            ):
                await self.state.set_active_chat_save_reasoning(False)
            await self.state.stop_generation()

        asyncio.run(scenario())

    def test_stop_generation_recovers_when_a_task_is_cancelled_before_it_starts(
        self,
    ) -> None:
        async def never_started() -> None:
            await asyncio.Event().wait()

        async def scenario() -> None:
            chat_id: str = self.state.active_chat.id
            session = self.state._by_id(chat_id)  # pyright: ignore[reportPrivateUsage]
            expected_idle_status: RuntimeStatus = session.runtime_status
            task: asyncio.Task[None] = asyncio.create_task(never_started())
            _ = task.cancel()
            await asyncio.sleep(0)
            session.is_generating = True
            session.generation_task = task
            session.runtime_status = RuntimeStatus(RuntimePhase.STARTING)

            await self.state.stop_generation()

            self.assertFalse(session.is_generating)
            self.assertIsNone(session.generation_task)
            self.assertEqual(session.runtime_status, expected_idle_status)

        asyncio.run(scenario())

    def test_unloading_a_generating_model_is_rejected(self) -> None:
        async def slow_stream(
            chat: Chat,
            model: str,
            generation: GenerationSettings,
        ) -> AsyncIterator[ChatStreamEvent]:
            _ = (chat, generation)
            yield ModelReady(model, "Demo Model", 4096)
            yield FirstToken()
            while True:
                await asyncio.sleep(0.001)
                yield PredictionFragment("x", 1, False)

        self.client.stream_chat = slow_stream

        async def scenario() -> None:
            await self.state.refresh_runtime_status()
            _ = await self.state.send_user_message("Keep the model loaded")
            await asyncio.sleep(0.01)
            with self.assertRaisesRegex(
                ValueError, "cannot unload a model while it is generating a response"
            ):
                await self.state.unload_model("demo-model")
            with self.assertRaisesRegex(
                ValueError, "cannot unload a model while it is generating a response"
            ):
                await self.state.unload_model_instance("demo-model")
            await self.state.stop_generation()

        asyncio.run(scenario())

    def test_chat_without_an_override_follows_the_global_default_model(self) -> None:
        async def scenario() -> None:
            _ = await self.state.new_chat()
            server = ServerSettings(
                base_url=self.state.config.server.base_url,
                api_key=self.state.config.server.api_key,
                default_model="replacement-model",
                model_aliases=dict(self.state.config.server.model_aliases),
                auto_unload_minutes=self.state.config.server.auto_unload_minutes,
            )
            await self.state.set_global_settings(self.state.config.generation, server)

        asyncio.run(scenario())

        self.assertEqual(self.state.active_chat.model, "")
        self.assertEqual(self.state.current_model_key(), "replacement-model")

    def test_runtime_refresh_reports_loaded_and_unloaded_models(self) -> None:
        async def scenario() -> tuple[RuntimePhase, RuntimePhase]:
            await self.state.refresh_runtime_status()
            loaded_phase = self.state.runtime_status.phase
            await self.state.set_active_chat_model("missing-model")
            return loaded_phase, self.state.runtime_status.phase

        loaded_phase, missing_phase = asyncio.run(scenario())

        self.assertIs(loaded_phase, RuntimePhase.READY)
        self.assertIs(missing_phase, RuntimePhase.MODEL_UNLOADED)

    def test_runtime_status_prefers_configured_model_alias(self) -> None:
        self.state.config.server.model_aliases["demo-model"] = "Local Demo"

        asyncio.run(self.state.refresh_runtime_status())

        self.assertEqual(self.state.runtime_status.model_name, "Local Demo")

    def test_runtime_monitor_detects_when_selected_model_is_unloaded(self) -> None:
        model_is_loaded = True

        async def mutable_inventory() -> tuple[ModelDescriptor, ...]:
            return (
                ModelDescriptor(
                    key="demo-model",
                    display_name="Demo Model",
                    loaded_instance_ids=("demo-model",) if model_is_loaded else (),
                    context_length=4096,
                ),
            )

        self.client.list_model_inventory = mutable_inventory

        async def scenario() -> None:
            nonlocal model_is_loaded
            with patch("jouzetsu.controllers._POLL_INTERVAL_SECONDS", 0.01):
                await self.state.start_runtime_monitor()
                self.assertIs(self.state.runtime_status.phase, RuntimePhase.READY)
                model_is_loaded = False
                for _ in range(100):
                    if self.state.runtime_status.phase is RuntimePhase.MODEL_UNLOADED:
                        return
                    await asyncio.sleep(0.01)
                self.fail("runtime monitor did not detect the unloaded model")

        asyncio.run(scenario())

    def test_unload_model_unloads_instances_and_refreshes_inventory(self) -> None:
        async def scenario() -> None:
            await self.state.refresh_runtime_status()
            await self.state.unload_model("demo-model")

        asyncio.run(scenario())

        self.assertEqual(self.client.unloaded_instance_ids, ["demo-model"])
        self.assertFalse(self.state.model_inventory[0].is_loaded)
        self.assertIs(self.state.runtime_status.phase, RuntimePhase.MODEL_UNLOADED)

    def test_generation_tracks_detailed_phases_and_final_metrics(self) -> None:
        phases: list[RuntimePhase] = []

        async def listener(kind: StateChangeKind) -> None:
            if kind in {StateChangeKind.STATUS, StateChangeKind.STREAM}:
                phases.append(self.state.runtime_status.phase)

        _ = self.state.add_listener(listener)

        async def scenario() -> None:
            _ = await self.state.send_user_message("Show detailed status")
            await self._wait_for_generation()

        asyncio.run(scenario())

        self.assertIn(RuntimePhase.PROCESSING_PROMPT, phases)
        self.assertIn(RuntimePhase.GENERATING, phases)
        self.assertIs(self.state.runtime_status.phase, RuntimePhase.READY)
        metrics = self.state.runtime_status.metrics
        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertEqual(metrics.tokens_per_second, 20.0)

    def test_generation_stage_timer_resets_without_resetting_the_overall_timer(
        self,
    ) -> None:
        async def scenario() -> tuple[RuntimeStatus, RuntimeStatus]:
            loading: RuntimeStatus = RuntimeStatus(
                RuntimePhase.LOADING_MODEL,
                model_key="demo-model",
                model_name="Demo Model",
                started_at=10.0,
                phase_started_at=10.0,
            )
            loading_update: RuntimeStatus = GenerationController.next_runtime_status(
                loading,
                RuntimePhase.LOADING_MODEL,
                model_key="demo-model",
                model_name="Demo Model",
                progress=0.5,
            )
            processing: RuntimeStatus = GenerationController.next_runtime_status(
                loading_update,
                RuntimePhase.PROCESSING_PROMPT,
                model_key="demo-model",
                model_name="Demo Model",
                progress=0.5,
            )
            return loading_update, processing

        loading_update, processing = asyncio.run(scenario())

        self.assertEqual(loading_update.started_at, 10.0)
        assert loading_update.phase_started_at is not None
        self.assertEqual(loading_update.phase_started_at, 10.0)
        self.assertEqual(processing.started_at, 10.0)
        assert processing.phase_started_at is not None
        self.assertGreater(processing.phase_started_at, loading_update.phase_started_at)

    def test_runtime_refresh_keeps_completed_generation_metrics(self) -> None:
        async def scenario() -> tuple[
            GenerationMetrics | None, GenerationMetrics | None
        ]:
            _ = await self.state.send_user_message("Show detailed status")
            await self._wait_for_generation()
            before_refresh = self.state.runtime_status.metrics
            await self.state.refresh_runtime_status()
            return before_refresh, self.state.runtime_status.metrics

        before_refresh, after_refresh = asyncio.run(scenario())

        self.assertIsNotNone(before_refresh)
        self.assertIs(after_refresh, before_refresh)

    def test_model_loading_and_reasoning_are_reported(self) -> None:
        phases: list[RuntimePhase] = []

        async def detailed_stream(
            chat: Chat,
            model: str,
            generation: GenerationSettings,
        ) -> AsyncIterator[ChatStreamEvent]:
            _ = (chat, model, generation)
            yield ModelLoadProgress(0.5)
            yield ModelReady("demo-model", "Demo Model", 4096)
            yield PromptProcessingProgress(0.75)
            yield FirstToken()
            yield PredictionFragment("private reasoning", 2, True)
            yield PredictionFragment("Answer", 1, False)
            yield PredictionComplete(GenerationMetrics(output_tokens=3))

        self.client.stream_chat = detailed_stream

        async def listener(kind: StateChangeKind) -> None:
            if kind in {StateChangeKind.STATUS, StateChangeKind.STREAM}:
                phases.append(self.state.runtime_status.phase)

        _ = self.state.add_listener(listener)

        async def scenario() -> None:
            _ = await self.state.send_user_message("Think first")
            await self._wait_for_generation()

        asyncio.run(scenario())

        self.assertIn(RuntimePhase.LOADING_MODEL, phases)
        self.assertEqual(
            phases[:2], [RuntimePhase.LOADING_MODEL, RuntimePhase.LOADING_MODEL]
        )
        self.assertIn(RuntimePhase.REASONING, phases)
        self.assertEqual(self.state.active_chat.messages[-1].content, "Answer")

    def test_generation_reasoning_is_available_live_and_is_not_persisted(self) -> None:
        reasoning_available: asyncio.Event = asyncio.Event()
        finish_response: asyncio.Event = asyncio.Event()
        self.state.config.generation.continuity_review = False

        async def reasoning_stream(
            chat: Chat,
            model: str,
            generation: GenerationSettings,
        ) -> AsyncIterator[ChatStreamEvent]:
            _ = (chat, model, generation)
            yield FirstToken()
            yield PredictionFragment("Inspecting the request", 2, True)
            _ = reasoning_available.set()
            _ = await finish_response.wait()
            yield PredictionFragment("Visible answer", 1, False)
            yield PredictionComplete(GenerationMetrics(output_tokens=3))

        self.client.stream_chat = reasoning_stream

        async def scenario() -> tuple[str, str, str, str]:
            _ = await self.state.send_user_message("Think first")
            _ = await asyncio.wait_for(reasoning_available.wait(), timeout=1.0)
            live_reasoning: str = self.state.active_generation_reasoning
            _ = finish_response.set()
            await self._wait_for_generation()
            return (
                live_reasoning,
                self.state.active_generation_reasoning,
                self.state.active_chat.messages[-1].content,
                self.state.active_chat.messages[-1].reasoning,
            )

        live_reasoning, completed_reasoning, response, saved_reasoning = asyncio.run(
            scenario()
        )

        self.assertEqual(live_reasoning, "Inspecting the request")
        self.assertEqual(completed_reasoning, "")
        self.assertEqual(response, "Visible answer")
        self.assertEqual(saved_reasoning, "Inspecting the request")
        reloaded = ChatStorage(self.state.config.chats_file).load_all()
        self.assertEqual(reloaded[0].messages[-1].reasoning, "Inspecting the request")

    def test_disabling_reasoning_persistence_clears_the_completed_trace(self) -> None:
        self.state.config.generation.continuity_review = False

        async def reasoning_stream(
            chat: Chat,
            model: str,
            generation: GenerationSettings,
        ) -> AsyncIterator[ChatStreamEvent]:
            _ = (chat, model, generation)
            yield PredictionFragment("Private trace", 1, True)
            yield PredictionFragment("Visible answer", 1, False)

        self.client.stream_chat = reasoning_stream

        async def scenario() -> None:
            assistant = self.state.active_chat.add_message(
                "assistant", "Earlier answer"
            )
            assistant.set_reasoning("Earlier trace")
            await self.state.set_active_chat_save_reasoning(False)
            _ = await self.state.send_user_message("Do not retain reasoning")
            await self._wait_for_generation()

        asyncio.run(scenario())

        self.assertFalse(self.state.active_chat.save_reasoning)
        self.assertEqual(self.state.active_chat.messages[-1].reasoning, "")
        reloaded = ChatStorage(self.state.config.chats_file).load_all()
        self.assertFalse(reloaded[0].save_reasoning)
        self.assertEqual(reloaded[0].messages[-1].reasoning, "")

    def test_generation_starts_in_model_loading_phase_for_an_unloaded_model(
        self,
    ) -> None:
        loading_gate: asyncio.Event = asyncio.Event()

        async def delayed_stream(
            chat: Chat,
            model: str,
            generation: GenerationSettings,
        ) -> AsyncIterator[ChatStreamEvent]:
            _ = (chat, model, generation)
            _ = await loading_gate.wait()
            yield ModelReady(model, "Demo Model", 4096)
            yield FirstToken()
            yield PredictionFragment("Ready", 1, False)
            yield PredictionComplete(GenerationMetrics(output_tokens=1))

        self.client.unloaded_instance_ids.append("demo-model")
        self.client.stream_chat = delayed_stream

        async def scenario() -> RuntimePhase:
            await self.state.refresh_runtime_status()
            _ = await self.state.send_user_message("Load the model")
            phase: RuntimePhase = self.state.runtime_status.phase
            _ = loading_gate.set()
            await self._wait_for_generation()
            return phase

        self.assertIs(asyncio.run(scenario()), RuntimePhase.LOADING_MODEL)

    def test_new_chat_creates_separate_active_conversation(self) -> None:
        async def scenario() -> tuple[str, str]:
            original_chat_id = self.state.active_chat.id
            _ = await self.state.send_user_message("First chat message")
            await self._wait_for_generation()
            new_chat = await self.state.new_chat()
            return original_chat_id, new_chat.id

        original_chat_id, new_chat_id = asyncio.run(scenario())

        self.assertNotEqual(original_chat_id, new_chat_id)
        self.assertEqual(self.state.active_chat.id, new_chat_id)
        self.assertEqual(self.state.active_chat.title, "New chat")
        self.assertEqual(self.state.active_chat.messages, [])

        chat_ids = [chat.id for chat in self.state.chats]
        self.assertEqual(chat_ids[0], new_chat_id)
        self.assertIn(original_chat_id, chat_ids)

    def test_fork_chat_copies_history_through_target_into_independent_chat(
        self,
    ) -> None:
        async def scenario() -> tuple[str, list[str], str]:
            await self.state.set_active_chat_system_prompt("Source prompt")
            _ = await self.state.send_user_message("First question")
            await self._wait_for_generation()
            target_message_id = self.state.active_chat.messages[-1].id
            _ = await self.state.send_user_message("Excluded follow up")
            await self._wait_for_generation()
            await self.state.set_active_chat_draft("Source draft")

            source_chat_id = self.state.active_chat.id
            source_message_ids = [
                message.id for message in self.state.active_chat.messages
            ]
            forked_chat = await self.state.fork_chat_at_message(target_message_id)
            return source_chat_id, source_message_ids, forked_chat.id

        source_chat_id, source_message_ids, forked_chat_id = asyncio.run(scenario())

        forked_chat = self.state.active_chat
        source_chat = next(
            chat for chat in self.state.chats if chat.id == source_chat_id
        )
        self.assertEqual(forked_chat.id, forked_chat_id)
        self.assertEqual(forked_chat.title, source_chat.title)
        self.assertEqual(forked_chat.model, source_chat.model)
        self.assertEqual(forked_chat.system_prompt, "Source prompt")
        self.assertEqual(forked_chat.draft, "")
        self.assertEqual(
            [(message.role, message.content) for message in forked_chat.messages],
            [("user", "First question"), ("assistant", "Hello world")],
        )
        self.assertTrue(
            set(source_message_ids).isdisjoint(
                message.id for message in forked_chat.messages
            )
        )
        self.assertEqual(len(source_chat.messages), 4)

        reloaded = ChatStorage(self.state.config.chats_file).load_all()
        self.assertIn(forked_chat_id, [chat.id for chat in reloaded])
        ui = _saved_config_section(self.state.config.config_file, "ui")
        self.assertEqual(ui["active_chat_id"], forked_chat_id)

    def test_fork_chat_rejects_unknown_message(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown message id"):
            _ = asyncio.run(self.state.fork_chat_at_message("missing-message"))

    def test_fork_chat_rejects_in_progress_generation(self) -> None:
        async def scenario() -> None:
            _ = await self.state.send_user_message("Still generating")
            _ = await self.state.fork_chat_at_message(
                self.state.active_chat.messages[0].id
            )

        with self.assertRaisesRegex(
            ValueError, "cannot fork while generation is in progress"
        ):
            asyncio.run(scenario())

    def test_select_chat_switches_active_chat(self) -> None:
        async def scenario() -> tuple[str, str]:
            first_chat_id = self.state.active_chat.id
            second_chat = await self.state.new_chat()
            await self.state.select_chat(first_chat_id)
            return first_chat_id, second_chat.id

        first_chat_id, second_chat_id = asyncio.run(scenario())

        self.assertEqual(self.state.active_chat.id, first_chat_id)
        self.assertNotEqual(first_chat_id, second_chat_id)
        ui = _saved_config_section(self.state.config.config_file, "ui")
        self.assertEqual(ui["active_chat_id"], first_chat_id)

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
                "continuity_review": True,
                "british_english": True,
                "british_spelling_replacements": [],
            },
        )

    def test_set_global_settings_persists_model_alias_and_auto_unload_minutes(
        self,
    ) -> None:
        generation = GenerationSettings(
            temperature=0.4,
            top_p=0.8,
            max_tokens=1024,
            system_prompt="Use compact answers.",
            british_spelling_replacements=[],
        )
        server = self.state.config.server
        server.model_aliases = {"demo-model": "Local Demo"}
        server.auto_unload_minutes = 15

        asyncio.run(self.state.set_global_settings(generation, server))

        server_json = _saved_config_section(self.state.config.config_file, "server")
        self.assertEqual(server_json["model_aliases"], {"demo-model": "Local Demo"})
        self.assertEqual(server_json["auto_unload_minutes"], 15)
        self.assertEqual(self.state.config.server.auto_unload_minutes, 15)

    def test_set_global_settings_retains_the_shared_server_settings_instance(
        self,
    ) -> None:
        original_server: ServerSettings = self.state.config.server
        updated_server = ServerSettings(
            base_url="http://example.test:4321/v1",
            api_key="updated-key",
            default_model="updated-model",
            model_aliases={"updated-model": "Updated Model"},
            auto_unload_minutes=10,
        )

        asyncio.run(
            self.state.set_global_settings(self.state.config.generation, updated_server)
        )

        self.assertIs(self.state.config.server, original_server)
        self.assertEqual(self.state.config.server, updated_server)

    def test_set_global_settings_persists_message_action_icon_style(self) -> None:
        asyncio.run(
            self.state.set_global_settings(
                self.state.config.generation,
                self.state.config.server,
                message_action_icon_style="muted_color",
            )
        )

        ui = _saved_config_section(self.state.config.config_file, "ui")
        self.assertEqual(ui["message_action_icon_style"], "muted_color")
        self.assertEqual(self.state.config.ui.message_action_icon_style, "muted_color")

    def test_set_global_settings_persists_icon_colors(self) -> None:
        colors = IconColorSettings(
            linework_color="#101112",
            accent_color="#131415",
            surface_color="#161718",
        )

        asyncio.run(
            self.state.set_global_settings(
                self.state.config.generation,
                self.state.config.server,
                icon_colors=colors,
            )
        )

        ui = _saved_config_section(self.state.config.config_file, "ui")
        self.assertEqual(
            ui["icon_colors"],
            {
                "linework_color": "#101112",
                "accent_color": "#131415",
                "surface_color": "#161718",
            },
        )
        self.assertEqual(self.state.config.ui.icon_colors, colors)

    def test_set_generation_defaults_rejects_invalid_values_without_mutating_config(
        self,
    ) -> None:
        original_settings = self.state.config.generation
        invalid_settings = GenerationSettings(temperature=2.1)

        with self.assertRaisesRegex(ValueError, "temperature must be between 0 and 2"):
            asyncio.run(self.state.set_generation_defaults(invalid_settings))

        self.assertIs(self.state.config.generation, original_settings)

    def test_set_active_chat_system_prompt_persists_to_chat_storage(self) -> None:
        asyncio.run(self.state.set_active_chat_system_prompt("Only answer in JSON."))

        reloaded = ChatStorage(self.state.config.chats_file).load_all()
        self.assertEqual(reloaded[0].system_prompt, "Only answer in JSON.")

    def test_chat_draft_persists_without_reordering_chats(self) -> None:
        async def scenario() -> tuple[str, str]:
            first_chat_id = self.state.active_chat.id
            _ = await self.state.send_user_message("First chat message")
            await self._wait_for_generation()
            newer_chat = await self.state.new_chat()
            await self.state.set_chat_draft(first_chat_id, "Saved draft")
            await self.state.persist_chats()
            return first_chat_id, newer_chat.id

        first_chat_id, newer_chat_id = asyncio.run(scenario())

        reloaded = ChatStorage(self.state.config.chats_file).load_all()
        saved_first_chat = next(chat for chat in reloaded if chat.id == first_chat_id)
        self.assertEqual(saved_first_chat.draft, "Saved draft")
        self.assertEqual(self.state.chats[0].id, newer_chat_id)

    def test_rapid_draft_updates_are_coalesced_into_one_storage_write(self) -> None:
        async def scenario() -> int:
            with patch.object(
                self.state.storage,
                "save_records",
                wraps=self.state.storage.save_records,
            ) as save_records:
                await self.state.set_active_chat_draft("First")
                await self.state.set_active_chat_draft("Second")
                await self.state.set_active_chat_draft("Final")
                await self.state.persist_chats()
                return save_records.call_count

        self.assertEqual(asyncio.run(scenario()), 1)
        self.assertEqual(self.state.active_chat.draft, "Final")

    def test_transient_chat_write_failure_is_retried_on_later_flush(self) -> None:
        _ = asyncio.run(self.state.persist_chats())
        original_save_records = self.state.storage.save_records
        save_attempts = 0

        def save_records_once_available(
            chat_records: Sequence[ChatJSON],
            *,
            deleted_chat_ids: Iterable[str] = (),
        ) -> None:
            nonlocal save_attempts
            save_attempts += 1
            if save_attempts == 1:
                raise OSError("transient disk failure")
            original_save_records(chat_records, deleted_chat_ids=deleted_chat_ids)

        async def scenario() -> None:
            with patch.object(
                self.state.storage,
                "save_records",
                side_effect=save_records_once_available,
            ):
                await self.state.set_active_chat_draft("Retry this draft")
                with self.assertRaisesRegex(OSError, "transient disk failure"):
                    await self.state._persistence.flush_chats()  # pyright: ignore[reportPrivateUsage]
                await self.state._persistence.flush_chats()  # pyright: ignore[reportPrivateUsage]

        asyncio.run(scenario())

        self.assertEqual(save_attempts, 2)
        reloaded = ChatStorage(self.state.config.chats_file).load_all()
        self.assertEqual(reloaded[0].draft, "Retry this draft")

    def test_concurrent_draft_updates_preserve_every_chat(self) -> None:
        async def scenario() -> tuple[str, str]:
            first_chat_id = self.state.active_chat.id
            second_chat = await self.state.new_chat()
            _ = await asyncio.gather(
                self.state.set_chat_draft(first_chat_id, "First draft"),
                self.state.set_chat_draft(second_chat.id, "Second draft"),
            )
            await self.state.persist_chats()
            return first_chat_id, second_chat.id

        first_chat_id, second_chat_id = asyncio.run(scenario())
        chats_by_id = {
            chat.id: chat
            for chat in ChatStorage(self.state.config.chats_file).load_all()
        }
        self.assertEqual(chats_by_id[first_chat_id].draft, "First draft")
        self.assertEqual(chats_by_id[second_chat_id].draft, "Second draft")

    def test_runtime_refresh_retries_transient_inventory_failure(self) -> None:
        original_inventory = self.client.list_model_inventory
        attempts = 0

        async def eventually_available() -> tuple[ModelDescriptor, ...]:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise RuntimeError("LM Studio is starting")
            return await original_inventory()

        self.client.list_model_inventory = eventually_available

        async def scenario() -> RuntimePhase:
            with patch(
                "jouzetsu.controllers._INVENTORY_RETRY_DELAYS_SECONDS", (0.0, 0.0)
            ):
                await self.state.refresh_runtime_status()
            return self.state.runtime_status.phase

        self.assertEqual(asyncio.run(scenario()), RuntimePhase.READY)
        self.assertEqual(attempts, 3)

    def test_shutdown_reports_success_after_flushing_state(self) -> None:
        self.assertTrue(asyncio.run(self.state.shutdown()))
        self.assertTrue(self.client.closed)

    def test_shutdown_stops_an_active_generation_and_flushes_chat_state(self) -> None:
        async def slow_stream(
            chat: Chat,
            model: str,
            generation: GenerationSettings,
        ) -> AsyncIterator[ChatStreamEvent]:
            _ = (chat, generation)
            yield ModelReady(model, "Demo Model", 4096)
            yield FirstToken()
            while True:
                await asyncio.sleep(0.001)
                yield PredictionFragment("x", 1, False)

        self.client.stream_chat = slow_stream

        async def scenario() -> bool:
            _ = await self.state.send_user_message("Generate until shutdown")
            await asyncio.sleep(0.01)
            return await self.state.shutdown()

        self.assertTrue(asyncio.run(scenario()))
        reloaded = ChatStorage(self.state.config.chats_file).load_all()
        self.assertEqual(reloaded[0].messages[0].content, "Generate until shutdown")

    def test_shutdown_stops_generations_for_every_chat(self) -> None:
        async def slow_stream(
            chat: Chat,
            model: str,
            generation: GenerationSettings,
        ) -> AsyncIterator[ChatStreamEvent]:
            _ = (chat, generation)
            yield ModelReady(model, "Demo Model", 4096)
            yield FirstToken()
            while True:
                await asyncio.sleep(0.001)
                yield PredictionFragment("x", 1, False)

        self.client.stream_chat = slow_stream

        async def scenario() -> tuple[bool, tuple[str, str], tuple[bool, bool]]:
            _ = await self.state.send_user_message("First generation")
            first_chat_id = self.state.active_chat.id
            second_chat = await self.state.new_chat()
            _ = await self.state.send_user_message("Second generation")
            await asyncio.sleep(0.01)
            shutdown_clean = await self.state.shutdown()
            chat_ids: tuple[str, str] = (first_chat_id, second_chat.id)
            generating_states: tuple[bool, bool] = (
                self.state.is_generating(first_chat_id),
                self.state.is_generating(second_chat.id),
            )
            return shutdown_clean, chat_ids, generating_states

        shutdown_clean, chat_ids, generating_states = asyncio.run(scenario())

        self.assertTrue(shutdown_clean)
        self.assertTrue(self.client.closed)
        self.assertEqual(generating_states, (False, False))
        reloaded_by_id = {
            chat.id: chat
            for chat in ChatStorage(self.state.config.chats_file).load_all()
        }
        self.assertEqual(
            reloaded_by_id[chat_ids[0]].messages[0].content, "First generation"
        )
        self.assertEqual(
            reloaded_by_id[chat_ids[1]].messages[0].content, "Second generation"
        )

    def test_shutdown_waits_for_generation_finalization_task(self) -> None:
        original_commit_chat = self.state._generation._commit_chat  # pyright: ignore[reportPrivateUsage]
        finalization_started = asyncio.Event()
        release_finalization = asyncio.Event()
        commit_count = 0

        async def block_final_commit(chat: Chat) -> None:
            nonlocal commit_count
            commit_count += 1
            if commit_count == 2:
                finalization_started.set()
                await release_finalization.wait()
            await original_commit_chat(chat)

        async def scenario() -> bool:
            self.state._generation._commit_chat = block_final_commit  # pyright: ignore[reportPrivateUsage]
            await self.state._start_generation()  # pyright: ignore[reportPrivateUsage]
            await finalization_started.wait()
            session = self.state._by_id(self.state.active_chat.id)  # pyright: ignore[reportPrivateUsage]
            self.assertFalse(session.is_generating)
            self.assertIsNotNone(session.generation_task)

            shutdown_task = asyncio.create_task(self.state.shutdown())
            try:
                with self.assertRaises(TimeoutError):
                    _ = await asyncio.wait_for(
                        asyncio.shield(shutdown_task), timeout=0.01
                    )
            finally:
                release_finalization.set()
            return await shutdown_task

        self.assertTrue(asyncio.run(scenario()))

    def test_lm_studio_disconnect_mid_generation_preserves_a_visible_error(
        self,
    ) -> None:
        async def disconnected_stream(
            chat: Chat,
            model: str,
            generation: GenerationSettings,
        ) -> AsyncIterator[ChatStreamEvent]:
            _ = (chat, generation)
            yield ModelReady(model, "Demo Model", 4096)
            raise LMStudioError("connection lost")

        self.client.stream_chat = disconnected_stream

        async def scenario() -> RuntimePhase:
            _ = await self.state.send_user_message("Trigger disconnect")
            await self._wait_for_generation()
            return self.state.runtime_status.phase

        self.assertIs(asyncio.run(scenario()), RuntimePhase.ERROR)
        self.assertIn("connection lost", self.state.active_chat.messages[-1].content)

    def test_send_user_message_clears_active_chat_draft(self) -> None:
        async def scenario() -> None:
            await self.state.set_active_chat_draft("Pending draft")
            _ = await self.state.send_user_message("Sent message")
            await self._wait_for_generation()

        asyncio.run(scenario())

        self.assertEqual(self.state.active_chat.draft, "")

    def test_rename_chat_updates_title(self) -> None:
        asyncio.run(self.state.rename_chat(self.state.active_chat.id, "Renamed chat"))

        self.assertEqual(self.state.active_chat.title, "Renamed chat")
        reloaded = ChatStorage(self.state.config.chats_file).load_all()
        self.assertEqual(reloaded[0].title, "Renamed chat")

    def test_delete_chat_removes_it_and_keeps_active_chat_valid(self) -> None:
        async def scenario() -> tuple[
            str, str, tuple[dict[str, object], ...], frozenset[str]
        ]:
            first_chat_id = self.state.active_chat.id
            await self.state.rename_chat(first_chat_id, "First chat")
            second_chat = await self.state.new_chat()
            await self.state.rename_chat(second_chat.id, "Second chat")
            await self.state.select_chat(first_chat_id)
            with patch.object(
                self.state.storage,
                "save_records",
                wraps=self.state.storage.save_records,
            ) as save_records:
                await self.state.delete_chat(first_chat_id)
                call = save_records.call_args
            self.assertIsNotNone(call)
            if call is None:
                raise AssertionError("chat delete did not reach storage")
            records = cast(tuple[dict[str, object], ...], call.args[0])
            deleted_ids = cast(frozenset[str], call.kwargs["deleted_chat_ids"])
            return first_chat_id, second_chat.id, records, deleted_ids

        deleted_chat_id, surviving_chat_id, changed_records, deleted_ids = asyncio.run(
            scenario()
        )

        self.assertEqual(self.state.active_chat.id, surviving_chat_id)
        self.assertNotIn(deleted_chat_id, [chat.id for chat in self.state.chats])
        self.assertEqual(changed_records, ())
        self.assertEqual(deleted_ids, frozenset({deleted_chat_id}))
        reloaded_chat_ids: set[str] = {
            chat.id for chat in ChatStorage(self.state.config.chats_file).load_all()
        }
        self.assertEqual(reloaded_chat_ids, {surviving_chat_id})
        self.assertTrue(
            (
                ChatStorage(self.state.config.chats_file).trash_directory
                / f"{deleted_chat_id}.json"
            ).exists()
        )

    def test_delete_chat_refuses_to_remove_a_chat_with_a_live_generation_task(
        self,
    ) -> None:
        async def cancellation_still_in_progress(_chat_id: str | None = None) -> bool:
            return False

        async def scenario() -> str:
            chat_id: str = self.state.active_chat.id
            chat_state = self.state._by_id(chat_id)  # pyright: ignore[reportPrivateUsage]
            chat_state.is_generating = True
            try:
                with (
                    patch.object(
                        self.state,
                        "_cancel_generation",
                        new=cancellation_still_in_progress,
                    ),
                    self.assertRaisesRegex(
                        RuntimeError, "generation is still stopping"
                    ),
                ):
                    await self.state.delete_chat(chat_id)
            finally:
                chat_state.is_generating = False
            return chat_id

        chat_id: str = asyncio.run(scenario())

        self.assertEqual(self.state.active_chat.id, chat_id)
        self.assertEqual([chat.id for chat in self.state.chats], [chat_id])

    def test_chat_mutation_serializes_only_the_changed_chat(self) -> None:
        async def scenario() -> tuple[str, tuple[dict[str, object], ...]]:
            first_chat_id: str = self.state.active_chat.id
            _ = await self.state.new_chat()
            _ = await self.state.new_chat()
            with patch.object(
                self.state.storage,
                "save_records",
                wraps=self.state.storage.save_records,
            ) as save_records:
                await self.state.rename_chat(first_chat_id, "Changed first chat")
                call = save_records.call_args
            self.assertIsNotNone(call)
            if call is None:
                raise AssertionError("chat mutation did not reach storage")
            return first_chat_id, cast(tuple[dict[str, object], ...], call.args[0])

        changed_chat_id, records = asyncio.run(scenario())

        self.assertEqual([record["id"] for record in records], [changed_chat_id])

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
                await self.state.rename_chat(self.state.active_chat.id, "First rename")
                await self.state.rename_chat(self.state.active_chat.id, "Second rename")

        asyncio.run(scenario())

        self.assertEqual(stale_calls, 1)
        self.assertEqual(healthy_calls, 2)
        asyncio.run(self.state.rename_chat(self.state.active_chat.id, "Third rename"))
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

        asyncio.run(
            self.state.rename_chat(self.state.active_chat.id, "No listener call")
        )
        self.assertEqual(calls, 0)

    def test_active_chat_selection_restores_from_config(self) -> None:
        async def scenario() -> tuple[str, str]:
            first_chat_id = self.state.active_chat.id
            second_chat = await self.state.new_chat()
            await self.state.select_chat(first_chat_id)
            return first_chat_id, second_chat.id

        first_chat_id, _second_chat_id = asyncio.run(scenario())
        saved_config = ConfigStore.for_config_file(self.state.config.config_file).load()

        reloaded_state = AppState(
            saved_config, ChatStorage(saved_config.chats_file), FakeLMStudioClient()
        )
        try:
            self.assertEqual(reloaded_state.active_chat.id, first_chat_id)
        finally:
            _ = asyncio.run(reloaded_state.shutdown())

    def test_reloaded_pristine_chat_has_an_empty_state_message(self) -> None:
        async def scenario() -> None:
            _ = await self.state.new_chat()

        asyncio.run(scenario())
        saved_config = ConfigStore.for_config_file(self.state.config.config_file).load()

        reloaded_state = AppState(
            saved_config, ChatStorage(saved_config.chats_file), FakeLMStudioClient()
        )
        try:
            self.assertTrue(reloaded_state.active_empty_state_message)
        finally:
            _ = asyncio.run(reloaded_state.shutdown())

    def test_edit_message_rejects_blank_content(self) -> None:
        message = self.state.active_chat.add_message("user", "Original question")

        async def scenario() -> None:
            await self.state.edit_message(message.id, "   ")

        with self.assertRaisesRegex(ValueError, "message cannot be empty"):
            asyncio.run(scenario())

    def test_edit_message_updates_in_place_without_truncating_history(self) -> None:
        async def scenario() -> tuple[str, str]:
            _ = await self.state.send_user_message("Original question")
            await self._wait_for_generation()
            original_assistant_id = self.state.active_chat.messages[-1].id
            user_message_id = self.state.active_chat.messages[0].id
            await self.state.edit_message(user_message_id, "Edited question")
            return user_message_id, original_assistant_id

        user_message_id, assistant_message_id = asyncio.run(scenario())

        messages = self.state.active_chat.messages
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].id, user_message_id)
        self.assertEqual(messages[0].content, "Edited question")
        self.assertEqual(messages[1].id, assistant_message_id)
        self.assertEqual(messages[1].content, "Hello world")

    def test_edit_assistant_message_does_not_continue_response(self) -> None:
        stream_calls = 0
        original_stream_chat = self.client.stream_chat

        async def counted_stream_chat(
            chat: Chat,
            model: str,
            generation: GenerationSettings,
        ) -> AsyncIterator[ChatStreamEvent]:
            nonlocal stream_calls
            stream_calls += 1
            async for event in original_stream_chat(chat, model, generation):
                yield event

        self.client.stream_chat = counted_stream_chat

        async def scenario() -> str:
            _ = await self.state.send_user_message("Original question")
            await self._wait_for_generation()
            assistant_message_id = self.state.active_chat.messages[-1].id
            await self.state.edit_message(assistant_message_id, "Edited answer")
            return assistant_message_id

        assistant_message_id = asyncio.run(scenario())

        messages = self.state.active_chat.messages
        self.assertEqual(stream_calls, 2)
        self.assertFalse(self.state.is_generating())
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1].id, assistant_message_id)
        self.assertEqual(messages[1].content, "Edited answer")

    def test_edit_message_while_generating_raises(self) -> None:
        async def scenario() -> None:
            _ = await self.state.send_user_message("Trigger generation")
            await self.state.edit_message(
                self.state.active_chat.messages[0].id, "Edited"
            )

        with self.assertRaisesRegex(
            ValueError, "cannot edit while generation is in progress"
        ):
            asyncio.run(scenario())

    def test_delete_message_removes_only_target_message(self) -> None:
        async def scenario() -> str:
            _ = await self.state.send_user_message("Keep question")
            await self._wait_for_generation()
            assistant_message_id = self.state.active_chat.messages[1].id
            await self.state.delete_message(assistant_message_id)
            return assistant_message_id

        deleted_message_id = asyncio.run(scenario())

        messages = self.state.active_chat.messages
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].content, "Keep question")
        self.assertNotEqual(messages[0].id, deleted_message_id)

    def test_delete_message_and_following_removes_the_selected_transcript_tail(
        self,
    ) -> None:
        async def scenario() -> str:
            _ = await self.state.send_user_message("First question")
            await self._wait_for_generation()
            second_user: Message = await self.state.send_user_message("Second question")
            await self._wait_for_generation()
            undo = await self.state.delete_message_and_following_with_undo(
                second_user.id
            )
            self.assertIsNotNone(undo)
            return second_user.id

        deleted_message_id = asyncio.run(scenario())

        messages = self.state.active_chat.messages
        self.assertEqual([message.role for message in messages], ["user", "assistant"])
        self.assertNotEqual(messages[-1].id, deleted_message_id)

    def test_truncate_chat_to_message_keeps_target_and_drops_following_messages(
        self,
    ) -> None:
        async def scenario() -> str:
            _ = await self.state.send_user_message("First question")
            await self._wait_for_generation()
            first_message_id = self.state.active_chat.messages[0].id
            _ = await self.state.send_user_message("Follow up question")
            await self._wait_for_generation()
            await self.state.truncate_chat_to_message(first_message_id)
            return first_message_id

        kept_message_id = asyncio.run(scenario())

        messages = self.state.active_chat.messages
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].id, kept_message_id)
        self.assertEqual(messages[0].content, "First question")

    async def _wait_for_generation(self) -> None:
        for _ in range(100):
            if not self.state.is_generating():
                return
            await asyncio.sleep(0.01)
        self.fail("generation did not complete")
