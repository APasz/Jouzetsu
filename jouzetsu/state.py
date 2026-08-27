"""Application-state façade coordinating chats, configuration, and runtime services."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import replace
from logging import Logger

from .character_storage import CharacterStorage
from .config import (
    AppConfig,
    ConfigStore,
    GenerationSettings,
    HostStatsDeviceSettings,
    IconColorSettings,
    MessageActionIconStyle,
    ServerSettings,
)
from .controllers import AccessController, PersistenceController, RuntimeController
from .empty_chat_messages import EmptyChatMessageProvider
from .events import StateChangeKind
from .host_stats import (
    HostStatsMonitor,
    HostStatsSnapshot,
    is_gpu_enabled,
    is_network_interface_enabled,
)
from .lmstudio import LMStudioClientProtocol
from .logging_config import chat_logger
from .models import (
    Character,
    CharacterChatBinding,
    CharacterField,
    CharacterPresetSelection,
    Chat,
    ChatSamplingOverrides,
    ChatTitleSource,
    Message,
)
from .runtime import ModelDescriptor, RuntimeStatus
from .state_events import ChangeListener, StateNotifier
from .state_generation import GenerationController, chat_with_continuation_prompt
from .state_registry import ChatRegistry
from .state_types import (
    NO_DELETED_CHAT_IDS,
    ChatMessageUndo,
    ChatSession,
    require_non_empty_text,
    restore_messages,
    snapshot_messages,
    trimmed_or_empty,
)
from .storage import ChatStorage

log: Logger = logging.getLogger(__name__)
chat_log: Logger = chat_logger()
_SHUTDOWN_TIMEOUT_SECONDS: float = 10.0


class AppState:
    def __init__(
        self,
        config: AppConfig,
        storage: ChatStorage,
        client: LMStudioClientProtocol,
        character_storage: CharacterStorage | None = None,
        config_store: ConfigStore | None = None,
    ) -> None:
        self.config: AppConfig = config
        self.storage: ChatStorage = storage
        self.character_storage: CharacterStorage = (
            character_storage or CharacterStorage(config.characters_directory)
        )
        self.client: LMStudioClientProtocol = client
        self._persistence: PersistenceController = PersistenceController(
            config,
            config_store or ConfigStore.for_paths(config.paths),
            storage,
        )
        self._access: AccessController = AccessController(
            config=config,
            persist_config=self._commit_config,
            persist_activity=self._commit_access_activity,
        )
        self._host_stats: HostStatsMonitor = HostStatsMonitor()
        self._empty_chat_messages: EmptyChatMessageProvider = EmptyChatMessageProvider(
            config.data_dir
        )
        self._chat_registry: ChatRegistry = ChatRegistry(
            self._empty_chat_messages.choose
        )
        self._characters: dict[str, Character] = {}
        self._notifier: StateNotifier = StateNotifier()
        self._runtime: RuntimeController = RuntimeController(
            config,
            client,
            lambda: self._chat_registry.sessions,
            self.model_display_name,
            self._notify,
        )
        self._generation: GenerationController = GenerationController(
            client=client,
            default_generation=lambda: self.config.generation,
            default_model=lambda: self.config.server.default_model,
            model_descriptor=self._runtime.model_descriptor,
            model_display_name=self.model_display_name,
            idle_status=self._runtime.idle_status,
            commit_chat=self._commit_chat,
            notify=self._notify,
        )

        self._load_from_disk()

    def add_listener(self, listener: ChangeListener) -> Callable[[], None]:
        """Register a listener and return an idempotent removal callback."""
        return self._notifier.add(listener)

    def remove_listener(self, listener: ChangeListener) -> None:
        """Remove a previously registered listener if it is still present."""
        self._notifier.remove(listener)

    async def _notify(self, kind: StateChangeKind = StateChangeKind.FULL) -> None:
        await self._notifier.publish(kind)

    # ----- accessors -----

    @property
    def chats(self) -> list[Chat]:
        """Return chats sorted by most recent activity first."""
        return self._chat_registry.chats_by_activity

    @property
    def characters(self) -> list[Character]:
        """Return reusable profiles ordered by most recently edited first."""

        return sorted(
            self._characters.values(),
            key=lambda character: character.updated_at,
            reverse=True,
        )

    def character(self, character_id: str) -> Character:
        """Resolve one character or fail clearly when a stale UI target is submitted."""

        try:
            return self._characters[character_id]
        except KeyError as exc:
            raise ValueError("unknown character") from exc

    @property
    def active_chat(self) -> Chat:
        if not self._chat_registry:
            return self._create_chat_internal()
        return self._chat_registry.active.chat

    @property
    def global_system_prompt(self) -> str:
        return self.config.generation.system_prompt

    def is_generating(self, chat_id: str | None = None) -> bool:
        session: ChatSession = self._by_id(
            chat_id or self._chat_registry.active_chat_id
        )
        return session.is_generating

    @property
    def active_generation_reasoning(self) -> str:
        """Return transient reasoning for the active in-progress response."""

        session: ChatSession = self._chat_registry.active
        return session.live_reasoning if session.is_generating else ""

    @property
    def active_empty_state_message(self) -> str:
        """Return the stable random message selected for the active pristine chat."""

        return self._chat_registry.active.empty_state_message

    @property
    def runtime_status(self) -> RuntimeStatus:
        """Return status information for the active chat."""
        return self._chat_registry.active.runtime_status

    @property
    def model_inventory(self) -> tuple[ModelDescriptor, ...]:
        return self._runtime.inventory

    def model_display_name(self, model_key: str, fallback: str = "") -> str:
        """Resolve the configured alias for a model, then its server name."""
        return self.config.server.model_aliases.get(model_key, fallback or model_key)

    def current_model_key(self) -> str:
        """Return the active resolved model key when one is known."""
        if self.runtime_status.model_key:
            return self.runtime_status.model_key
        selected_model: str = self.active_chat.model or self.config.server.default_model
        if selected_model:
            descriptor: ModelDescriptor | None = self._runtime.model_descriptor(
                selected_model
            )
            return descriptor.key if descriptor is not None else selected_model
        loaded: ModelDescriptor | None = next(
            (model for model in self.model_inventory if model.is_loaded), None
        )
        return loaded.key if loaded is not None else ""

    def current_model_alias(self) -> str:
        model_key = self.current_model_key()
        return self.config.server.model_aliases.get(model_key, "") if model_key else ""

    def current_model_auto_unload_minutes(self) -> int | None:
        model_key: str = self.current_model_key()
        descriptor: ModelDescriptor | None = self._runtime.model_descriptor(model_key)
        return descriptor.auto_unload_minutes if descriptor is not None else None

    def is_model_generating(self, model_key: str) -> bool:
        """Return whether any generation may be using the named model."""

        return any(
            session.is_generating and self._generation_model_key(session) == model_key
            for session in self._chat_registry.sessions
        )

    def _ensure_model_is_idle(self, model_key: str) -> None:
        """Reject unloading a model that may serve an active generation."""

        if self.is_model_generating(model_key):
            raise ValueError("cannot unload a model while it is generating a response")

    def _generation_model_key(self, session: ChatSession) -> str:
        """Resolve the server model key selected by one active generation."""

        requested_model: str = (
            session.runtime_status.model_key
            or session.chat.model
            or self.config.server.default_model
        )
        descriptor: ModelDescriptor | None = self._runtime.model_descriptor(
            requested_model
        )
        return descriptor.key if descriptor is not None else requested_model

    def host_stats_snapshot(self) -> HostStatsSnapshot:
        """Return the latest monitor sample, bootstrapping only before monitoring starts."""
        return self._host_stats.latest_snapshot()

    async def refresh_host_stats_snapshot(self) -> HostStatsSnapshot:
        """Collect one explicit host sample without blocking the web event loop."""

        snapshot: HostStatsSnapshot = await self._host_stats.sample_async()
        if self._register_host_stats_devices(snapshot):
            await self._commit(config=True, notify=False)
        return snapshot

    def _register_host_stats_devices(self, snapshot: HostStatsSnapshot) -> bool:
        """Persist newly discovered GPU and network devices with their default visibility."""

        settings = self.config.host_stats
        changed: bool = False
        for gpu in snapshot.gpus:
            if gpu.id not in settings.gpus:
                settings.gpus[gpu.id] = HostStatsDeviceSettings(
                    visible=is_gpu_enabled(gpu, {}),
                    label=gpu.name,
                )
                changed = True
            elif not settings.gpus[gpu.id].label:
                settings.gpus[gpu.id].label = gpu.name
                changed = True
        for network in snapshot.networks:
            if network.name not in settings.interfaces:
                settings.interfaces[network.name] = HostStatsDeviceSettings(
                    visible=is_network_interface_enabled(network, {}),
                    label=network.name,
                )
                changed = True
            elif not settings.interfaces[network.name].label:
                settings.interfaces[network.name].label = network.name
                changed = True
        return changed

    async def unload_model(self, model_key: str) -> None:
        """Unload every currently loaded instance for one model."""
        self._ensure_model_is_idle(model_key)
        descriptor: ModelDescriptor = await self._runtime.unload_model(model_key)
        log.info(
            "model unloaded model_key=%s instance_count=%d",
            descriptor.key,
            len(descriptor.loaded_instance_ids),
        )

    async def unload_model_instance(self, instance_id: str) -> None:
        """Unload one currently loaded model instance."""
        descriptor: ModelDescriptor | None = self._runtime.model_descriptor(instance_id)
        if descriptor is not None:
            self._ensure_model_is_idle(descriptor.key)
        unloaded_descriptor: ModelDescriptor = (
            await self._runtime.unload_model_instance(instance_id)
        )
        log.info(
            "model instance unloaded model_key=%s instance_id=%s",
            unloaded_descriptor.key,
            instance_id,
        )

    async def start_runtime_monitor(self) -> None:
        """Start the single application-wide LM Studio status poller."""
        await self._runtime.start()

    async def start_host_stats_monitor(self) -> None:
        """Start the application-wide host metrics sampler."""
        await self._host_stats.start()
        if self._register_host_stats_devices(self._host_stats.latest_snapshot()):
            await self._commit(config=True, notify=False)

    async def refresh_runtime_status(self) -> None:
        """Refresh reachability, downloaded models, and loaded instances."""
        await self._runtime.refresh()

    def _idle_runtime_status(self, chat: Chat) -> RuntimeStatus:
        return self._runtime.idle_status(chat)

    def _load_from_disk(self) -> None:
        self._characters = {
            character.id: character for character in self.character_storage.load_all()
        }
        log.info("loaded characters count=%d", len(self._characters))
        loaded: list[Chat] = self.storage.load_all()
        if not loaded:
            chat: Chat = self._create_chat_internal()
            self.config.ui.active_chat_id = chat.id
            log.info("initialised empty chat store with chat_id=%s", chat.id)
            chat_log.info(
                "chat_created chat_id=%s reason=initial_empty_state model=%s message_count=%d",
                chat.id,
                chat.model,
                len(chat.messages),
            )
            return
        for chat in loaded:
            self._chat_registry.add(
                chat,
                runtime_status=self._idle_runtime_status(chat),
                persist=False,
                activate=False,
            )
        configured_id: str = self.config.ui.active_chat_id
        if configured_id and configured_id in self._chat_registry:
            _ = self._chat_registry.select(configured_id)
        else:
            self.config.ui.active_chat_id = self._chat_registry.select_most_recent()
        log.info(
            "loaded chats count=%d active_chat_id=%s",
            len(self._chat_registry),
            self._chat_registry.active_chat_id,
        )

    def _create_chat_internal(self) -> Chat:
        chat: Chat = Chat()
        self._add_and_activate_chat(chat)
        return chat

    def _add_and_activate_chat(self, chat: Chat) -> None:
        """Add a chat to state and make it the active conversation."""
        self._chat_registry.add(
            chat,
            runtime_status=self._idle_runtime_status(chat),
            persist=True,
            activate=True,
        )
        self.config.ui.active_chat_id = chat.id

    async def new_chat(self) -> Chat:
        """Create a new chat and make it active."""
        chat: Chat = self._create_chat_internal()
        await self._commit_chat(chat, config=True)
        chat_log.info(
            "chat_created chat_id=%s reason=user model=%s message_count=%d",
            chat.id,
            chat.model,
            len(chat.messages),
        )
        return chat

    async def new_character(self) -> Character:
        """Create a blank flexible profile without imposing a fixed character schema."""

        character: Character = Character()
        self.character_storage.save(character)
        self._characters[character.id] = character
        await self._notify(StateChangeKind.CHARACTERS)
        log.info("character_created character_id=%s", character.id)
        return character

    async def update_character(
        self,
        character_id: str,
        *,
        expected_revision: int,
        name: str,
        fields: list[CharacterField],
        presets: CharacterPresetSelection | None = None,
    ) -> Character:
        """Persist one current profile revision before exposing it to the active application."""

        character: Character = self.character(character_id)
        if character.revision != expected_revision:
            raise ValueError("character changed elsewhere; reload before saving")
        updated_character: Character = character.revised_profile(
            name, fields, presets=presets
        )
        if updated_character is character:
            return character
        self.character_storage.save(updated_character)
        self._characters[character_id] = updated_character
        await self._notify(StateChangeKind.CHARACTERS)
        log.info(
            "character_updated character_id=%s revision=%d field_count=%d",
            updated_character.id,
            updated_character.revision,
            len(updated_character.fields),
        )
        return updated_character

    async def delete_character(self, character_id: str) -> None:
        """Delete one profile while retaining all chat prompt snapshots that used it."""

        character: Character = self.character(character_id)
        self.character_storage.delete(character.id)
        del self._characters[character.id]
        await self._notify(StateChangeKind.CHARACTERS)
        log.info("character_deleted character_id=%s", character.id)

    async def start_chat_from_character(self, character_id: str) -> Chat:
        """Start a conversation using an immutable copy of the character's current prompt."""

        character: Character = self.character(character_id)
        chat: Chat = Chat(
            title=character.name,
            title_source=ChatTitleSource.MANUAL,
            system_prompt=character.compiled_system_prompt(),
            character=CharacterChatBinding(
                id=character.id,
                name=character.name,
                revision=character.revision,
            ),
        )
        self._add_and_activate_chat(chat)
        await self._commit_chat(chat, config=True)
        chat_log.info(
            "chat_created chat_id=%s reason=character character_id=%s character_revision=%d",
            chat.id,
            character.id,
            character.revision,
        )
        return chat

    async def fork_chat_at_message(self, message_id: str) -> Chat:
        """Fork the active chat through a message and select the new chat."""
        source_state: ChatSession = self._idle_active_state("fork")

        forked_chat: Chat = source_state.chat.fork_through(message_id)
        self._add_and_activate_chat(forked_chat)
        await self._commit_chat(forked_chat, config=True)
        chat_log.info(
            "chat_forked source_chat_id=%s chat_id=%s through_message_id=%s message_count=%d",
            source_state.chat.id,
            forked_chat.id,
            message_id,
            len(forked_chat.messages),
        )
        return forked_chat

    async def select_chat(self, chat_id: str) -> None:
        if not self._chat_registry.select(chat_id):
            return
        self.config.ui.active_chat_id = chat_id
        await self._commit(config=True)
        chat_log.info("chat_selected chat_id=%s", chat_id)

    async def set_chat_draft(self, chat_id: str, draft: str | None) -> None:
        """Persist in-progress input text without affecting chat ordering."""
        if not self.update_chat_draft(chat_id, draft):
            return
        await self._commit_chat(
            self._by_id(chat_id).chat, flush_chats=False, notify=False
        )

    def update_chat_draft(self, chat_id: str, draft: str | None) -> bool:
        """Update a draft in memory and report whether it changed."""
        chat: Chat = self._by_id(chat_id).chat
        next_draft: str = draft or ""
        if chat.draft == next_draft:
            return False
        chat.draft = next_draft
        return True

    async def persist_chats(self) -> None:
        """Persist the current in-memory chats."""
        await self._commit(
            changed_chats=tuple(
                session.chat for session in self._chat_registry.sessions
            ),
            notify=False,
        )

    async def set_active_chat_draft(self, draft: str | None) -> None:
        await self.set_chat_draft(self._chat_registry.active_chat_id, draft)

    def _by_id(self, chat_id: str) -> ChatSession:
        return self._chat_registry.get(chat_id)

    def _idle_active_state(self, operation: str) -> ChatSession:
        """Return the active state, rejecting mutations during generation."""
        state: ChatSession = self._chat_registry.active
        if state.is_generating:
            raise ValueError(f"cannot {operation} while generation is in progress")
        return state

    def _continuity_rewrite_target(
        self, message_id: str, *, operation: str
    ) -> tuple[ChatSession, Message]:
        """Return the final assistant rewrite selected for an explicit user decision."""

        st: ChatSession = self._idle_active_state(operation)
        message: Message | None = st.chat.find_message(message_id)
        if message is None or message.role != "assistant":
            raise ValueError("continuity rewrites apply only to assistant messages")
        if not st.chat.messages or st.chat.messages[-1] is not message:
            raise ValueError(
                "continuity rewrites are available only for the last assistant message"
            )
        if not message.continuity_rewrite:
            raise ValueError("this continuity rewrite is no longer available")
        return st, message

    async def send_user_message(self, content: str) -> Message:
        """Append a user message and trigger a streaming assistant reply."""
        content = require_non_empty_text((content or "").strip(), field_name="message")

        st: ChatSession = self._idle_active_state("send")
        chat: Chat = st.chat
        is_first_message: bool = not chat.messages
        user_message: Message = chat.add_message("user", content)
        chat.draft = ""

        if is_first_message:
            chat.set_auto_title_from_first_message(content)

        await self._start_generation()
        chat_log.info(
            "message_sent chat_id=%s message_id=%s role=user content_length=%d",
            chat.id,
            user_message.id,
            len(content),
        )
        return user_message

    async def regenerate_last(self) -> None:
        """Replace the final assistant turn with one fresh response or continuation."""
        st: ChatSession = self._idle_active_state("regenerate")
        chat: Chat = st.chat
        if not chat.messages or chat.messages[-1].role != "assistant":
            raise ValueError(
                "regeneration requires the last message to be an assistant response"
            )
        removed_message: Message = chat.messages.pop()
        regenerating_continuation: bool = bool(
            chat.messages and chat.messages[-1].role == "assistant"
        )
        request_chat: Chat | None = (
            chat_with_continuation_prompt(chat) if regenerating_continuation else None
        )
        chat_log.info(
            "message_deleted chat_id=%s message_id=%s role=assistant reason=regenerate_last",
            chat.id,
            removed_message.id,
        )
        # _start_generation adds and persists the replacement assistant turn in
        # one state publication. Publishing the removal first permits an older
        # browser fragment request to restore the deleted reply visually.
        await self._start_generation(request_chat=request_chat)

    async def resend_user_message(self, message_id: str) -> Message:
        """Generate a reply to the active chat's final user message without duplicating it."""
        st: ChatSession = self._idle_active_state("resend")
        message: Message | None = st.chat.find_message(message_id)
        if (
            message is None
            or message.role != "user"
            or message is not st.chat.messages[-1]
        ):
            raise ValueError("message must be the final user message")
        _ = require_non_empty_text(message.content, field_name="message")
        await self._start_generation()
        chat_log.info(
            "message_resent chat_id=%s source_message_id=%s",
            st.chat.id,
            message.id,
        )
        return message

    async def continue_last_response(self) -> None:
        """Continue the last completed assistant response as a new message."""
        st: ChatSession = self._idle_active_state("continue")
        if (
            not st.chat.messages
            or st.chat.messages[-1].role != "assistant"
            or not st.chat.messages[-1].content.strip()
        ):
            raise ValueError("last message must be a completed assistant response")
        request_chat: Chat = chat_with_continuation_prompt(st.chat)
        assistant: Message = st.chat.add_message("assistant", "")
        await self._start_generation(
            assistant=assistant,
            request_chat=request_chat,
        )
        chat_log.info(
            "generation_continued chat_id=%s message_id=%s",
            st.chat.id,
            assistant.id,
        )

    async def edit_message(self, message_id: str, new_content: str) -> None:
        """Edit a message in place without truncating or regenerating."""
        new_content = require_non_empty_text(new_content, field_name="message")

        st: ChatSession = self._idle_active_state("edit")
        chat: Chat = st.chat
        message: Message | None = chat.update_message(message_id, new_content)
        if message is None:
            return
        await self._commit_chat(chat)
        chat_log.info(
            "message_edited chat_id=%s message_id=%s role=%s content_length=%d",
            chat.id,
            message_id,
            message.role,
            len(new_content),
        )

    async def apply_continuity_rewrite(self, message_id: str) -> None:
        """Replace one assistant response with its reviewed rewrite after confirmation."""

        st, message = self._continuity_rewrite_target(
            message_id, operation="apply continuity rewrite"
        )
        rewrite: str = message.continuity_rewrite
        message.update_content(rewrite)
        st.chat.touch()
        await self._commit_chat(st.chat)
        chat_log.info(
            "continuity_rewrite_applied chat_id=%s message_id=%s content_length=%d",
            st.chat.id,
            message.id,
            len(rewrite),
        )

    async def discard_continuity_rewrite(self, message_id: str) -> None:
        """Discard one reviewed rewrite while retaining the original response."""

        st, message = self._continuity_rewrite_target(
            message_id, operation="discard continuity rewrite"
        )
        message.discard_continuity_rewrite()
        st.chat.touch()
        await self._commit_chat(st.chat)
        chat_log.info(
            "continuity_rewrite_discarded chat_id=%s message_id=%s",
            st.chat.id,
            message.id,
        )

    async def delete_message(self, message_id: str) -> None:
        """Delete a single message from the active chat."""

        _ = await self.delete_message_with_undo(message_id)

    async def delete_message_with_undo(self, message_id: str) -> ChatMessageUndo | None:
        """Delete a message and return the exact transcript state needed to undo it."""
        st: ChatSession = self._idle_active_state("delete")
        chat: Chat = st.chat
        previous_messages = snapshot_messages(chat.messages)
        message: Message | None = chat.remove_message(message_id)
        if message is None:
            return None
        await self._commit_chat(chat)
        chat_log.info(
            "message_deleted chat_id=%s message_id=%s role=%s reason=user",
            chat.id,
            message_id,
            message.role,
        )
        return ChatMessageUndo(
            chat_id=chat.id,
            previous_messages=previous_messages,
            resulting_messages=snapshot_messages(chat.messages),
        )

    async def delete_message_and_following_with_undo(
        self, message_id: str
    ) -> ChatMessageUndo | None:
        """Delete a selected message plus every later transcript entry, with undo support."""
        st: ChatSession = self._idle_active_state("delete messages")
        chat: Chat = st.chat
        previous_messages = snapshot_messages(chat.messages)
        removed_messages: list[Message] = chat.remove_messages_from(message_id)
        if not removed_messages:
            return None
        await self._commit_chat(chat)
        chat_log.info(
            "messages_deleted chat_id=%s first_message_id=%s removed_count=%d reason=user_and_following",
            chat.id,
            message_id,
            len(removed_messages),
        )
        return ChatMessageUndo(
            chat_id=chat.id,
            previous_messages=previous_messages,
            resulting_messages=snapshot_messages(chat.messages),
        )

    async def merge_message_with_previous(self, message_id: str) -> Message:
        """Merge a message into the preceding message when both roles match."""
        st: ChatSession = self._idle_active_state("merge")
        chat: Chat = st.chat
        merged_message: Message = chat.merge_message_with_previous(message_id)
        await self._commit_chat(chat)
        chat_log.info(
            "message_merged chat_id=%s source_message_id=%s target_message_id=%s role=%s content_length=%d",
            chat.id,
            message_id,
            merged_message.id,
            merged_message.role,
            len(merged_message.content),
        )
        return merged_message

    async def truncate_chat_to_message(self, message_id: str) -> None:
        """Keep the target message and remove everything after it."""

        _ = await self.truncate_chat_to_message_with_undo(message_id)

    async def truncate_chat_to_message_with_undo(
        self, message_id: str
    ) -> ChatMessageUndo | None:
        """Truncate the active transcript and return an undo state when it changed."""
        st: ChatSession = self._idle_active_state("truncate")
        chat: Chat = st.chat
        previous_messages = snapshot_messages(chat.messages)
        removed_count: int = chat.truncate_to_message(message_id)
        if removed_count <= 0:
            return None
        await self._commit_chat(chat)
        chat_log.info(
            "chat_truncated chat_id=%s through_message_id=%s removed_count=%d",
            chat.id,
            message_id,
            removed_count,
        )
        return ChatMessageUndo(
            chat_id=chat.id,
            previous_messages=previous_messages,
            resulting_messages=snapshot_messages(chat.messages),
        )

    async def restore_message_undo(self, undo: ChatMessageUndo) -> None:
        """Restore an undo snapshot only when the active transcript still matches its result."""

        if self._chat_registry.active_chat_id != undo.chat_id:
            raise ValueError("the active chat changed before this undo action")
        st: ChatSession = self._idle_active_state("undo")
        chat: Chat = st.chat
        if snapshot_messages(chat.messages) != undo.resulting_messages:
            raise ValueError("the conversation changed before this undo action")
        chat.messages = restore_messages(undo.previous_messages)
        chat.touch()
        await self._commit_chat(chat)
        chat_log.info(
            "chat_undo_restored chat_id=%s message_count=%d",
            chat.id,
            len(chat.messages),
        )

    async def stop_generation(self) -> None:
        st: ChatSession = self._chat_registry.active
        if self._generation.request_stop(st):
            chat_log.info("generation_stop_requested chat_id=%s", st.chat.id)
            await self._notify(StateChangeKind.STATUS)
        _ = await self._cancel_generation()

    async def set_active_chat_model(self, model: str | None) -> None:
        """Persist the selected model for the active chat."""
        st: ChatSession = self._idle_active_state("set model")
        chat: Chat = st.chat
        chat.model = trimmed_or_empty(model)
        chat.touch()
        st.runtime_status = self._idle_runtime_status(chat)
        await self._commit_chat(chat)
        chat_log.info("chat_model_changed chat_id=%s model=%s", chat.id, chat.model)

    async def set_active_chat_sampling_overrides(
        self, overrides: ChatSamplingOverrides
    ) -> None:
        """Persist validated per-chat sampling values, leaving null fields inherited."""

        st: ChatSession = self._idle_active_state("set sampling overrides")
        chat: Chat = st.chat
        _ = overrides.resolve(self.config.generation)
        if chat.sampling_overrides == overrides:
            return
        chat.sampling_overrides = overrides
        chat.touch()
        await self._commit_chat(chat)
        chat_log.info(
            "chat_sampling_overrides_updated chat_id=%s temperature=%s top_p=%s max_tokens=%s",
            chat.id,
            overrides.temperature,
            overrides.top_p,
            overrides.max_tokens,
        )

    async def set_global_system_prompt(self, prompt: str) -> None:
        """Persist the global fallback system prompt to config.json."""
        await self.set_generation_defaults(
            replace(self.config.generation, system_prompt=trimmed_or_empty(prompt))
        )

    async def set_generation_defaults(self, settings: GenerationSettings) -> None:
        """Validate and atomically persist global generation defaults."""
        settings.validate()
        self.config.generation = settings
        await self._commit(config=True)
        log.info(
            "generation defaults updated max_tokens=%d temperature=%s top_p=%s",
            settings.max_tokens,
            settings.temperature,
            settings.top_p,
        )
        chat_log.info(
            "global_system_prompt_updated content_length=%d",
            len(settings.system_prompt),
        )

    async def set_global_settings(
        self,
        generation: GenerationSettings,
        server: ServerSettings,
        message_action_icon_style: MessageActionIconStyle | None = None,
        icon_colors: IconColorSettings | None = None,
    ) -> None:
        """Validate and atomically persist settings edited in the global dialog."""
        generation.validate()
        server.validate()
        if icon_colors is not None:
            icon_colors.validate()
        self.config.generation = generation
        self.config.server.apply(server)
        if message_action_icon_style is not None:
            self.config.ui.message_action_icon_style = message_action_icon_style
        if icon_colors is not None:
            self.config.ui.icon_colors = icon_colors
        for session in self._chat_registry.sessions:
            if not session.is_generating and not session.chat.model:
                session.runtime_status = self._idle_runtime_status(session.chat)
        await self._commit(config=True)
        log.info(
            "global settings updated default_model=%s base_url=%s",
            server.default_model,
            server.base_url,
        )

    async def set_access_defaults(
        self,
        *,
        default_private: bool,
        allow_localhost_without_approval: bool,
        global_settings_for_approved: bool,
        allow_network_device_reassociation: bool,
    ) -> None:
        """Persist global access-control switches."""
        await self._access.set_defaults(
            default_private=default_private,
            allow_localhost_without_approval=allow_localhost_without_approval,
            global_settings_for_approved=global_settings_for_approved,
            allow_network_device_reassociation=allow_network_device_reassociation,
        )
        log.info(
            (
                "access settings updated default_private=%s allow_localhost_without_approval=%s "
                "global_settings_for_approved=%s allow_network_device_reassociation=%s"
            ),
            default_private,
            allow_localhost_without_approval,
            global_settings_for_approved,
            allow_network_device_reassociation,
        )

    async def set_device_access_allowed(
        self, device_id: str, access_allowed: bool
    ) -> None:
        """Persist one known browser/device allow flag."""
        await self._access.set_device_access(device_id, access_allowed)
        log.info(
            "device access updated device_id=%s access_allowed=%s",
            device_id,
            access_allowed,
        )

    async def forget_pending_access_device(self, device_id: str) -> None:
        """Remove one rejected pending browser from the access configuration."""

        await self._access.forget_pending_device(device_id)
        log.info("pending device forgotten device_id=%s", device_id)

    async def register_access_device(
        self,
        device_id: str,
        label: str,
        *,
        last_ip: str = "",
        hostname: str = "",
    ) -> None:
        """Register or refresh a browser/device entry and notify admin views."""
        if not await self._access.register_device(
            device_id,
            label,
            last_ip=last_ip,
            hostname=hostname,
        ):
            return

    async def refresh_access_device_activity(
        self,
        device_id: str,
        label: str,
        *,
        last_ip: str = "",
        hostname: str = "",
        hostname_observed: bool = False,
    ) -> None:
        """Quietly persist a known browser's latest active time."""

        await self._access.refresh_device_activity(
            device_id,
            label,
            last_ip=last_ip,
            hostname=hostname,
            hostname_observed=hostname_observed,
        )

    async def set_access_device_label(self, device_id: str, label: str) -> None:
        """Persist one known browser/device label."""
        await self._access.set_device_label(device_id, label)
        log.info(
            "device label updated device_id=%s label_length=%d",
            device_id,
            len(label.strip()),
        )

    async def set_active_chat_system_prompt(self, prompt: str | None) -> None:
        """Persist the system prompt override for the active chat."""
        st: ChatSession = self._idle_active_state("set system prompt")
        chat: Chat = st.chat
        chat.system_prompt = trimmed_or_empty(prompt)
        chat.touch()
        await self._commit_chat(chat)
        chat_log.info(
            "chat_system_prompt_updated chat_id=%s content_length=%d",
            chat.id,
            len(chat.system_prompt),
        )

    async def set_active_chat_postprocess_british_spellings(
        self, enabled: bool
    ) -> None:
        """Persist whether the active chat applies British spelling replacements to output."""
        st: ChatSession = self._idle_active_state(
            "set British spelling post-processing"
        )
        chat: Chat = st.chat
        if chat.postprocess_british_spellings == enabled:
            return
        chat.postprocess_british_spellings = enabled
        chat.touch()
        await self._commit_chat(chat)
        chat_log.info(
            "chat_british_spelling_postprocess_updated chat_id=%s enabled=%s",
            chat.id,
            enabled,
        )

    async def set_active_chat_save_reasoning(self, enabled: bool) -> None:
        """Persist whether completed responses retain their transient reasoning trace."""

        st: ChatSession = self._idle_active_state("set reasoning persistence")
        chat: Chat = st.chat
        if chat.save_reasoning == enabled:
            return
        chat.save_reasoning = enabled
        chat.touch()
        await self._commit_chat(chat)
        chat_log.info(
            "chat_reasoning_persistence_updated chat_id=%s enabled=%s", chat.id, enabled
        )

    async def rename_chat(self, chat_id: str, title: str) -> None:
        """Rename a chat explicitly."""
        chat: Chat = self._by_id(chat_id).chat
        chat.set_manual_title(
            require_non_empty_text(trimmed_or_empty(title), field_name="chat title")
        )
        await self._commit_chat(chat)
        chat_log.info(
            "chat_renamed chat_id=%s title_length=%d", chat.id, len(chat.title)
        )

    async def delete_chat(self, chat_id: str) -> None:
        """Delete a chat and choose a replacement active chat if needed."""
        session: ChatSession = self._by_id(chat_id)
        cancellation_was_clean: bool = await self._cancel_generation(chat_id)
        if session.is_generating or session.generation_task is not None:
            raise RuntimeError(
                "cannot delete a chat while its generation is still stopping"
            )
        if not cancellation_was_clean:
            log.warning(
                "chat deletion followed a forced generation cancellation chat_id=%s",
                chat_id,
            )
        was_active: bool = self._chat_registry.active_chat_id == chat_id
        self._chat_registry.remove(chat_id)

        replacement: Chat | None = None
        if not self._chat_registry:
            replacement = self._create_chat_internal()
        elif was_active:
            self.config.ui.active_chat_id = self._chat_registry.select_most_recent()
        replacement_id: str = self._chat_registry.active_chat_id

        await self._commit(
            changed_chats=(replacement,) if replacement is not None else (),
            deleted_chat_ids=frozenset({chat_id}),
            config=True,
        )
        chat_log.info(
            "chat_deleted chat_id=%s replacement_chat_id=%s remaining_chats=%d",
            chat_id,
            replacement_id,
            len(self._chat_registry),
        )

    async def _start_generation(
        self,
        *,
        assistant: Message | None = None,
        request_chat: Chat | None = None,
    ) -> None:
        await self._generation.start(
            self._chat_registry.active, assistant=assistant, request_chat=request_chat
        )

    async def _cancel_generation(self, chat_id: str | None = None) -> bool:
        return await self._generation.cancel(
            self._by_id(chat_id or self._chat_registry.active_chat_id)
        )

    async def _save(
        self,
        changed_chats: tuple[Chat, ...],
        *,
        deleted_chat_ids: frozenset[str],
        flush: bool,
    ) -> None:
        """Queue only changed chat documents, retaining any never-persisted new chat."""

        changes: dict[str, Chat] = self._chat_registry.changes_for_save(
            changed_chats,
            deleted_chat_ids=deleted_chat_ids,
        )
        if not changes and not deleted_chat_ids:
            return

        self._persistence.queue_chat_changes(
            changes.values(),
            deleted_chat_ids=deleted_chat_ids,
        )
        if flush:
            await self._persistence.flush_chats()
            self._chat_registry.mark_persisted(changes)

    async def _commit_chat(
        self,
        chat: Chat,
        *,
        config: bool = False,
        flush_chats: bool = True,
        notify: bool = True,
    ) -> None:
        """Persist one changed chat without serialising every other chat."""

        await self._commit(
            changed_chats=(chat,),
            config=config,
            flush_chats=flush_chats,
            notify=notify,
        )

    async def _commit_config(self) -> None:
        await self._commit(config=True)

    async def _commit_access_activity(self) -> None:
        """Save a heartbeat without prompting every connected browser to refresh."""

        await self._commit(config=True, notify=False)

    async def _commit(
        self,
        *,
        changed_chats: tuple[Chat, ...] | None = None,
        deleted_chat_ids: frozenset[str] = NO_DELETED_CHAT_IDS,
        config: bool = False,
        flush_chats: bool = True,
        notify: bool = True,
    ) -> None:
        """Persist changed state once, then notify UI listeners once."""
        if changed_chats is not None or deleted_chat_ids:
            await self._save(
                changed_chats or (),
                deleted_chat_ids=deleted_chat_ids,
                flush=flush_chats,
            )
        if config:
            await self._persistence.save_config()
        if notify:
            await self._notify()

    async def shutdown(self) -> bool:
        """Stop background work and flush state within the shutdown deadline."""
        try:
            return await asyncio.wait_for(
                self._shutdown(), timeout=_SHUTDOWN_TIMEOUT_SECONDS
            )
        except TimeoutError:
            log.error(
                "application state shutdown exceeded timeout_seconds=%s",
                _SHUTDOWN_TIMEOUT_SECONDS,
            )
            return False
        except Exception:
            log.exception("application state shutdown failed")
            return False

    async def _shutdown(self) -> bool:
        await self._host_stats.stop()
        await self._runtime.shutdown()
        generation_stopped_cleanly: bool = await self._cancel_all_generations()
        await self._commit(changed_chats=(), notify=False)
        await self.client.close()
        return generation_stopped_cleanly

    async def _cancel_all_generations(self) -> bool:
        """Stop every live chat task concurrently before closing the client."""

        return await self._generation.cancel_all(self._chat_registry.sessions)
