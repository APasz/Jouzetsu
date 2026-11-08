"""Shared application services; chat and configuration commands belong to the runtime."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from dataclasses import replace
from functools import wraps
from typing import Concatenate, cast

from . import access
from .async_workers import complete_before_cancelling, run_in_worker
from .backend_models import BackendConfig
from .backend_protocol import InferenceBackend
from .character_storage import CharacterStorage
from .config import (
    AppConfig,
    ConfigStore,
    GenerationSettings,
    IconColorSettings,
    MessageActionStyle,
    ThemeColorway,
    ThemeSettings,
    default_config,
)
from .controllers import PersistenceController
from .empty_chat_messages import EmptyChatMessageProvider
from .events import StateChangeKind
from .image_storage import ImageStorage
from .inference import ChangeKind, InferenceRuntime, RuntimeChange
from .models import (
    Character,
    CharacterField,
    CharacterName,
    CharacterPresetSelection,
    Chat,
    ChatPromptMode,
    ChatTitleSource,
    ImageAttachment,
    character_cast_chat_title,
    compiled_character_chat_system_prompt,
)
from .runtime import RuntimePhase, RuntimeStatus
from .state_events import ChangeListener, StateNotifier
from .state_registry import ChatRegistry
from .state_types import ChatSession
from .storage import ChatStorage

log = logging.getLogger(__name__)


def _state_mutation[**P, T](
    method: Callable[Concatenate[AppState, P], Awaitable[T]],
) -> Callable[Concatenate[AppState, P], Coroutine[object, object, T]]:
    """Serialize character commands through durable publication."""

    @wraps(method)
    async def execute(self: AppState, *args: P.args, **kwargs: P.kwargs) -> T:
        async def locked() -> T:
            async with self._character_lock:  # pyright: ignore[reportPrivateUsage]
                if self._closing:  # pyright: ignore[reportPrivateUsage]
                    raise RuntimeError("application is shutting down")
                return await method(self, *args, **kwargs)

        return await complete_before_cancelling(locked())

    return execute


class AppState:
    def __init__(
        self,
        config: AppConfig,
        storage: ChatStorage,
        character_storage: CharacterStorage | None = None,
        config_store: ConfigStore | None = None,
        image_storage: ImageStorage | None = None,
        *,
        backend_factory: Callable[[BackendConfig], InferenceBackend] | None = None,
    ) -> None:
        self.config = config
        self.storage = storage
        self.character_storage = character_storage or CharacterStorage(
            config.characters_directory
        )
        self.images = image_storage or ImageStorage(config.paths.images_directory)
        self._character_lock = asyncio.Lock()
        self._closing = False
        self._notifier = StateNotifier()
        empty_messages = EmptyChatMessageProvider(config.data_dir)
        self.registry = ChatRegistry(empty_messages.choose)
        self._characters = {
            character.id: character for character in self.character_storage.load_all()
        }
        for chat in storage.load_all():
            self.registry.add(chat, runtime_status=RuntimeStatus(RuntimePhase.READY))
        self.persistence = PersistenceController(
            config, config_store or ConfigStore.for_paths(config.paths), storage
        )
        self.inference = InferenceRuntime(
            self.registry,
            self.persistence,
            backend_factory=backend_factory,
            notify=self._runtime_changed,
        )

    async def _runtime_changed(self, change: RuntimeChange) -> None:
        session = (
            self.session(change.chat_id)
            if change.chat_id and change.chat_id in self.registry
            else None
        )
        kind = (
            StateChangeKind.STREAM
            if change.kind is ChangeKind.STATUS
            and session is not None
            and session.is_generating
            and session.live_response is not None
            and session.latest_error is None
            else StateChangeKind.FULL
        )
        if change.kind is ChangeKind.CONFIG:
            kind = StateChangeKind.CONFIG
        elif change.kind is ChangeKind.DRAFT:
            kind = StateChangeKind.DRAFT
        await self._notify(kind)

    def session(self, chat_id: str) -> ChatSession:
        return self.registry.get(chat_id)

    async def new_chat(self) -> Chat:
        return await self.inference.create_chat()

    async def start(self) -> None:
        await self.inference.start()

    async def shutdown(self) -> bool:
        self._closing = True
        characters_drained = False
        try:
            await asyncio.wait_for(self._character_lock.acquire(), timeout=10.0)
        except TimeoutError:
            log.error("character persistence did not drain before shutdown")
        else:
            self._character_lock.release()
            characters_drained = True
        return await self.inference.shutdown() and characters_drained

    def add_listener(self, listener: ChangeListener) -> Callable[[], None]:
        """Register a listener and return an idempotent removal callback."""
        return self._notifier.add(listener)

    def remove_listener(self, listener: ChangeListener) -> None:
        """Remove a previously registered listener if it is still present."""
        self._notifier.remove(listener)

    async def _notify(self, kind: StateChangeKind = StateChangeKind.FULL) -> None:
        await self._notifier.publish(kind)

    @property
    def chats(self) -> list[Chat]:
        """Return chats sorted by most recent activity first."""
        return self.registry.chats_by_activity

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
    def global_system_prompt(self) -> str:
        return self.config.generation.system_prompt

    @_state_mutation
    async def new_character(self) -> Character:
        """Create a blank flexible profile without imposing a fixed character schema."""

        character: Character = Character.draft()
        await run_in_worker(self.character_storage.save, character)
        self._characters[character.id] = character
        await self._notify(StateChangeKind.CHARACTERS)
        log.info("character_created character_id=%s", character.id)
        return character

    @_state_mutation
    async def update_character(
        self,
        character_id: str,
        *,
        expected_revision: int,
        name: CharacterName | None,
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
        await run_in_worker(self.character_storage.save, updated_character)
        self._characters[character_id] = updated_character
        await self._notify(StateChangeKind.CHARACTERS)
        log.info(
            "character_updated character_id=%s revision=%d field_count=%d",
            updated_character.id,
            updated_character.revision,
            len(updated_character.fields),
        )
        return updated_character

    @_state_mutation
    async def delete_character(self, character_id: str) -> None:
        """Delete one profile while retaining all chat prompt snapshots that used it."""

        character: Character = self.character(character_id)
        await run_in_worker(self.character_storage.delete, character.id)
        del self._characters[character.id]
        await self._notify(StateChangeKind.CHARACTERS)
        log.info("character_deleted character_id=%s", character.id)

    async def start_chat_from_character(
        self,
        character_id: str,
        *,
        prompt_mode: ChatPromptMode = ChatPromptMode.ROLEPLAY,
        custom_instruction: str = "",
        story_direction: str = "",
    ) -> Chat:
        """Start a single-character conversation through the cast-based workflow."""

        return await self.start_chat_from_characters(
            (character_id,),
            prompt_mode=prompt_mode,
            custom_instruction=custom_instruction,
            story_direction=story_direction,
        )

    @_state_mutation
    async def start_chat_from_characters(
        self,
        character_ids: Sequence[str],
        *,
        prompt_mode: ChatPromptMode = ChatPromptMode.ROLEPLAY,
        custom_instruction: str = "",
        story_direction: str = "",
    ) -> Chat:
        """Start a mode-specific chat from immutable snapshots of an ordered cast."""

        characters: tuple[Character, ...] = self._characters_for_cast(character_ids)
        chat: Chat = Chat(
            title=character_cast_chat_title(characters),
            title_source=ChatTitleSource.MANUAL,
            system_prompt=compiled_character_chat_system_prompt(
                characters,
                mode=prompt_mode,
                custom_instruction=custom_instruction,
                story_direction=story_direction,
            ),
            prompt_mode=prompt_mode,
            character_cast=tuple(character.chat_binding() for character in characters),
        )
        chat = await self.inference.create_chat(chat)
        return chat

    def _characters_for_cast(
        self, character_ids: Sequence[str]
    ) -> tuple[Character, ...]:
        """Resolve a non-empty, duplicate-free ordered set of current profiles."""

        normalized_ids: list[str] = []
        for candidate_id in character_ids:
            raw_character_id: object = cast(object, candidate_id)
            if not isinstance(raw_character_id, str):
                raise TypeError("character cast ids must be text")
            character_id: str = raw_character_id.strip()
            if not character_id:
                raise ValueError("character cast ids cannot be empty")
            normalized_ids.append(character_id)
        if not normalized_ids:
            raise ValueError("character cast must contain at least one character")
        if len(normalized_ids) != len(set(normalized_ids)):
            raise ValueError("character cast cannot contain duplicate characters")
        return tuple(self.character(character_id) for character_id in normalized_ids)

    def is_generating(self, chat_id: str) -> bool:
        return self.session(chat_id).is_generating

    def image_attachment(self, image_id: str) -> ImageAttachment:
        for session in self.registry.sessions:
            for image in (
                *session.chat.draft_images,
                *(
                    image
                    for message in session.chat.messages
                    for image in message.images
                ),
            ):
                if image.id == image_id:
                    return image
        raise ValueError("unknown image")

    async def set_global_system_prompt(self, prompt: str) -> None:
        await self.inference.update_settings(
            lambda config: setattr(config.generation, "system_prompt", prompt.strip())
        )

    async def reset_global_system_prompt(self) -> None:
        await self.set_global_system_prompt(
            default_config(self.config.paths).generation.system_prompt
        )

    async def set_generation_defaults(self, settings: GenerationSettings) -> None:
        await self.inference.update_settings(
            lambda config: setattr(config, "generation", settings)
        )

    async def reset_generation_defaults(self) -> None:
        defaults = default_config(self.config.paths).generation

        def edit(config: AppConfig) -> None:
            config.generation = replace(
                defaults, system_prompt=config.generation.system_prompt
            )

        await self.inference.update_settings(edit)

    async def set_appearance_settings(
        self,
        theme: ThemeSettings,
        icon_colors: IconColorSettings,
        *,
        dark_mode: bool,
        message_action_style: MessageActionStyle,
    ) -> None:
        def edit(config: AppConfig) -> None:
            config.theme = theme
            config.ui = replace(
                config.ui,
                dark_mode=dark_mode,
                message_action_style=message_action_style,
                icon_colors=icon_colors,
            )

        await self.inference.update_settings(edit)

    async def reset_appearance_settings(self) -> None:
        defaults = default_config(self.config.paths)
        await self.set_appearance_settings(
            defaults.theme,
            defaults.ui.icon_colors,
            dark_mode=defaults.ui.dark_mode,
            message_action_style=defaults.ui.message_action_style,
        )

    async def reset_theme_colorway(self, colorway: ThemeColorway) -> None:
        def edit(config: AppConfig) -> None:
            defaults = default_config(config.paths).theme
            config.theme = replace(
                config.theme, **{colorway.value: getattr(defaults, colorway.value)}
            )

        await self.inference.update_settings(edit)

    async def register_access_device(
        self,
        device_id: str,
        label: str,
        *,
        last_ip: str = "",
        hostname: str = "",
        approved: bool = False,
    ) -> None:
        def edit(config: AppConfig) -> None:
            access.register_seen_device(
                config,
                device_id=device_id,
                label=label,
                last_ip=last_ip,
                hostname=hostname,
            )
            if approved:
                access.set_device_label(config, device_id=device_id, label=label)
                access.set_device_access(
                    config, device_id=device_id, access_allowed=True
                )

        await self.inference.update_settings(edit, notify=approved)

    async def refresh_access_device_activity(
        self,
        device_id: str,
        label: str,
        *,
        last_ip: str = "",
        hostname: str = "",
        hostname_observed: bool = False,
    ) -> None:
        def edit(config: AppConfig) -> None:
            access.refresh_seen_device(
                config,
                device_id=device_id,
                label=label,
                last_ip=last_ip,
                hostname=hostname,
                hostname_observed=hostname_observed,
            )

        await self.inference.update_settings(edit, notify=False)

    async def set_access_device_label(self, device_id: str, label: str) -> None:
        await self.inference.update_settings(
            lambda config: access.set_device_label(
                config, device_id=device_id, label=label
            )
        )

    async def set_device_access_allowed(
        self, device_id: str, access_allowed: bool
    ) -> None:
        await self.inference.update_settings(
            lambda config: access.set_device_access(
                config, device_id=device_id, access_allowed=access_allowed
            )
        )

    async def forget_pending_access_device(self, device_id: str) -> None:
        await self.inference.update_settings(
            lambda config: access.forget_pending_device(config, device_id=device_id)
        )

    async def set_approval_phrase(self, phrase: str) -> None:
        await self.inference.update_settings(
            lambda config: setattr(config.access, "approval_phrase", phrase)
        )
