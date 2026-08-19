"""Domain models for Jouzetsu"""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Literal, TypeAlias, TypedDict, cast

from .character_presets import CharacterFieldKind
from .config import GenerationSettings

Role: TypeAlias = Literal["system", "user", "assistant"]


class ChatTitleSource(str, Enum):
    """Whether a chat title may be generated from its first user message."""

    AUTO = "auto"
    MANUAL = "manual"


class MessageJSON(TypedDict):
    id: str
    role: Role
    content: str
    model: str
    continuity_rewrite: str
    reasoning: str
    created_at: float
    updated_at: float


class ChatSamplingOverridesJSON(TypedDict):
    """Nullable sampling values persisted as overrides to global defaults."""

    temperature: float | None
    top_p: float | None
    max_tokens: int | None


class CharacterChatBindingJSON(TypedDict):
    """The character revision whose compiled prompt was copied into a chat."""

    id: str
    name: str
    revision: int


class ChatJSON(TypedDict):
    id: str
    title: str
    title_source: ChatTitleSource
    created_at: float
    updated_at: float
    model: str
    system_prompt: str
    sampling_overrides: ChatSamplingOverridesJSON
    postprocess_british_spellings: bool
    save_reasoning: bool
    draft: str
    character: CharacterChatBindingJSON | None
    messages: list[MessageJSON]


class CharacterFieldJSON(TypedDict):
    """One user-defined profile field and its presentation metadata."""

    id: str
    label: str
    kind: CharacterFieldKind
    value: str


class CharacterJSON(TypedDict):
    """The complete standalone document persisted for one character."""

    id: str
    name: str
    revision: int
    created_at: float
    updated_at: float
    fields: list[CharacterFieldJSON]


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _required_string(raw: object, *, field_name: str) -> str:
    """Read a non-empty string from a persisted document."""

    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    return raw.strip()


def _integer_from_json(raw: object, *, field_name: str, minimum: int) -> int:
    """Read an integer above a defined lower bound from persisted data."""

    if isinstance(raw, bool) or not isinstance(raw, int) or raw < minimum:
        raise ValueError(f"{field_name} must be an integer greater than or equal to {minimum}")
    return raw


def _role_from_json(raw: object) -> Role:
    if raw not in {"system", "user", "assistant"}:
        raise ValueError(f"invalid message role: {raw!r}")
    return cast(Role, raw)


def _float_from_json(raw: object, *, default: float) -> float:
    if raw is None:
        return default
    if isinstance(raw, bool) or not isinstance(raw, int | float | str):
        raise ValueError(f"expected a number, got {raw!r}")
    try:
        value: float = float(raw)
    except ValueError as exc:
        raise ValueError(f"expected a number, got {raw!r}") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"expected a non-negative finite number, got {raw!r}")
    return value


def _bool_from_json(raw: object, *, default: bool) -> bool:
    if raw is None:
        return default
    if not isinstance(raw, bool):
        raise ValueError(f"expected a boolean, got {raw!r}")
    return raw


def _optional_float_from_json(raw: object, *, field_name: str) -> float | None:
    """Read a nullable finite number from a persisted chat field."""

    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise ValueError(f"{field_name} must be a number or null")
    value: float = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


def _optional_integer_from_json(raw: object, *, field_name: str) -> int | None:
    """Read a nullable integer from a persisted chat field."""

    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{field_name} must be an integer or null")
    return raw


def _string_mapping(raw: object, *, field_name: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError(f"{field_name} must be an object")
    mapping: dict[object, object] = cast(dict[object, object], raw)
    if not all(isinstance(key, str) for key in mapping):
        raise ValueError(f"{field_name} must use string keys")
    return {cast(str, key): value for key, value in mapping.items()}


def _merge_message_content(previous_content: str, current_content: str) -> str:
    previous: str = previous_content.rstrip()
    current: str = current_content.lstrip()
    if not previous:
        return current
    if not current:
        return previous
    return f"{previous}\n\n{current}"


def _title_source_from_json(raw: object) -> ChatTitleSource:
    """Read and validate an explicitly persisted chat title source."""

    if not isinstance(raw, str):
        raise ValueError("chat.title_source must be text")
    try:
        return ChatTitleSource(raw)
    except ValueError as exc:
        raise ValueError(f"invalid chat.title_source: {raw!r}") from exc


def _is_legacy_new_chat_placeholder(messages: list[Message]) -> bool:
    """Identify the empty user turn created by versions before pristine chats."""

    return len(messages) == 1 and messages[0].role == "user" and not messages[0].content


@dataclass(frozen=True, slots=True)
class ChatSamplingOverrides:
    """Optional per-chat sampling values that supersede global generation defaults."""

    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None

    def resolve(self, defaults: GenerationSettings) -> GenerationSettings:
        """Return validated effective generation settings for this chat."""

        settings: GenerationSettings = replace(
            defaults,
            temperature=defaults.temperature if self.temperature is None else self.temperature,
            top_p=defaults.top_p if self.top_p is None else self.top_p,
            max_tokens=defaults.max_tokens if self.max_tokens is None else self.max_tokens,
        )
        settings.validate()
        return settings

    def to_dict(self) -> ChatSamplingOverridesJSON:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ChatSamplingOverrides":
        """Load and validate optional sampling values from persisted chat data."""

        overrides: ChatSamplingOverrides = cls(
            temperature=_optional_float_from_json(raw.get("temperature"), field_name="chat.sampling_overrides.temperature"),
            top_p=_optional_float_from_json(raw.get("top_p"), field_name="chat.sampling_overrides.top_p"),
            max_tokens=_optional_integer_from_json(raw.get("max_tokens"), field_name="chat.sampling_overrides.max_tokens"),
        )
        _ = overrides.resolve(GenerationSettings())
        return overrides


@dataclass(frozen=True, slots=True)
class CharacterChatBinding:
    """Identity of the immutable character revision used to start a chat."""

    id: str
    name: str
    revision: int

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("character binding id cannot be empty")
        if not self.name.strip():
            raise ValueError("character binding name cannot be empty")
        if self.revision < 1:
            raise ValueError("character binding revision must be positive")

    def to_dict(self) -> CharacterChatBindingJSON:
        return {"id": self.id, "name": self.name, "revision": self.revision}

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "CharacterChatBinding":
        return cls(
            id=_required_string(raw.get("id"), field_name="chat.character.id"),
            name=_required_string(raw.get("name"), field_name="chat.character.name"),
            revision=_integer_from_json(raw.get("revision"), field_name="chat.character.revision", minimum=1),
        )


@dataclass(frozen=True, slots=True)
class CharacterField:
    """A named profile value, including the widget used to edit it."""

    id: str = field(default_factory=_new_id)
    label: str = ""
    kind: CharacterFieldKind = "short_text"
    value: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("character field id cannot be empty")
        if not self.label.strip():
            raise ValueError("character field label cannot be empty")
        if self.kind not in {"short_text", "long_text"}:
            raise ValueError(f"invalid character field kind: {self.kind!r}")

    def to_dict(self) -> CharacterFieldJSON:
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "CharacterField":
        raw_kind: object = raw.get("kind", "short_text")
        if raw_kind not in {"short_text", "long_text"}:
            raise ValueError(f"invalid character field kind: {raw_kind!r}")
        raw_value: object = raw.get("value", "")
        if not isinstance(raw_value, str):
            raise ValueError("character field value must be text")
        return cls(
            id=_required_string(raw.get("id"), field_name="character.fields.id"),
            label=_required_string(raw.get("label"), field_name="character.fields.label"),
            kind=cast(CharacterFieldKind, raw_kind),
            value=raw_value,
        )


@dataclass
class Character:
    """A reusable, field-configured character profile stored as one document."""

    id: str = field(default_factory=_new_id)
    name: str = "New character"
    revision: int = 1
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    fields: list[CharacterField] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not self.id:
            raise ValueError("character id cannot be empty")
        if not self.name.strip():
            raise ValueError("character name cannot be empty")
        if self.revision < 1:
            raise ValueError("character revision must be positive")
        field_ids: list[str] = [profile_field.id for profile_field in self.fields]
        if len(field_ids) != len(set(field_ids)):
            raise ValueError("character field ids must be unique")

    def revised_profile(self, name: str, fields: list[CharacterField]) -> "Character":
        """Return the next profile revision without changing this persisted version."""

        next_name: str = name.strip()
        if not next_name:
            raise ValueError("character name cannot be empty")
        next_fields: list[CharacterField] = list(fields)
        field_ids: list[str] = [profile_field.id for profile_field in next_fields]
        if len(field_ids) != len(set(field_ids)):
            raise ValueError("character field ids must be unique")
        if self.name == next_name and self.fields == next_fields:
            return self
        return Character(
            id=self.id,
            name=next_name,
            revision=self.revision + 1,
            created_at=self.created_at,
            updated_at=time.time(),
            fields=next_fields,
        )

    def compiled_system_prompt(self) -> str:
        """Compile the character's current structured profile for one chat snapshot."""

        populated_fields: list[CharacterField] = [profile_field for profile_field in self.fields if profile_field.value.strip()]
        profile_lines: list[str] = [
            f"{profile_field.label.strip()}: {profile_field.value.strip()}" for profile_field in populated_fields
        ]
        profile: str = "\n".join(profile_lines) if profile_lines else "No additional profile details are defined."
        return (
            f"You are roleplaying as {self.name.strip()}.\n\n"
            f"Character profile:\n{profile}\n\n"
            "Stay consistent with this profile while responding naturally to the user."
        )

    def to_dict(self) -> CharacterJSON:
        return {
            "id": self.id,
            "name": self.name,
            "revision": self.revision,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "fields": [profile_field.to_dict() for profile_field in self.fields],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "Character":
        now: float = time.time()
        raw_fields: object = raw.get("fields", [])
        if not isinstance(raw_fields, list):
            raise ValueError("character.fields must be a list")
        fields: list[CharacterField] = [
            CharacterField.from_dict(_string_mapping(item, field_name="character.fields item"))
            for item in cast(list[object], raw_fields)
        ]
        return cls(
            id=_required_string(raw.get("id"), field_name="character.id"),
            name=_required_string(raw.get("name"), field_name="character.name"),
            revision=_integer_from_json(raw.get("revision", 1), field_name="character.revision", minimum=1),
            created_at=_float_from_json(raw.get("created_at"), default=now),
            updated_at=_float_from_json(raw.get("updated_at"), default=now),
            fields=fields,
        )


@dataclass
class Message:
    """A single message in a chat.

    `id` is a stable, short identifier so the UI can target a specific
    message for edits, copy, or regeneration without relying on its
    position in the list.
    """

    id: str = field(default_factory=_new_id)
    role: Role = "user"
    content: str = ""
    model: str = ""
    continuity_rewrite: str = ""
    reasoning: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.model and self.role != "assistant":
            raise ValueError("only assistant messages can have models")
        if self.continuity_rewrite and self.role != "assistant":
            raise ValueError("only assistant messages can have continuity rewrites")
        if self.reasoning and self.role != "assistant":
            raise ValueError("only assistant messages can have reasoning")

    def update_content(self, content: str) -> None:
        """Replace content, clearing stale reasoning and review metadata."""
        self.content = content
        self.continuity_rewrite = ""
        self.reasoning = ""
        self.updated_at = time.time()

    def set_model(self, model: str) -> None:
        """Record the model that generated this assistant response."""

        if self.role != "assistant":
            raise ValueError("only assistant messages can have models")
        model_key: str = model.strip()
        if not model_key:
            raise ValueError("assistant model cannot be empty")
        self.model = model_key
        self.updated_at = time.time()

    def clear_model(self) -> None:
        """Clear model provenance when one message no longer has one source model."""

        if self.role != "assistant":
            raise ValueError("only assistant messages can have models")
        if not self.model:
            return
        self.model = ""
        self.updated_at = time.time()

    def set_reasoning(self, reasoning: str) -> None:
        """Attach the transient trace that produced one assistant response."""

        if self.role != "assistant":
            raise ValueError("only assistant messages can have reasoning")
        self.reasoning = reasoning
        self.updated_at = time.time()

    def set_continuity_rewrite(self, rewrite: str) -> None:
        """Attach a non-empty proposed replacement to an assistant response."""

        if self.role != "assistant":
            raise ValueError("only assistant messages can have continuity rewrites")
        if not rewrite.strip():
            raise ValueError("continuity rewrite cannot be empty")
        self.continuity_rewrite = rewrite
        self.updated_at = time.time()

    def discard_continuity_rewrite(self) -> None:
        """Remove a proposed replacement without changing the response text."""

        if not self.continuity_rewrite:
            return
        self.continuity_rewrite = ""
        self.updated_at = time.time()

    def to_dict(self) -> MessageJSON:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "model": self.model,
            "continuity_rewrite": self.continuity_rewrite,
            "reasoning": self.reasoning,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "Message":
        now: float = time.time()
        created_at: float = _float_from_json(raw.get("created_at"), default=now)
        role: Role = _role_from_json(raw.get("role", "user"))
        model: object = raw.get("model", "")
        continuity_rewrite: object = raw.get("continuity_rewrite", "")
        reasoning: object = raw.get("reasoning", "")
        if not isinstance(model, str):
            raise ValueError("message.model must be text")
        if not isinstance(continuity_rewrite, str):
            raise ValueError("message.continuity_rewrite must be text")
        if not isinstance(reasoning, str):
            raise ValueError("message.reasoning must be text")
        if continuity_rewrite and role != "assistant":
            raise ValueError("only assistant messages can have continuity rewrites")
        if reasoning and role != "assistant":
            raise ValueError("only assistant messages can have reasoning")
        if model and role != "assistant":
            raise ValueError("only assistant messages can have models")
        return cls(
            id=str(raw.get("id") or _new_id()),
            role=role,
            content=str(raw.get("content", "")),
            model=model,
            continuity_rewrite=continuity_rewrite,
            reasoning=reasoning,
            created_at=created_at,
            updated_at=_float_from_json(raw.get("updated_at"), default=created_at),
        )


@dataclass
class Chat:
    """A single conversation thread.

    `messages` is ordered chronologically and may be empty for a pristine
    chat. The system prompt is maintained separately and resolved at request
    time when this chat does not define an override.
    """

    id: str = field(default_factory=_new_id)
    title: str = "New chat"
    title_source: ChatTitleSource = ChatTitleSource.AUTO
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    model: str = ""
    system_prompt: str = ""
    sampling_overrides: ChatSamplingOverrides = field(default_factory=ChatSamplingOverrides)
    postprocess_british_spellings: bool = True
    save_reasoning: bool = True
    draft: str = ""
    character: CharacterChatBinding | None = None
    messages: list[Message] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("chat id cannot be empty")
        if not self.title.strip():
            raise ValueError("chat title cannot be empty")

    def touch(self) -> None:
        self.updated_at = time.time()

    def add_message(self, role: Role, content: str = "") -> Message:
        msg: Message = Message(role=role, content=content)
        self.messages.append(msg)
        self.touch()
        return msg

    def set_manual_title(self, title: str) -> None:
        """Set a user-chosen title that future first-message sends must retain."""

        if not title.strip():
            raise ValueError("chat title cannot be empty")
        self.title = title
        self.title_source = ChatTitleSource.MANUAL
        self.touch()

    def set_auto_title_from_first_message(self, content: str) -> None:
        """Generate the title only while this chat still uses an automatic title."""

        if self.title_source is not ChatTitleSource.AUTO:
            return
        self.title = auto_title_from(content)
        self.touch()

    def update_message(self, message_id: str, content: str) -> Message | None:
        """Update a message and return it, or None when the id is unknown."""
        message: Message | None = self.find_message(message_id)
        if message is None:
            return None
        message.update_content(content)
        self.touch()
        return message

    def find_message(self, message_id: str) -> Message | None:
        for m in self.messages:
            if m.id == message_id:
                return m
        return None

    def remove_message(self, message_id: str) -> Message | None:
        """Remove and return one message, or None when the id is unknown."""
        for index, message in enumerate[Message](self.messages):
            if message.id != message_id:
                continue
            del self.messages[index]
            self.touch()
            return message
        return None

    def merge_message_with_previous(self, message_id: str) -> Message:
        """Merge a message into its immediate predecessor and remove it."""
        target_index: int | None = next(
            (index for index, message in enumerate[Message](self.messages) if message.id == message_id),
            None,
        )
        if target_index is None:
            raise ValueError(f"unknown message id: {message_id}")
        if target_index == 0:
            raise ValueError("message has no previous message")

        previous_message: Message = self.messages[target_index - 1]
        target_message: Message = self.messages[target_index]
        if previous_message.role != target_message.role:
            raise ValueError("previous message must have the same role")

        models_match: bool = previous_message.model == target_message.model
        previous_message.update_content(_merge_message_content(previous_message.content, target_message.content))
        if previous_message.role == "assistant" and not models_match:
            previous_message.clear_model()
        del self.messages[target_index]
        self.touch()
        return previous_message

    def fork_through(self, message_id: str) -> "Chat":
        """Create an independent chat containing messages through the target."""
        target_index: int | None = next(
            (index for index, message in enumerate[Message](self.messages) if message.id == message_id),
            None,
        )
        if target_index is None:
            raise ValueError(f"unknown message id: {message_id}")

        copied_messages: list[Message] = [
            Message(
                role=message.role,
                content=message.content,
                model=message.model,
                continuity_rewrite=message.continuity_rewrite,
                reasoning=message.reasoning,
                created_at=message.created_at,
                updated_at=message.updated_at,
            )
            for message in self.messages[: target_index + 1]
        ]
        return Chat(
            title=self.title,
            title_source=self.title_source,
            model=self.model,
            system_prompt=self.system_prompt,
            sampling_overrides=self.sampling_overrides,
            postprocess_british_spellings=self.postprocess_british_spellings,
            save_reasoning=self.save_reasoning,
            character=self.character,
            messages=copied_messages,
        )

    def truncate_to_message(self, message_id: str) -> int:
        """Keep messages up to and including `message_id`.

        Returns the number of messages removed after the target.
        """
        for index, message in enumerate[Message](self.messages):
            if message.id != message_id:
                continue
            removed: int = len(self.messages) - (index + 1)
            del self.messages[index + 1 :]
            self.touch()
            return max(removed, 0)
        return 0

    def remove_messages_from(self, message_id: str) -> list[Message]:
        """Remove a message and every later message, preserving prior history."""
        for index, message in enumerate[Message](self.messages):
            if message.id != message_id:
                continue
            removed: list[Message] = self.messages[index:]
            del self.messages[index:]
            self.touch()
            return removed
        return []

    def to_dict(self) -> ChatJSON:
        return {
            "id": self.id,
            "title": self.title,
            "title_source": self.title_source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "sampling_overrides": self.sampling_overrides.to_dict(),
            "postprocess_british_spellings": self.postprocess_british_spellings,
            "save_reasoning": self.save_reasoning,
            "draft": self.draft,
            "character": self.character.to_dict() if self.character is not None else None,
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "Chat":
        now: float = time.time()
        raw_messages: object = raw.get("messages", [])
        if not isinstance(raw_messages, list):
            raise ValueError("chat.messages must be a list")
        messages: list[Message] = [
            Message.from_dict(_string_mapping(item, field_name="chat.messages item"))
            for item in cast(list[object], raw_messages)
        ]
        if _is_legacy_new_chat_placeholder(messages):
            messages = []
        raw_sampling_overrides: object = raw.get("sampling_overrides")
        sampling_overrides: ChatSamplingOverrides = (
            ChatSamplingOverrides()
            if raw_sampling_overrides is None
            else ChatSamplingOverrides.from_dict(
                _string_mapping(raw_sampling_overrides, field_name="chat.sampling_overrides")
            )
        )
        raw_character: object = raw.get("character")
        character: CharacterChatBinding | None = (
            None
            if raw_character is None
            else CharacterChatBinding.from_dict(_string_mapping(raw_character, field_name="chat.character"))
        )
        title: str = str(raw.get("title", "New chat"))
        title_source: ChatTitleSource = (
            _title_source_from_json(raw["title_source"])
            if "title_source" in raw
            else (ChatTitleSource.AUTO if title == "New chat" else ChatTitleSource.MANUAL)
        )
        return cls(
            id=str(raw.get("id") or _new_id()),
            title=title,
            title_source=title_source,
            created_at=_float_from_json(raw.get("created_at"), default=now),
            updated_at=_float_from_json(raw.get("updated_at"), default=now),
            model=str(raw.get("model", "")),
            system_prompt=str(raw.get("system_prompt", "")),
            sampling_overrides=sampling_overrides,
            postprocess_british_spellings=_bool_from_json(
                raw.get("postprocess_british_spellings"),
                default=True,
            ),
            save_reasoning=_bool_from_json(raw.get("save_reasoning"), default=True),
            draft=str(raw.get("draft", "")),
            character=character,
            messages=messages,
        )


def auto_title_from(text: str, max_len: int = 48) -> str:
    """Derive a short chat title from the first user message."""
    string: str = (text or "").strip().replace("\n", " ")
    if not string:
        return "New chat"
    if len(string) > max_len:
        string = string[: max_len - 1].rstrip() + "…"
    return string
