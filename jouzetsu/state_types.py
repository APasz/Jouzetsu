"""Internal state data structures shared by the application controllers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from .models import Message, Role
from .runtime import RuntimePhase, RuntimeStatus

if TYPE_CHECKING:
    from .models import Chat


@dataclass(slots=True)
class ChatSession:
    """Mutable runtime state associated with one persisted chat."""

    chat: Chat
    is_generating: bool = False
    cancellation_requested: asyncio.Event = field(default_factory=asyncio.Event)
    generation_task: asyncio.Task[None] | None = None
    runtime_status: RuntimeStatus = field(
        default_factory=lambda: RuntimeStatus(RuntimePhase.CHECKING)
    )
    live_reasoning: str = ""
    empty_state_message: str = ""


@dataclass(frozen=True, slots=True)
class MessageSnapshot:
    """Immutable message content used to validate and restore one undo action."""

    id: str
    role: Role
    content: str
    model: str
    continuity_rewrite: str
    reasoning: str
    created_at: float
    updated_at: float


@dataclass(frozen=True, slots=True)
class ChatMessageUndo:
    """One restorable transcript mutation, valid only while its result is unchanged."""

    chat_id: str
    previous_messages: tuple[MessageSnapshot, ...]
    resulting_messages: tuple[MessageSnapshot, ...]


def snapshot_messages(messages: list[Message]) -> tuple[MessageSnapshot, ...]:
    """Create an immutable transcript snapshot suitable for optimistic undo."""

    return tuple(
        MessageSnapshot(
            id=message.id,
            role=message.role,
            content=message.content,
            model=message.model,
            continuity_rewrite=message.continuity_rewrite,
            reasoning=message.reasoning,
            created_at=message.created_at,
            updated_at=message.updated_at,
        )
        for message in messages
    )


def restore_messages(snapshot: tuple[MessageSnapshot, ...]) -> list[Message]:
    """Rebuild mutable message instances from an undo snapshot."""

    return [
        Message(
            id=message.id,
            role=message.role,
            content=message.content,
            model=message.model,
            continuity_rewrite=message.continuity_rewrite,
            reasoning=message.reasoning,
            created_at=message.created_at,
            updated_at=message.updated_at,
        )
        for message in snapshot
    ]


def require_non_empty_text(content: str, *, field_name: str) -> str:
    """Reject blank user-editable text."""

    if not content.strip():
        raise ValueError(f"{field_name} cannot be empty")
    return content


def trimmed_or_empty(content: str | None) -> str:
    """Normalize optional form text to its persisted representation."""

    return (content or "").strip()


NO_DELETED_CHAT_IDS: Final[frozenset[str]] = frozenset()
