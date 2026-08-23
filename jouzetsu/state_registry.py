"""Indexed ownership of chats and their transient runtime sessions."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from .models import Chat
from .runtime import RuntimeStatus
from .state_types import ChatSession


class ChatRegistry:
    """Keep chat identity, selection, and unsaved-document tracking consistent."""

    def __init__(self, empty_state_message: Callable[[], str]) -> None:
        self._empty_state_message: Callable[[], str] = empty_state_message
        self._sessions_by_id: dict[str, ChatSession] = {}
        self._active_chat_id: str = ""
        self._unpersisted_chat_ids: set[str] = set()

    @property
    def sessions(self) -> tuple[ChatSession, ...]:
        """Return every live session in insertion order."""

        return tuple(self._sessions_by_id.values())

    @property
    def active_chat_id(self) -> str:
        return self._active_chat_id

    @property
    def active(self) -> ChatSession:
        return self.get(self._active_chat_id)

    @property
    def chats_by_activity(self) -> list[Chat]:
        """Return chats ordered for presentation without changing registry order."""

        return [
            session.chat
            for session in sorted(
                self._sessions_by_id.values(),
                key=lambda session: session.chat.updated_at,
                reverse=True,
            )
        ]

    def __bool__(self) -> bool:
        return bool(self._sessions_by_id)

    def __len__(self) -> int:
        return len(self._sessions_by_id)

    def __contains__(self, chat_id: str) -> bool:
        return chat_id in self._sessions_by_id

    def get(self, chat_id: str) -> ChatSession:
        """Resolve a session or make stale chat identifiers explicit."""

        try:
            return self._sessions_by_id[chat_id]
        except KeyError as exc:
            raise KeyError(f"unknown chat id: {chat_id}") from exc

    def add(
        self,
        chat: Chat,
        *,
        runtime_status: RuntimeStatus,
        persist: bool,
        activate: bool,
    ) -> ChatSession:
        """Register one chat exactly once, optionally selecting and persisting it."""

        if chat.id in self._sessions_by_id:
            raise ValueError(f"duplicate chat id: {chat.id}")
        session = ChatSession(
            chat=chat,
            runtime_status=runtime_status,
            empty_state_message=self._empty_state_message(),
        )
        self._sessions_by_id[chat.id] = session
        if persist:
            self._unpersisted_chat_ids.add(chat.id)
        if activate:
            self._active_chat_id = chat.id
        return session

    def select(self, chat_id: str) -> bool:
        """Select a known chat and report whether the selection changed."""

        if chat_id not in self or chat_id == self._active_chat_id:
            return False
        self._active_chat_id = chat_id
        return True

    def select_most_recent(self) -> str:
        """Select and return the most recently updated live chat."""

        if not self._sessions_by_id:
            raise RuntimeError("cannot select a chat from an empty registry")
        self._active_chat_id = self.chats_by_activity[0].id
        return self._active_chat_id

    def remove(self, chat_id: str) -> ChatSession:
        """Remove a chat and discard its unsaved-document marker."""

        session = self.get(chat_id)
        del self._sessions_by_id[chat_id]
        self._unpersisted_chat_ids.discard(chat_id)
        if self._active_chat_id == chat_id:
            self._active_chat_id = ""
        return session

    def changes_for_save(
        self,
        changed_chats: Iterable[Chat],
        *,
        deleted_chat_ids: frozenset[str],
    ) -> dict[str, Chat]:
        """Combine explicit changes with every still-unpersisted live chat."""

        changes: dict[str, Chat] = {}
        for chat in changed_chats:
            if chat.id in changes:
                raise ValueError(f"duplicate changed chat id: {chat.id}")
            changes[chat.id] = chat
        for chat_id in self._unpersisted_chat_ids - deleted_chat_ids:
            try:
                changes.setdefault(chat_id, self._sessions_by_id[chat_id].chat)
            except KeyError as exc:
                raise RuntimeError(
                    f"unpersisted chat is absent from state: {chat_id}"
                ) from exc
        for chat_id in deleted_chat_ids:
            changes.pop(chat_id, None)
        return changes

    def mark_persisted(self, chat_ids: Iterable[str]) -> None:
        """Forget unsaved markers only after durable storage acknowledges them."""

        self._unpersisted_chat_ids.difference_update(chat_ids)
