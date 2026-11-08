"""Indexed ownership of chats and their transient runtime sessions."""

from __future__ import annotations

from collections.abc import Callable

from .models import Chat
from .runtime import RuntimeStatus
from .state_types import ChatSession


class ChatRegistry:
    """Keep shared chat identity and transient sessions consistent."""

    def __init__(self, empty_state_message: Callable[[], str]) -> None:
        self._empty_state_message: Callable[[], str] = empty_state_message
        self._sessions_by_id: dict[str, ChatSession] = {}

    @property
    def sessions(self) -> tuple[ChatSession, ...]:
        """Return every live session in insertion order."""

        return tuple(self._sessions_by_id.values())

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
    ) -> ChatSession:
        """Register one durably saved chat exactly once."""

        if chat.id in self._sessions_by_id:
            raise ValueError(f"duplicate chat id: {chat.id}")
        session = ChatSession(
            chat=chat,
            runtime_status=runtime_status,
            empty_state_message=self._empty_state_message(),
        )
        self._sessions_by_id[chat.id] = session
        return session

    def remove(self, chat_id: str) -> ChatSession:
        """Remove a durably deleted chat."""

        session = self.get(chat_id)
        del self._sessions_by_id[chat_id]
        return session
