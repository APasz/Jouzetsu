"""Guarded mutations and short-lived message undo records."""

from __future__ import annotations

import logging
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Final, Literal
from urllib.parse import urlencode

from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import RedirectResponse

from ..models import Message
from ..state import AppState, ChatMessageUndo
from .request_access import RequestAccess

log: logging.Logger = logging.getLogger(__name__)
_NOTICE_MAXIMUM_LENGTH: Final[int] = 240
_UNDO_MAX_AGE_SECONDS: Final[float] = 12.0

type DialogName = Literal["", "global", "access", "models", "host"]
type DialogTabName = Literal["", "behaviour", "appearance", "loaded", "defaults"]


@dataclass(frozen=True, slots=True)
class _MessageUndoRecord:
    undo: ChatMessageUndo
    expires_at: float


class Mutations:
    """Run user mutations with access checks and predictable redirect feedback."""

    def __init__(self, state: AppState, access: RequestAccess) -> None:
        self._state: AppState = state
        self._access: RequestAccess = access
        self._message_undo_records: dict[str, _MessageUndoRecord] = {}

    async def perform(
        self,
        request: Request,
        operation: Callable[[], Awaitable[object]],
        *,
        notice: str,
        require_global_settings: bool = False,
        require_access_management: bool = False,
        dialog: DialogName = "",
        dialog_tab: DialogTabName = "",
    ) -> RedirectResponse:
        """Run a guarded mutation and turn expected failures into an app notice."""

        try:
            _ = await self._access.require(
                request,
                require_csrf=True,
                require_global_settings=require_global_settings,
                require_access_management=require_access_management,
            )
            _ = await operation()
        except HTTPException:
            raise
        except Exception as exc:
            log.exception("web mutation failed route=%s", request.url.path)
            return self.redirect(error=str(exc), dialog=dialog, dialog_tab=dialog_tab)
        return self.redirect(notice=notice, dialog=dialog, dialog_tab=dialog_tab)

    def require_visible_message(self, message_id: str) -> Message:
        """Return a visible user/assistant message or reject a crafted system target."""

        if not message_id:
            raise ValueError("message id is required")
        message: Message | None = self._state.active_chat.find_message(message_id)
        if message is None or message.role not in {"user", "assistant"}:
            raise ValueError(
                "message action requires a visible user or assistant message"
            )
        return message

    def require_last_assistant_message(self, message_id: str) -> Message:
        """Reject actions whose target is no longer the active final reply."""

        message: Message = self.require_visible_message(message_id)
        messages = self._state.active_chat.messages
        if not messages or messages[-1].id != message.id or message.role != "assistant":
            raise ValueError(
                "this action is only available for the last assistant message"
            )
        return message

    def require_last_user_message(self, message_id: str) -> Message:
        """Reject a resend target that is not the active final user message."""

        message: Message = self.require_visible_message(message_id)
        messages = self._state.active_chat.messages
        if not messages or messages[-1].id != message.id or message.role != "user":
            raise ValueError("this action is only available for the last user message")
        return message

    def store_message_undo(self, undo: ChatMessageUndo) -> str:
        """Store one undo snapshot with a short expiry."""

        now: float = time.monotonic()
        self._message_undo_records = {
            token: record
            for token, record in self._message_undo_records.items()
            if record.expires_at > now
        }
        token: str = secrets.token_urlsafe(18)
        self._message_undo_records[token] = _MessageUndoRecord(
            undo=undo, expires_at=now + _UNDO_MAX_AGE_SECONDS
        )
        return token

    def consume_message_undo(self, token: str) -> ChatMessageUndo:
        """Consume an unexpired undo snapshot exactly once."""

        record: _MessageUndoRecord | None = self._message_undo_records.pop(token, None)
        if record is None or record.expires_at <= time.monotonic():
            raise ValueError("this undo action is no longer available")
        return record.undo

    @staticmethod
    def redirect(
        *,
        notice: str = "",
        error: str = "",
        dialog: DialogName = "",
        dialog_tab: DialogTabName = "",
        undo: str = "",
    ) -> RedirectResponse:
        """Redirect a form submission while retaining its source dialog."""

        values: dict[str, str] = {}
        if dialog:
            values["dialog"] = dialog
        if dialog_tab:
            values["tab"] = dialog_tab
        if notice:
            values["notice"] = notice[:_NOTICE_MAXIMUM_LENGTH]
        if error:
            values["error"] = error[:_NOTICE_MAXIMUM_LENGTH]
        if undo:
            values["undo"] = undo
        url: str = "/chats" if not values else f"/chats?{urlencode(values)}"
        return RedirectResponse(url, status_code=303)
