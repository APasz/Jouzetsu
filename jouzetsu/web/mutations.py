"""Guarded browser mutations with explicit chat targets and visible rejections."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Literal
from urllib.parse import urlencode

from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import (
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
)

from ..inference import ChatBusy, RevisionConflict, SendResult
from ..models import Chat, Message
from ..state import AppState
from .form_data import form_values
from .request_access import RequestAccess

type DialogName = Literal["", "global", "access"]
type DialogTabName = Literal["", "behaviour", "appearance", "backends"]
log = logging.getLogger(__name__)


class Mutations:
    """Authorize commands and retain their browser-local destination."""

    def __init__(self, state: AppState, access: RequestAccess) -> None:
        self._state = state
        self._access = access

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
    ) -> Response:
        _ = await self._access.require(
            request,
            require_csrf=True,
            require_global_settings=require_global_settings,
            require_access_management=require_access_management,
        )
        chat_id = ""
        try:
            form = await form_values(request)
            chat_id = form.text("chat_id") or form.text("return_chat_id")
            target = request.path_params.get("chat_id")
            if target is not None and target != chat_id:
                raise ValueError("chat target mismatch")
            result = await operation()
            if isinstance(result, Chat):
                chat_id = result.id
            if chat_id and chat_id not in self._state.registry:
                chat_id = ""
        except HTTPException:
            raise
        except (RevisionConflict, ChatBusy) as exc:
            return PlainTextResponse(str(exc), status_code=409)
        except Exception as exc:  # noqa: BLE001
            # Exception messages can contain submitted content; log only the type.
            log.error(
                "web mutation failed route=%s error_type=%s",
                request.url.path,
                type(exc).__name__,
            )
            return PlainTextResponse(str(exc), status_code=400)
        destination = self.redirect(
            chat_id=chat_id, notice=notice, dialog=dialog, dialog_tab=dialog_tab
        )
        if request.headers.get("HX-Request") == "true":
            headers = {"X-Jouzetsu-Location": destination.headers["location"]}
            if (
                isinstance(result, SendResult)
                and result.draft_cleared
                and result.draft_revision is not None
            ):
                headers["X-Jouzetsu-Draft-Cleared"] = "true"
                headers["X-Jouzetsu-Draft-Revision"] = str(result.draft_revision)
            if chat_id:
                headers["X-Jouzetsu-Chat-Revision"] = str(
                    self._state.session(chat_id).chat.revision
                )
            revision = (
                self._state.session(chat_id).chat.draft_revision if chat_id else None
            )
            return JSONResponse({"draft_revision": revision}, headers=headers)
        return destination

    def require_visible_message(self, chat_id: str, message_id: str) -> Message:
        message = self._state.session(chat_id).chat.find_message(message_id)
        if message is None or message.role not in {"user", "assistant"}:
            raise ValueError("message action requires a visible message")
        return message

    @staticmethod
    def redirect(
        *,
        chat_id: str = "",
        notice: str = "",
        error: str = "",
        dialog: DialogName = "",
        dialog_tab: DialogTabName = "",
    ) -> RedirectResponse:
        values = {
            key: value[:240]
            for key, value in {
                "notice": notice,
                "error": error,
                "dialog": dialog,
                "tab": dialog_tab,
            }.items()
            if value
        }
        path = f"/chats/{chat_id}" if chat_id else "/chats"
        return RedirectResponse(
            path + (f"?{urlencode(values)}" if values else ""), status_code=303
        )
