"""Renderers for user-visible operation feedback."""

from __future__ import annotations

from fastcore.xml import FT  # pyright: ignore[reportMissingTypeStubs]

from ...state import AppState
from ..html import Div, Span

type HTML = FT


def storage_warning(state: AppState) -> str:
    """Explain why isolated documents are absent without exposing their contents."""
    chats = state.storage.isolated_count
    characters = state.character_storage.isolated_count
    if not chats and not characters:
        return ""
    return (
        f"Some documents could not be loaded ({chats} chat files, "
        f"{characters} character files). Their files were left unchanged. "
        "Check the server logs for details."
    )


def render_notice_area(*, notice: str, error: str, warning: str = "") -> HTML:
    """Render server-side feedback in an ARIA live region."""

    message: str = error or notice
    classes: str = (
        "jouzetsu-notice jouzetsu-notice-error" if error else "jouzetsu-notice"
    )
    return Div(
        Div(
            Span(message, cls="jouzetsu-notice-message"),
            cls=classes,
            role="alert" if error else "status",
            data_notice_kind="error" if error else "success",
            data_notice_dismiss="true",
        )
        if message
        else None,
        Div(
            Span(warning, cls="jouzetsu-notice-message"),
            cls="jouzetsu-notice",
            role="alert",
            data_storage_warning="true",
        )
        if warning
        else None,
        id="notice-area",
        cls="jouzetsu-notice-area",
    )


def dialog_feedback(*, notice: str, error: str) -> HTML | None:
    """Render mutation feedback inside the dialog that initiated it."""

    message: str = error or notice
    if not message:
        return None
    classes: str = (
        "jouzetsu-dialog-feedback is-error" if error else "jouzetsu-dialog-feedback"
    )
    return Div(message, cls=classes, role="alert" if error else "status")
