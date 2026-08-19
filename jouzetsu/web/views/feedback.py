"""Renderers for user-visible operation feedback."""

from __future__ import annotations

from fastcore.xml import FT  # pyright: ignore[reportMissingTypeStubs]

from ..html import Button, Div, Form, Input, Span

type HTML = FT


def render_notice_area(*, notice: str, error: str, undo_token: str = "") -> HTML:
    """Render server-side feedback in an ARIA live region."""

    message: str = error or notice
    if not message:
        return Div(id="notice-area", cls="jouzetsu-notice-area")
    classes: str = (
        "jouzetsu-notice jouzetsu-notice-error" if error else "jouzetsu-notice"
    )
    undo_control: HTML | None = None
    if undo_token and not error:
        undo_control = Form(
            Input(type="hidden", name="undo_token", value=undo_token),
            Button(
                "Undo",
                type="submit",
                cls="jouzetsu-notice-action",
                data_notice_action="undo",
            ),
            action="/messages/undo",
            method="post",
        )
    return Div(
        Div(
            Span(message, cls="jouzetsu-notice-message"),
            undo_control,
            cls=classes,
            role="alert" if error else "status",
            data_notice_kind="error" if error else "success",
            data_notice_dismiss="true",
        ),
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
