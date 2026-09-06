"""Shared HTML controls used across Jouzetsu workspaces and dialogs."""

from __future__ import annotations

from typing import Literal

from ..html import HTML, Button, Div, Form, Input, Label, Span
from .context import PageContext


def post_button(
    label: str | HTML,
    *,
    action: str,
    marker: str,
    classes: str = "jouzetsu-button",
    confirm: str = "",
    disabled: bool = False,
    title: str = "",
    aria_label: str = "",
    chat_mutation: bool = False,
    message_action: Literal["continue"] | None = None,
) -> HTML:
    """Render a consistently styled form containing one POST action button."""

    button_attributes: dict[str, object] = {
        "type": "submit",
        "cls": classes,
        "data_testid": marker,
        "disabled": disabled,
    }
    if confirm:
        button_attributes["data_confirm"] = confirm
    if title:
        button_attributes["title"] = title
    if aria_label:
        button_attributes["aria_label"] = aria_label
    if message_action is not None:
        button_attributes["data_message_action"] = message_action
    return Form(
        Button(label, **button_attributes),
        action=action,
        method="post",
        cls="jouzetsu-inline-form",
        data_chat_mutation="true" if chat_mutation else None,
    )


def close_dialog_button(dialog_id: str, *, marker: str = "") -> HTML:
    """Render a button that closes a native dialog through the local script."""

    return Button(
        "Close",
        type="button",
        cls="jouzetsu-button jouzetsu-button-muted",
        data_dialog_close=dialog_id,
        data_testid=marker or f"{dialog_id}-close-button",
    )


def reset_form_button(
    *,
    action: str,
    marker: str,
    label: str = "Reset to defaults",
    confirmation: str = "",
    disabled: bool = False,
    live_notice: str = "",
    form_id: str = "",
) -> HTML:
    """Render a reset submitter that can target one section of a shared form."""

    attributes: dict[str, object] = {
        "type": "submit",
        "formaction": action,
        "formmethod": "post",
        "formnovalidate": True,
        "cls": "jouzetsu-button jouzetsu-button-muted",
        "data_testid": marker,
        "disabled": disabled,
    }
    if confirmation:
        attributes["data_confirm"] = confirmation
    if live_notice:
        attributes["data_live_notice"] = live_notice
    if form_id:
        attributes["form"] = form_id
    return Button(label, **attributes)


def form_submit_actions(
    label: str,
    marker: str,
    *,
    reset_action: str | None = None,
    reset_marker: str = "",
    reset_label: str = "Reset to defaults",
    reset_confirmation: str = "",
) -> HTML:
    """Render optional reset and primary save actions for one settings form."""

    return Div(
        Button(
            label,
            type="submit",
            cls="jouzetsu-button jouzetsu-button-primary",
            data_testid=marker,
        ),
        (
            reset_form_button(
                action=reset_action,
                marker=reset_marker,
                label=reset_label,
                confirmation=reset_confirmation,
            )
            if reset_action is not None
            else None
        ),
        cls="jouzetsu-dialog-actions",
    )


def field(label: str, control: HTML, *, classes: str = "") -> HTML:
    """Associate one visible field label with a control without generated IDs."""

    return Label(
        Span(label, cls="jouzetsu-field-label"),
        control,
        cls=f"jouzetsu-field {classes}".strip(),
    )


def checkbox(
    name: str,
    *,
    checked: bool,
    marker: str,
    label: str,
    disabled: bool = False,
    live_autosave: bool = False,
) -> HTML:
    """Render a checkbox that submits ``true`` when selected."""

    return Label(
        Input(
            type="checkbox",
            name=name,
            value="true",
            checked=checked,
            disabled=disabled,
            data_live_autosave="true" if live_autosave else None,
            data_testid=marker,
        ),
        Span(label),
        cls="jouzetsu-check-label",
    )


def csrf_context(context: PageContext) -> HTML:
    """Expose the request-bound CSRF token to same-origin browser code."""

    return Div(
        id="jouzetsu-csrf-context",
        hidden=True,
        data_csrf_token=context.csrf_token,
    )
