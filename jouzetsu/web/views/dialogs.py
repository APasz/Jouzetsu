"""Dialogs attached to the chat workspace."""

from __future__ import annotations

from ...models import Message
from ..html import (
    H2,
    HTML,
    Button,
    Dialog,
    Div,
    Form,
    P,
    Pre,
    Span,
)
from .controls import close_dialog_button


def render_message_context_menu() -> HTML:
    """Render the shared pointer-positioned menu for a selected message."""

    return Div(
        Button(
            "Copy",
            type="button",
            role="menuitem",
            cls="jouzetsu-message-context-menu-item",
            data_message_copy="true",
            data_testid="message-copy-context-action",
        ),
        _message_context_fork_form(),
        Button(
            "Message details",
            type="button",
            role="menuitem",
            cls="jouzetsu-message-context-menu-item",
            data_message_details_open="true",
            data_testid="message-details-context-action",
        ),
        id="message-context-menu",
        role="menu",
        aria_label="Message actions",
        cls="jouzetsu-message-context-menu",
        hidden=True,
        data_testid="message-context-menu",
    )


def _message_context_fork_form() -> HTML:
    """Render the menu action whose target is set for the selected message."""

    return Form(
        Button(
            "Fork chat",
            type="submit",
            role="menuitem",
            cls="jouzetsu-message-context-menu-item",
            disabled=True,
            title="Fork a chat through this message",
        ),
        action="",
        method="post",
        cls="jouzetsu-message-context-menu-form",
        hidden=True,
        data_chat_mutation="true",
        data_message_context_fork_form="true",
        data_testid="message-fork-context-action",
    )


def render_message_details_dialog() -> HTML:
    """Render the reusable dialog whose metadata body loads on demand."""

    return Dialog(
        Div(
            H2(
                "Message details",
                id="message-details-dialog-title",
                cls="jouzetsu-modal-title",
            ),
            P(
                "Timestamps, model, and saved reasoning.",
                id="message-details-dialog-description",
                cls="jouzetsu-dialog-description",
            ),
            render_message_details_content(),
            Div(
                close_dialog_button(
                    "message-details-dialog", marker="message-details-close-button"
                ),
                cls="jouzetsu-dialog-actions",
            ),
            cls="jouzetsu-modal-card jouzetsu-message-details-card",
        ),
        id="message-details-dialog",
        cls="jouzetsu-dialog",
        aria_labelledby="message-details-dialog-title",
        aria_describedby="message-details-dialog-description",
        data_testid="message-details-dialog",
    )


def render_message_details_content(message: Message | None = None) -> HTML:
    """Render one message's detailed metadata, fetched only when requested."""

    if message is None:
        return Div(
            P(
                "Select a message to view its details.",
                cls="jouzetsu-message-details-status",
            ),
            id="message-details-content",
            data_testid="message-details-content",
        )

    reasoning: HTML = (
        Pre(
            message.reasoning,
            cls="jouzetsu-message-details-reasoning",
            data_testid="message-details-reasoning",
        )
        if message.reasoning
        else P(
            "No saved reasoning.",
            cls="jouzetsu-message-details-reasoning-empty",
            data_testid="message-details-reasoning-empty",
        )
    )
    model_detail: HTML | None = (
        Div(
            Span("Model", cls="jouzetsu-message-detail-label"),
            Pre(
                "\n".join(
                    f"{item.backend_name} / {item.model_name} ({item.model.backend_id} / {item.model.model_id})"
                    for item in message.attributions
                )
                or "Not recorded",
                cls="jouzetsu-message-detail-value",
            ),
            cls="jouzetsu-message-detail-row",
            data_testid="message-details-model",
        )
        if message.role == "assistant"
        else None
    )
    return Div(
        Div(
            Span("Message ID", cls="jouzetsu-message-detail-label"),
            Pre(message.id, cls="jouzetsu-message-detail-value"),
            cls="jouzetsu-message-detail-row",
        ),
        model_detail,
        _message_timestamp_row("Created", message.created_at),
        _message_timestamp_row("Updated", message.updated_at),
        Div(
            Span("Reasoning", cls="jouzetsu-message-detail-label"),
            reasoning,
            cls="jouzetsu-message-detail-row is-reasoning",
        ),
        id="message-details-content",
        cls="jouzetsu-message-details",
        data_testid="message-details-content",
    )


def _message_timestamp_row(label: str, timestamp: float) -> HTML:
    return Div(
        Span(label, cls="jouzetsu-message-detail-label"),
        Span(
            cls="jouzetsu-message-detail-value",
            data_message_details_timestamp=timestamp,
        ),
        cls="jouzetsu-message-detail-row",
    )
