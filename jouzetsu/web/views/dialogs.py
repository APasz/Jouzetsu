"""Dialogs attached to the chat workspace."""

from __future__ import annotations

from ...models import Message
from ...runtime import ModelDescriptor, ModelInstanceDescriptor
from ...state import AppState
from ..html import (
    H2,
    H3,
    HTML,
    Button,
    Details,
    Dialog,
    Div,
    Form,
    P,
    Pre,
    Span,
    Summary,
    Textarea,
)
from .controls import close_dialog_button, post_button
from .feedback import dialog_feedback
from .host_stats import render_content as _render_host_stats_content


def render_message_delete_dialog() -> HTML:
    """Render the reusable confirmation dialog for one message deletion choice."""

    return Dialog(
        Div(
            H2(
                "Delete message",
                id="message-delete-dialog-title",
                cls="jouzetsu-modal-title",
            ),
            P(
                "Choose whether to remove only this message or this message and every later message.",
                id="message-delete-dialog-description",
                cls="jouzetsu-dialog-description",
            ),
            Div(
                Form(
                    Button("Delete only", type="submit", cls="jouzetsu-button"),
                    method="post",
                    cls="jouzetsu-inline-form",
                    data_delete_choice_form="single",
                ),
                Form(
                    Button(
                        "Delete this and all following",
                        type="submit",
                        cls="jouzetsu-button jouzetsu-button-danger",
                    ),
                    method="post",
                    cls="jouzetsu-inline-form",
                    data_delete_choice_form="following",
                ),
                close_dialog_button(
                    "message-delete-dialog", marker="message-delete-cancel-button"
                ),
                cls="jouzetsu-dialog-actions",
            ),
            cls="jouzetsu-modal-card",
        ),
        id="message-delete-dialog",
        cls="jouzetsu-dialog",
        aria_labelledby="message-delete-dialog-title",
        aria_describedby="message-delete-dialog-description",
        data_testid="message-delete-dialog",
    )


def render_message_context_menu() -> HTML:
    """Render the shared pointer-positioned menu for a selected message."""

    return Div(
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
                "Metadata and any saved reasoning trace for this message.",
                id="message-details-dialog-description",
                cls="jouzetsu-dialog-description",
            ),
            render_message_details_content(),
            Div(
                _message_details_fork_form(),
                close_dialog_button(
                    "message-details-dialog", marker="message-details-close-button"
                ),
                cls="jouzetsu-dialog-actions",
                data_message_details_actions="true",
            ),
            cls="jouzetsu-modal-card jouzetsu-message-details-card",
        ),
        id="message-details-dialog",
        cls="jouzetsu-dialog",
        aria_labelledby="message-details-dialog-title",
        aria_describedby="message-details-dialog-description",
        data_testid="message-details-dialog",
    )


def _message_details_fork_form() -> HTML:
    return Form(
        Button(
            "Fork chat",
            type="submit",
            cls="jouzetsu-button",
            disabled=True,
            title="Fork a chat through this message",
        ),
        action="",
        method="post",
        cls="jouzetsu-inline-form",
        hidden=True,
        data_chat_mutation="true",
        data_message_details_fork_form="true",
        data_testid="message-details-fork-action",
    )


def render_message_details_content(
    message: Message | None = None, *, can_fork: bool = True
) -> HTML:
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
            "No reasoning text was saved for this message.",
            cls="jouzetsu-message-details-reasoning-empty",
            data_testid="message-details-reasoning-empty",
        )
    )
    model_detail: HTML | None = (
        Div(
            Span("Model", cls="jouzetsu-message-detail-label"),
            Pre(message.model or "Not recorded", cls="jouzetsu-message-detail-value"),
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
        data_message_details_fork_action=f"/messages/{message.id}/fork"
        if can_fork
        else None,
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


def render_continuity_rewrite_dialog(message: Message) -> HTML:
    """Render the user-confirmed replacement proposed for one assistant response."""

    return Dialog(
        Div(
            H2(
                "Review proposed rewrite",
                id="continuity-rewrite-dialog-title",
                cls="jouzetsu-modal-title",
            ),
            P(
                "The model found a possible continuity issue. Apply this rewrite, or cancel to keep the current reply.",
                id="continuity-rewrite-dialog-description",
                cls="jouzetsu-dialog-description",
            ),
            Textarea(
                message.continuity_rewrite,
                rows=16,
                readonly=True,
                aria_label="Proposed rewrite",
                data_testid="continuity-rewrite-preview",
            ),
            Div(
                post_button(
                    "Apply rewrite",
                    action=f"/messages/{message.id}/continuity-rewrite/apply",
                    marker="apply-continuity-rewrite-button",
                    classes="jouzetsu-button jouzetsu-button-primary",
                    chat_mutation=True,
                ),
                post_button(
                    "Cancel",
                    action=f"/messages/{message.id}/continuity-rewrite/discard",
                    marker="cancel-continuity-rewrite-button",
                    classes="jouzetsu-button jouzetsu-button-muted",
                    chat_mutation=True,
                ),
                cls="jouzetsu-dialog-actions",
            ),
            cls="jouzetsu-modal-card",
        ),
        id="continuity-rewrite-dialog",
        cls="jouzetsu-dialog",
        aria_labelledby="continuity-rewrite-dialog-title",
        aria_describedby="continuity-rewrite-dialog-description",
        data_testid="continuity-rewrite-dialog",
    )


def render_models_dialog(state: AppState, *, notice: str = "", error: str = "") -> HTML:
    """Render metadata for every loaded LM Studio model instance."""

    model_cards: tuple[HTML, ...] = tuple(
        render_loaded_model_card(state, model, instance)
        for model in state.model_inventory
        for instance in model.loaded_instances
    )
    model_content: tuple[HTML, ...] = model_cards or (
        P("No loaded models.", cls="jouzetsu-empty-state"),
    )
    return Dialog(
        Div(
            H2("Models", id="models-dialog-title", cls="jouzetsu-modal-title"),
            P(
                "Loaded LM Studio instances.",
                id="models-dialog-description",
                cls="jouzetsu-dialog-description",
            ),
            dialog_feedback(notice=notice, error=error),
            Form(
                Button(
                    "Refresh",
                    type="submit",
                    cls="jouzetsu-button",
                    data_testid="refresh-models-button",
                ),
                action="/models/refresh",
                method="post",
                cls="jouzetsu-inline-form",
            ),
            H3("Loaded models", data_testid="models-tab"),
            Div(
                *model_content,
                cls="jouzetsu-model-list",
                data_testid="loaded-model-list",
            ),
            close_dialog_button("models-dialog", marker="loaded-models-close-button"),
            cls="jouzetsu-modal-card jouzetsu-models-card",
        ),
        id="models-dialog",
        cls="jouzetsu-dialog",
        aria_labelledby="models-dialog-title",
        aria_describedby="models-dialog-description",
    )


def render_loaded_model_card(
    state: AppState, model: ModelDescriptor, instance: ModelInstanceDescriptor
) -> HTML:
    """Render one loaded instance with detailed metadata in a disclosure element."""

    capabilities: tuple[str, ...] = tuple(
        label
        for enabled, label in (
            (model.capabilities.vision, "vision"),
            (model.capabilities.trained_for_tool_use, "tool use"),
            (model.capabilities.reasoning, "reasoning"),
        )
        if enabled
    )
    is_generating: bool = state.is_model_generating(model.key)
    return Details(
        Summary(
            model.display_name or state.model_display_name(model.key),
            cls="jouzetsu-model-summary",
        ),
        Div(
            P(f"Model key: {model.key}", cls="jouzetsu-mono"),
            P(f"Instance ID: {instance.id}", cls="jouzetsu-mono"),
            P(
                f"Context: {_optional_int(instance.context_length or model.context_length)}"
            ),
            P(f"Format: {model.format or 'Unknown'}"),
            P(f"Quantization: {model.quantization_name or 'Unknown'}"),
            P(
                f"Capabilities: {', '.join(capabilities) if capabilities else 'None advertised'}"
            ),
            post_button(
                "Unload instance",
                action=f"/models/instances/{instance.id}/unload",
                marker=f"unload-model-{instance.id}",
                classes="jouzetsu-button jouzetsu-button-danger",
                confirm="Unload this model instance?",
                disabled=is_generating,
                title="Cannot unload a model while it is generating a response"
                if is_generating
                else "",
            ),
            cls="jouzetsu-model-details",
            data_testid="loaded-model-detail",
        ),
        cls="jouzetsu-model-card",
        data_testid="loaded-model-card",
    )


def _optional_int(value: int | None) -> str:
    return str(value) if value is not None else "Unknown"


def render_host_stats_dialog(
    state: AppState, *, notice: str = "", error: str = ""
) -> HTML:
    """Render the host telemetry dialog and its independently refreshable content."""

    return Dialog(
        Div(
            H2("Host stats", id="host-stats-dialog-title", cls="jouzetsu-modal-title"),
            P(
                "Live host telemetry and ten-minute rolling averages.",
                id="host-stats-dialog-description",
                cls="jouzetsu-dialog-description",
            ),
            dialog_feedback(notice=notice, error=error),
            render_host_stats_content(state),
            Div(
                Button(
                    "Refresh now",
                    type="button",
                    cls="jouzetsu-button",
                    data_host_stats_refresh="true",
                    data_testid="refresh-host-stats-button",
                ),
                close_dialog_button(
                    "host-stats-dialog", marker="host-stats-close-button"
                ),
                cls="jouzetsu-dialog-actions",
            ),
            cls="jouzetsu-modal-card jouzetsu-host-stats-card",
            data_testid="host-stats-dialog",
        ),
        id="host-stats-dialog",
        cls="jouzetsu-dialog",
        data_host_stats_dialog="true",
        aria_labelledby="host-stats-dialog-title",
        aria_describedby="host-stats-dialog-description",
    )


def render_host_stats_content(state: AppState) -> HTML:
    """Render the independently refreshable host-telemetry body."""

    return _render_host_stats_content(
        state.host_stats_snapshot(), state.config.host_stats
    )
