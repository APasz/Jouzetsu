"""Renderers and read models for the chat workspace."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Final, Literal
from urllib.parse import urlencode

from ...config import GenerationSettings, MessageActionIconStyle, StarterPrompt
from ...markdown import render_markdown
from ...models import Chat, ChatSamplingOverrides, Message
from ...runtime import GenerationMetrics, RuntimePhase, RuntimeStatus
from ..html import (
    H1,
    H2,
    H3,
    HTML,
    A,
    Article,
    Aside,
    Button,
    Details,
    Div,
    Footer,
    Form,
    Header,
    Img,
    Input,
    Main,
    Option,
    P,
    Pre,
    Section,
    Select,
    Small,
    Span,
    Summary,
    Textarea,
)
from .controls import checkbox as _checkbox
from .controls import field as _field
from .controls import post_button as _post_button
from .icons import render_icon as _icon


@dataclass(frozen=True, slots=True)
class ChatView:
    """All data required to render one chat workspace and its live fragments."""

    chat: Chat
    generating: bool
    runtime_status: RuntimeStatus
    generation_reasoning: str
    empty_state_message: str
    message_action_icon_style: MessageActionIconStyle
    starter_prompts: tuple[StarterPrompt, ...]


@dataclass(frozen=True, slots=True)
class ModelChoice:
    """One selectable model rendered in a chat-scoped model selector."""

    key: str
    label: str


@dataclass(frozen=True, slots=True)
class ChatSettingsView:
    """All data required to render settings for one active chat."""

    chat: Chat
    generating: bool
    generation_defaults: GenerationSettings
    model_choices: tuple[ModelChoice, ...]


@dataclass(frozen=True, slots=True)
class NavigationView:
    """All data required to render the chat navigation drawer."""

    chats: tuple[Chat, ...]
    active_chat_id: str
    generating_chat_ids: frozenset[str]
    app_icon_url: str
    can_use_global_settings: bool
    can_manage_access: bool
    current_device_label: str | None


_COMPOSER_SUBMIT_SHORTCUT_HINT: Final[str] = "Shift+Enter / Cmd+Enter"
_COMPOSER_SUBMIT_ARIA_SHORTCUTS: Final[str] = "Shift+Enter Meta+Enter"


@dataclass(frozen=True, slots=True)
class _ComposerStatus:
    """Three-column content shared by full and incremental composer renders."""

    state: _ComposerStatusState
    detail: str = ""
    stage_elapsed_seconds: float | None = None
    overall_elapsed_seconds: float | None = None


class _ComposerStatusState(Enum):
    """Finite state labels displayed in the composer status's left column."""

    CHECKING = "Checking"
    EDITING = "Editing"
    ERROR = "Error"
    GENERATING = "Generating"
    LOADING = "Loading"
    NO_MODEL = "No model"
    OFFLINE = "Offline"
    READY = "Ready"
    REASONING = "Reasoning"
    UNLOADED = "Unloaded"


def _query_url(**values: str) -> str:
    """Build the canonical chat-workspace URL while omitting empty query values."""

    query: dict[str, str] = {key: value for key, value in values.items() if value}
    return "/chats" if not query else f"/chats?{urlencode(query)}"


def _sampling_override_value(value: float | None) -> float | int | str:
    """Render an empty form value for a sampling setting inherited from global defaults."""

    return "" if value is None else value


def _format_chat_timestamp(timestamp: float) -> str:
    """Format a chat timestamp in the host's local time for the settings drawer."""

    return datetime.fromtimestamp(timestamp).astimezone().strftime("%d %b %Y · %H:%M")


def _safe_edit_message(chat: Chat, message_id: str) -> Message | None:
    """Return an editable visible message, rejecting unknown/system messages."""

    message: Message | None = chat.find_message(message_id) if message_id else None
    if message is None or message.role not in {"user", "assistant"}:
        return None
    return message


def _streaming_assistant_message(view: ChatView) -> Message | None:
    """Return the active assistant reply while the current chat is generating."""

    if not view.generating or not view.chat.messages:
        return None
    message: Message = view.chat.messages[-1]
    return message if message.role == "assistant" else None


def _composer_status(
    view: ChatView, *, editing_message: Message | None
) -> _ComposerStatus:
    """Derive the composer status once for full and incremental chat renders."""

    return (
        _ComposerStatus(_ComposerStatusState.EDITING, "Message")
        if editing_message is not None
        else _format_runtime_status(view.runtime_status)
    )


def _render_composer_status_content(status: _ComposerStatus) -> tuple[HTML, ...]:
    """Render stable state, detail, and timer columns for live updates."""

    return (
        Span(
            status.state.value,
            cls="jouzetsu-composer-status-state",
            data_composer_status_state="true",
        ),
        Span(
            status.detail,
            cls="jouzetsu-composer-status-detail",
            data_composer_status_detail="true",
        ),
        Div(
            _render_composer_status_timer(status.stage_elapsed_seconds, timer="stage"),
            _render_composer_status_timer(
                status.overall_elapsed_seconds, timer="overall"
            ),
            cls="jouzetsu-composer-status-timers",
            data_composer_status_timers="true",
        ),
    )


def _render_composer_status_timer(
    elapsed_seconds: float | None,
    *,
    timer: Literal["stage", "overall"],
) -> HTML:
    """Render one persistent timer so the client can update it without rebuilding the status."""

    elapsed_marker: str = f"data_composer_status_{timer}_elapsed"
    timer_marker: str = f"data_composer_status_{timer}_timer"
    return Span(
        Span(
            f"{elapsed_seconds:.1f}s" if elapsed_seconds is not None else "",
            **{elapsed_marker: "true"},
        ),
        cls="jouzetsu-composer-status-timer",
        hidden=elapsed_seconds is None,
        **{timer_marker: "true"},
    )


def _composer_status_elapsed_attribute(elapsed_seconds: float | None) -> str | None:
    """Encode a monotonic elapsed baseline only when the client should tick it."""

    return f"{elapsed_seconds:.3f}" if elapsed_seconds is not None else None


def render_navigation_panel(view: NavigationView) -> HTML:
    """Render the chat list and application utility launchers."""

    chat_items: list[HTML] = []
    for chat in view.chats:
        generating: bool = chat.id in view.generating_chat_ids
        classes: str = (
            "jouzetsu-chat-item is-active"
            if chat.id == view.active_chat_id
            else "jouzetsu-chat-item"
        )
        title: str = f"● {chat.title}" if generating else chat.title
        chat_items.append(
            Form(
                Button(
                    title,
                    type="submit",
                    cls=classes,
                    data_testid=f"chat-item-{chat.id}",
                ),
                action=f"/chats/{chat.id}/select",
                method="post",
                data_chat_mutation="true",
            )
        )

    utility_links: list[HTML] = []
    if view.can_use_global_settings:
        utility_links.append(
            Button(
                _icon("settings", classes="jouzetsu-svg-icon jouzetsu-utility-icon"),
                Span("Global settings", cls="jouzetsu-utility-label"),
                type="button",
                cls="jouzetsu-utility-link",
                data_dialog_open="global-settings-dialog",
                data_testid="global-settings-button",
            )
        )
    if view.can_manage_access:
        utility_links.append(
            Button(
                _icon("shield", classes="jouzetsu-svg-icon jouzetsu-utility-icon"),
                Span("Access & Logs", cls="jouzetsu-utility-label"),
                type="button",
                cls="jouzetsu-utility-link",
                data_dialog_open="access-dialog",
                data_testid="access-settings-button",
            )
        )
    if view.can_use_global_settings:
        utility_links.extend(
            [
                A(
                    _icon("box", classes="jouzetsu-svg-icon jouzetsu-utility-icon"),
                    Span("Models", cls="jouzetsu-utility-label"),
                    href=_query_url(dialog="models"),
                    cls="jouzetsu-utility-link",
                    data_testid="loaded-models-button",
                ),
                A(
                    _icon(
                        "activity", classes="jouzetsu-svg-icon jouzetsu-utility-icon"
                    ),
                    Span("Host stats", cls="jouzetsu-utility-label"),
                    href=_query_url(dialog="host"),
                    cls="jouzetsu-utility-link",
                    data_testid="host-stats-button",
                ),
            ]
        )

    device_form: HTML | None = None
    if view.current_device_label is not None:
        device_form = Form(
            _field(
                "This device",
                Input(
                    value=view.current_device_label,
                    name="label",
                    autocomplete="off",
                    data_live_autosave="true",
                    data_testid="current-device-label-input",
                ),
            ),
            action="/access/current-device/label",
            method="post",
            cls="jouzetsu-device-form",
            data_live_submit="true",
            data_live_notice="Device label saved",
        )

    return Aside(
        Div(
            Div(
                Div(
                    Img(
                        src=view.app_icon_url,
                        alt="",
                        cls="jouzetsu-sheet-brand-icon",
                        data_testid="navigation-brand-icon",
                    ),
                    Span("Jouzetsu", cls="jouzetsu-sheet-brand"),
                    cls="jouzetsu-sheet-brand-lockup",
                ),
                Button(
                    _icon("close"),
                    type="button",
                    cls="jouzetsu-icon-button",
                    data_panel_close="navigation",
                    aria_label="Close chats",
                ),
                cls="jouzetsu-panel-header",
            ),
            H3("Chats", cls="jouzetsu-section-title"),
            Div(
                Form(
                    Button(
                        _icon("plus"),
                        "New chat",
                        type="submit",
                        cls="jouzetsu-button jouzetsu-button-primary jouzetsu-new-chat-button",
                        data_testid="new-chat-button",
                    ),
                    action="/chats/new",
                    method="post",
                    data_chat_mutation="true",
                ),
                Div(*chat_items, cls="jouzetsu-chat-list", data_testid="chat-list"),
                cls="jouzetsu-chats-section",
            ),
            Div(
                H3("Workspace", cls="jouzetsu-section-title"),
                A(
                    _icon("box", classes="jouzetsu-svg-icon jouzetsu-utility-icon"),
                    Span("Characters", cls="jouzetsu-utility-label"),
                    href="/characters",
                    cls="jouzetsu-utility-link",
                    data_testid="characters-workspace-link",
                ),
                cls="jouzetsu-utilities",
            ),
            Div(cls="jouzetsu-panel-divider"),
            Div(
                H3("Utilities", cls="jouzetsu-section-title"),
                *utility_links,
                cls="jouzetsu-utilities",
            ),
            Div(
                Span(f"{len(view.chats)} chat(s)", cls="jouzetsu-sheet-footnote"),
                device_form,
                cls="jouzetsu-sheet-footer-meta",
            ),
            cls="jouzetsu-panel-content",
        ),
        id="navigation-panel",
        cls="jouzetsu-panel jouzetsu-navigation-panel",
        aria_label="Chats",
        aria_hidden="true",
        inert=True,
        data_panel_name="navigation",
    )


def render_settings_panel(view: ChatSettingsView) -> HTML:
    """Render settings scoped to the active chat."""

    chat: Chat = view.chat
    generating: bool = view.generating
    sampling: ChatSamplingOverrides = chat.sampling_overrides
    generation_defaults: GenerationSettings = view.generation_defaults
    current_model: str = chat.model
    model_options: list[HTML] = [
        Option("Use global default", value="", selected=not current_model)
    ]
    model_options.extend(
        Option(choice.label, value=choice.key, selected=choice.key == current_model)
        for choice in view.model_choices
    )

    return Aside(
        Div(
            Div(
                Div(
                    H2(chat.title, cls="jouzetsu-sheet-title"),
                    Small(
                        "Chat prompt override active"
                        if chat.system_prompt
                        else "Using global system prompt",
                        cls="jouzetsu-sheet-subtitle",
                    ),
                ),
                Button(
                    _icon("close"),
                    type="button",
                    cls="jouzetsu-icon-button",
                    data_panel_close="settings",
                    aria_label="Close settings",
                ),
                cls="jouzetsu-panel-header",
            ),
            H3("Chat", cls="jouzetsu-section-title"),
            (
                Div(
                    Span(
                        f"Character: {chat.character.name} (revision {chat.character.revision})",
                        cls="jouzetsu-form-help",
                    ),
                    A(
                        "Open character",
                        href=f"/characters/{chat.character.id}",
                        cls="jouzetsu-settings-character-link",
                    ),
                    cls="jouzetsu-settings-character-source",
                )
                if chat.character is not None
                else None
            ),
            Form(
                _field(
                    "Title",
                    Input(
                        value=chat.title,
                        name="title",
                        required=True,
                        data_live_autosave="true",
                        data_testid="rename-chat-input",
                    ),
                ),
                action=f"/chats/{chat.id}/rename",
                method="post",
                cls="jouzetsu-settings-form",
                data_live_submit="true",
                data_live_notice="Chat renamed",
            ),
            H3("Model", cls="jouzetsu-section-title"),
            Form(
                _field(
                    "Model",
                    Select(
                        *model_options,
                        name="model",
                        disabled=generating,
                        data_live_autosave="true",
                        data_testid="active-chat-model",
                    ),
                ),
                action="/chat/model",
                method="post",
                cls="jouzetsu-settings-form",
                data_live_submit="true",
                data_live_notice="Chat model saved",
            ),
            H3("Sampling", cls="jouzetsu-section-title"),
            Form(
                Div(
                    _field(
                        "Temperature",
                        Input(
                            value=_sampling_override_value(sampling.temperature),
                            name="temperature",
                            type="number",
                            min=GenerationSettings.MIN_TEMPERATURE,
                            max=GenerationSettings.MAX_TEMPERATURE,
                            step="0.05",
                            placeholder=f"Global: {generation_defaults.temperature:g}",
                            disabled=generating,
                            data_testid="active-chat-temperature",
                        ),
                    ),
                    _field(
                        "Top P",
                        Input(
                            value=_sampling_override_value(sampling.top_p),
                            name="top_p",
                            type="number",
                            min=GenerationSettings.MIN_TOP_P,
                            max=GenerationSettings.MAX_TOP_P,
                            step="0.05",
                            placeholder=f"Global: {generation_defaults.top_p:g}",
                            disabled=generating,
                            data_testid="active-chat-top-p",
                        ),
                    ),
                    _field(
                        "Maximum output tokens",
                        Input(
                            value=_sampling_override_value(sampling.max_tokens),
                            name="max_tokens",
                            type="number",
                            min=GenerationSettings.MIN_MAX_TOKENS,
                            step=1,
                            placeholder=f"Global: {generation_defaults.max_tokens:,}",
                            disabled=generating,
                            data_testid="active-chat-max-tokens",
                        ),
                    ),
                    cls="jouzetsu-form-grid is-chat-sampling-grid",
                ),
                Small(
                    "Using global sampling defaults"
                    if sampling == ChatSamplingOverrides()
                    else "Using per-chat sampling overrides",
                    cls="jouzetsu-form-help",
                    data_testid="active-chat-sampling-source",
                ),
                Small(
                    "Leave any value blank to inherit its global default.",
                    cls="jouzetsu-form-help",
                ),
                Button(
                    "Save sampling",
                    type="submit",
                    cls="jouzetsu-button",
                    disabled=generating,
                    data_testid="save-chat-sampling-button",
                ),
                action="/chat/sampling",
                method="post",
                cls="jouzetsu-settings-form",
                data_live_submit="true",
                data_live_notice="Sampling settings saved",
            ),
            H3("Chat system prompt", cls="jouzetsu-section-title"),
            Form(
                _field(
                    "System prompt override",
                    Textarea(
                        chat.system_prompt,
                        name="prompt",
                        rows=6,
                        placeholder="Leave empty to use the global system prompt.",
                        disabled=generating,
                        data_testid="active-chat-prompt",
                    ),
                ),
                Small(
                    "Chat prompt override active"
                    if chat.system_prompt
                    else "Using global system prompt",
                    cls="jouzetsu-form-help",
                    data_testid="chat-prompt-source",
                ),
                Button(
                    "Save prompt",
                    type="submit",
                    cls="jouzetsu-button",
                    disabled=generating,
                    data_testid="save-chat-prompt-button",
                ),
                action="/chat/prompt",
                method="post",
                cls="jouzetsu-settings-form",
            ),
            H3("Writing", cls="jouzetsu-section-title"),
            Form(
                _checkbox(
                    "enabled",
                    checked=chat.postprocess_british_spellings,
                    label="Post-process British spellings",
                    marker="active-chat-british-spellings",
                    disabled=generating,
                    live_autosave=True,
                ),
                action="/chat/british-spellings",
                method="post",
                cls="jouzetsu-settings-form",
                data_live_submit="true",
                data_live_notice="Spelling setting saved",
            ),
            H3("Reasoning", cls="jouzetsu-section-title"),
            Form(
                _checkbox(
                    "enabled",
                    checked=chat.save_reasoning,
                    label="Save reasoning traces",
                    marker="active-chat-save-reasoning",
                    disabled=generating,
                    live_autosave=True,
                ),
                Small(
                    "Stores each response's reasoning with its assistant message.",
                    cls="jouzetsu-form-help",
                ),
                action="/chat/reasoning",
                method="post",
                cls="jouzetsu-settings-form",
                data_live_submit="true",
                data_live_notice="Reasoning setting saved",
            ),
            H3("Chat metadata", cls="jouzetsu-section-title"),
            Div(
                Div(
                    Span("Created", cls="jouzetsu-chat-metadata-label"),
                    Span(
                        _format_chat_timestamp(chat.created_at),
                        cls="jouzetsu-chat-metadata-value",
                        data_testid="chat-created-at",
                    ),
                    cls="jouzetsu-chat-metadata-item",
                ),
                Div(
                    Span("Last active", cls="jouzetsu-chat-metadata-label"),
                    Span(
                        _format_chat_timestamp(chat.updated_at),
                        cls="jouzetsu-chat-metadata-value",
                        data_testid="chat-last-active-at",
                    ),
                    cls="jouzetsu-chat-metadata-item",
                ),
                Div(
                    Span("Messages", cls="jouzetsu-chat-metadata-label"),
                    Span(
                        str(len(chat.messages)),
                        cls="jouzetsu-chat-metadata-value",
                        data_testid="chat-message-count",
                    ),
                    cls="jouzetsu-chat-metadata-item",
                ),
                cls="jouzetsu-chat-metadata",
                data_testid="chat-metadata",
            ),
            H3("Danger zone", cls="jouzetsu-section-title"),
            Form(
                Button(
                    "Delete chat",
                    type="submit",
                    cls="jouzetsu-button jouzetsu-button-danger",
                    data_testid="delete-chat-confirm-button",
                    data_confirm="Delete this chat permanently?",
                ),
                action=f"/chats/{chat.id}/delete",
                method="post",
                cls="jouzetsu-settings-form jouzetsu-danger-zone",
            ),
            cls="jouzetsu-panel-content",
        ),
        id="settings-panel",
        cls="jouzetsu-panel jouzetsu-settings-panel",
        aria_label="Chat settings",
        aria_hidden="true",
        inert=True,
        data_panel_name="settings",
    )


def render_chat_fragment(
    view: ChatView,
    *,
    edit_message_id: str = "",
) -> HTML:
    """Render the active chat surface; this is the SSE refresh target."""

    chat: Chat = view.chat
    editing_message: Message | None = _safe_edit_message(chat, edit_message_id)
    if editing_message is None:
        edit_message_id = ""
    return Div(
        Header(
            Button(
                _icon("menu"),
                type="button",
                cls="jouzetsu-icon-button",
                data_panel_open="navigation",
                data_testid="open-nav-button",
                aria_label="Open chats",
            ),
            H1(chat.title, cls="jouzetsu-title", data_testid="active-chat-title"),
            Button(
                _icon("settings"),
                type="button",
                cls="jouzetsu-icon-button",
                data_panel_open="settings",
                data_testid="open-settings-button",
                aria_label="Open chat settings",
            ),
            cls="jouzetsu-topbar",
        ),
        Main(
            render_message_list(view, editing_message_id=edit_message_id),
            cls="jouzetsu-main-content",
        ),
        render_composer(
            view, edit_message_id=edit_message_id, editing_message=editing_message
        ),
        id="chat-fragment",
        cls="jouzetsu-chat-shell",
        data_chat_id=chat.id,
    )


def render_chat_live_fragment(
    view: ChatView,
    *,
    edit_message_id: str = "",
) -> HTML:
    """Render only the values that can change during a streamed response.

    The browser applies this tiny fragment in place so token updates do not
    recreate the full transcript or restart its entry animations.
    """

    chat: Chat = view.chat
    editing_message: Message | None = _safe_edit_message(chat, edit_message_id)
    status: _ComposerStatus = _composer_status(view, editing_message=editing_message)
    streaming_message: Message | None = _streaming_assistant_message(view)
    reasoning: str = view.generation_reasoning
    streaming_content: HTML | None = None
    if streaming_message is not None and streaming_message.content:
        streaming_content = Span(
            render_markdown(streaming_message.content),
            data_live_streaming_message_id=streaming_message.id,
            data_live_streaming_text=streaming_message.content,
        )
    return Div(
        streaming_content,
        Div(
            data_live_generation_reasoning="true",
            data_live_generation_reasoning_text=reasoning,
        ),
        Div(
            *_render_composer_status_content(status),
            data_live_composer_status="true",
            data_composer_status_stage_elapsed_seconds=_composer_status_elapsed_attribute(
                status.stage_elapsed_seconds
            ),
            data_composer_status_overall_elapsed_seconds=_composer_status_elapsed_attribute(
                status.overall_elapsed_seconds
            ),
        ),
        id="chat-live-fragment",
        data_chat_id=chat.id,
    )


def render_message_list(view: ChatView, *, editing_message_id: str) -> HTML:
    """Render visible conversation messages and their guarded action controls."""

    chat: Chat = view.chat
    visible_messages: list[Message] = [
        message for message in chat.messages if message.role in {"user", "assistant"}
    ]
    if not visible_messages:
        contents: tuple[HTML, ...] = (
            P(
                view.empty_state_message,
                cls="jouzetsu-empty-state",
                data_testid="empty-chat-message",
            ),
        )
    else:
        contents = tuple(
            render_message(
                view, visible_messages, index, editing_message_id=editing_message_id
            )
            for index in range(len(visible_messages))
        )
    return Section(
        *contents,
        id="message-list",
        cls="jouzetsu-messages",
        data_testid="message-list",
    )


def render_message(
    view: ChatView, messages: list[Message], index: int, *, editing_message_id: str
) -> HTML:
    """Render one bubble with only actions valid for the current state."""

    message: Message = messages[index]
    previous: Message | None = messages[index - 1] if index else None
    generating: bool = view.generating
    editing: bool = bool(editing_message_id)
    is_last: bool = index == len(messages) - 1
    is_last_assistant: bool = is_last and message.role == "assistant"
    is_streaming: bool = is_last_assistant and generating
    can_change: bool = not generating and not editing
    can_merge: bool = (
        can_change and previous is not None and previous.role == message.role
    )
    can_resend: bool = (
        can_change
        and message.role == "user"
        and is_last
        and bool(message.content.strip())
    )
    can_continue: bool = (
        can_change and is_last_assistant and bool(message.content.strip())
    )
    classes: list[str] = ["jouzetsu-message", f"is-{message.role}"]
    if is_streaming:
        classes.append("is-streaming")
    if message.id == editing_message_id:
        classes.append("is-editing")

    actions: list[HTML] = []
    if can_change:
        actions.append(
            Button(
                _icon("trash"),
                type="button",
                cls="jouzetsu-message-action is-delete",
                data_delete_choice_open=message.id,
                data_testid=f"delete-message-{message.id}",
                title="Delete this message",
                aria_label="Delete this message",
            )
        )
    if can_change and is_last_assistant:
        actions.append(
            _post_button(
                _icon("refresh"),
                action=f"/messages/{message.id}/regenerate",
                marker=f"regenerate-message-{message.id}",
                classes="jouzetsu-message-action is-regenerate is-action-divider",
                title="Regenerate the last assistant message",
                aria_label="Regenerate the last assistant message",
                chat_mutation=True,
            )
        )
    if can_change and is_last_assistant and message.continuity_rewrite:
        actions.append(
            A(
                _icon("review"),
                href=_query_url(continuity_rewrite=message.id),
                cls="jouzetsu-message-action is-continuity-rewrite",
                data_testid=f"review-continuity-rewrite-{message.id}",
                title="Review the proposed continuity rewrite",
                aria_label="Review the proposed continuity rewrite",
            )
        )
    if can_merge:
        actions.append(
            _post_button(
                _icon("merge"),
                action=f"/messages/{message.id}/merge",
                marker=f"merge-message-{message.id}",
                classes="jouzetsu-message-action is-merge",
                title="Merge with the previous message",
                aria_label="Merge with the previous message",
                chat_mutation=True,
            )
        )
    if can_change:
        actions.append(
            A(
                _icon("pencil"),
                href=_query_url(edit=message.id),
                cls="jouzetsu-message-action is-edit",
                data_testid=f"edit-message-{message.id}",
                title="Edit this message",
                aria_label="Edit this message",
                data_chat_navigation="true",
            )
        )
    if can_resend:
        actions.append(
            _post_button(
                _icon("play"),
                action=f"/messages/{message.id}/resend",
                marker=f"resend-message-{message.id}",
                classes="jouzetsu-message-action is-resend",
                title="Resend this message",
                aria_label="Resend this message",
                chat_mutation=True,
            )
        )
    if can_continue:
        actions.append(
            _post_button(
                _icon("play"),
                action=f"/messages/{message.id}/continue",
                marker=f"continue-message-{message.id}",
                classes="jouzetsu-message-action is-continue",
                title="Continue this response",
                aria_label="Continue this response",
                chat_mutation=True,
                message_action="continue",
            )
        )

    text: HTML | str
    if is_streaming and not message.content:
        text = Span(
            Span(cls="jouzetsu-streaming-pending-dot", aria_hidden="true"),
            Span(cls="jouzetsu-streaming-pending-dot", aria_hidden="true"),
            Span(cls="jouzetsu-streaming-pending-dot", aria_hidden="true"),
            cls="jouzetsu-streaming-pending",
            role="status",
            aria_label="Jouzetsu is responding",
            data_streaming_pending="true",
        )
    else:
        text = render_markdown(message.content)
    action_style: MessageActionIconStyle = view.message_action_icon_style
    action_classes: str = "jouzetsu-message-actions"
    if action_style == "muted_color":
        action_classes += " jouzetsu-action-icons-muted"
    return Article(
        Div(
            Div(
                text,
                cls="jouzetsu-message-text",
                data_streaming_message_id=message.id if is_streaming else None,
                data_streaming_text=message.content if is_streaming else None,
            ),
            Div(
                Small(
                    cls="jouzetsu-message-updated-at",
                    data_message_updated_at=message.updated_at,
                    data_testid=f"message-updated-at-{message.id}",
                ),
                Div(*actions, cls=action_classes),
                cls="jouzetsu-message-footer",
            ),
            cls="jouzetsu-message-surface",
        ),
        cls=" ".join(classes),
        data_message_id=message.id,
    )


def render_composer(
    view: ChatView, *, edit_message_id: str, editing_message: Message | None
) -> HTML:
    """Render the compact shared composer in normal or explicit edit mode."""

    generating: bool = view.generating
    editing: bool = editing_message is not None
    content: str = (
        editing_message.content if editing_message is not None else view.chat.draft
    )
    status: _ComposerStatus = _composer_status(view, editing_message=editing_message)
    action: str
    primary_label: str
    primary_content: HTML
    primary_marker: str
    primary_classes: str
    primary_disabled: bool
    primary_title: str
    if generating:
        action = "/chat/stop"
        primary_label = "Stop generating"
        primary_content = _icon("stop")
        primary_marker = "stop-generation-button"
        primary_classes = (
            "jouzetsu-button jouzetsu-button-danger jouzetsu-composer-primary is-icon"
        )
        primary_disabled = False
        primary_title = primary_label
    elif editing:
        action = "/messages/edit"
        primary_label = "Save edit"
        primary_content = _icon("save")
        primary_marker = "save-edit-button"
        primary_classes = (
            "jouzetsu-button jouzetsu-button-primary jouzetsu-composer-primary is-icon"
        )
        primary_disabled = not content.strip()
        primary_title = f"{primary_label} ({_COMPOSER_SUBMIT_SHORTCUT_HINT})"
    else:
        action = "/chat/send"
        primary_label = "Send"
        primary_content = _icon("send")
        primary_marker = "send-message-button"
        primary_classes = (
            "jouzetsu-button jouzetsu-button-primary jouzetsu-composer-primary is-icon"
        )
        primary_disabled = not content.strip()
        primary_title = f"{primary_label} ({_COMPOSER_SUBMIT_SHORTCUT_HINT})"
    input_description: str = (
        "composer-status composer-edit-validation" if editing else "composer-status"
    )
    message_input: HTML = Textarea(
        content,
        name="content",
        rows=1,
        placeholder="Message Jouzetsu…",
        required=not generating,
        cls="jouzetsu-message-input" + (" is-editing" if editing else ""),
        aria_describedby=input_description,
        aria_errormessage="composer-edit-validation" if editing else None,
        aria_invalid="false" if editing else None,
        aria_keyshortcuts=_COMPOSER_SUBMIT_ARIA_SHORTCUTS if not generating else None,
        data_composer_input="true",
        data_draft_sync="true" if not editing else None,
        data_testid="message-input",
    )
    suggestion_buttons: list[HTML] = []
    if not generating and not editing:
        suggestion_buttons = [
            Button(
                prompt.label,
                type="button",
                cls="jouzetsu-suggestion-button",
                data_suggestion=prompt.content,
                data_testid=f"composer-suggestion-{index}",
            )
            for index, prompt in enumerate(view.starter_prompts)
        ]

    hidden_edit: HTML | None = (
        Input(type="hidden", name="message_id", value=edit_message_id)
        if editing
        else None
    )
    cancel_control: HTML | None = None
    if editing:
        cancel_control = A(
            "Cancel",
            href="/",
            cls="jouzetsu-button jouzetsu-button-muted",
            data_testid="cancel-edit-button",
            data_chat_navigation="true",
        )

    status_classes: str = "jouzetsu-composer-status"
    if generating:
        status_classes += " jouzetsu-generation-status"
    status_display: HTML = Div(
        *_render_composer_status_content(status),
        id="composer-status",
        cls=status_classes,
        aria_live="polite",
        data_composer_status="true",
        data_composer_status_stage_elapsed_seconds=_composer_status_elapsed_attribute(
            status.stage_elapsed_seconds
        ),
        data_composer_status_overall_elapsed_seconds=_composer_status_elapsed_attribute(
            status.overall_elapsed_seconds
        ),
        data_testid="composer-action-status",
    )
    footer: HTML
    if generating:
        footer = Div(
            status_display,
            cls="jouzetsu-composer-footer jouzetsu-generation-strip",
        )
    else:
        edit_validation: HTML | None = None
        if editing:
            edit_validation = Span(
                "Message cannot be empty.",
                id="composer-edit-validation",
                cls="jouzetsu-composer-validation",
                role="alert",
                hidden=True,
                data_composer_validation="true",
            )
        footer = Div(
            Div(
                status_display,
                edit_validation,
                cls="jouzetsu-composer-status-stack",
            ),
            Div(*suggestion_buttons, cancel_control, cls="jouzetsu-composer-actions"),
            cls="jouzetsu-composer-footer",
        )

    return Footer(
        Div(
            cls="jouzetsu-composer-resize-handle",
            role="separator",
            tabindex="0",
            aria_label="Resize composer",
            data_testid="composer-resize-handle",
        ),
        _render_generation_reasoning(view.generation_reasoning),
        Form(
            Div(
                message_input,
                Button(
                    primary_content,
                    type="submit",
                    cls=primary_classes,
                    disabled=primary_disabled,
                    title=primary_title,
                    aria_label=primary_label,
                    data_composer_primary="true",
                    data_testid=primary_marker,
                ),
                cls="jouzetsu-composer-input-row",
            ),
            hidden_edit,
            action=action,
            method="post",
            cls="jouzetsu-composer-form",
            data_chat_id=view.chat.id,
            data_composer_generating="true" if generating else "false",
            data_composer_mode="edit" if editing else "compose",
            data_chat_mutation="true",
        ),
        footer,
        cls="jouzetsu-composer",
    )


def _render_generation_reasoning(reasoning: str) -> HTML:
    """Render an expandable, live-only view of the model's reasoning trace."""

    return Details(
        Summary("Thinking", cls="jouzetsu-generation-reasoning-summary"),
        Pre(reasoning, cls="jouzetsu-generation-reasoning-content"),
        id="generation-reasoning",
        cls="jouzetsu-generation-reasoning",
        hidden=True if not reasoning else None,
        data_generation_reasoning="true",
        data_testid="generation-reasoning",
    )


def _format_runtime_status(status: RuntimeStatus) -> _ComposerStatus:
    """Render the typed runtime phase as state, detail, and independently ticking timers."""

    model_name: str = status.model_name or status.model_key or "model"
    stage_started_at: float | None = (
        status.phase_started_at
        if status.phase_started_at is not None
        else status.started_at
    )
    stage_elapsed_seconds: float | None = _elapsed_seconds(stage_started_at)
    overall_elapsed_seconds: float | None = _elapsed_seconds(status.started_at)
    progress: str = (
        f"{round(status.progress * 100):d}%"
        if status.progress is not None
        else "in progress"
    )

    def timed_status(state: _ComposerStatusState, detail: str = "") -> _ComposerStatus:
        return _ComposerStatus(
            state, detail, stage_elapsed_seconds, overall_elapsed_seconds
        )

    if status.phase is RuntimePhase.CHECKING:
        return _ComposerStatus(_ComposerStatusState.CHECKING, "LM Studio")
    if status.phase is RuntimePhase.OFFLINE:
        return _ComposerStatus(_ComposerStatusState.OFFLINE, "LM Studio unavailable")
    if status.phase is RuntimePhase.NO_MODEL_SELECTED:
        return _ComposerStatus(_ComposerStatusState.NO_MODEL, "Select a model")
    if status.phase is RuntimePhase.MODEL_UNLOADED:
        return _ComposerStatus(
            _ComposerStatusState.UNLOADED, _status_detail(model_name, status.detail)
        )
    if status.phase is RuntimePhase.STARTING:
        return timed_status(
            _ComposerStatusState.GENERATING,
            _status_detail(f"Preparing {model_name}", status.detail),
        )
    if status.phase is RuntimePhase.LOADING_MODEL:
        return timed_status(
            _ComposerStatusState.LOADING, _status_detail(model_name, progress)
        )
    if status.phase is RuntimePhase.PROCESSING_PROMPT:
        return timed_status(
            _ComposerStatusState.GENERATING,
            _status_detail("Processing prompt", progress),
        )
    if status.phase is RuntimePhase.REASONING:
        token_detail: str = (
            f"{status.output_tokens} tokens" if status.output_tokens else ""
        )
        return timed_status(_ComposerStatusState.REASONING, token_detail)
    if status.phase is RuntimePhase.GENERATING:
        token_detail = f"{status.output_tokens} tokens" if status.output_tokens else ""
        return timed_status(_ComposerStatusState.GENERATING, token_detail)
    if status.phase is RuntimePhase.REVIEWING:
        return timed_status(
            _ComposerStatusState.GENERATING, "Reviewing response continuity"
        )
    if status.phase is RuntimePhase.STOPPING:
        return timed_status(_ComposerStatusState.GENERATING, "Stopping")
    if status.phase is RuntimePhase.ERROR:
        detail: str = " ".join(status.detail.split())
        return _ComposerStatus(_ComposerStatusState.ERROR, detail[:72])
    return _ComposerStatus(
        _ComposerStatusState.READY, _ready_status_detail(model_name, status.metrics)
    )


def _ready_status_detail(model_name: str, metrics: GenerationMetrics | None) -> str:
    """Return the informative middle-column content for a ready model."""

    details: list[str] = [model_name]
    if metrics is not None:
        if metrics.tokens_per_second is not None:
            details.append(f"{metrics.tokens_per_second:.1f} tok/s")
        if metrics.output_tokens is not None:
            details.append(f"{metrics.output_tokens} tokens")
        if metrics.time_to_first_token_seconds is not None:
            details.append(f"TTFT {metrics.time_to_first_token_seconds:.2f}s")
    return _status_detail(*details)


def _status_detail(*parts: str) -> str:
    """Join non-empty contextual status details in their display order."""

    return " · ".join(part for part in parts if part)


def _elapsed_seconds(started_at: float | None) -> float | None:
    return max(0.0, time.monotonic() - started_at) if started_at is not None else None
