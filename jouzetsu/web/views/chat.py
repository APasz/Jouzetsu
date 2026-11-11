"""Renderers and read models for the chat workspace."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Final, Literal
from urllib.parse import urlencode

from ...backend_models import BackendConfiguration, GenerationProfile
from ...backend_protocol import BackendError
from ...config import GenerationSettings, MessageActionStyle, StarterPrompt
from ...markdown import render_markdown
from ...models import Chat, Message
from ...runtime import GenerationMetrics, ReasoningOption, RuntimePhase, RuntimeStatus
from ...sampling import SamplingParameters
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
    scope_chat_forms,
)
from .controls import (
    checkbox as _checkbox,
    count_label,
    field as _field,
    post_button as _post_button,
    reset_form_button as _reset_form_button,
)
from .icons import render_icon as _icon
from .images import (
    render_image_gallery,
    render_image_upload,
    render_image_upload_control,
)
from .model_controls import ModelChoice, render_model_picker


@dataclass(frozen=True, slots=True)
class ChatView:
    """All data required to render one chat workspace and its live fragments."""

    chat: Chat
    generating: bool
    runtime_status: RuntimeStatus
    generation_reasoning: str
    empty_state_message: str
    starter_prompts: tuple[StarterPrompt, ...]
    message_action_style: MessageActionStyle
    generation_block_reason: str = ""
    history_generation_block_reason: str = ""
    live_response: Message | None = None
    replacement_id: str | None = None
    latest_error: BackendError | None = None
    save_failed: bool = False
    draft_write_error: bool = False
    reasoning_options: tuple[ReasoningOption, ...] = ()
    reasoning_default: ReasoningOption | None = None
    reasoning_unavailable: bool = False


@dataclass(frozen=True, slots=True)
class ChatSettingsView:
    """All data required to render settings for one explicitly selected chat."""

    chat: Chat
    generating: bool
    model_choices: tuple[ModelChoice, ...]
    backends: BackendConfiguration
    profile: GenerationProfile
    inactive_parameters: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NavigationView:
    """All data required to render the chat navigation drawer."""

    chats: tuple[Chat, ...]
    selected_chat_id: str
    generating_chat_ids: frozenset[str]
    app_icon_url: str
    can_use_global_settings: bool
    can_manage_access: bool
    current_device_label: str | None


_COMPOSER_SUBMIT_SHORTCUT_HINT: Final[str] = "Shift+Enter / Cmd+Enter"
_COMPOSER_SUBMIT_ARIA_SHORTCUTS: Final[str] = "Shift+Enter Meta+Enter"


@dataclass(frozen=True, slots=True)
class _ComposerStatus:
    """Status-line content shared by full and incremental composer renders."""

    state: _ComposerStatusState
    detail: str = ""
    stage_elapsed_seconds: float | None = None
    overall_elapsed_seconds: float | None = None
    load_progress: float | None = None


class _ComposerStatusState(Enum):
    """Finite state labels displayed in the composer status's left column."""

    QUEUED = "Queued"
    CHECKING = "Checking"
    EDITING = "Editing"
    ERROR = "Error"
    GENERATING = "Generating"
    LOADING = "Loading"
    NO_MODEL = "No model"
    OFFLINE = "Offline"
    PREPARING = "Preparing"
    PROCESSING = "Processing"
    READY = "Ready"
    REASONING = "Reasoning"
    STOPPING = "Stopping"
    UNLOADED = "Unloaded"


def _query_url(chat_id: str, **values: str) -> str:
    """Build the canonical chat-workspace URL while omitting empty query values."""

    query: dict[str, str] = {key: value for key, value in values.items() if value}
    path = f"/chats/{chat_id}"
    return path if not query else f"{path}?{urlencode(query)}"


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

    return view.live_response


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
            aria_live="off",
            data_composer_status_timers="true",
        ),
        _render_composer_status_load_progress(status.load_progress),
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


def _render_composer_status_load_progress(progress: float | None) -> HTML:
    """Render a subtle, accessible load meter when numerical progress is available."""

    percent: float | None = _progress_percent(progress)
    percent_text: str | None = f"{percent:.3f}" if percent is not None else None
    return Span(
        Span(
            cls="jouzetsu-composer-status-progress-fill",
            style=(
                f"--jouzetsu-composer-status-load-progress: {percent_text}%"
                if percent_text is not None
                else None
            ),
            data_composer_status_progress_fill="true",
        ),
        cls="jouzetsu-composer-status-progress",
        hidden=percent is None,
        role="progressbar",
        aria_live="off",
        aria_label="Model loading progress",
        aria_valuemin="0",
        aria_valuemax="100",
        aria_valuenow=percent_text,
        aria_valuetext=(f"{round(percent):d}% loaded" if percent is not None else None),
        data_composer_status_progress="true",
        data_composer_status_load_progress=percent_text,
    )


def _progress_percent(progress: float | None) -> float | None:
    """Convert the runtime's fractional progress value into a percentage."""

    return progress * 100.0 if progress is not None else None


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
            if chat.id == view.selected_chat_id
            else "jouzetsu-chat-item"
        )
        chat_items.append(
            A(
                _icon("activity") if generating else None,
                chat.title,
                href=f"/chats/{chat.id}",
                cls=classes,
                data_testid=f"chat-item-{chat.id}",
                data_chat_navigation="true",
                aria_label=f"{chat.title} (generating)" if generating else None,
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
            Div(
                H3("Chats", cls="jouzetsu-section-title"),
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
                    cls="jouzetsu-inline-form",
                    data_chat_mutation="true",
                ),
                cls="jouzetsu-chats-header",
            ),
            Div(
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
                Span(
                    count_label(len(view.chats), "chat"),
                    cls="jouzetsu-sheet-footnote",
                ),
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
    """Render settings scoped to the selected chat."""

    chat: Chat = view.chat
    generating: bool = view.generating
    sampling: SamplingParameters = view.profile.parameters
    current_model: str = view.profile.model_id or ""
    return scope_chat_forms(
        Aside(
            Div(
                Div(
                    Div(
                        H2("Chat settings", cls="jouzetsu-sheet-title"),
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
                _render_character_cast_source(chat),
                Form(
                    _field(
                        "Title",
                        Input(
                            value=chat.title,
                            name="title",
                            required=True,
                            disabled=generating,
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
                Form(
                    _field(
                        "Backend",
                        Select(
                            Option(
                                "No backend selected",
                                value="",
                                selected=chat.inference.selected_backend_id is None,
                            ),
                            *(
                                Option(
                                    backend.name
                                    + (" (disabled)" if not backend.enabled else ""),
                                    value=backend.id,
                                    selected=chat.inference.selected_backend_id
                                    == backend.id,
                                )
                                for backend in view.backends.backends
                            ),
                            *(
                                Option(
                                    f"Missing backend: {backend_id}",
                                    value=backend_id,
                                    selected=True,
                                )
                                for backend_id in (chat.inference.selected_backend_id,)
                                if backend_id is not None
                                and all(
                                    backend.id != backend_id
                                    for backend in view.backends.backends
                                )
                            ),
                            name="backend_id",
                            disabled=generating,
                            data_live_autosave="true",
                            data_testid="active-chat-backend",
                        ),
                    ),
                    action="/chat/backend",
                    method="post",
                    data_live_submit="true",
                ),
                Form(
                    render_model_picker(
                        view.model_choices,
                        current_model,
                        marker="active-chat-model",
                        disabled=generating,
                        autosave=True,
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
                                value=sampling.temperature,
                                name="temperature",
                                type="number",
                                min=GenerationSettings.MIN_TEMPERATURE,
                                max=GenerationSettings.MAX_TEMPERATURE,
                                step="0.05",
                                required=True,
                                disabled=generating,
                                data_testid="active-chat-temperature",
                            ),
                        ),
                        _field(
                            "Top P",
                            Input(
                                value=sampling.top_p,
                                name="top_p",
                                type="number",
                                min=GenerationSettings.MIN_TOP_P,
                                max=GenerationSettings.MAX_TOP_P,
                                step="0.05",
                                required=True,
                                disabled=generating,
                                data_testid="active-chat-top-p",
                            ),
                        ),
                        _field(
                            "Maximum output tokens",
                            Input(
                                value=sampling.max_tokens,
                                name="max_tokens",
                                type="number",
                                min=GenerationSettings.MIN_MAX_TOKENS,
                                step=1,
                                required=True,
                                disabled=generating,
                                data_testid="active-chat-max-tokens",
                            ),
                        ),
                        cls="jouzetsu-form-grid is-chat-sampling-grid",
                    ),
                    Small(
                        "Saved for this backend in this chat."
                        + (
                            " Inactive parameters: "
                            + ", ".join(view.inactive_parameters)
                            if view.inactive_parameters
                            else ""
                        ),
                        cls="jouzetsu-form-help",
                        data_testid="active-chat-sampling-source",
                    ),
                    Div(
                        Button(
                            "Save sampling",
                            type="submit",
                            cls="jouzetsu-button",
                            disabled=generating,
                            data_testid="save-chat-sampling-button",
                        ),
                        _reset_form_button(
                            action="/chat/sampling/reset",
                            marker="reset-chat-sampling-button",
                            label="Reset to backend defaults",
                            confirmation="Reset this chat's profile to backend defaults?",
                            disabled=generating,
                            live_notice="Profile reset to backend defaults",
                        ),
                        cls="jouzetsu-dialog-actions",
                    ),
                    action="/chat/sampling",
                    method="post",
                    cls="jouzetsu-settings-form",
                    data_live_submit="true",
                    data_live_notice="Sampling settings saved",
                ),
                H3("System prompt", cls="jouzetsu-section-title"),
                Form(
                    _field(
                        "Chat override",
                        Textarea(
                            chat.system_prompt,
                            name="prompt",
                            rows=6,
                            placeholder="Instructions for this chat…",
                            disabled=generating,
                            data_testid="active-chat-prompt",
                        ),
                    ),
                    Small(
                        "Leave blank to use the global prompt.",
                        cls="jouzetsu-form-help",
                        data_testid="chat-prompt-source",
                    ),
                    Div(
                        Button(
                            "Save prompt",
                            type="submit",
                            cls="jouzetsu-button",
                            disabled=generating,
                            data_testid="save-chat-prompt-button",
                        ),
                        _reset_form_button(
                            action="/chat/prompt/reset",
                            marker="reset-chat-prompt-button",
                            label="Use global default",
                            confirmation="Reset this chat's prompt to the global default?",
                            disabled=generating,
                        ),
                        cls="jouzetsu-dialog-actions",
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
                        label="Save model reasoning",
                        marker="active-chat-save-reasoning",
                        disabled=generating,
                        live_autosave=True,
                    ),
                    Small(
                        "View saved reasoning in Message details.",
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
                        disabled=generating,
                        data_confirm="Delete this chat? Its file will be moved to recoverable trash.",
                    ),
                    action=f"/chats/{chat.id}/delete",
                    method="post",
                    cls="jouzetsu-settings-form jouzetsu-danger-zone",
                ),
                cls="jouzetsu-panel-content",
            ),
            id="settings-panel",
            data_chat_id=chat.id,
            cls="jouzetsu-panel jouzetsu-settings-panel",
            aria_label="Chat settings",
            aria_hidden="true",
            inert=True,
            data_panel_name="settings",
        ),
        view.chat,
    )


def _render_character_cast_source(chat: Chat) -> HTML | None:
    """Render the immutable profile revisions that supplied this chat's prompt."""

    if not chat.character_cast:
        return None
    member_sources: tuple[HTML, ...] = tuple(
        Div(
            Span(
                f"{binding.name} (revision {binding.revision})",
                cls="jouzetsu-form-help",
            ),
            A(
                "Open character",
                href=f"/characters/{binding.id}",
                cls="jouzetsu-settings-character-link",
            ),
            cls="jouzetsu-settings-character-cast-member",
        )
        for binding in chat.character_cast
    )
    return Div(
        Span(
            "Character" if len(chat.character_cast) == 1 else "Cast",
            cls="jouzetsu-form-help",
        ),
        (
            Span(
                f"Started as: {chat.prompt_mode.label}",
                cls="jouzetsu-form-help",
            )
            if chat.prompt_mode is not None
            else None
        ),
        Div(
            *member_sources,
            cls="jouzetsu-settings-character-cast-members",
        ),
        cls="jouzetsu-settings-character-source",
    )


def render_chat_fragment(
    view: ChatView,
    *,
    edit_message_id: str = "",
) -> HTML:
    """Render the selected chat surface; this is the SSE refresh target."""

    chat: Chat = view.chat
    editing_message: Message | None = _safe_edit_message(chat, edit_message_id)
    if editing_message is None:
        edit_message_id = ""
    return scope_chat_forms(
        Div(
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
            data_revision=chat.revision,
            data_draft_revision=chat.draft_revision,
        ),
        view.chat,
    )


def render_chat_composer_fragment(view: ChatView, *, edit_message_id: str = "") -> HTML:
    """Refresh the shared draft without rendering committed transcript messages."""
    chat = view.chat
    editing = _safe_edit_message(chat, edit_message_id)
    return scope_chat_forms(
        Div(
            render_composer(
                view,
                edit_message_id=editing.id if editing else "",
                editing_message=editing,
            ),
            id="chat-composer-fragment",
            data_chat_id=chat.id,
            data_revision=chat.revision,
            data_draft_revision=chat.draft_revision,
        ),
        chat,
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
        data_revision=chat.revision,
        data_draft_revision=chat.draft_revision,
    )


def render_message_list(view: ChatView, *, editing_message_id: str) -> HTML:
    """Render visible conversation messages and their guarded action controls."""

    chat: Chat = view.chat
    visible_messages: list[Message] = [
        message
        for message in chat.messages
        if message.role in {"user", "assistant"}
        and (view.live_response is None or message.id != view.replacement_id)
    ]
    if view.live_response is not None:
        visible_messages.append(view.live_response)
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
        Article(
            P(
                "A draft could not be saved to disk. The shared draft remains available in memory."
            ),
            _post_button(
                "Retry draft save",
                action="/chat/retry-draft-save",
                marker="retry-draft-save",
                chat_mutation=True,
            ),
            cls="jouzetsu-message is-system jouzetsu-generation-error",
            data_message_id="draft-error",
            role="alert",
        )
        if view.draft_write_error
        else None,
        Article(
            P(str(view.latest_error)),
            _post_button(
                "Dismiss",
                action="/chat/dismiss-error",
                marker="dismiss-generation-error",
                chat_mutation=True,
            ),
            cls="jouzetsu-message is-system jouzetsu-generation-error",
            data_message_id="generation-error",
            role="alert",
        )
        if view.latest_error
        else None,
        Article(
            P("The generated response could not be saved. It remains available here."),
            _post_button(
                "Retry save",
                action="/chat/retry-save",
                marker="retry-generation-save",
                chat_mutation=True,
            ),
            cls="jouzetsu-message is-system jouzetsu-generation-error",
            data_message_id="save-error",
            role="alert",
        )
        if view.save_failed
        else None,
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
    is_streaming: bool = (
        view.live_response is not None and message.id == view.live_response.id
    )
    can_change: bool = not generating and not editing
    can_merge: bool = (
        can_change and previous is not None and previous.role == message.role
    )
    can_resend: bool = (
        can_change
        and message.role == "user"
        and is_last
        and bool(message.content.strip() or message.images)
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
            _post_button(
                _icon("trash"),
                action=f"/messages/{message.id}/delete",
                marker=f"delete-message-{message.id}",
                classes="jouzetsu-message-action is-delete",
                title="Delete this message",
                aria_label="Delete this message",
                chat_mutation=True,
            )
        )
        if not is_last:
            actions.append(
                _post_button(
                    _icon("scissors"),
                    title="Truncate after this message",
                    aria_label="Truncate after this message",
                    action=f"/messages/{message.id}/truncate",
                    marker=f"truncate-{message.id}",
                    classes="jouzetsu-message-action is-delete",
                    confirm="Remove all messages after this one?",
                    chat_mutation=True,
                )
            )
    if can_change and is_last_assistant:
        actions.append(
            _post_button(
                _icon("refresh"),
                action=f"/messages/{message.id}/regenerate",
                marker=f"regenerate-message-{message.id}",
                classes="jouzetsu-message-action is-regenerate",
                title="Regenerate the last assistant message",
                disabled=bool(view.history_generation_block_reason),
                aria_label="Regenerate the last assistant message",
                chat_mutation=True,
            )
        )
    if actions:
        actions.append(Span(cls="jouzetsu-message-action-divider", aria_hidden="true"))
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
                href=_query_url(view.chat.id, edit=message.id),
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
                disabled=bool(view.history_generation_block_reason),
                aria_label="Resend this message",
                chat_mutation=True,
            )
        )
    if can_continue:
        actions.append(
            _post_button(
                _icon("play", armed_name="play-bold"),
                action=f"/messages/{message.id}/continue",
                marker=f"continue-message-{message.id}",
                classes="jouzetsu-message-action is-continue",
                title="Continue this response",
                disabled=bool(view.history_generation_block_reason),
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
    action_classes: str = "jouzetsu-message-actions"
    if view.message_action_style is MessageActionStyle.SEMANTIC:
        action_classes += " is-semantic"

    return Article(
        Div(
            render_image_gallery(message.images) if message.images else None,
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
    has_images: bool = bool(
        editing_message.images if editing_message else view.chat.draft_images
    )
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
        primary_disabled = not content.strip() and not has_images
        primary_title = f"{primary_label} ({_COMPOSER_SUBMIT_SHORTCUT_HINT})"
    else:
        action = "/chat/send"
        primary_label = "Send"
        primary_content = _icon("send")
        primary_marker = "send-message-button"
        primary_classes = (
            "jouzetsu-button jouzetsu-button-primary jouzetsu-composer-primary is-icon"
        )
        primary_disabled = not content.strip() and not has_images
        primary_title = f"{primary_label} ({_COMPOSER_SUBMIT_SHORTCUT_HINT})"
    input_description: str = (
        "composer-status composer-edit-validation" if editing else "composer-status"
    )
    message_input: HTML = Textarea(
        content,
        name="content",
        rows=1,
        placeholder="Message Jouzetsu…",
        aria_label="Edit message" if editing else "Message",
        required=not generating and not has_images,
        cls="jouzetsu-message-input" + (" is-editing" if editing else ""),
        aria_describedby=input_description,
        aria_errormessage="composer-edit-validation" if editing else None,
        aria_invalid="false" if editing else None,
        aria_keyshortcuts=_COMPOSER_SUBMIT_ARIA_SHORTCUTS if not generating else None,
        data_composer_input="true",
        data_draft_sync="true" if not editing else None,
        data_testid="message-input",
    )
    hidden_edit: HTML | None = (
        Input(type="hidden", name="message_id", value=edit_message_id)
        if editing
        else None
    )
    cancel_control: HTML | None = None
    if editing:
        cancel_control = A(
            "Cancel",
            href=_query_url(view.chat.id),
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
            Div(cancel_control, cls="jouzetsu-composer-actions")
            if cancel_control is not None
            else None,
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
        (
            render_image_gallery(editing_message.images)
            if editing_message.images
            else None
        )
        if editing_message is not None
        else render_image_upload(view.chat),
        P(
            view.generation_block_reason,
            role="status",
            cls="jouzetsu-image-model-notice",
            data_testid="vision-model-required",
        )
        if view.generation_block_reason and not editing
        else None,
        P(
            "Previous reasoning choice is unavailable; using Automatic.",
            role="status",
            cls="jouzetsu-image-model-notice",
            data_testid="reasoning-unavailable",
        )
        if view.reasoning_unavailable
        else None,
        Form(
            action="/chat/reasoning-level",
            method="post",
            id="chat-reasoning-form",
            data_chat_mutation="true",
        )
        if view.reasoning_options and not editing
        else None,
        Form(
            Div(
                _render_composer_menu(
                    generating=generating,
                    send_disabled=primary_disabled,
                    starter_prompts=view.starter_prompts,
                )
                if not editing
                else None,
                _render_reasoning_selector(view)
                if view.reasoning_options and not editing
                else None,
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
            Input(type="hidden", name="chat_id", value=view.chat.id),
            Input(type="hidden", name="submission_id", value=""),
            Input(
                type="hidden",
                name="images",
                value=json.dumps([image.to_dict() for image in view.chat.draft_images]),
            ),
            action=action,
            method="post",
            cls="jouzetsu-composer-form",
            data_chat_id=view.chat.id,
            data_composer_generating="true" if generating else "false",
            data_composer_mode="edit" if editing else "compose",
            data_has_images="true" if has_images else "false",
            data_generation_blocked="false",
            data_chat_mutation="true",
        ),
        footer,
        cls="jouzetsu-composer",
    )


def _render_reasoning_selector(view: ChatView) -> HTML:
    profile = view.chat.inference.selected_profile
    selected = profile.reasoning if profile else "model_default"
    available = ("model_default", "automatic", *view.reasoning_options)
    invalid = view.reasoning_unavailable or selected not in available
    if invalid:
        selected = "automatic"
    effective = view.reasoning_default if selected == "model_default" else selected
    state: Literal["automatic", "off", "on"] = (
        "automatic"
        if effective in (None, "automatic")
        else "off"
        if effective == "off"
        else "on"
    )
    label = (
        f"Model default ({effective.title()})"
        if selected == "model_default" and effective is not None
        else "Model default (Automatic)"
        if selected == "model_default"
        else selected.title()
    )
    return Details(
        Summary(
            _icon("lightbulb"),
            cls="jouzetsu-button jouzetsu-composer-menu-toggle",
            aria_label=f"Reasoning level: {label}",
            title=f"Reasoning level: {label}",
            data_reasoning_state=state,
            data_testid="reasoning-level-toggle",
        ),
        Div(
            *(
                Button(
                    label,
                    type="submit",
                    name="reasoning_level",
                    value=value,
                    form="chat-reasoning-form",
                    disabled=view.generating,
                    cls="jouzetsu-button jouzetsu-composer-menu-item",
                    aria_pressed="true" if value == selected else "false",
                )
                for value, label in (
                    ("model_default", "Model default"),
                    ("automatic", "Automatic"),
                    *((option, option.title()) for option in view.reasoning_options),
                )
            ),
            cls="jouzetsu-composer-menu-options jouzetsu-reasoning-options",
        ),
        cls="jouzetsu-composer-menu",
        data_reasoning_menu="true",
    )


def _render_composer_menu(
    *,
    generating: bool,
    send_disabled: bool,
    starter_prompts: tuple[StarterPrompt, ...],
) -> HTML:
    """Group secondary actions behind one compact, keyboard-accessible disclosure."""

    return Details(
        Summary(
            _icon("plus"),
            cls="jouzetsu-button jouzetsu-composer-menu-toggle",
            aria_label="Message actions",
            title="Message actions",
            data_testid="composer-menu-toggle",
        ),
        Div(
            render_image_upload_control(),
            Button(
                _icon("send"),
                "Send only",
                type="submit",
                name="generate",
                value="false",
                disabled=send_disabled,
                hidden=generating,
                data_send_only="true",
                data_testid="send-only-button",
                cls="jouzetsu-button jouzetsu-composer-menu-item",
            ),
            Div(
                Small("Quick templates", cls="jouzetsu-composer-menu-label"),
                *(
                    Button(
                        prompt.label,
                        type="button",
                        cls="jouzetsu-button jouzetsu-composer-menu-item",
                        data_suggestion=prompt.content,
                        data_testid=f"composer-suggestion-{index}",
                    )
                    for index, prompt in enumerate(starter_prompts)
                ),
                cls="jouzetsu-composer-templates",
                role="group",
                aria_label="Quick templates",
            )
            if starter_prompts and not generating
            else None,
            role="group",
            aria_label="Message actions",
            cls="jouzetsu-composer-menu-options",
        ),
        cls="jouzetsu-composer-menu",
        data_composer_menu="true",
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
    progress_percent: float | None = _progress_percent(status.progress)
    progress: str = (
        f"{round(progress_percent):d}%" if progress_percent is not None else ""
    )

    def timed_status(
        state: _ComposerStatusState,
        detail: str = "",
        *,
        load_progress: float | None = None,
    ) -> _ComposerStatus:
        return _ComposerStatus(
            state=state,
            detail=detail,
            stage_elapsed_seconds=stage_elapsed_seconds,
            overall_elapsed_seconds=overall_elapsed_seconds,
            load_progress=load_progress,
        )

    if status.phase is RuntimePhase.CHECKING:
        return _ComposerStatus(_ComposerStatusState.CHECKING, "Backend")
    if status.phase is RuntimePhase.OFFLINE:
        return _ComposerStatus(
            _ComposerStatusState.OFFLINE, status.detail or "Backend unavailable"
        )
    if status.phase is RuntimePhase.NO_MODEL_SELECTED:
        return _ComposerStatus(
            _ComposerStatusState.NO_MODEL, status.detail or "Select a model"
        )
    if status.phase is RuntimePhase.MODEL_UNLOADED:
        return _ComposerStatus(
            _ComposerStatusState.UNLOADED, _status_detail(model_name, status.detail)
        )
    if status.phase is RuntimePhase.STARTING:
        return timed_status(
            _ComposerStatusState.PREPARING,
            _status_detail(model_name, status.detail),
        )
    if status.phase is RuntimePhase.LOADING_MODEL:
        return timed_status(
            _ComposerStatusState.LOADING,
            _status_detail(model_name, progress),
            load_progress=status.progress,
        )
    if status.phase is RuntimePhase.PROCESSING_PROMPT:
        return timed_status(
            _ComposerStatusState.PROCESSING, _status_detail("Prompt", progress)
        )
    if status.phase is RuntimePhase.REASONING:
        token_detail: str = (
            count_label(status.output_tokens, "token") if status.output_tokens else ""
        )
        return timed_status(_ComposerStatusState.REASONING, token_detail)
    if status.phase is RuntimePhase.GENERATING:
        token_detail = (
            count_label(status.output_tokens, "token") if status.output_tokens else ""
        )
        return timed_status(_ComposerStatusState.GENERATING, token_detail)
    if status.phase is RuntimePhase.QUEUED:
        return timed_status(_ComposerStatusState.QUEUED)
    if status.phase is RuntimePhase.STOPPING:
        return timed_status(_ComposerStatusState.STOPPING)
    if status.phase is RuntimePhase.ERROR:
        detail: str = " ".join(status.detail.split())
        return _ComposerStatus(_ComposerStatusState.ERROR, detail[:72])
    if status.phase is RuntimePhase.READY:
        return _ComposerStatus(
            _ComposerStatusState.READY, _ready_status_detail(status.metrics)
        )
    raise ValueError(f"unsupported runtime phase: {status.phase!r}")


def _ready_status_detail(metrics: GenerationMetrics | None) -> str:
    """Return completion metrics for the ready-status middle column."""

    details: list[str] = []
    if metrics is not None:
        if metrics.tokens_per_second is not None:
            details.append(f"{metrics.tokens_per_second:.1f} tok/s")
        if metrics.output_tokens is not None:
            details.append(count_label(metrics.output_tokens, "token"))
        if metrics.time_to_first_token_seconds is not None:
            details.append(f"TTFT {metrics.time_to_first_token_seconds:.2f}s")
    return _status_detail(*details)


def _status_detail(*parts: str) -> str:
    """Join non-empty contextual status details in their display order."""

    return " · ".join(part for part in parts if part)


def _elapsed_seconds(started_at: float | None) -> float | None:
    return max(0.0, time.monotonic() - started_at) if started_at is not None else None
