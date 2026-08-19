"""Composition roots for Jouzetsu's top-level web workspaces."""

from __future__ import annotations

from ...models import Message
from ...state import AppState
from ..html import H1, H2, HTML, A, Div, Header, Img, Main, P, Section, Span
from ..icon import app_icon_url
from .access import render_access_dialog
from .chat import (
    ChatSettingsView,
    ChatView,
    NavigationView,
    render_chat_fragment,
    render_navigation_panel,
    render_settings_panel,
)
from .context import PageContext
from .controls import csrf_context
from .dialogs import (
    render_continuity_rewrite_dialog,
    render_host_stats_dialog,
    render_message_context_menu,
    render_message_delete_dialog,
    render_message_details_dialog,
    render_models_dialog,
)
from .feedback import render_notice_area
from .global_settings import render_global_settings_dialog

_DIALOG_ELEMENT_ID_BY_NAME: dict[str, str] = {
    "global": "global-settings-dialog",
    "access": "access-dialog",
    "models": "models-dialog",
    "host": "host-stats-dialog",
}


def render_main_page(
    state: AppState,
    chat_view: ChatView,
    chat_settings_view: ChatSettingsView,
    navigation_view: NavigationView,
    context: PageContext,
    *,
    edit_message_id: str = "",
    continuity_rewrite_message_id: str = "",
    active_dialog: str = "",
    notice: str = "",
    error: str = "",
    undo_token: str = "",
) -> HTML:
    """Compose the chat workspace around independently refreshable fragments."""

    continuity_rewrite: Message | None = _continuity_rewrite_message(
        chat_view.chat.messages,
        continuity_rewrite_message_id,
    )
    dialogs: tuple[HTML, ...] = _workspace_dialogs(
        state,
        context,
        active_dialog=active_dialog,
        notice=notice,
        error=error,
        continuity_rewrite=continuity_rewrite,
    )
    open_dialog: str = _DIALOG_ELEMENT_ID_BY_NAME.get(active_dialog, "")
    if not open_dialog and continuity_rewrite is not None:
        open_dialog = "continuity-rewrite-dialog"
    return Div(
        csrf_context(context),
        render_notice_area(
            notice="" if active_dialog else notice,
            error="" if active_dialog else error,
            undo_token=undo_token if not active_dialog else "",
        ),
        render_navigation_panel(navigation_view),
        render_settings_panel(chat_settings_view),
        Div(cls="jouzetsu-panel-backdrop", data_panel_backdrop="true"),
        render_chat_fragment(chat_view, edit_message_id=edit_message_id),
        render_message_context_menu(),
        render_message_delete_dialog(),
        render_message_details_dialog(),
        *dialogs,
        cls="jouzetsu-app",
        data_open_dialog=open_dialog,
    )


def _continuity_rewrite_message(
    messages: list[Message], message_id: str
) -> Message | None:
    """Return the final assistant message when it has a reviewed rewrite available."""

    if not message_id or not messages:
        return None
    message: Message | None = next(
        (message for message in messages if message.id == message_id), None
    )
    if (
        message is None
        or message is not messages[-1]
        or message.role != "assistant"
        or not message.continuity_rewrite
    ):
        return None
    return message


def _workspace_dialogs(
    state: AppState,
    context: PageContext,
    *,
    active_dialog: str,
    notice: str,
    error: str,
    continuity_rewrite: Message | None,
) -> tuple[HTML, ...]:
    dialogs: list[HTML] = []
    if context.decision.can_use_global_settings:
        dialogs.append(
            render_global_settings_dialog(
                state,
                context,
                notice=notice if active_dialog == "global" else "",
                error=error if active_dialog == "global" else "",
            )
        )
    if context.decision.can_manage_access:
        dialogs.append(
            render_access_dialog(
                state,
                context,
                notice=notice if active_dialog == "access" else "",
                error=error if active_dialog == "access" else "",
            )
        )
    if active_dialog == "models" and context.decision.can_use_global_settings:
        dialogs.append(render_models_dialog(state, notice=notice, error=error))
    elif active_dialog == "host" and context.decision.can_use_global_settings:
        dialogs.append(render_host_stats_dialog(state, notice=notice, error=error))
    if continuity_rewrite is not None:
        dialogs.append(render_continuity_rewrite_dialog(continuity_rewrite))
    return tuple(dialogs)


def render_launch_page(
    state: AppState,
    context: PageContext,
    *,
    notice: str = "",
    error: str = "",
) -> HTML:
    """Render the protected workspace launcher served at the application root."""

    return Div(
        csrf_context(context),
        render_notice_area(notice=notice, error=error),
        Header(
            Div(
                Img(
                    src=app_icon_url(state.config.ui.icon_colors),
                    alt="",
                    cls="jouzetsu-workspace-brand-icon",
                ),
                Span("Jouzetsu", cls="jouzetsu-workspace-brand"),
                cls="jouzetsu-workspace-brand-lockup",
            ),
            cls="jouzetsu-workspace-topbar",
        ),
        Main(
            Section(
                H1("Choose a workspace", cls="jouzetsu-launch-title"),
                P(
                    "Continue an existing conversation or build a reusable character profile.",
                    cls="jouzetsu-character-intro",
                ),
                Div(
                    _workspace_link(
                        "Chats",
                        "Conversations, models, prompts, and generation controls.",
                        "/chats",
                        "launch-chats-link",
                    ),
                    _workspace_link(
                        "Characters",
                        "Field-configured reusable profiles that can start a new chat.",
                        "/characters",
                        "launch-characters-link",
                    ),
                    cls="jouzetsu-launch-card-grid",
                ),
                cls="jouzetsu-launch-content",
            ),
            cls="jouzetsu-launch-main",
        ),
        cls="jouzetsu-app jouzetsu-launch-app",
    )


def _workspace_link(title: str, description: str, href: str, marker: str) -> HTML:
    return A(
        H2(title, cls="jouzetsu-launch-card-title"),
        P(description, cls="jouzetsu-launch-card-copy"),
        href=href,
        cls="jouzetsu-launch-card",
        data_testid=marker,
    )
