"""Composition roots for Jouzetsu's top-level web workspaces."""

from __future__ import annotations

from ...state import AppState
from ..html import (
    H1,
    H2,
    HTML,
    A,
    Button,
    Div,
    Form,
    Header,
    Img,
    Main,
    P,
    Section,
    Span,
)
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
    render_message_context_menu,
    render_message_details_dialog,
)
from .feedback import render_notice_area, storage_warning
from .global_settings import (
    global_settings_tab_from_text,
    render_global_settings_dialog,
)
from .icons import render_icon

_DIALOG_ELEMENT_ID_BY_NAME: dict[str, str] = {
    "global": "global-settings-dialog",
    "access": "access-dialog",
}


def render_main_page(
    state: AppState,
    chat_view: ChatView,
    chat_settings_view: ChatSettingsView,
    navigation_view: NavigationView,
    context: PageContext,
    *,
    edit_message_id: str = "",
    active_dialog: str = "",
    active_dialog_tab: str = "",
    notice: str = "",
    error: str = "",
) -> HTML:
    """Compose the chat workspace around independently refreshable fragments."""

    dialogs: tuple[HTML, ...] = _workspace_dialogs(
        state,
        context,
        active_dialog=active_dialog,
        active_dialog_tab=active_dialog_tab,
        notice=notice,
        error=error,
    )
    open_dialog: str = _DIALOG_ELEMENT_ID_BY_NAME.get(active_dialog, "")
    return Div(
        csrf_context(context),
        render_notice_area(
            notice="" if active_dialog else notice,
            error="" if active_dialog else error,
            warning=storage_warning(state),
        ),
        render_navigation_panel(navigation_view),
        render_settings_panel(chat_settings_view),
        Div(cls="jouzetsu-panel-backdrop", data_panel_backdrop="true"),
        render_chat_fragment(chat_view, edit_message_id=edit_message_id),
        render_message_context_menu(),
        render_message_details_dialog(),
        *dialogs,
        cls="jouzetsu-app",
        data_open_dialog=open_dialog,
    )


def _workspace_dialogs(
    state: AppState,
    context: PageContext,
    *,
    active_dialog: str,
    active_dialog_tab: str,
    notice: str,
    error: str,
) -> tuple[HTML, ...]:
    dialogs: list[HTML] = []
    if context.decision.can_use_global_settings:
        dialogs.append(
            render_global_settings_dialog(
                state,
                context,
                active_tab=global_settings_tab_from_text(active_dialog_tab),
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
    return tuple(dialogs)


def render_empty_chat_page(
    state: AppState,
    navigation: NavigationView,
    context: PageContext,
    *,
    active_dialog: str = "",
    active_dialog_tab: str = "",
    notice: str = "",
    error: str = "",
) -> HTML:
    """An empty shared library needs an explicit create action, not a GET-time write."""
    return Div(
        csrf_context(context),
        render_notice_area(notice=notice, error=error, warning=storage_warning(state)),
        render_navigation_panel(navigation),
        Div(cls="jouzetsu-panel-backdrop", data_panel_backdrop="true"),
        Div(
            Header(
                Button(
                    render_icon("menu"),
                    type="button",
                    cls="jouzetsu-icon-button",
                    data_panel_open="navigation",
                    data_testid="open-nav-button",
                    aria_label="Open chats",
                ),
                H1("Chats", cls="jouzetsu-title"),
                Button(
                    render_icon("settings"),
                    type="button",
                    cls="jouzetsu-icon-button",
                    data_dialog_open="global-settings-dialog",
                    aria_label="Global settings",
                ),
                cls="jouzetsu-topbar",
            ),
            Main(
                Section(
                    P("No chats yet.", cls="jouzetsu-empty-state"),
                    Form(
                        Button(
                            "New chat",
                            type="submit",
                            cls="jouzetsu-button jouzetsu-button-primary",
                            data_testid="start-first-chat",
                        ),
                        action="/chats/new",
                        method="post",
                    ),
                    cls="jouzetsu-messages",
                ),
                cls="jouzetsu-main-content",
            ),
            cls="jouzetsu-chat-shell",
        ),
        *_workspace_dialogs(
            state,
            context,
            active_dialog=active_dialog,
            active_dialog_tab=active_dialog_tab,
            notice=notice,
            error=error,
        ),
        cls="jouzetsu-app",
        data_open_dialog=_DIALOG_ELEMENT_ID_BY_NAME.get(active_dialog, ""),
    )


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
        render_notice_area(notice=notice, error=error, warning=storage_warning(state)),
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
                Div(
                    _workspace_link(
                        "Chats",
                        "Conversations with your local models.",
                        "/chats",
                        "launch-chats-link",
                    ),
                    _workspace_link(
                        "Characters",
                        "Reusable profiles for character chats.",
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
