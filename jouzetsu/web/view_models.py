"""Read-model builders for the web views."""

from __future__ import annotations

from ..state import AppState
from .icon import app_icon_url
from .views.chat import ChatSettingsView, ChatView, ModelChoice, NavigationView
from .views.context import PageContext


def chat_view(state: AppState) -> ChatView:
    """Capture the current chat data required by the pure chat renderers."""

    return ChatView(
        chat=state.active_chat,
        generating=state.is_generating(),
        runtime_status=state.runtime_status,
        generation_reasoning=state.active_generation_reasoning,
        empty_state_message=state.active_empty_state_message,
        message_action_icon_style=state.config.ui.message_action_icon_style,
        starter_prompts=tuple(state.config.ui.starter_prompts),
    )


def chat_settings_view(state: AppState) -> ChatSettingsView:
    """Capture the active chat's independently refreshable settings data."""

    chat = state.active_chat
    option_keys: tuple[str, ...] = (
        *state.config.server.model_aliases,
        *(model.key for model in state.model_inventory),
        chat.model,
    )
    model_choices: tuple[ModelChoice, ...] = tuple(
        ModelChoice(key=model_key, label=state.model_display_name(model_key))
        for model_key in dict.fromkeys(key for key in option_keys if key)
    )
    return ChatSettingsView(
        chat=chat,
        generating=state.is_generating(),
        generation_defaults=state.config.generation,
        model_choices=model_choices,
    )


def navigation_view(state: AppState, context: PageContext) -> NavigationView:
    """Capture the request-aware data required by the navigation drawer."""

    device = state.config.access.devices.get(context.decision.device_id)
    current_device_label: str | None = None
    if device is not None:
        current_device_label = device.label or context.client_label
    chats = tuple(state.chats)
    return NavigationView(
        chats=chats,
        active_chat_id=state.active_chat.id,
        generating_chat_ids=frozenset(
            chat.id for chat in chats if state.is_generating(chat.id)
        ),
        app_icon_url=app_icon_url(state.config.ui.icon_colors),
        can_use_global_settings=context.decision.can_use_global_settings,
        can_manage_access=context.decision.can_manage_access,
        current_device_label=current_device_label,
    )
