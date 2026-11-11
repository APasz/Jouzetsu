"""Explicit chat read models; navigation belongs to each browser URL."""

from __future__ import annotations

from dataclasses import replace

from ..backend_models import BackendKind, GenerationProfile
from ..backend_protocol import BackendReachability
from ..runtime import RuntimePhase
from ..state import AppState
from .icon import app_icon_url
from .views.chat import ChatSettingsView, ChatView, NavigationView
from .views.context import PageContext
from .views.model_controls import model_choices


def chat_view(state: AppState, chat_id: str) -> ChatView:
    session = state.session(chat_id)
    job = state.inference.job(chat_id)
    status = session.runtime_status
    selected = session.chat.inference.selected_backend_id
    profile = session.chat.inference.selected_profile
    backend = next(
        (item for item in state.config.backends.backends if item.id == selected), None
    )
    model = (
        next(
            (
                item
                for item in backend.models
                if profile and item.descriptor.key == profile.model_id
            ),
            None,
        )
        if backend and backend.kind is BackendKind.LM_STUDIO
        else None
    )
    if not session.is_generating and session.latest_error is None:
        if selected is None or session.chat.inference.selected_model is None:
            status = replace(
                status,
                phase=RuntimePhase.NO_MODEL_SELECTED,
                detail="Select a backend" if selected is None else "Select a model",
            )
        else:
            if backend is None:
                status = replace(
                    status,
                    phase=RuntimePhase.ERROR,
                    detail="Selected backend was removed",
                )
            elif (
                not backend.enabled
                or state.inference.health(selected).state is BackendReachability.OFFLINE
            ):
                status = replace(
                    status,
                    phase=RuntimePhase.OFFLINE,
                    detail=backend.name + " unavailable",
                )
    return ChatView(
        chat=session.chat,
        generating=session.is_generating,
        runtime_status=status,
        generation_reasoning=session.live_reasoning,
        empty_state_message=session.empty_state_message,
        starter_prompts=tuple(state.config.ui.starter_prompts),
        message_action_style=state.config.ui.message_action_style,
        live_response=session.live_response,
        replacement_id=job.replacement_id if job else None,
        latest_error=session.latest_error,
        save_failed=job is not None and job.save_error is not None,
        draft_write_error=state.persistence.chat_write_error is not None,
        reasoning_options=model.reasoning_options if model else (),
        reasoning_default=model.reasoning_default if model else None,
        reasoning_unavailable=bool(
            model
            and profile
            and (
                not model.resolve_reasoning(profile.reasoning)[1]
                or (
                    profile.reasoning == "model_default"
                    and model.reasoning_default_unavailable
                )
            )
        ),
    )


def chat_settings_view(state: AppState, chat_id: str) -> ChatSettingsView:
    chat = state.session(chat_id).chat
    backend_id = chat.inference.selected_backend_id
    backend = next(
        (
            backend
            for backend in state.config.backends.backends
            if backend.id == backend_id
        ),
        None,
    )
    profile = chat.inference.selected_profile or GenerationProfile()
    support = backend.support_for_model(profile.model_id) if backend else None
    inactive = tuple(
        name
        for name in ("temperature", "top_p", "max_tokens")
        if support is None or getattr(support, name) is not True
    )
    return ChatSettingsView(
        chat=chat,
        generating=state.is_generating(chat_id),
        model_choices=model_choices(backend, profile.model_id),
        backends=state.config.backends,
        profile=profile,
        inactive_parameters=inactive,
    )


def navigation_view(
    state: AppState, context: PageContext, chat_id: str = ""
) -> NavigationView:
    device = state.config.access.devices.get(context.decision.device_id)
    chats = tuple(state.chats)
    return NavigationView(
        chats=chats,
        selected_chat_id=chat_id,
        generating_chat_ids=frozenset(
            chat.id for chat in chats if state.is_generating(chat.id)
        ),
        app_icon_url=app_icon_url(state.config.ui.icon_colors),
        can_use_global_settings=context.decision.can_use_global_settings,
        can_manage_access=context.decision.can_manage_access,
        current_device_label=(device.label or context.client_label) if device else None,
    )
