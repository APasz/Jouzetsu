"""HTTP route registration for the FastHTML application."""

# FastHTML's route decorator registers nested handlers dynamically.
# pyright: reportUnusedFunction=false

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast
from urllib.parse import urlencode

from fastcore.xml import FT  # pyright: ignore[reportMissingTypeStubs]
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import (
    PlainTextResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)

from ..access import is_device_access_approval_phrase
from ..character_presets import CharacterPresetCatalog
from ..config import (
    AppConfig,
    GenerationSettings,
    IconColorSettings,
    MessageActionIconStyle,
    ServerSettings,
)
from ..models import Character, CharacterField, ChatSamplingOverrides, Message
from ..state import AppState, ChatMessageUndo
from .form_data import (
    FormValues as _FormValues,
    character_fields_from_form as _character_fields_from_form,
    form_values as _form_values,
    missing_preset_fields as _missing_preset_fields,
    optional_float as _optional_float,
    optional_integer as _optional_integer,
    updated_generation_settings as _updated_generation_settings,
    updated_icon_color_settings as _updated_icon_color_settings,
    updated_server_settings as _updated_server_settings,
)
from .icon import configured_icon_svg, icon_artwork_path
from .mutations import (
    DialogName as _DialogName,
    Mutations,
)
from .request_access import (
    RequestAccess,
    RequestContext as _RequestContext,
)
from .security import require_csrf_token
from .theme import render_theme_css
from .ui_events import (
    UiEvent as _UiEvent,
    UiEventBroker,
    stream_events as _stream_events,
)
from .views.access import render_locked_access_page
from .views.characters import render_character_page
from .views.chat import (
    ChatSettingsView,
    ChatView,
    NavigationView,
    render_chat_fragment,
    render_chat_live_fragment,
    render_navigation_panel,
    render_settings_panel,
)
from .views.context import PageContext
from .views.dialogs import render_message_details_content
from .views.host_stats import render_content as render_host_stats_content
from .views.pages import (
    render_launch_page,
    render_main_page,
)

log: logging.Logger = logging.getLogger(__name__)
_NOTICE_MAXIMUM_LENGTH: Final[int] = 240
_ASSET_DIRECTORY: Path = Path(__file__).resolve().parent / "static"
_DIALOG_NAMES: Final[frozenset[str]] = frozenset({"global", "access", "models", "host"})

type HttpMethod = Literal["GET", "POST"]
type RouteResult = FT | Response
type RouteHandler = Callable[..., Awaitable[RouteResult]]
type RouteDecorator = Callable[[RouteHandler], RouteHandler]
type RouteRegistrar = Callable[[HttpMethod, str, str], RouteDecorator]
type CharacterUpdater = Callable[..., Awaitable[Character]]


@dataclass(frozen=True, slots=True)
class RouteContext:
    """Explicit services and callbacks required to register HTTP routes."""

    route: RouteRegistrar
    config: AppConfig
    state: AppState
    character_presets: CharacterPresetCatalog
    events: UiEventBroker
    access: RequestAccess
    mutations: Mutations
    chat_view: Callable[[], ChatView]
    chat_settings_view: Callable[[], ChatSettingsView]
    navigation_view: Callable[[PageContext], NavigationView]
    update_character: CharacterUpdater
    request_exit: Callable[[], bool]


def register_routes(context: RouteContext) -> None:
    """Register every route by its application domain."""

    _register_page_routes(context)
    _register_character_routes(context)
    _register_chat_routes(context)
    _register_settings_routes(context)
    _register_access_routes(context)


def _register_page_routes(context: RouteContext) -> None:
    @context.route("GET", "/icon.svg", "app_icon")
    async def app_icon(_request: Request) -> Response:
        """Serve the canonical icon artwork with the active palette."""

        return Response(
            configured_icon_svg(
                icon_artwork_path(_ASSET_DIRECTORY), context.config.ui.icon_colors
            ),
            media_type="image/svg+xml",
            headers={"Cache-Control": "no-store"},
        )

    @context.route("GET", "/theme.css", "theme_css")
    async def theme_css(_request: Request) -> Response:
        return Response(
            render_theme_css(context.config),
            media_type="text/css",
            headers={"Cache-Control": "no-store"},
        )

    async def render_chat_page(request: Request) -> FT:
        request_context: _RequestContext = await context.access.context(
            request, register_device=True
        )
        if not request_context.page.decision.access_allowed:
            return render_locked_access_page(
                request_context.page,
                bootstrap_device=request_context.page.decision.reason
                == "missing_device_id",
                approval_phrase_enabled=bool(context.config.access.approval_phrase),
            )
        return render_main_page(
            context.state,
            context.chat_view(),
            context.chat_settings_view(),
            context.navigation_view(request_context.page),
            request_context.page,
            edit_message_id=_query_text(request, "edit"),
            continuity_rewrite_message_id=_query_text(request, "continuity_rewrite"),
            active_dialog=_query_dialog_name(request),
            notice=_query_text(request, "notice"),
            error=_query_text(request, "error"),
            undo_token=_query_text(request, "undo"),
        )

    @context.route("GET", "/", "index")
    async def index(request: Request) -> FT:
        request_context: _RequestContext = await context.access.context(
            request, register_device=True
        )
        if not request_context.page.decision.access_allowed:
            return render_locked_access_page(
                request_context.page,
                bootstrap_device=request_context.page.decision.reason
                == "missing_device_id",
                approval_phrase_enabled=bool(context.config.access.approval_phrase),
            )
        return render_launch_page(
            context.state,
            request_context.page,
            notice=_query_text(request, "notice"),
            error=_query_text(request, "error"),
        )

    @context.route("GET", "/access/denied", "access_denied_preview")
    async def access_denied_preview(request: Request) -> FT:
        """Let localhost administrators view the page shown to unapproved browsers."""

        request_context: _RequestContext = await context.access.require(
            request, require_access_management=True
        )
        return render_locked_access_page(
            request_context.page,
            bootstrap_device=False,
            approval_phrase_enabled=bool(context.config.access.approval_phrase),
            preview=True,
        )

    @context.route("GET", "/chats", "chats")
    async def chats(request: Request) -> FT:
        return await render_chat_page(request)

    @context.route("GET", "/characters", "characters")
    async def characters(request: Request) -> FT:
        request_context: _RequestContext = await context.access.context(
            request, register_device=True
        )
        if not request_context.page.decision.access_allowed:
            return render_locked_access_page(
                request_context.page,
                bootstrap_device=request_context.page.decision.reason
                == "missing_device_id",
                approval_phrase_enabled=bool(context.config.access.approval_phrase),
            )
        return render_character_page(
            context.state,
            request_context.page,
            context.character_presets,
            notice=_query_text(request, "notice"),
            error=_query_text(request, "error"),
        )

    @context.route("GET", "/characters/{character_id}", "character")
    async def character(request: Request) -> FT:
        request_context: _RequestContext = await context.access.context(
            request, register_device=True
        )
        if not request_context.page.decision.access_allowed:
            return render_locked_access_page(
                request_context.page,
                bootstrap_device=request_context.page.decision.reason
                == "missing_device_id",
                approval_phrase_enabled=bool(context.config.access.approval_phrase),
            )
        character_id: str = _path_text(request, "character_id")
        try:
            _ = context.state.character(character_id)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        return render_character_page(
            context.state,
            request_context.page,
            context.character_presets,
            selected_character_id=character_id,
            notice=_query_text(request, "notice"),
            error=_query_text(request, "error"),
        )

    @context.route("GET", "/fragments/chat", "chat_fragment")
    async def chat_fragment(request: Request) -> FT:
        _ = await context.access.require(request)
        return render_chat_fragment(
            context.chat_view(), edit_message_id=_query_text(request, "edit")
        )

    @context.route("GET", "/fragments/chat/live", "chat_live_fragment")
    async def chat_live_fragment(request: Request) -> FT:
        _ = await context.access.require(request)
        return render_chat_live_fragment(
            context.chat_view(), edit_message_id=_query_text(request, "edit")
        )

    @context.route(
        "GET", "/fragments/messages/{message_id}/details", "message_details_fragment"
    )
    async def message_details_fragment(request: Request) -> FT:
        _ = await context.access.require(request)
        message_id: str = _path_text(request, "message_id")
        try:
            message: Message = context.mutations.require_visible_message(message_id)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        return render_message_details_content(message)

    @context.route("GET", "/fragments/navigation", "navigation_fragment")
    async def navigation_fragment(request: Request) -> FT:
        request_context: _RequestContext = await context.access.require(request)
        return render_navigation_panel(context.navigation_view(request_context.page))

    @context.route("GET", "/fragments/settings", "settings_fragment")
    async def settings_fragment(request: Request) -> FT:
        _ = await context.access.require(request)
        return render_settings_panel(context.chat_settings_view())

    @context.route("GET", "/fragments/host-stats", "host_stats_fragment")
    async def host_stats_fragment(request: Request) -> FT:
        _ = await context.access.require(request, require_global_settings=True)
        _ = await context.state.refresh_host_stats_snapshot()
        return render_host_stats_content(
            context.state.host_stats_snapshot(), context.state.config.host_stats
        )

    @context.route(
        "POST", "/fragments/host-stats/refresh", "refresh_host_stats_fragment"
    )
    async def refresh_host_stats_fragment(request: Request) -> FT:
        _ = await context.access.require(
            request,
            require_csrf=True,
            require_global_settings=True,
        )
        _ = await context.state.refresh_host_stats_snapshot()
        return render_host_stats_content(
            context.state.host_stats_snapshot(), context.state.config.host_stats
        )

    @context.route("GET", "/events", "events")
    async def events(request: Request) -> Response:
        _ = await context.access.require(request)
        queue: asyncio.Queue[_UiEvent] = context.events.subscribe()
        return StreamingResponse(
            _stream_events(context.events, queue),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )


def _register_character_routes(context: RouteContext) -> None:
    @context.route("POST", "/characters/new", "create_character")
    async def create_character(request: Request) -> Response:
        try:
            _ = await context.access.require(request, require_csrf=True)
            character = await context.state.new_character()
        except HTTPException:
            raise
        except Exception as exc:
            log.exception("character creation failed")
            return _character_redirect(error=str(exc))
        return _character_redirect(character.id, notice="Character created")

    @context.route("POST", "/characters/{character_id}", "update_character")
    async def update_character(request: Request) -> Response:
        character_id: str = _path_text(request, "character_id")
        form: _FormValues = await _form_values(request)
        try:
            _ = await context.access.require(request, require_csrf=True)
            character: Character = await context.update_character(character_id, form)
        except HTTPException:
            raise
        except Exception as exc:
            log.exception("character update failed character_id=%s", character_id)
            return _character_redirect(character_id, error=str(exc))
        return _character_redirect(character.id, notice="Character saved")

    @context.route(
        "POST", "/characters/{character_id}/fields/add", "add_character_field"
    )
    async def add_character_field(request: Request) -> Response:
        """Add a field through ordinary form navigation when browser enhancement is unavailable."""

        character_id: str = _path_text(request, "character_id")
        form: _FormValues = await _form_values(request)
        try:
            _ = await context.access.require(request, require_csrf=True)
            character: Character = await context.update_character(
                character_id,
                form,
                additional_fields=[CharacterField(label="New field")],
            )
        except HTTPException:
            raise
        except Exception as exc:
            log.exception("character field add failed character_id=%s", character_id)
            return _character_redirect(character_id, error=str(exc))
        return _character_redirect(character.id, notice="Field added")

    @context.route(
        "POST", "/characters/{character_id}/presets/apply", "apply_character_presets"
    )
    async def apply_character_presets(request: Request) -> Response:
        """Apply selected field templates through ordinary form navigation."""

        character_id: str = _path_text(request, "character_id")
        form: _FormValues = await _form_values(request)
        try:
            _ = await context.access.require(request, require_csrf=True)
            fields: list[CharacterField] = _character_fields_from_form(form)
            additions: list[CharacterField] = _missing_preset_fields(
                fields,
                context.character_presets.selected(
                    form.text("base_preset"), form.texts("extra_preset")
                ),
            )
            character: Character = await context.update_character(
                character_id,
                form,
                additional_fields=additions,
            )
        except HTTPException:
            raise
        except Exception as exc:
            log.exception("character preset apply failed character_id=%s", character_id)
            return _character_redirect(character_id, error=str(exc))
        return _character_redirect(character.id, notice="Selected presets applied")

    @context.route("POST", "/characters/{character_id}/delete", "delete_character")
    async def delete_character(request: Request) -> Response:
        character_id: str = _path_text(request, "character_id")
        try:
            _ = await context.access.require(request, require_csrf=True)
            await context.state.delete_character(character_id)
        except HTTPException:
            raise
        except Exception as exc:
            log.exception("character deletion failed character_id=%s", character_id)
            return _character_redirect(character_id, error=str(exc))
        return _character_redirect(notice="Character deleted")

    @context.route("POST", "/characters/{character_id}/chat", "start_character_chat")
    async def start_character_chat(request: Request) -> Response:
        character_id: str = _path_text(request, "character_id")
        form: _FormValues = await _form_values(request)
        try:
            _ = await context.access.require(request, require_csrf=True)
            character: Character = await context.update_character(character_id, form)
            chat = await context.state.start_chat_from_character(character.id)
        except HTTPException:
            raise
        except Exception as exc:
            log.exception(
                "character chat creation failed character_id=%s", character_id
            )
            return _character_redirect(character_id, error=str(exc))
        return RedirectResponse(
            f"/chats?{urlencode({'notice': f'Chat started with {chat.title}'})}",
            status_code=303,
        )


def _register_chat_routes(context: RouteContext) -> None:
    @context.route("POST", "/chats/new", "create_chat")
    async def create_chat(request: Request) -> Response:
        return await context.mutations.perform(
            request, context.state.new_chat, notice="New chat created"
        )

    @context.route("POST", "/chats/{chat_id}/select", "select_chat")
    async def select_chat(request: Request) -> Response:
        chat_id: str = _path_text(request, "chat_id")
        return await context.mutations.perform(
            request, lambda: context.state.select_chat(chat_id), notice="Chat selected"
        )

    @context.route("POST", "/chats/{chat_id}/rename", "rename_chat")
    async def rename_chat(request: Request) -> Response:
        chat_id: str = _path_text(request, "chat_id")
        form: _FormValues = await _form_values(request)
        return await context.mutations.perform(
            request,
            lambda: context.state.rename_chat(chat_id, form.text("title")),
            notice="Chat renamed",
        )

    @context.route("POST", "/chats/{chat_id}/delete", "delete_chat")
    async def delete_chat(request: Request) -> Response:
        chat_id: str = _path_text(request, "chat_id")

        async def operation() -> None:
            if chat_id != context.state.active_chat.id:
                raise ValueError("only the active chat can be deleted from this page")
            await context.state.delete_chat(chat_id)

        return await context.mutations.perform(
            request, operation, notice="Chat deleted"
        )

    @context.route("POST", "/chat/draft", "save_draft")
    async def save_draft(request: Request) -> Response:
        form: _FormValues = await _form_values(request)
        _ = await context.access.require(request, require_csrf=True)
        try:
            await context.state.set_chat_draft(
                form.required_text("chat_id"), form.text("draft")
            )
        except Exception as exc:
            log.exception("draft save failed")
            return PlainTextResponse(str(exc), status_code=400)
        return Response(status_code=204)

    @context.route("POST", "/chat/send", "send_message")
    async def send_message(request: Request) -> Response:
        form: _FormValues = await _form_values(request)
        return await context.mutations.perform(
            request,
            lambda: context.state.send_user_message(form.text("content")),
            notice="",
        )

    @context.route("POST", "/chat/stop", "stop_generation")
    async def stop_generation(request: Request) -> Response:
        return await context.mutations.perform(
            request, context.state.stop_generation, notice=""
        )

    @context.route("POST", "/messages/edit", "edit_message")
    async def edit_message(request: Request) -> Response:
        form: _FormValues = await _form_values(request)
        message_id: str = form.required_text("message_id")

        async def operation() -> None:
            _ = context.mutations.require_visible_message(message_id)
            await context.state.edit_message(message_id, form.text("content"))

        return await context.mutations.perform(
            request, operation, notice="Message updated"
        )

    @context.route("POST", "/messages/{message_id}/delete", "delete_message")
    async def delete_message(request: Request) -> Response:
        message_id: str = _path_text(request, "message_id")

        async def operation() -> ChatMessageUndo | None:
            _ = context.mutations.require_visible_message(message_id)
            return await context.state.delete_message_with_undo(message_id)

        try:
            _ = await context.access.require(request, require_csrf=True)
            undo: ChatMessageUndo | None = await operation()
        except HTTPException:
            raise
        except Exception as exc:
            log.exception("web mutation failed route=%s", request.url.path)
            return context.mutations.redirect(error=str(exc))
        return context.mutations.redirect(
            notice="Message deleted",
            undo=context.mutations.store_message_undo(undo) if undo else "",
        )

    @context.route("POST", "/messages/{message_id}/truncate", "truncate_message")
    async def truncate_message(request: Request) -> Response:
        message_id: str = _path_text(request, "message_id")

        async def operation() -> ChatMessageUndo | None:
            _ = context.mutations.require_visible_message(message_id)
            return await context.state.truncate_chat_to_message_with_undo(message_id)

        try:
            _ = await context.access.require(request, require_csrf=True)
            undo: ChatMessageUndo | None = await operation()
        except HTTPException:
            raise
        except Exception as exc:
            log.exception("web mutation failed route=%s", request.url.path)
            return context.mutations.redirect(error=str(exc))
        return context.mutations.redirect(
            notice="Chat truncated",
            undo=context.mutations.store_message_undo(undo) if undo else "",
        )

    @context.route(
        "POST",
        "/messages/{message_id}/delete-following",
        "delete_message_and_following",
    )
    async def delete_message_and_following(request: Request) -> Response:
        message_id: str = _path_text(request, "message_id")

        async def operation() -> ChatMessageUndo | None:
            _ = context.mutations.require_visible_message(message_id)
            return await context.state.delete_message_and_following_with_undo(
                message_id
            )

        try:
            _ = await context.access.require(request, require_csrf=True)
            undo: ChatMessageUndo | None = await operation()
        except HTTPException:
            raise
        except Exception as exc:
            log.exception("web mutation failed route=%s", request.url.path)
            return context.mutations.redirect(error=str(exc))
        return context.mutations.redirect(
            notice="Messages deleted",
            undo=context.mutations.store_message_undo(undo) if undo else "",
        )

    @context.route("POST", "/messages/{message_id}/regenerate", "regenerate_message")
    async def regenerate_message(request: Request) -> Response:
        message_id: str = _path_text(request, "message_id")

        async def operation() -> None:
            _ = context.mutations.require_last_assistant_message(message_id)
            await context.state.regenerate_last()

        return await context.mutations.perform(request, operation, notice="")

    @context.route("POST", "/messages/{message_id}/continue", "continue_message")
    async def continue_message(request: Request) -> Response:
        message_id: str = _path_text(request, "message_id")

        async def operation() -> None:
            _ = context.mutations.require_last_assistant_message(message_id)
            await context.state.continue_last_response()

        return await context.mutations.perform(request, operation, notice="")

    @context.route(
        "POST",
        "/messages/{message_id}/continuity-rewrite/apply",
        "apply_continuity_rewrite",
    )
    async def apply_continuity_rewrite(request: Request) -> Response:
        message_id: str = _path_text(request, "message_id")

        async def operation() -> None:
            _ = context.mutations.require_visible_message(message_id)
            await context.state.apply_continuity_rewrite(message_id)

        return await context.mutations.perform(
            request, operation, notice="Continuity rewrite applied"
        )

    @context.route(
        "POST",
        "/messages/{message_id}/continuity-rewrite/discard",
        "discard_continuity_rewrite",
    )
    async def discard_continuity_rewrite(request: Request) -> Response:
        message_id: str = _path_text(request, "message_id")

        async def operation() -> None:
            _ = context.mutations.require_visible_message(message_id)
            await context.state.discard_continuity_rewrite(message_id)

        return await context.mutations.perform(
            request, operation, notice="Continuity rewrite discarded"
        )

    @context.route("POST", "/messages/{message_id}/resend", "resend_message")
    async def resend_message(request: Request) -> Response:
        message_id: str = _path_text(request, "message_id")

        async def operation() -> None:
            _ = context.mutations.require_last_user_message(message_id)
            _ = await context.state.resend_user_message(message_id)

        return await context.mutations.perform(request, operation, notice="")

    @context.route("POST", "/messages/undo", "undo_message_action")
    async def undo_message_action(request: Request) -> Response:
        form: _FormValues = await _form_values(request)
        try:
            _ = await context.access.require(request, require_csrf=True)
            undo: ChatMessageUndo = context.mutations.consume_message_undo(
                form.required_text("undo_token")
            )
            await context.state.restore_message_undo(undo)
        except HTTPException:
            raise
        except Exception as exc:
            log.exception("message undo failed")
            return context.mutations.redirect(error=str(exc))
        return context.mutations.redirect(notice="Change undone")

    @context.route("POST", "/messages/{message_id}/fork", "fork_message")
    async def fork_message(request: Request) -> Response:
        message_id: str = _path_text(request, "message_id")

        async def operation() -> None:
            _ = context.mutations.require_visible_message(message_id)
            _ = await context.state.fork_chat_at_message(message_id)

        return await context.mutations.perform(request, operation, notice="Chat forked")

    @context.route("POST", "/messages/{message_id}/merge", "merge_message")
    async def merge_message(request: Request) -> Response:
        message_id: str = _path_text(request, "message_id")

        async def operation() -> None:
            _ = context.mutations.require_visible_message(message_id)
            _ = await context.state.merge_message_with_previous(message_id)

        return await context.mutations.perform(
            request, operation, notice="Messages merged"
        )

    @context.route("POST", "/chat/model", "set_chat_model")
    async def set_chat_model(request: Request) -> Response:
        form: _FormValues = await _form_values(request)
        model: str = form.text("model")
        return await context.mutations.perform(
            request,
            lambda: context.state.set_active_chat_model(model or None),
            notice="Chat model saved",
        )

    @context.route("POST", "/chat/sampling", "set_chat_sampling")
    async def set_chat_sampling(request: Request) -> Response:
        form: _FormValues = await _form_values(request)
        overrides: ChatSamplingOverrides = ChatSamplingOverrides(
            temperature=_optional_float(
                form.text("temperature"), field_name="temperature"
            ),
            top_p=_optional_float(form.text("top_p"), field_name="top p"),
            max_tokens=_optional_integer(
                form.text("max_tokens"), field_name="maximum output tokens"
            ),
        )
        return await context.mutations.perform(
            request,
            lambda: context.state.set_active_chat_sampling_overrides(overrides),
            notice="Sampling settings saved",
        )

    @context.route("POST", "/chat/prompt", "set_chat_prompt")
    async def set_chat_prompt(request: Request) -> Response:
        form: _FormValues = await _form_values(request)
        return await context.mutations.perform(
            request,
            lambda: context.state.set_active_chat_system_prompt(form.text("prompt")),
            notice="Chat prompt saved",
        )

    @context.route("POST", "/chat/british-spellings", "set_british_spellings")
    async def set_british_spellings(request: Request) -> Response:
        form: _FormValues = await _form_values(request)
        return await context.mutations.perform(
            request,
            lambda: context.state.set_active_chat_postprocess_british_spellings(
                form.flag("enabled")
            ),
            notice="Spelling setting saved",
        )

    @context.route("POST", "/chat/reasoning", "set_reasoning_persistence")
    async def set_reasoning_persistence(request: Request) -> Response:
        form: _FormValues = await _form_values(request)
        return await context.mutations.perform(
            request,
            lambda: context.state.set_active_chat_save_reasoning(form.flag("enabled")),
            notice="Reasoning setting saved",
        )


def _register_settings_routes(context: RouteContext) -> None:
    @context.route("POST", "/settings/global", "save_global_settings")
    async def save_global_settings(request: Request) -> Response:
        form: _FormValues = await _form_values(request)

        async def operation() -> None:
            generation: GenerationSettings = _updated_generation_settings(
                context.state,
                form,
                system_prompt=form.text("system_prompt").strip(),
            )
            server: ServerSettings = _updated_server_settings(
                context.state,
                default_model=form.text(
                    "default_model", default=context.state.config.server.default_model
                ),
                alias=form.text("model_alias"),
                auto_unload_minutes=form.text("auto_unload_minutes"),
            )
            icon_style: MessageActionIconStyle = (
                "muted_color" if form.flag("muted_color_icons") else "monochrome"
            )
            icon_colors: IconColorSettings = _updated_icon_color_settings(
                context.state, form
            )
            await context.state.set_global_settings(
                generation,
                server,
                message_action_icon_style=icon_style,
                icon_colors=icon_colors,
            )

        return await context.mutations.perform(
            request,
            operation,
            notice="Global settings saved",
            require_global_settings=True,
            dialog="global",
        )

    @context.route("POST", "/settings/global/prompt", "save_global_prompt")
    async def save_global_prompt(request: Request) -> Response:
        form: _FormValues = await _form_values(request)
        return await context.mutations.perform(
            request,
            lambda: context.state.set_global_system_prompt(form.text("system_prompt")),
            notice="Global system prompt saved",
            require_global_settings=True,
            dialog="global",
        )

    @context.route(
        "POST", "/settings/global/model-defaults", "save_global_model_defaults"
    )
    async def save_global_model_defaults(request: Request) -> Response:
        form: _FormValues = await _form_values(request)

        async def operation() -> None:
            server: ServerSettings = _updated_server_settings(
                context.state,
                default_model=form.text(
                    "default_model", default=context.state.config.server.default_model
                ),
                alias=form.text("model_alias"),
                auto_unload_minutes=form.text("auto_unload_minutes"),
            )
            await context.state.set_global_settings(
                context.state.config.generation, server
            )

        return await context.mutations.perform(
            request,
            operation,
            notice="Model defaults saved",
            require_global_settings=True,
            dialog="global",
        )

    @context.route("POST", "/settings/global/interface", "save_global_interface")
    async def save_global_interface(request: Request) -> Response:
        form: _FormValues = await _form_values(request)
        icon_style: MessageActionIconStyle = (
            "muted_color" if form.flag("muted_color_icons") else "monochrome"
        )
        icon_colors: IconColorSettings = _updated_icon_color_settings(
            context.state, form
        )
        return await context.mutations.perform(
            request,
            lambda: context.state.set_global_settings(
                context.state.config.generation,
                context.state.config.server,
                message_action_icon_style=icon_style,
                icon_colors=icon_colors,
            ),
            notice="Interface settings saved",
            require_global_settings=True,
            dialog="global",
        )

    @context.route("POST", "/settings/global/generation", "save_global_generation")
    async def save_global_generation(request: Request) -> Response:
        form: _FormValues = await _form_values(request)

        async def operation() -> None:
            generation: GenerationSettings = _updated_generation_settings(
                context.state,
                form,
                system_prompt=context.state.config.generation.system_prompt,
            )
            await context.state.set_generation_defaults(generation)

        return await context.mutations.perform(
            request,
            operation,
            notice="Generation defaults saved",
            require_global_settings=True,
            dialog="global",
        )


def _register_access_routes(context: RouteContext) -> None:
    @context.route(
        "POST", "/access/current-device/activity", "record_current_device_activity"
    )
    async def record_current_device_activity(request: Request) -> Response:
        """Accept a CSRF-protected, visible-tab browser activity heartbeat."""

        await require_csrf_token(request)
        await context.access.record_current_device_activity(request)
        return Response(status_code=204, headers={"Cache-Control": "no-store"})

    @context.route("POST", "/access/current-device/approve", "approve_current_device")
    async def approve_current_device(request: Request) -> Response:
        async def operation() -> None:
            request_context: _RequestContext = await context.access.context(
                request, register_device=True
            )
            if not request_context.page.decision.device_id:
                raise ValueError(
                    "this browser does not have a valid device identifier yet"
                )
            await context.state.set_device_access_allowed(
                request_context.page.decision.device_id, True
            )

        return await context.mutations.perform(
            request,
            operation,
            notice="Browser approved",
            require_access_management=True,
        )

    @context.route("POST", "/access/current-device/pending", "save_pending_device")
    async def save_pending_device(request: Request) -> Response:
        form: _FormValues = await _form_values(request)
        try:
            await require_csrf_token(request)
            request_context: _RequestContext = await context.access.context(
                request, register_device=True
            )
            device_id: str = request_context.page.decision.device_id
            if not device_id:
                raise ValueError(
                    "this browser does not have a valid device identifier yet"
                )
            await context.state.set_access_device_label(device_id, form.text("label"))
            approved: bool = is_device_access_approval_phrase(
                form.text("approval_phrase"),
                required_phrase=context.config.access.approval_phrase,
            )
            if approved:
                await context.state.set_device_access_allowed(device_id, True)
        except HTTPException:
            raise
        except Exception as exc:
            log.exception("pending device update failed")
            return context.mutations.redirect(error=str(exc))
        return context.mutations.redirect(
            notice="Browser approved" if approved else "Browser details saved"
        )

    @context.route("POST", "/access/current-device/label", "label_current_device")
    async def label_current_device(request: Request) -> Response:
        form: _FormValues = await _form_values(request)

        async def operation() -> None:
            request_context: _RequestContext = await context.access.context(
                request, register_device=False
            )
            if not request_context.page.decision.device_id:
                raise ValueError(
                    "this browser does not have a valid device identifier yet"
                )
            await context.state.set_access_device_label(
                request_context.page.decision.device_id, form.text("label")
            )

        return await context.mutations.perform(
            request, operation, notice="Device label saved"
        )

    @context.route("POST", "/access/defaults", "save_access_defaults")
    async def save_access_defaults(request: Request) -> Response:
        form: _FormValues = await _form_values(request)
        return await context.mutations.perform(
            request,
            lambda: context.state.set_access_defaults(
                default_private=form.flag("default_private"),
                allow_localhost_without_approval=form.flag(
                    "allow_localhost_without_approval"
                ),
                global_settings_for_approved=form.flag("global_settings_for_approved"),
                allow_network_device_reassociation=form.flag(
                    "allow_network_device_reassociation"
                ),
            ),
            notice="Access defaults saved",
            require_access_management=True,
            dialog="access",
        )

    @context.route("POST", "/access/devices/{device_id}", "save_device_access")
    async def save_device_access(request: Request) -> Response:
        device_id: str = _path_text(request, "device_id")
        form: _FormValues = await _form_values(request)
        return await context.mutations.perform(
            request,
            lambda: context.state.set_device_access_allowed(
                device_id, form.flag("access_allowed")
            ),
            notice="Device access saved",
            require_access_management=True,
            dialog="access",
        )

    @context.route(
        "POST", "/access/devices/{device_id}/forget", "forget_pending_device"
    )
    async def forget_pending_device(request: Request) -> Response:
        device_id: str = _path_text(request, "device_id")

        async def operation() -> None:
            request_context: _RequestContext = await context.access.context(
                request, register_device=False
            )
            if device_id == request_context.page.decision.device_id:
                raise ValueError("the current browser cannot be forgotten")
            await context.state.forget_pending_access_device(device_id)

        return await context.mutations.perform(
            request,
            operation,
            notice="Pending device forgotten",
            require_access_management=True,
            dialog="access",
        )

    @context.route("POST", "/access/devices/{device_id}/label", "save_device_label")
    async def save_device_label(request: Request) -> Response:
        device_id: str = _path_text(request, "device_id")
        form: _FormValues = await _form_values(request)
        return await context.mutations.perform(
            request,
            lambda: context.state.set_access_device_label(
                device_id, form.text("label")
            ),
            notice="Device label saved",
            require_access_management=True,
            dialog="access",
        )

    @context.route("POST", "/models/refresh", "refresh_models")
    async def refresh_models(request: Request) -> Response:
        return await context.mutations.perform(
            request,
            context.state.refresh_runtime_status,
            notice="Model inventory refreshed",
            require_global_settings=True,
            dialog="models",
        )

    @context.route(
        "POST", "/models/instances/{instance_id}/unload", "unload_model_instance"
    )
    async def unload_model_instance(request: Request) -> Response:
        instance_id: str = _path_text(request, "instance_id")
        return await context.mutations.perform(
            request,
            lambda: context.state.unload_model_instance(instance_id),
            notice="Model instance unloaded",
            require_global_settings=True,
            dialog="models",
        )

    @context.route("POST", "/host-stats/refresh", "refresh_host_stats")
    async def refresh_host_stats(request: Request) -> Response:
        async def operation() -> None:
            _ = await context.state.refresh_host_stats_snapshot()

        return await context.mutations.perform(
            request,
            operation,
            notice="Host stats refreshed",
            require_global_settings=True,
            dialog="host",
        )

    @context.route("POST", "/app/exit", "exit_application")
    async def exit_application(request: Request) -> Response:
        _ = await context.access.require(
            request,
            require_csrf=True,
            require_access_management=True,
        )
        if not context.request_exit():
            return context.mutations.redirect(
                error="The server cannot accept an Exit request in this context"
            )
        return context.mutations.redirect(notice="Jouzetsu is stopping")


def _character_redirect(
    character_id: str = "", *, notice: str = "", error: str = ""
) -> RedirectResponse:
    """Redirect a character form back to its library or its own editor."""

    path: str = f"/characters/{character_id}" if character_id else "/characters"
    values: dict[str, str] = {}
    if notice:
        values["notice"] = notice[:_NOTICE_MAXIMUM_LENGTH]
    if error:
        values["error"] = error[:_NOTICE_MAXIMUM_LENGTH]
    destination: str = path if not values else f"{path}?{urlencode(values)}"
    return RedirectResponse(destination, status_code=303)


def _path_text(request: Request, name: str) -> str:
    """Return one non-empty FastHTML path parameter."""

    value: object = request.path_params.get(name)
    if not isinstance(value, str) or not value:
        raise HTTPException(400, f"invalid {name}")
    return value


def _query_text(request: Request, name: str) -> str:
    """Return a single query value without mutating-route binding precedence."""

    return request.query_params.get(name, "")


def _query_dialog_name(request: Request) -> _DialogName:
    """Return a known dialog name, treating arbitrary query values as absent."""

    value: str = _query_text(request, "dialog")
    return cast(_DialogName, value) if value in _DIALOG_NAMES else ""
