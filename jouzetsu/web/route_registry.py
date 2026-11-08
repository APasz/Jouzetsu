"""HTTP route registration for the FastHTML application."""

# FastHTML's route decorator registers nested handlers dynamically.
# pyright: reportUnusedFunction=false

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Literal, Protocol, cast
from urllib.parse import urlencode

from fastcore.xml import FT  # pyright: ignore[reportMissingTypeStubs]
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)

from ..async_workers import run_in_worker
from ..backend_models import (
    BackendConfig,
    BackendId,
    BackendKind,
    GenerationProfile,
    ModelManagementSettings,
    ModelRef,
)
from ..backend_protocol import BackendReachability
from ..character_presets import CharacterNameSuggestionsCatalog, CharacterPresetCatalog
from ..config import (
    AppConfig,
    GenerationSettings,
    IconColorSettings,
    MessageActionStyle,
    ThemeColorway,
    ThemeSettings,
)
from ..inference import GenerationMode, RevisionConflict
from ..models import (
    Character,
    CharacterField,
    Chat,
    ChatPromptMode,
    ImageAttachment,
    Message,
    chat_prompt_mode_from_text,
    image_attachments_from_json,
)
from ..sampling import SamplingParameters
from ..state import AppState
from .form_data import (
    FormValues as _FormValues,
    character_fields_from_form as _character_fields_from_form,
    form_values as _form_values,
    message_action_style_from_form as _message_action_style_from_form,
    missing_preset_fields as _missing_preset_fields,
    optional_integer as _optional_integer,
    updated_generation_settings as _updated_generation_settings,
    updated_icon_color_settings as _updated_icon_color_settings,
    updated_theme_settings as _updated_theme_settings,
)
from .icon import configured_icon_svg, icon_artwork_path
from .mutations import (
    DialogName as _DialogName,
    Mutations,
)
from .navigation import LAST_CHAT_COOKIE_NAME
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
    render_chat_composer_fragment,
    render_chat_fragment,
    render_chat_live_fragment,
    render_navigation_panel,
    render_settings_panel,
)
from .views.context import PageContext
from .views.dialogs import render_message_details_content
from .views.pages import (
    render_empty_chat_page,
    render_launch_page,
    render_main_page,
)

log: logging.Logger = logging.getLogger(__name__)
_NOTICE_MAXIMUM_LENGTH: Final[int] = 240
_ASSET_DIRECTORY: Path = Path(__file__).resolve().parent / "static"
_DIALOG_NAMES: Final[frozenset[str]] = frozenset({"global", "access"})

type HttpMethod = Literal["GET", "POST"]
type RouteResult = FT | Response
type RouteHandler = Callable[..., Awaitable[RouteResult]]
type RouteDecorator = Callable[[RouteHandler], RouteHandler]
type RouteRegistrar = Callable[[HttpMethod, str, str], RouteDecorator]


class CharacterUpdater(Protocol):
    """Persist one character editor submission for a character route."""

    def __call__(
        self,
        character_id: str,
        form: _FormValues,
        *,
        additional_fields: list[CharacterField] | None = None,
        preserve_existing_name_if_blank: bool = False,
    ) -> Awaitable[Character]: ...


@dataclass(frozen=True, slots=True)
class RouteContext:
    """Explicit services and callbacks required to register HTTP routes."""

    route: RouteRegistrar
    config: AppConfig
    state: AppState
    character_presets: CharacterPresetCatalog
    character_name_suggestions: CharacterNameSuggestionsCatalog
    events: UiEventBroker
    access: RequestAccess
    mutations: Mutations
    chat_view: Callable[[str], ChatView]
    chat_settings_view: Callable[[str], ChatSettingsView]
    navigation_view: Callable[[PageContext, str], NavigationView]
    update_character: CharacterUpdater


def register_routes(context: RouteContext) -> None:
    """Register every route by its application domain."""

    _register_page_routes(context)
    _register_character_routes(context)
    _register_chat_routes(context)
    _register_settings_routes(context)
    _register_access_routes(context)
    _register_backend_routes(context)


def _register_page_routes(context: RouteContext) -> None:
    @context.route("GET", "/images/{image_id}", "image_attachment")
    async def image_attachment(request: Request) -> Response:
        _ = await context.access.require(request)
        try:
            image: ImageAttachment = context.state.image_attachment(
                _path_text(request, "image_id")
            )
            thumbnail: bool = _query_text(request, "thumbnail") == "1"
            path: Path = await run_in_worker(
                context.state.images.available_path, image, thumbnail=thumbnail
            )
            if not path.is_file():
                raise ValueError("image file is missing")
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        return FileResponse(
            path,
            media_type="image/jpeg" if thumbnail else image.media_type.value,
            filename=image.name,
            content_disposition_type="inline",
            headers={"Cache-Control": "private, no-store"},
        )

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
            context.chat_view(_existing_chat_id(context.state, request)),
            context.chat_settings_view(_existing_chat_id(context.state, request)),
            context.navigation_view(request_context.page, _chat_id(request)),
            request_context.page,
            edit_message_id=_query_text(request, "edit"),
            active_dialog=_query_dialog_name(request),
            active_dialog_tab=_query_text(request, "tab"),
            notice=_query_text(request, "notice"),
            error=_query_text(request, "error"),
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
    async def chats(request: Request) -> FT | Response:
        request_context = await context.access.context(request, register_device=True)
        if not request_context.page.decision.access_allowed:
            return render_locked_access_page(
                request_context.page,
                bootstrap_device=request_context.page.decision.reason
                == "missing_device_id",
                approval_phrase_enabled=bool(context.config.access.approval_phrase),
            )
        if context.state.chats:
            remembered = request.cookies.get(LAST_CHAT_COOKIE_NAME, "")
            chat_id = (
                remembered
                if remembered in context.state.registry
                else context.state.chats[0].id
            )
            return RedirectResponse(
                f"/chats/{chat_id}"
                + (f"?{request.url.query}" if request.url.query else ""),
                status_code=303,
            )
        return render_empty_chat_page(
            context.state,
            context.navigation_view(request_context.page, ""),
            request_context.page,
            active_dialog=_query_dialog_name(request),
            active_dialog_tab=_query_text(request, "tab"),
            notice=_query_text(request, "notice"),
            error=_query_text(request, "error"),
        )

    @context.route("GET", "/chats/{chat_id}", "chat_page")
    async def chat_page(request: Request) -> FT:
        request_context = await context.access.context(request, register_device=True)
        if not request_context.page.decision.access_allowed:
            return render_locked_access_page(
                request_context.page,
                bootstrap_device=request_context.page.decision.reason
                == "missing_device_id",
                approval_phrase_enabled=bool(context.config.access.approval_phrase),
            )
        try:
            _ = context.state.session(_chat_id(request))
        except KeyError as exc:
            raise HTTPException(404, "chat no longer exists") from exc
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
            context.character_name_suggestions,
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
            context.character_name_suggestions,
            selected_character_id=character_id,
            notice=_query_text(request, "notice"),
            error=_query_text(request, "error"),
        )

    @context.route("GET", "/fragments/chat", "chat_fragment")
    async def chat_fragment(request: Request) -> FT:
        _ = await context.access.require(request)
        return render_chat_fragment(
            context.chat_view(_existing_chat_id(context.state, request)),
            edit_message_id=_query_text(request, "edit"),
        )

    @context.route("GET", "/fragments/chat/live", "chat_live_fragment")
    async def chat_live_fragment(request: Request) -> FT:
        _ = await context.access.require(request)
        return render_chat_live_fragment(
            context.chat_view(_existing_chat_id(context.state, request)),
            edit_message_id=_query_text(request, "edit"),
        )

    @context.route("GET", "/fragments/chat/composer", "chat_composer_fragment")
    async def chat_composer_fragment(request: Request) -> FT:
        _ = await context.access.require(request)
        return render_chat_composer_fragment(
            context.chat_view(_existing_chat_id(context.state, request)),
            edit_message_id=_query_text(request, "edit"),
        )

    @context.route(
        "GET", "/fragments/messages/{message_id}/details", "message_details_fragment"
    )
    async def message_details_fragment(request: Request) -> FT:
        _ = await context.access.require(request)
        message_id: str = _path_text(request, "message_id")
        try:
            message: Message = context.mutations.require_visible_message(
                _chat_id(request), message_id
            )
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        return render_message_details_content(message)

    @context.route("GET", "/fragments/navigation", "navigation_fragment")
    async def navigation_fragment(request: Request) -> FT:
        request_context: _RequestContext = await context.access.require(request)
        return render_navigation_panel(
            context.navigation_view(request_context.page, _chat_id(request))
        )

    @context.route("GET", "/fragments/settings", "settings_fragment")
    async def settings_fragment(request: Request) -> FT:
        _ = await context.access.require(request)
        return render_settings_panel(
            context.chat_settings_view(_existing_chat_id(context.state, request))
        )

    @context.route("GET", "/fragments/global", "global_fragment")
    async def global_fragment(request: Request) -> FT:
        from .views.global_settings import (
            global_settings_tab_from_text,
            render_global_settings_dialog,
        )

        request_context = await context.access.require(
            request, require_global_settings=True
        )
        return render_global_settings_dialog(
            context.state,
            request_context.page,
            active_tab=global_settings_tab_from_text(_query_text(request, "tab")),
        )

    @context.route("GET", "/fragments/access", "access_fragment")
    async def access_fragment(request: Request) -> FT:
        from .views.access import render_access_dialog

        request_context = await context.access.require(
            request, require_access_management=True
        )
        return render_access_dialog(context.state, request_context.page)

    @context.route("GET", "/events", "events")
    async def events(request: Request) -> Response:
        _ = await context.access.require(request)
        queue: asyncio.Queue[_UiEvent] = context.events.subscribe()
        return StreamingResponse(
            _stream_events(
                context.events, queue, lambda: context.access.allowed(request)
            ),
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
        except Exception as exc:  # noqa: BLE001
            log.error("character creation failed error_type=%s", type(exc).__name__)
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
        except Exception as exc:  # noqa: BLE001
            log.error(
                "character update failed character_id=%s error_type=%s",
                character_id,
                type(exc).__name__,
            )
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
                preserve_existing_name_if_blank=True,
            )
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            log.error(
                "character field add failed character_id=%s error_type=%s",
                character_id,
                type(exc).__name__,
            )
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
                preserve_existing_name_if_blank=True,
            )
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            log.error(
                "character preset apply failed character_id=%s error_type=%s",
                character_id,
                type(exc).__name__,
            )
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
        except Exception as exc:  # noqa: BLE001
            log.error(
                "character deletion failed character_id=%s error_type=%s",
                character_id,
                type(exc).__name__,
            )
            return _character_redirect(character_id, error=str(exc))
        return _character_redirect(notice="Character deleted")

    @context.route("POST", "/characters/{character_id}/chat", "start_character_chat")
    async def start_character_chat(request: Request) -> Response:
        character_id: str = _path_text(request, "character_id")
        form: _FormValues = await _form_values(request)
        try:
            _ = await context.access.require(request, require_csrf=True)
            character: Character = await context.update_character(character_id, form)
            prompt_mode: ChatPromptMode = chat_prompt_mode_from_text(
                form.text("prompt_mode", default=ChatPromptMode.ROLEPLAY.value)
            )
            chat = await context.state.start_chat_from_characters(
                (character.id, *form.texts("cast_member_id")),
                prompt_mode=prompt_mode,
                custom_instruction=form.text("custom_instruction"),
                story_direction=form.text("story_direction"),
            )
        except HTTPException:
            raise
        except Exception as exc:
            log.exception(
                "character chat creation failed character_id=%s", character_id
            )
            return _character_redirect(character_id, error=str(exc))
        return RedirectResponse(
            f"/chats/{chat.id}",
            status_code=303,
        )


def _register_chat_routes(context: RouteContext) -> None:
    @context.route("POST", "/chats/new", "create_chat")
    async def create_chat(request: Request) -> Response:
        return await context.mutations.perform(
            request, context.state.new_chat, notice="New chat created"
        )

    @context.route("POST", "/chat/draft", "save_draft")
    async def save_draft(request: Request) -> Response:
        _ = await context.access.require(request, require_csrf=True)
        form = await _form_values(request)
        try:
            revision = await context.state.inference.update_draft(
                form.required_text("chat_id"),
                _revision(form, "draft_revision"),
                form.text("draft"),
                images=context.state.session(
                    form.required_text("chat_id")
                ).chat.draft_images,
            )
        except RevisionConflict as exc:
            return PlainTextResponse(str(exc), status_code=409)
        except (ValueError, TypeError, KeyError) as exc:
            return PlainTextResponse(str(exc), status_code=400)
        return JSONResponse({"draft_revision": revision})

    async def command(request: Request) -> Response:
        form = await _form_values(request)
        action = request.url.path.rsplit("/", 1)[-1]
        runtime = context.state.inference

        async def operation() -> object:
            chat_id = form.required_text("chat_id")
            chat = context.state.session(chat_id).chat
            revision = _revision(form)
            message_id = _path_text(request, "message_id", required=False) or form.text(
                "message_id"
            )
            if action == "send":
                return await runtime.send_message(
                    chat_id,
                    revision,
                    form.required_text("submission_id"),
                    form.text("content"),
                    images=image_attachments_from_json(
                        json.loads(form.text("images", default="[]"))
                    ),
                    generate=form.flag("generate") if form.has("generate") else True,
                    expected_draft_revision=_revision(form, "draft_revision"),
                )
            if action == "stop":
                await runtime.cancel(chat_id)
                return None
            if action == "retry-draft-save":
                await runtime.flush_drafts()
                return None
            if action == "retry-save":
                await runtime.retry_save(chat_id)
                return None
            if action == "dismiss-error":
                await runtime.dismiss_error(chat_id)
                return None
            if action == "delete" and not message_id:
                await runtime.delete_chat(chat_id, revision)
                return None
            if action == "fork":
                return await runtime.fork_chat(chat_id, revision, message_id)
            if action in {"regenerate", "continue", "resend"}:
                message = context.mutations.require_visible_message(chat_id, message_id)
                if not chat.messages or message is not chat.messages[-1]:
                    raise ValueError("generation action requires the final message")
                required_role = "user" if action == "resend" else "assistant"
                if message.role != required_role:
                    raise ValueError("invalid generation target")
                mode = (
                    GenerationMode.REPLY
                    if action == "resend"
                    else GenerationMode.REGENERATE
                    if action == "regenerate"
                    else GenerationMode.CONTINUE
                )
                return await runtime.generate(chat_id, revision, mode=mode)
            if action == "backend":
                await runtime.select_backend(
                    chat_id, revision, BackendId(form.required_text("backend_id"))
                )
                return None
            if action in {"model", "sampling"}:
                selected = chat.inference.selected_backend_id
                if selected is None:
                    raise ValueError("select a backend first")
                profile = chat.inference.profiles[selected]
                if action == "model":
                    profile = replace(profile, model_id=form.text("model") or None)
                else:
                    profile = replace(profile, parameters=_sampling(form))
                await runtime.set_profile(chat_id, revision, profile)
                return None
            if action == "reset" and request.url.path.startswith("/chat/sampling"):
                await runtime.reset_profile(chat_id, revision)
                return None

            def edit(candidate: Chat) -> None:
                if action == "rename":
                    title = form.required_text("title").strip()
                    if not title:
                        raise ValueError("title cannot be blank")
                    candidate.set_manual_title(title)
                elif action in {"prompt", "reset"}:
                    candidate.system_prompt = (
                        form.text("prompt").strip() if action == "prompt" else ""
                    )
                elif action == "british-spellings":
                    candidate.postprocess_british_spellings = form.flag("enabled")
                elif action == "reasoning":
                    candidate.save_reasoning = form.flag("enabled")
                else:
                    message = candidate.find_message(message_id)
                    if message is None or message.role not in {"user", "assistant"}:
                        raise ValueError("message action requires a visible message")
                    if action == "edit":
                        content = form.text("content")
                        if not content.strip() and not message.images:
                            raise ValueError("message cannot be empty")
                        candidate.update_message(message_id, content)
                    elif action == "delete":
                        candidate.remove_message(message_id)
                    elif action == "delete-following":
                        candidate.remove_messages_from(message_id)
                    elif action == "truncate":
                        candidate.truncate_to_message(message_id)
                    elif action == "merge":
                        merged = candidate.merge_message_with_previous(message_id)
                        context.state.images.limits.validate_image_count(
                            len(merged.images)
                        )
                    else:
                        raise ValueError("unknown chat action")

            await runtime.edit_chat(chat_id, revision, edit)
            return None

        return await context.mutations.perform(request, operation, notice="")

    for path in (
        "/chat/send",
        "/chat/stop",
        "/chat/retry-draft-save",
        "/chat/retry-save",
        "/chat/dismiss-error",
        "/chats/{chat_id}/rename",
        "/chats/{chat_id}/delete",
        "/messages/edit",
        *(
            f"/messages/{{message_id}}/{action}"
            for action in (
                "delete",
                "truncate",
                "delete-following",
                "regenerate",
                "continue",
                "resend",
                "fork",
                "merge",
            )
        ),
        *(
            f"/chat/{action}"
            for action in (
                "backend",
                "model",
                "sampling",
                "sampling/reset",
                "prompt",
                "prompt/reset",
                "british-spellings",
                "reasoning",
            )
        ),
    ):
        context.route(
            "POST",
            path,
            "chat_" + path.replace("/", "_").replace("{", "").replace("}", ""),
        )(command)

    @context.route("POST", "/chats/{chat_id}/images", "upload_chat_images")
    async def upload_chat_images(request: Request) -> Response:
        _ = await context.access.require(request)
        form = _FormValues(
            await request.form(
                max_files=context.state.images.limits.max_images_per_message,
                max_part_size=context.state.images.limits.max_image_bytes,
            )
        )

        async def operation() -> None:
            chat_id = form.required_text("chat_id")
            if chat_id != _path_text(request, "chat_id"):
                raise ValueError("chat target mismatch")
            uploads = form.uploads("images")
            if not uploads:
                raise ValueError("select at least one image")
            images: list[ImageAttachment] = []
            total_bytes = 0
            limits = context.state.images.limits
            for upload in uploads:
                data = await upload.read(limits.max_image_bytes + 1)
                total_bytes += len(data)
                if total_bytes > limits.max_batch_bytes:
                    raise ValueError("the combined images exceed the upload limit")
                images.append(
                    await run_in_worker(
                        context.state.images.store, data, upload.filename or "image"
                    )
                )
            chat = context.state.session(chat_id).chat
            merged_images = tuple(
                {image.id: image for image in (*chat.draft_images, *images)}.values()
            )
            limits.validate_image_count(len(merged_images))
            await context.state.inference.update_draft(
                chat_id,
                _revision(form, "draft_revision"),
                chat.draft,
                images=merged_images,
            )

        try:
            result = await context.mutations.perform(request, operation, notice="")
            if result.status_code != 303:
                return result
            chat = context.state.session(form.required_text("chat_id")).chat
            return JSONResponse({"draft_revision": chat.draft_revision})
        finally:
            await request.close()

    @context.route(
        "POST", "/chats/{chat_id}/images/{image_id}/remove", "remove_chat_image"
    )
    async def remove_chat_image(request: Request) -> Response:
        form = await _form_values(request)

        async def operation() -> None:
            chat_id = form.required_text("chat_id")
            if chat_id != _path_text(request, "chat_id"):
                raise ValueError("chat target mismatch")
            chat = context.state.session(chat_id).chat
            image_id = _path_text(request, "image_id")
            await context.state.inference.update_draft(
                chat_id,
                _revision(form, "draft_revision"),
                chat.draft,
                images=tuple(
                    image for image in chat.draft_images if image.id != image_id
                ),
            )

        result = await context.mutations.perform(request, operation, notice="")
        if result.status_code != 303:
            return result
        return JSONResponse(
            {
                "draft_revision": context.state.session(
                    form.required_text("chat_id")
                ).chat.draft_revision
            }
        )


def _revision(form: _FormValues, field: str = "revision") -> int:
    value = int(form.required_text(field))
    if value < 0:
        raise ValueError("revision cannot be negative")
    return value


def _sampling(form: _FormValues) -> SamplingParameters:
    return SamplingParameters(
        temperature=float(form.required_text("temperature")),
        top_p=float(form.required_text("top_p")),
        max_tokens=int(form.required_text("max_tokens")),
    )


def _register_settings_routes(context: RouteContext) -> None:

    @context.route("POST", "/settings/global/prompt", "save_global_prompt")
    async def save_global_prompt(request: Request) -> Response:
        form: _FormValues = await _form_values(request)
        return await context.mutations.perform(
            request,
            lambda: context.state.set_global_system_prompt(form.text("system_prompt")),
            notice="Global system prompt saved",
            require_global_settings=True,
            dialog="global",
            dialog_tab="behaviour",
        )

    @context.route("POST", "/settings/global/prompt/reset", "reset_global_prompt")
    async def reset_global_prompt(request: Request) -> Response:
        return await context.mutations.perform(
            request,
            context.state.reset_global_system_prompt,
            notice="Global system prompt reset",
            require_global_settings=True,
            dialog="global",
            dialog_tab="behaviour",
        )

    @context.route("POST", "/settings/global/appearance", "save_global_appearance")
    async def save_global_appearance(request: Request) -> Response:
        form: _FormValues = await _form_values(request)

        async def operation() -> None:
            theme: ThemeSettings = _updated_theme_settings(context.state, form)
            icon_colors: IconColorSettings = _updated_icon_color_settings(
                context.state, form
            )
            message_action_style: MessageActionStyle = _message_action_style_from_form(
                context.state, form
            )
            await context.state.set_appearance_settings(
                theme,
                icon_colors,
                dark_mode=form.flag("dark_mode"),
                message_action_style=message_action_style,
            )

        return await context.mutations.perform(
            request,
            operation,
            notice="Appearance settings saved",
            require_global_settings=True,
            dialog="global",
            dialog_tab="appearance",
        )

    @context.route(
        "POST", "/settings/global/appearance/reset", "reset_global_appearance"
    )
    async def reset_global_appearance(request: Request) -> Response:
        return await context.mutations.perform(
            request,
            context.state.reset_appearance_settings,
            notice="Appearance settings reset",
            require_global_settings=True,
            dialog="global",
            dialog_tab="appearance",
        )

    @context.route(
        "POST",
        "/settings/global/appearance/{colorway}/reset",
        "reset_theme_colorway",
    )
    async def reset_theme_colorway(request: Request) -> Response:
        colorway: ThemeColorway = _theme_colorway_from_text(
            _path_text(request, "colorway")
        )
        return await context.mutations.perform(
            request,
            lambda: context.state.reset_theme_colorway(colorway),
            notice=f"{colorway.label} colourway reset",
            require_global_settings=True,
            dialog="global",
            dialog_tab="appearance",
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
            dialog_tab="behaviour",
        )

    @context.route(
        "POST", "/settings/global/generation/reset", "reset_global_generation"
    )
    async def reset_global_generation(request: Request) -> Response:
        return await context.mutations.perform(
            request,
            context.state.reset_generation_defaults,
            notice="Generation defaults reset",
            require_global_settings=True,
            dialog="global",
            dialog_tab="behaviour",
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

    @context.route("POST", "/access/current-device/pending", "save_pending_device")
    async def save_pending_device(request: Request) -> Response:
        await require_csrf_token(request)
        form = await _form_values(request)
        try:
            await context.access.approve_with_phrase(
                request, form.text("approval_phrase"), label=form.text("label")
            )
        except (ValueError, TypeError, KeyError) as exc:
            return PlainTextResponse(str(exc), status_code=400)
        return context.mutations.redirect(notice="Browser approved")

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

    @context.route("POST", "/access/phrase", "save_approval_phrase")
    async def save_approval_phrase(request: Request) -> Response:
        form = await _form_values(request)
        return await context.mutations.perform(
            request,
            lambda: context.state.set_approval_phrase(form.text("approval_phrase")),
            notice="Approval phrase saved",
            require_access_management=True,
            dialog="access",
        )

    @context.route("POST", "/access/devices/{device_id}", "save_device_access")
    async def save_device_access(request: Request) -> Response:
        device_id: str = _path_text(request, "device_id")
        form: _FormValues = await _form_values(request)
        if form.flag("access_allowed"):
            raise HTTPException(400, "enrollment requires the approval phrase")
        return await context.mutations.perform(
            request,
            lambda: context.state.set_device_access_allowed(device_id, False),
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


def _path_text(request: Request, name: str, *, required: bool = True) -> str:
    """Return one non-empty FastHTML path parameter."""

    value: object = request.path_params.get(name)
    if not isinstance(value, str) or not value:
        if not required:
            return ""
        raise HTTPException(400, f"invalid {name}")
    return value


def _theme_colorway_from_text(value: str) -> ThemeColorway:
    """Parse one URL colourway identifier at the HTTP boundary."""

    try:
        return ThemeColorway(value)
    except ValueError as exc:
        raise HTTPException(400, "invalid colourway") from exc


def _query_text(request: Request, name: str) -> str:
    """Return a single query value without mutating-route binding precedence."""

    return request.query_params.get(name, "")


def _query_dialog_name(request: Request) -> _DialogName:
    """Return a known dialog name, treating arbitrary query values as absent."""

    value: str = _query_text(request, "dialog")
    return cast(_DialogName, value) if value in _DIALOG_NAMES else ""


def _chat_id(request: Request) -> str:
    value = _path_text(request, "chat_id", required=False) or _query_text(
        request, "chat_id"
    )
    if not value:
        raise HTTPException(400, "explicit chat_id is required")
    return value


def _register_backend_routes(context: RouteContext) -> None:
    @context.route("POST", "/settings/backends/new", "new_backend")
    async def new_backend(request: Request) -> Response:
        form = await _form_values(request)

        async def operation() -> None:
            backend = BackendConfig.create(
                BackendKind(form.required_text("kind")),
                form.required_text("name").strip(),
                form.required_text("base_url").strip(),
            )
            await context.state.inference.update_backend(backend)

        return await context.mutations.perform(
            request,
            operation,
            notice="Backend added",
            require_global_settings=True,
            dialog="global",
            dialog_tab="backends",
        )

    @context.route("POST", "/settings/backends/default", "default_backend")
    async def default_backend(request: Request) -> Response:
        form = await _form_values(request)
        return await context.mutations.perform(
            request,
            lambda: context.state.inference.set_default_backend(
                BackendId(form.text("backend_id")) if form.text("backend_id") else None,
                expected_fingerprint=form.required_text("backend_fingerprint"),
            ),
            notice="Default backend saved",
            require_global_settings=True,
            dialog="global",
            dialog_tab="backends",
        )

    @context.route(
        "POST", "/settings/backends/{backend_id}/{action}", "backend_command"
    )
    async def backend_command(request: Request) -> Response:
        form = await _form_values(request)

        async def operation() -> None:
            backend_id = BackendId(_path_text(request, "backend_id"))
            runtime = context.state.inference
            try:
                backend = context.config.backends.backend(backend_id)
            except KeyError as exc:
                raise RevisionConflict(
                    "This backend was removed. Your edits are preserved; copy them before reloading."
                ) from exc
            action = _path_text(request, "action")
            if action == "test":
                await runtime.refresh_backend(backend_id)
                health = runtime.health(backend_id)
                if health.state is not BackendReachability.ONLINE:
                    raise ValueError("backend is unavailable: " + health.state.value)
                error = runtime.discovery_error(backend_id)
                if error:
                    raise error
                return
            if action == "remove":
                await runtime.remove_backend(
                    backend_id,
                    expected_fingerprint=form.required_text("backend_fingerprint"),
                )
                return
            if action in {"load", "unload"}:
                model = ModelRef(backend_id, form.required_text("model_id"))
                if action == "load":
                    await runtime.load_model(model)
                else:
                    await runtime.unload_model(model)
                return
            runtime.validate_backend_snapshot(
                backend, form.required_text("backend_fingerprint")
            )
            if action == "save":
                management = (
                    ModelManagementSettings(
                        automatic_loading=form.flag("automatic_loading"),
                        idle_unload_minutes=_optional_integer(
                            form.text("idle_unload_minutes"),
                            field_name="idle unload minutes",
                        ),
                    )
                    if backend.model_management
                    else None
                )
                candidate = replace(
                    backend,
                    name=form.required_text("name").strip(),
                    base_url=form.required_text("base_url").strip(),
                    enabled=form.flag("enabled"),
                    concurrency_limit=int(form.required_text("concurrency_limit")),
                    defaults=GenerationProfile(
                        model_id=form.text("model") or None, parameters=_sampling(form)
                    ),
                    model_management=management,
                )
            elif action == "alias":
                candidate = backend.set_model_alias(
                    form.required_text("model_id"), form.text("alias").strip()
                )
            elif action == "register-model":
                model_id = form.required_text("model_id").strip()
                candidate = backend.register_model(
                    model_id, alias=form.text("alias").strip()
                )
            else:
                raise ValueError("unknown backend action")
            await runtime.update_backend(
                candidate,
                expected=backend,
                expected_fingerprint=form.required_text("backend_fingerprint"),
            )

        return await context.mutations.perform(
            request,
            operation,
            notice="Backend updated",
            require_global_settings=True,
            dialog="global",
            dialog_tab="backends",
        )


def _character_redirect(
    character_id: str = "", *, notice: str = "", error: str = ""
) -> RedirectResponse:
    values = {
        key: value[:240]
        for key, value in {"notice": notice, "error": error}.items()
        if value
    }
    path = f"/characters/{character_id}" if character_id else "/characters"
    return RedirectResponse(
        path + (f"?{urlencode(values)}" if values else ""), status_code=303
    )


def _existing_chat_id(state: AppState, request: Request) -> str:
    chat_id = _chat_id(request)
    if chat_id not in state.registry:
        raise HTTPException(404, "chat no longer exists")
    return chat_id
