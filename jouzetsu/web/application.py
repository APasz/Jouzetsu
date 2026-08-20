"""FastHTML application composition root."""

# FastHTML's route decorator registers nested handlers dynamically.
# pyright: reportUnusedFunction=false

from __future__ import annotations

import mimetypes
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Final, Literal, cast

from fastcore.xml import FT  # pyright: ignore[reportMissingTypeStubs]
from fasthtml.core import FastHTML
from starlette.requests import Request
from starlette.responses import Response

from ..async_workers import run_in_worker
from ..character_presets import CharacterPresetCatalog
from ..config import (
    AppConfig,
)
from ..models import Character, CharacterField
from ..state import AppState
from .form_data import (
    FormValues as _FormValues,
)
from .form_data import (
    character_fields_from_form as _character_fields_from_form,
)
from .form_data import (
    require_integer as _require_integer,
)
from .html import Link, Meta, Script
from .icon import app_icon_url
from .mutations import Mutations as _Mutations
from .request_access import RequestAccess as _RequestAccess
from .route_registry import RouteContext, register_routes
from .security import (
    BrowserSecurityMiddleware,
)
from .ui_events import UiEventBroker as _UiEventBroker
from .view_models import chat_settings_view as _chat_settings_view
from .view_models import chat_view as _chat_view
from .view_models import navigation_view as _navigation_view
from .views.chat import (
    ChatSettingsView,
    ChatView,
    NavigationView,
)
from .views.context import PageContext

_ASSET_DIRECTORY: Path = Path(__file__).resolve().parent / "static"
_THEME_STYLESHEETS: Final[tuple[str, ...]] = (
    "theme/foundation.css",
    "theme/chat.css",
    "theme/composer.css",
    "theme/panels.css",
    "theme/workspace.css",
    "theme/diagnostics.css",
    "theme/notices.css",
    "theme/responsive.css",
)
# FastHTML reads or creates ``.sesskey`` even with sessions disabled.  This
# value is deliberately unused because ``sess_cls=None``; it just prevents a
# surprising filesystem write during app construction.
_UNUSED_FASTHTML_SESSION_KEY: Final[str] = "jouzetsu-no-session-key"

type StartupCallback = Callable[[], Awaitable[None]]
type ShutdownCallback = Callable[[], Awaitable[None]]
type ExitCallback = Callable[[], None]
type HttpMethod = Literal["GET", "POST"]
type RouteResult = FT | Response
type RouteHandler = Callable[..., Awaitable[RouteResult]]


def _static_asset_url(filename: str) -> str:
    """Return a deployment-specific static asset URL so browser caches refresh safely."""

    asset_path: Path = _ASSET_DIRECTORY / filename
    try:
        version: int = asset_path.stat().st_mtime_ns
    except OSError:
        return f"/static/{filename}"
    return f"/static/{filename}?v={version}"


async def _static_asset_response(asset_path: str) -> Response:
    """Serve one packaged static asset without the default async worker pool."""

    static_directory: Path = _ASSET_DIRECTORY.resolve()
    resolved_asset: Path = (static_directory / asset_path).resolve()
    if not resolved_asset.is_relative_to(static_directory) or not resolved_asset.is_file():
        return Response(status_code=404)
    try:
        content: bytes = await run_in_worker(resolved_asset.read_bytes)
    except OSError:
        return Response(status_code=404)
    media_type: str = mimetypes.guess_type(resolved_asset.name)[0] or "application/octet-stream"
    return Response(content, media_type=media_type)


class WebApplication:
    """Own the FastHTML layer while keeping application state framework-agnostic."""

    def __init__(
        self,
        config: AppConfig,
        state: AppState,
        *,
        on_startup: StartupCallback,
        on_shutdown: ShutdownCallback,
    ) -> None:
        self.config: AppConfig = config
        self.state: AppState = state
        self.character_preset_catalog: CharacterPresetCatalog = (
            CharacterPresetCatalog.load(config.character_presets_directory)
        )
        self._on_startup: StartupCallback = on_startup
        self._on_shutdown: ShutdownCallback = on_shutdown
        self._event_broker: _UiEventBroker = _UiEventBroker(state)
        self._access: _RequestAccess = _RequestAccess(config, state)
        self._mutations: _Mutations = _Mutations(state, self._access)
        self._exit_callback: ExitCallback | None = None
        self.app: FastHTML = FastHTML(
            title="Jouzetsu",
            hdrs=(
                Meta(charset="utf-8"),
                Meta(
                    name="viewport",
                    content="width=device-width, initial-scale=1, viewport-fit=cover",
                ),
                Link(
                    rel="icon",
                    href=app_icon_url(config.ui.icon_colors),
                    type="image/svg+xml",
                ),
                *(
                    Link(rel="stylesheet", href=_static_asset_url(stylesheet))
                    for stylesheet in _THEME_STYLESHEETS
                ),
                Link(rel="stylesheet", href="/theme.css"),
                Script(src=_static_asset_url("app.js"), type="module"),
            ),
            default_hdrs=False,
            htmx=False,
            surreal=False,
            canonical=False,
            sess_cls=None,  # pyright: ignore[reportArgumentType]
            secret_key=_UNUSED_FASTHTML_SESSION_KEY,
            lifespan=self._lifespan,
            htmlkw={"lang": "en"},
        )
        self.app.add_middleware(BrowserSecurityMiddleware)

        @self._route("GET", "/static/{asset_path:path}", "static_asset")
        async def static_asset(request: Request) -> Response:
            asset_path: object = request.path_params.get("asset_path", "")
            if not isinstance(asset_path, str):
                return Response(status_code=404)
            return await _static_asset_response(asset_path)

        register_routes(
            RouteContext(
                route=self._route,
                config=config,
                state=state,
                character_presets=self.character_preset_catalog,
                events=self._event_broker,
                access=self._access,
                mutations=self._mutations,
                chat_view=self._chat_view,
                chat_settings_view=self._chat_settings_view,
                navigation_view=self._navigation_view,
                update_character=self._update_character_from_form,
                request_exit=self._request_exit,
            )
        )

    def set_exit_callback(self, callback: ExitCallback) -> None:
        """Allow the production server runner to honour localhost Exit requests."""

        self._exit_callback = callback

    def _request_exit(self) -> bool:
        """Request process shutdown when this hosting context supports it."""

        if self._exit_callback is None:
            return False
        self._exit_callback()
        return True

    async def _lifespan(self, _app: FastHTML) -> AsyncIterator[None]:
        self._event_broker.start()
        try:
            await self._on_startup()
        except Exception:
            self._event_broker.close()
            raise
        try:
            yield
        finally:
            self._event_broker.close()
            await self._on_shutdown()

    def _route(
        self, method: HttpMethod, path: str, name: str
    ) -> Callable[[RouteHandler], RouteHandler]:
        """Register one typed FastHTML endpoint through its declared route API."""

        route: Callable[..., object] = cast(Callable[..., object], self.app.route)
        decorator: Callable[[RouteHandler], object] = cast(
            Callable[[RouteHandler], object],
            route(path, methods=method, name=name),
        )

        def register(handler: RouteHandler) -> RouteHandler:
            _ = decorator(handler)
            return handler

        return register

    def _chat_view(self) -> ChatView:
        """Capture the current chat data required by the pure chat renderers."""

        return _chat_view(self.state)

    def _chat_settings_view(self) -> ChatSettingsView:
        """Capture the active chat's independently refreshable settings data."""

        return _chat_settings_view(self.state)

    def _navigation_view(self, context: PageContext) -> NavigationView:
        """Capture the request-aware data required by the navigation drawer."""

        return _navigation_view(self.state, context)

    async def _update_character_from_form(
        self,
        character_id: str,
        form: _FormValues,
        *,
        additional_fields: list[CharacterField] | None = None,
    ) -> Character:
        """Validate and persist one editor submission before a character-dependent action."""

        revision: int = _require_integer(
            form.required_text("revision"), field_name="character revision"
        )
        fields: list[CharacterField] = _character_fields_from_form(form)
        if additional_fields:
            fields.extend(additional_fields)
        return await self.state.update_character(
            character_id,
            expected_revision=revision,
            name=form.required_text("name"),
            fields=fields,
        )
