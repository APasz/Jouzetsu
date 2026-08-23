"""Browser security primitives for the FastHTML presentation layer.

The application deliberately avoids framework sessions.  A per-browser,
HTTP-only double-submit token gives ordinary HTML forms CSRF protection without
coupling the domain layer to a web-session implementation.
"""

from __future__ import annotations

import re
import secrets
from http.cookies import SimpleCookie
from typing import Final, cast

from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

CSRF_COOKIE_NAME: Final[str] = "jouzetsu_csrf"
CSRF_FORM_FIELD: Final[str] = "_jouzetsu_csrf"
CSRF_HEADER_NAME: Final[str] = "x-jouzetsu-csrf"
CSRF_TOKEN_STATE_KEY: Final[str] = "jouzetsu_csrf_token"
CSRF_COOKIE_MAX_AGE_SECONDS: Final[int] = 60 * 60 * 24 * 30
_CSRF_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_-]{32,128}$")
_CONTENT_SECURITY_POLICY: Final[str] = (
    "default-src 'self'; base-uri 'self'; object-src 'none'; frame-ancestors 'none'; "
    "form-action 'self'; connect-src 'self'; img-src 'self' data:; "
    "style-src 'self'; script-src 'self'"
)


class BrowserSecurityMiddleware:
    """Issue a CSRF token and add browser-facing defensive response headers."""

    def __init__(self, app: ASGIApp) -> None:
        self._app: ASGIApp = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        token: str = _csrf_token_from_cookie(scope)
        issue_cookie: bool = not bool(token)
        if not token:
            token = secrets.token_urlsafe(32)
        state: dict[str, object] = cast(
            dict[str, object], scope.setdefault("state", {})
        )
        state[CSRF_TOKEN_STATE_KEY] = token

        async def send_with_security_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                _ = headers.setdefault(
                    "Content-Security-Policy", _CONTENT_SECURITY_POLICY
                )
                _ = headers.setdefault("X-Frame-Options", "DENY")
                _ = headers.setdefault("X-Content-Type-Options", "nosniff")
                _ = headers.setdefault("Referrer-Policy", "same-origin")
                content_type: str = headers.get("content-type", "")
                if content_type.startswith("text/html"):
                    _ = headers.setdefault("Cache-Control", "no-store")
                if issue_cookie:
                    headers.append("Set-Cookie", _csrf_cookie_header(token))
            await send(message)

        await self._app(scope, receive, send_with_security_headers)


def csrf_token_for_request(request: Request) -> str:
    """Return the middleware-issued CSRF token for a rendered browser page."""

    token: object = getattr(request.state, CSRF_TOKEN_STATE_KEY, "")
    if not isinstance(token, str) or not _CSRF_TOKEN_PATTERN.fullmatch(token):
        raise RuntimeError("CSRF middleware did not provide a valid request token")
    return token


async def require_csrf_token(request: Request) -> None:
    """Reject an unsafe request lacking the current browser's CSRF token."""

    expected: str = csrf_token_for_request(request)
    supplied: str = request.headers.get(CSRF_HEADER_NAME, "")
    if not supplied:
        form = await request.form()
        raw_token: object = form.get(CSRF_FORM_FIELD, "")
        supplied = raw_token if isinstance(raw_token, str) else ""
    if not secrets.compare_digest(expected, supplied):
        raise HTTPException(403, "invalid or missing CSRF token")

    origin: str = request.headers.get("origin", "")
    if origin and not secrets.compare_digest(
        origin.rstrip("/"), str(request.base_url).rstrip("/")
    ):
        raise HTTPException(403, "cross-origin mutation request rejected")


def _csrf_token_from_cookie(scope: Scope) -> str:
    """Read a syntactically valid CSRF cookie without constructing a Request."""

    raw_headers: object = cast(object, scope.get("headers", []))
    if not isinstance(raw_headers, list):
        return ""
    headers: list[object] = cast(list[object], raw_headers)
    for raw_header in headers:
        if not isinstance(raw_header, tuple):
            continue
        header: tuple[object, ...] = cast(tuple[object, ...], raw_header)
        if (
            len(header) != 2
            or not isinstance(header[0], bytes)
            or not isinstance(header[1], bytes)
        ):
            continue
        raw_name: bytes = header[0]
        raw_value: bytes = header[1]
        if raw_name.lower() != b"cookie":
            continue
        cookies: SimpleCookie = SimpleCookie()
        try:
            cookies.load(raw_value.decode("latin-1"))
        except UnicodeDecodeError, ValueError:
            return ""
        morsel = cookies.get(CSRF_COOKIE_NAME)
        value: str = morsel.value if morsel is not None else ""
        return value if _CSRF_TOKEN_PATTERN.fullmatch(value) else ""
    return ""


def _csrf_cookie_header(token: str) -> str:
    """Build the one HTTP-only cookie header used by the double-submit token."""

    cookies: SimpleCookie = SimpleCookie()
    cookies[CSRF_COOKIE_NAME] = token
    morsel = cookies[CSRF_COOKIE_NAME]
    _ = morsel.__setitem__("path", "/")
    _ = morsel.__setitem__("max-age", str(CSRF_COOKIE_MAX_AGE_SECONDS))
    _ = morsel.__setitem__("samesite", "Strict")
    _ = morsel.__setitem__("httponly", True)
    return morsel.OutputString()
