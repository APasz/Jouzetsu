"""Shared HTTP test helpers for protected application routes."""

from __future__ import annotations

import re

import httpx

from jouzetsu.state import AppState
from jouzetsu.web.application import WebApplication
from jouzetsu.web.security import CSRF_HEADER_NAME


async def noop() -> None:
    return None


def csrf_headers(page: httpx.Response) -> dict[str, str]:
    """Read the browser token from rendered markup for the next mutation."""

    match = re.search(r'data-csrf-token="([A-Za-z0-9_-]{32,128})"', page.text)
    assert match is not None, "rendered page did not include a CSRF token"
    return {CSRF_HEADER_NAME: match.group(1)}


async def close_web_application(web: WebApplication, state: AppState) -> None:
    web._event_broker.close()  # pyright: ignore[reportPrivateUsage]
    _ = await state.shutdown()


def transport(web: WebApplication, *, client_ip: str) -> httpx.ASGITransport:
    return httpx.ASGITransport(
        app=web.app, raise_app_exceptions=True, client=(client_ip, 43123)
    )
