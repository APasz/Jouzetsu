"""Request-scoped device access evaluation and enforcement."""

from __future__ import annotations

import asyncio
import logging
import socket
from dataclasses import dataclass
from typing import Final

from starlette.exceptions import HTTPException
from starlette.requests import Request

from ..access import (
    DEVICE_ID_COOKIE_NAME,
    AccessDecision,
    access_decision,
    is_localhost_ip,
)
from ..async_workers import run_in_worker
from ..config import AppConfig, DeviceAccessSettings
from ..state import AppState
from .security import csrf_token_for_request, require_csrf_token
from .views.context import PageContext

log: logging.Logger = logging.getLogger(__name__)
_REVERSE_DNS_TIMEOUT_SECONDS: Final[float] = 1.0
_MAX_PENDING_DEVICES: Final[int] = 64


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Access result plus display information derived from one request."""

    page: PageContext
    client_ip: str


class RequestAccess:
    """Evaluate and enforce browser-device permissions for web endpoints."""

    def __init__(self, config: AppConfig, state: AppState) -> None:
        self._config: AppConfig = config
        self._state: AppState = state
        self._device_registration_lock: asyncio.Lock = asyncio.Lock()

    async def context(
        self, request: Request, *, register_device: bool
    ) -> RequestContext:
        """Evaluate a request without trusting browser-provided identity fields."""

        client_ip: str = request.client.host if request.client is not None else ""
        is_localhost: bool = is_localhost_ip(client_ip)
        client_label: str = _host_label() if is_localhost else client_ip
        raw_device_id: object = request.cookies.get(DEVICE_ID_COOKIE_NAME, "")
        decision: AccessDecision = access_decision(
            self._config,
            client_ip=client_ip,
            raw_device_id=raw_device_id,
        )
        if (
            register_device
            and self._config.access.default_private
            and decision.device_id
        ):
            await self._record_device_activity(
                decision.device_id,
                client_ip=client_ip,
                client_label=client_label,
                is_localhost=is_localhost,
            )
            decision = access_decision(
                self._config,
                client_ip=client_ip,
                raw_device_id=raw_device_id,
            )
        return RequestContext(
            page=PageContext(
                decision=decision,
                client_label=client_label,
                csrf_token=csrf_token_for_request(request),
            ),
            client_ip=client_ip,
        )

    async def require(
        self,
        request: Request,
        *,
        require_csrf: bool = False,
        require_global_settings: bool = False,
        require_access_management: bool = False,
    ) -> RequestContext:
        """Guard streams and mutations, including localhost recovery flows."""

        if require_csrf:
            await require_csrf_token(request)
        context: RequestContext = await self.context(request, register_device=False)
        decision: AccessDecision = context.page.decision
        allow_pending_local_admin: bool = (
            require_access_management and decision.can_manage_access
        )
        if not decision.access_allowed and not allow_pending_local_admin:
            raise HTTPException(403, "device access has not been approved")
        if require_global_settings and not decision.can_use_global_settings:
            raise HTTPException(403, "global settings are not allowed for this device")
        if require_access_management and not decision.can_manage_access:
            raise HTTPException(403, "access management is localhost-only")
        return context

    async def record_current_device_activity(self, request: Request) -> None:
        """Record a visible browser's activity without granting it any access."""

        context: RequestContext = await self.context(request, register_device=False)
        if not self._config.access.default_private:
            return
        device_id: str = context.page.decision.device_id
        if not device_id:
            return
        await self._refresh_known_device_activity(
            device_id,
            client_ip=context.client_ip,
            client_label=context.page.client_label,
            is_localhost=context.page.decision.is_localhost,
        )

    async def _record_device_activity(
        self,
        device_id: str,
        *,
        client_ip: str,
        client_label: str,
        is_localhost: bool,
    ) -> None:
        """Register an unknown browser or quietly refresh a known one."""

        if device_id in self._config.access.devices:
            await self._refresh_known_device_activity(
                device_id,
                client_ip=client_ip,
                client_label=client_label,
                is_localhost=is_localhost,
            )
            return
        await self._register_pending_device(
            device_id,
            client_ip=client_ip,
            client_label=client_label,
            is_localhost=is_localhost,
        )

    async def _refresh_known_device_activity(
        self,
        device_id: str,
        *,
        client_ip: str,
        client_label: str,
        is_localhost: bool,
    ) -> None:
        """Refresh an existing device, resolving hostname only after an IP change."""

        device: DeviceAccessSettings | None = self._config.access.devices.get(device_id)
        if device is None:
            return
        hostname: str = ""
        hostname_observed: bool = False
        if is_localhost:
            hostname = _host_label()
            hostname_observed = True
        elif device.last_ip != client_ip:
            hostname = await _reverse_hostname(client_ip)
            hostname_observed = True
        await self._state.refresh_access_device_activity(
            device_id,
            client_label,
            last_ip=client_ip,
            hostname=hostname,
            hostname_observed=hostname_observed,
        )

    async def _register_pending_device(
        self,
        device_id: str,
        *,
        client_ip: str,
        client_label: str,
        is_localhost: bool,
    ) -> None:
        if device_id in self._config.access.devices:
            return
        async with self._device_registration_lock:
            if device_id in self._config.access.devices:
                return
            pending_count: int = sum(
                1
                for device in self._config.access.devices.values()
                if not device.access_allowed
            )
            if pending_count >= _MAX_PENDING_DEVICES and not is_localhost:
                log.warning(
                    "pending device registration limit reached client_ip=%s", client_ip
                )
                return
            hostname: str = (
                _host_label() if is_localhost else await _reverse_hostname(client_ip)
            )
            await self._state.register_access_device(
                device_id,
                client_label,
                last_ip=client_ip,
                hostname=hostname,
            )


def _host_label() -> str:
    """Return a useful local label without allowing an empty host name into config."""

    return socket.gethostname().strip() or "localhost"


async def _reverse_hostname(client_ip: str) -> str:
    """Resolve a remote host label with a bounded wait outside the event loop."""

    try:
        hostname, _, _ = await asyncio.wait_for(
            run_in_worker(socket.gethostbyaddr, client_ip),
            timeout=_REVERSE_DNS_TIMEOUT_SECONDS,
        )
    except TimeoutError, OSError, socket.herror:
        return ""
    return hostname.strip()
