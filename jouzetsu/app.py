"""Application factory and ASGI lifecycle glue for Jouzetsu.

The domain layer remains independent of the web framework. This module builds
it once, hands it to the FastHTML presentation layer, and owns process-level
startup, shutdown, locking, and browser-opening concerns.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import webbrowser
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from logging import Logger
from pathlib import Path
from typing import Final, cast

import uvicorn

from .atomic_write import atomic_write_text
from .character_storage import CharacterStorage
from .config import (
    AppConfig,
    AppPaths,
    ConfigStore,
    EnvironmentOverrides,
    resolve_app_home,
)
from .lmstudio import LMStudioClient
from .logging_config import configure_bootstrap_logging, configure_logging
from .process_lock import LinuxDataDirectoryLock
from .state import AppState
from .storage import ChatStorage
from .web.application import WebApplication

log: Logger = logging.getLogger(__name__)
UNCLEAN_SHUTDOWN_MARKER_FILE_NAME: str = "unclean_shutdown.marker"
_UVICORN_GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS: Final[int] = 5

_state: AppState | None = None
_web_application: WebApplication | None = None
_data_directory_lock: LinuxDataDirectoryLock | None = None
_config_store: ConfigStore | None = None


def _browser_host(bind_host: str) -> str:
    """Prefer a browser-safe local host when binding all interfaces."""

    return "127.0.0.1" if bind_host in {"0.0.0.0", "::", ""} else bind_host


def _detect_lan_ip() -> str | None:
    """Best-effort LAN IP detection for startup logs."""

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.1", 80))
            address: tuple[str, int] = cast(tuple[str, int], sock.getsockname())
            host: str = address[0]
    except OSError:
        return None
    return None if host.startswith("127.") else host


def _log_access_urls(config: AppConfig) -> None:
    lan_ip: str | None = _detect_lan_ip()
    if lan_ip is not None:
        log.info("Jouzetsu available on LAN at http://%s:%s/", lan_ip, config.ui.port)


def unclean_shutdown_marker_path(config: AppConfig) -> Path:
    """Return the marker retained when the prior shutdown could not complete."""

    return config.data_dir / UNCLEAN_SHUTDOWN_MARKER_FILE_NAME


def _warn_if_unclean_shutdown(config: AppConfig) -> None:
    marker_path: Path = unclean_shutdown_marker_path(config)
    if marker_path.exists():
        log.warning("previous shutdown was incomplete marker_path=%s", marker_path)


def mark_unclean_shutdown(config: AppConfig) -> None:
    """Record that graceful shutdown could not complete before its deadline."""

    marker_path: Path = unclean_shutdown_marker_path(config)
    content: str = f"recorded_at={datetime.now(UTC).isoformat()}\n"
    try:
        atomic_write_text(marker_path, content)
    except Exception:
        log.exception("could not record incomplete shutdown marker path=%s", marker_path)


def clear_unclean_shutdown_marker(config: AppConfig) -> None:
    """Remove an old incomplete-shutdown marker after a clean shutdown."""

    marker_path: Path = unclean_shutdown_marker_path(config)
    try:
        marker_path.unlink(missing_ok=True)
    except OSError:
        log.exception("could not clear incomplete shutdown marker path=%s", marker_path)


def create_state(config: AppConfig, config_store: ConfigStore) -> AppState:
    """Construct the framework-independent state and its infrastructure."""

    storage: ChatStorage = ChatStorage(config.chats_file)
    character_storage: CharacterStorage = CharacterStorage(config.characters_directory)
    client: LMStudioClient = LMStudioClient(config.server)
    return AppState(config, storage, client, character_storage, config_store)


async def _open_browser_after_start(config: AppConfig) -> None:
    """Open the local browser after the ASGI server has accepted connections."""

    if not config.ui.auto_open_browser:
        return
    await asyncio.sleep(0.6)
    url: str = f"http://{_browser_host(config.ui.host)}:{config.ui.port}/"
    try:
        _ = webbrowser.open(url)
    except Exception:
        log.exception("could not open browser url=%s", url)


def _build_lifecycle_callbacks(
    config: AppConfig,
) -> tuple[Callable[[], Awaitable[None]], Callable[[], Awaitable[None]]]:
    """Return the one process lifecycle pair consumed by ``WebApplication``."""

    async def startup() -> None:
        if _state is None:
            raise RuntimeError("application state is not initialised")
        await _state.start_runtime_monitor()
        await _state.start_host_stats_monitor()
        _ = asyncio.create_task(_open_browser_after_start(config))

    async def shutdown() -> None:
        global _data_directory_lock
        shutdown_clean: bool = False
        try:
            if _state is not None:
                log.info("Shutting down — flushing state and closing client")
                shutdown_clean = await _state.shutdown()
                if shutdown_clean:
                    clear_unclean_shutdown_marker(config)
                else:
                    mark_unclean_shutdown(config)
                    log.error("application shutdown was incomplete")
        finally:
            if _data_directory_lock is not None:
                _data_directory_lock.release()
                _data_directory_lock = None
            log.info("application shutdown handler complete clean=%s", shutdown_clean)

    return startup, shutdown


def build_app(config: AppConfig | None = None, *, app_home: Path | None = None) -> AppConfig:
    """Load configuration and construct the ASGI application exactly once."""

    global _config_store, _data_directory_lock, _state, _web_application

    if config is not None and app_home is not None:
        raise ValueError("supply either an application configuration or an application home, not both")
    if _web_application is not None:
        if config is not None and _state is not None and _state.config.data_dir != config.data_dir:
            raise RuntimeError("Jouzetsu is already initialised with a different data directory")
        if app_home is not None and _state is not None:
            requested_config_file: Path = AppPaths.for_home(resolve_app_home(app_home)).config_file
            if _state.config.config_file != requested_config_file:
                raise RuntimeError("Jouzetsu is already initialised with a different application home")
        if _state is not None:
            return _state.config
        if config is not None:
            return config
        return ConfigStore.for_home(app_home).load()

    startup_records = configure_bootstrap_logging() if config is None else []
    _config_store = (
        ConfigStore.for_paths(config.paths, overrides=EnvironmentOverrides())
        if config is not None
        else ConfigStore.for_home(app_home)
    )
    resolved_config: AppConfig = config or _config_store.load()
    _data_directory_lock = LinuxDataDirectoryLock.acquire(resolved_config.data_dir)
    try:
        configure_logging(resolved_config, startup_records=startup_records)
        _warn_if_unclean_shutdown(resolved_config)
        _state = create_state(resolved_config, _config_store)
        startup, shutdown = _build_lifecycle_callbacks(resolved_config)
        _web_application = WebApplication(
            resolved_config,
            _state,
            on_startup=startup,
            on_shutdown=shutdown,
        )
    except Exception:
        log.exception("application initialisation failed data_dir=%s", resolved_config.data_dir)
        _data_directory_lock.release()
        _data_directory_lock = None
        _config_store = None
        _state = None
        raise

    log.info(
        "application initialised data_dir=%s logging_enabled=%s host=%s port=%d",
        resolved_config.data_dir,
        resolved_config.logging.enabled,
        resolved_config.ui.host,
        resolved_config.ui.port,
    )
    return resolved_config


def asgi_application() -> object:
    """Return the constructed FastHTML ASGI application for external runners/tests."""

    if _web_application is None:
        _ = build_app()
    if _web_application is None:
        raise RuntimeError("application could not be initialised")
    return _web_application.app


def run(config: AppConfig | None = None, *, app_home: Path | None = None) -> None:
    """Run Jouzetsu through Uvicorn without a JavaScript build step."""

    resolved_config: AppConfig = build_app(config, app_home=app_home)
    _log_access_urls(resolved_config)
    if _web_application is None:
        raise RuntimeError("application could not be initialised")
    server: uvicorn.Server = uvicorn.Server(
        uvicorn.Config(
            _web_application.app,
            host=resolved_config.ui.host,
            port=resolved_config.ui.port,
            log_level="warning",
            # Browser clients keep the SSE stream open indefinitely. Uvicorn's
            # default timeout is unbounded, which can otherwise stall Ctrl+C
            # before application lifespan shutdown begins.
            timeout_graceful_shutdown=_UVICORN_GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS,
        )
    )
    _web_application.set_exit_callback(lambda: setattr(server, "should_exit", True))
    server.run()
