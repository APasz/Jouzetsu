"""Small real-server harness for the focused Playwright browser tests."""

from __future__ import annotations

import socket
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Final

import uvicorn

from jouzetsu.config import (
    AccessSettings,
    AppConfig,
    AppPaths,
    GenerationSettings,
    LoggingSettings,
    ServerSettings,
    UiSettings,
)
from jouzetsu.models import Chat
from jouzetsu.runtime import (
    ChatStreamEvent,
    FirstToken,
    GenerationMetrics,
    ModelDescriptor,
    ModelReady,
    PredictionComplete,
    PredictionFragment,
)
from jouzetsu.state import AppState
from jouzetsu.storage import ChatStorage
from jouzetsu.web.application import WebApplication

_SERVER_START_TIMEOUT_SECONDS: Final[float] = 5.0
_SERVER_STOP_TIMEOUT_SECONDS: Final[float] = 5.0


class BrowserTestLMStudioClient:
    """Predictable local runtime that keeps browser tests independent of LM Studio."""

    async def list_models(self) -> list[str]:
        return ["browser-test-model"]

    async def list_model_inventory(self) -> tuple[ModelDescriptor, ...]:
        return (
            ModelDescriptor(
                key="browser-test-model", display_name="Browser test model"
            ),
        )

    async def unload_model_instance(self, instance_id: str) -> None:
        _ = instance_id

    async def stream_chat(
        self,
        chat: Chat,
        model: str,
        generation: GenerationSettings,
    ) -> AsyncIterator[ChatStreamEvent]:
        _ = (chat, model, generation)
        yield ModelReady("browser-test-model", "Browser test model", 4096)
        yield FirstToken()
        yield PredictionFragment("Browser test reply", 1, False)
        yield PredictionComplete(
            GenerationMetrics(output_tokens=1, tokens_per_second=20.0)
        )

    async def close(self) -> None:
        return None


async def _noop() -> None:
    return None


@dataclass(slots=True)
class BrowserServer:
    """A temporary Jouzetsu server with mutable state for one browser test."""

    base_url: str
    state: AppState
    server: uvicorn.Server
    thread: Thread
    listener: socket.socket

    def close(self) -> None:
        """Stop the web server before disposing of its test-only application state."""

        self.server.should_exit = True
        self.thread.join(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
        if self.thread.is_alive():
            raise RuntimeError("browser test server did not stop")
        try:
            self.listener.close()
        except OSError:
            pass


def create_browser_server(root: Path) -> BrowserServer:
    """Start an isolated local ASGI server suitable for one Playwright test."""

    paths: AppPaths = AppPaths.for_home(root)
    config = AppConfig(
        paths=paths,
        server=ServerSettings(default_model="browser-test-model"),
        ui=UiSettings(auto_open_browser=False),
        logging=LoggingSettings(enabled=False, directory=root / "logs"),
        access=AccessSettings(default_private=False),
    )
    state = AppState(config, ChatStorage(paths.chats_file), BrowserTestLMStudioClient())

    async def shutdown() -> None:
        if not await state.shutdown():
            raise RuntimeError("browser test state did not shut down cleanly")

    web = WebApplication(config, state, on_startup=_noop, on_shutdown=shutdown)
    listener = _listening_socket()
    host, port = listener.getsockname()
    server = uvicorn.Server(
        uvicorn.Config(
            web.app,
            host=host,
            port=port,
            # Uvicorn's optional uvloop path emits one deprecation warning per asset request.
            loop="asyncio",
            log_level="error",
            access_log=False,
            timeout_graceful_shutdown=1,
        )
    )
    thread = Thread(
        target=server.run,
        kwargs={"sockets": [listener]},
        daemon=True,
        name="jouzetsu-browser-test-server",
    )
    thread.start()
    try:
        _wait_for_server_start(server, thread)
    except Exception:
        server.should_exit = True
        thread.join(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
        try:
            listener.close()
        except OSError:
            pass
        web._event_broker.close()  # pyright: ignore[reportPrivateUsage]
        raise
    return BrowserServer(
        base_url=f"http://{host}:{port}",
        state=state,
        server=server,
        thread=thread,
        listener=listener,
    )


def _listening_socket() -> socket.socket:
    """Reserve a loopback port so parallel test processes cannot race for one."""

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    return listener


def _wait_for_server_start(server: uvicorn.Server, thread: Thread) -> None:
    """Wait until Uvicorn accepts connections or reports a startup failure."""

    deadline: float = time.monotonic() + _SERVER_START_TIMEOUT_SECONDS
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("browser test server stopped during startup")
        if time.monotonic() >= deadline:
            raise RuntimeError("browser test server did not start")
        time.sleep(0.01)
