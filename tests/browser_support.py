"""Small real-server harness for the focused Playwright browser tests."""

from __future__ import annotations

import asyncio
import socket
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event, Thread
from typing import Final

import uvicorn
from inference_support import FakeBackend, backend

from jouzetsu.backend_models import (
    BackendConfig,
    BackendConfiguration,
    ChatInferenceSettings,
    GenerationProfile,
    RegisteredModel,
)
from jouzetsu.backend_protocol import GenerationRequest, ModelDiscovery
from jouzetsu.config import (
    AppConfig,
    AppPaths,
    LoggingSettings,
    UiSettings,
)
from jouzetsu.models import Chat
from jouzetsu.runtime import (
    ChatStreamEvent,
    FirstToken,
    GenerationMetrics,
    ModelCapabilities,
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


class BrowserTestBackend(FakeBackend):
    """Controlled inference with actual browser/server scheduling."""

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        self.response_content = "Browser test reply"
        self.first_token_delay_seconds = 0.0
        self.completion_gate: Event | None = None
        self.descriptors = tuple(model.descriptor for model in config.models)

    async def discover_models(self) -> ModelDiscovery:
        return ModelDiscovery(True, self.descriptors)

    async def stream(
        self, request: GenerationRequest
    ) -> AsyncGenerator[ChatStreamEvent]:
        self.requests.append(request)
        yield ModelReady(request.model.model_id, "Browser test model", 4096)
        if self.first_token_delay_seconds:
            await asyncio.sleep(self.first_token_delay_seconds)
        yield FirstToken()
        yield PredictionFragment(self.response_content, 1, False)
        if self.completion_gate is not None:
            await asyncio.to_thread(self.completion_gate.wait)
        yield PredictionComplete(
            GenerationMetrics(output_tokens=1, tokens_per_second=20.0)
        )


async def _noop() -> None:
    return None


@dataclass(slots=True)
class BrowserServer:
    """A temporary Jouzetsu server with mutable state for one browser test."""

    base_url: str
    state: AppState
    client: BrowserTestBackend
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
    configured = replace(
        backend(),
        defaults=GenerationProfile("browser-test-model"),
        models=(
            RegisteredModel(
                ModelDescriptor(
                    "browser-test-model",
                    "Browser test model",
                    capabilities=ModelCapabilities(vision=True),
                )
            ),
            RegisteredModel(
                ModelDescriptor(
                    "text-model",
                    "Text model",
                    capabilities=ModelCapabilities(vision=False),
                )
            ),
        ),
    )
    configuration = BackendConfiguration(
        backends=(configured,), default_backend_id=configured.id
    )
    config = AppConfig(
        paths=paths,
        backends=configuration,
        ui=UiSettings(auto_open_browser=False),
        logging=LoggingSettings(enabled=False, directory=root / "logs"),
    )
    client = BrowserTestBackend(configured)
    storage = ChatStorage(paths.chats_file)
    storage.save_records(
        (Chat(inference=ChatInferenceSettings.new_chat(configuration)).to_dict(),)
    )
    state = AppState(config, storage, backend_factory=lambda _: client)

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
            proxy_headers=False,
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
        client=client,
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
