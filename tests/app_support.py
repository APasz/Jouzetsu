"""Temporary real-file application with controlled inference adapters."""

from __future__ import annotations

from pathlib import Path

from inference_support import FakeBackend
from web_support import noop

from jouzetsu.backend_models import BackendConfig, BackendConfiguration, BackendId
from jouzetsu.config import ConfigStore
from jouzetsu.state import AppState
from jouzetsu.storage import ChatStorage
from jouzetsu.web.application import WebApplication


class WebHarness:
    def __init__(self, root: Path, backends: tuple[BackendConfig, ...] = ()) -> None:
        store = ConfigStore.for_home(root)
        config = store.load()
        config.backends = BackendConfiguration(
            backends=backends, default_backend_id=backends[0].id if backends else None
        )
        store.save(config)
        self.adapters: dict[BackendId, FakeBackend] = {}

        def factory(backend: BackendConfig) -> FakeBackend:
            adapter = FakeBackend(backend)
            self.adapters[backend.id] = adapter
            return adapter

        self.state = AppState(
            config,
            ChatStorage(config.chats_file),
            config_store=store,
            backend_factory=factory,
        )
        self.web = WebApplication(config, self.state, on_startup=noop, on_shutdown=noop)

    async def close(self) -> None:
        self.web._event_broker.close()  # pyright: ignore[reportPrivateUsage]
        assert await self.state.shutdown()
