"""Controlled backend and real-file harness shared by Gate 2 tests."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from jouzetsu.backend_models import (
    BackendConfig,
    BackendConfiguration,
    BackendId,
    BackendKind,
    GenerationProfile,
    RegisteredModel,
)
from jouzetsu.backend_protocol import (
    BackendError,
    BackendErrorCategory,
    BackendHealth,
    BackendReachability,
    GenerationRequest,
    InferenceBackend,
    ModelDiscovery,
    ModelManagement,
)
from jouzetsu.config import ConfigStore
from jouzetsu.controllers import PersistenceController
from jouzetsu.inference import InferenceRuntime, RuntimeChange
from jouzetsu.models import Chat
from jouzetsu.runtime import (
    ChatStreamEvent,
    GenerationMetrics,
    ModelDescriptor,
    ModelLoadProgress,
    ModelReady,
    PredictionComplete,
    PredictionFragment,
)
from jouzetsu.state_registry import ChatRegistry
from jouzetsu.storage import ChatStorage


def backend(
    name: str = "Local", *, limit: int = 1, native: bool = False
) -> BackendConfig:
    return replace(
        BackendConfig.create(
            BackendKind.LM_STUDIO if native else BackendKind.OPENAI_COMPATIBLE,
            name,
            "http://localhost:1234/v1",
        ),
        concurrency_limit=limit,
        defaults=GenerationProfile("model"),
        models=(RegisteredModel(ModelDescriptor("model", "Model"), alias="Alias"),),
    )


@dataclass(slots=True)
class Plan:
    text: str = "Reply"
    reasoning: str = ""
    error: BackendError | None = None
    hold: bool = False
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    produced: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)


class FakeBackend:
    def __init__(self, config: BackendConfig) -> None:
        self.id: BackendId = config.id
        self.native: bool = config.model_management is not None
        self.online: bool = True
        self.loaded: bool = False
        self.closed: bool = False
        self.load_error: BackendError | None = None
        self.plans: deque[Plan] = deque()
        self.requests: list[GenerationRequest] = []
        self.loads: list[str] = []
        self.unloads: list[str] = []
        self.probes: int = 0
        self.discoveries: int = 0

    @property
    def model_management(self) -> ModelManagement | None:
        return self if self.native else None

    async def probe(self) -> BackendHealth:
        self.probes += 1
        return BackendHealth(
            BackendReachability.ONLINE if self.online else BackendReachability.OFFLINE,
            datetime.now(UTC),
        )

    async def discover_models(self) -> ModelDiscovery:
        self.discoveries += 1
        if not self.online:
            raise BackendError(BackendErrorCategory.OFFLINE, "Backend is offline.")
        return ModelDiscovery(
            True,
            (ModelDescriptor("model", "Model", ("instance",) if self.loaded else ()),),
        )

    async def load_model(
        self, model_id: str
    ) -> AsyncGenerator[ModelLoadProgress | ModelReady]:
        self.loads.append(model_id)
        if self.load_error:
            raise self.load_error
        self.loaded = True
        yield ModelReady(model_id, "Model", 4096, "instance")

    async def unload_model(self, instance_id: str) -> None:
        self.unloads.append(instance_id)
        self.loaded = False

    async def stream(
        self, request: GenerationRequest
    ) -> AsyncGenerator[ChatStreamEvent]:
        self.requests.append(request)
        plan = self.plans.popleft() if self.plans else Plan()
        plan.entered.set()
        yield ModelReady(
            request.model.model_id, "Model", 4096, "instance" if self.native else None
        )
        if plan.reasoning:
            yield PredictionFragment(plan.reasoning, None, True)
        if plan.text:
            yield PredictionFragment(plan.text, None, False)
        plan.produced.set()
        if plan.hold:
            await plan.release.wait()
        if plan.error:
            raise plan.error
        yield PredictionComplete(GenerationMetrics(output_tokens=1))

    async def close(self) -> None:
        self.closed = True


class Harness:
    def __init__(
        self,
        home: Path,
        configs: tuple[BackendConfig, ...],
        *,
        adapter_factory: Callable[[BackendConfig], FakeBackend] = FakeBackend,
    ) -> None:
        self.store = ConfigStore.for_home(home)
        config = self.store.load()
        config.backends = BackendConfiguration(
            configs, configs[0].id if configs else None
        )
        self.store.save(config)
        self.storage = ChatStorage(config.chats_file)
        self.persistence = PersistenceController(config, self.store, self.storage)
        self.registry = ChatRegistry(lambda: "")
        self.adapters: dict[BackendId, FakeBackend] = {}
        self.changes: list[RuntimeChange] = []

        def factory(item: BackendConfig) -> InferenceBackend:
            adapter = adapter_factory(item)
            self.adapters[item.id] = adapter
            return adapter

        async def notify(change: RuntimeChange) -> None:
            self.changes.append(change)

        self.runtime = InferenceRuntime(
            self.registry, self.persistence, backend_factory=factory, notify=notify
        )

    async def chat(self, text: str = "Hello") -> Chat:
        chat = await self.runtime.create_chat()
        _ = await self.runtime.send_message(
            chat.id, chat.revision, "first", text, generate=False
        )
        return chat

    def reload(self, chat: Chat) -> Chat:
        return next(item for item in self.storage.load_all() if item.id == chat.id)
