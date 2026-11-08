"""Regression coverage for Gate 2 lifecycle and resource ownership review."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import replace
from pathlib import Path
from typing import override
from unittest.mock import patch

import pytest
from inference_support import FakeBackend, Harness, backend

from jouzetsu.atomic_write import CommitUncertainError
from jouzetsu.backend_models import BackendConfig, ModelRef
from jouzetsu.backend_protocol import GenerationRequest, ModelDiscovery
from jouzetsu.config import AppConfig, ConfigStore
from jouzetsu.inference import (
    ChangeKind,
    ChatBusy,
    GenerationJob,
    JobPhase,
    RuntimeChange,
)
from jouzetsu.models import ChatJSON
from jouzetsu.runtime import (
    ChatStreamEvent,
    ModelLoadProgress,
    ModelReady,
    PredictionFragment,
)


class ControlledBackend(FakeBackend):
    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        self.discovery_entered = asyncio.Event()
        self.discovery_release = asyncio.Event()
        self.discovery_release.set()
        self.load_entered = asyncio.Event()
        self.load_release = asyncio.Event()
        self.load_release.set()
        self.stream_closed = asyncio.Event()
        self.streams: list[AsyncGenerator[ChatStreamEvent]] = []
        self.close_error: Exception | None = None

    @override
    async def discover_models(self) -> ModelDiscovery:
        self.discovery_entered.set()
        await self.discovery_release.wait()
        return await super().discover_models()

    @override
    async def load_model(
        self, model_id: str
    ) -> AsyncGenerator[ModelLoadProgress | ModelReady]:
        self.load_entered.set()
        await self.load_release.wait()
        async for event in super().load_model(model_id):
            yield event

    @override
    def stream(self, request: GenerationRequest) -> AsyncGenerator[ChatStreamEvent]:
        iterator = self._stream(request)
        self.streams.append(iterator)
        return iterator

    async def _stream(
        self, request: GenerationRequest
    ) -> AsyncGenerator[ChatStreamEvent]:
        try:
            async with aclosing(super().stream(request)) as events:
                async for event in events:
                    yield event
        finally:
            self.stream_closed.set()

    @override
    async def close(self) -> None:
        if self.close_error:
            raise self.close_error
        await super().close()


async def worker_started(event: threading.Event) -> None:
    while not event.is_set():
        await asyncio.sleep(0)


def test_shutdown_drains_user_commit_without_starting_late_generation(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        config = backend()
        h = Harness(tmp_path, (config,))
        chat = await h.runtime.create_chat()
        entered, release = threading.Event(), threading.Event()
        original = h.storage.save_records

        def slow(
            records: tuple[ChatJSON, ...],
            *,
            deleted_chat_ids: frozenset[str] = frozenset(),
        ) -> None:
            entered.set()
            assert release.wait(5)
            original(records, deleted_chat_ids=deleted_chat_ids)

        with patch.object(h.storage, "save_records", side_effect=slow):
            sending = asyncio.create_task(
                h.runtime.send_message(chat.id, 0, "send", "Durable")
            )
            await worker_started(entered)
            stopping = asyncio.create_task(h.runtime.shutdown())
            await asyncio.sleep(0)
            assert not stopping.done() and not h.adapters[config.id].closed
            release.set()
            sent = await sending
            assert sent.job is None
            assert await stopping
        assert h.reload(chat).messages[0].content == "Durable"
        assert not h.adapters[config.id].requests
        with pytest.raises(RuntimeError, match="shutting down"):
            _ = await h.runtime.generate(chat.id, chat.revision)

    asyncio.run(exercise())


def test_shutdown_waits_for_adapter_replacement_publication(tmp_path: Path) -> None:
    async def exercise() -> None:
        config = backend()
        h = Harness(tmp_path, (config,))
        previous = h.adapters[config.id]
        entered, release = threading.Event(), threading.Event()
        original = ConfigStore.save

        def slow(store: ConfigStore, candidate: AppConfig) -> None:
            entered.set()
            assert release.wait(5)
            original(store, candidate)

        with patch.object(ConfigStore, "save", new=slow):
            update = asyncio.create_task(
                h.runtime.update_backend(replace(config, base_url="http://changed/v1"))
            )
            await worker_started(entered)
            replacement = h.adapters[config.id]
            stopping = asyncio.create_task(h.runtime.shutdown())
            await asyncio.sleep(0)
            assert not stopping.done() and not replacement.closed
            release.set()
            await update
            assert await stopping
        assert previous.closed and replacement.closed
        assert (
            h.store.load().backends.backend(config.id).base_url == "http://changed/v1"
        )

    asyncio.run(exercise())


def test_shutdown_timeout_keeps_resources_open_until_maintenance_finishes(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        config = backend()
        adapter = ControlledBackend(config)
        h = Harness(tmp_path, (config,), adapter_factory=lambda _: adapter)
        adapter.discovery_release.clear()
        refresh = asyncio.create_task(h.runtime.refresh_backend(config.id))
        await adapter.discovery_entered.wait()
        with patch("jouzetsu.inference._SHUTDOWN_SECONDS", 0):
            assert not await h.runtime.shutdown()
        assert not adapter.closed
        adapter.discovery_release.set()
        await refresh
        assert await h.runtime.shutdown()
        assert adapter.closed

    asyncio.run(exercise())


def test_unpublished_adapter_is_closed_when_configuration_save_fails(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        config = backend()
        h = Harness(tmp_path, (config,))
        previous = h.adapters[config.id]
        with (
            patch.object(ConfigStore, "save", side_effect=OSError("full")),
            pytest.raises(OSError),
        ):
            await h.runtime.update_backend(
                replace(config, base_url="http://changed/v1")
            )
        replacement = h.adapters[config.id]
        assert replacement is not previous and replacement.closed
        assert not previous.closed
        assert h.runtime.configuration.backend(config.id) == config
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_cleanup_failure_does_not_reject_an_already_committed_config_change(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        config = backend()
        previous = ControlledBackend(config)
        previous.close_error = OSError("cleanup failure")
        h = Harness(tmp_path, (config,), adapter_factory=lambda _: previous)
        await h.runtime.set_default_backend(None)
        h.changes.clear()
        await h.runtime.remove_backend(config.id)
        assert not h.runtime.configuration.backends
        assert RuntimeChange(ChangeKind.CONFIG) in h.changes
        assert not await h.runtime.shutdown()
        previous.close_error = None
        assert await h.runtime.shutdown()
        assert previous.closed

    asyncio.run(exercise())


def test_cancel_during_notification_closes_stream_before_unlocking_chat(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        config = backend()
        adapter = ControlledBackend(config)
        h = Harness(tmp_path, (config,), adapter_factory=lambda _: adapter)
        chat = await h.chat()
        notified = asyncio.Event()

        async def notify(change: RuntimeChange) -> None:
            if (
                change.kind is ChangeKind.STATUS
                and h.registry.get(chat.id).live_response
            ):
                notified.set()
                await asyncio.Event().wait()

        with patch.object(h.runtime, "_notify_callback", new=notify):
            job = await h.runtime.generate(chat.id, chat.revision)
            assert job
            await notified.wait()
            # Let final persistence notifications complete after cancellation.
            h.runtime._notify_callback = lambda_change  # pyright: ignore[reportPrivateUsage]
            await h.runtime.cancel(chat.id)
        assert adapter.stream_closed.is_set()
        assert job.phase is JobPhase.CANCELLED
        assert h.reload(chat).messages[-1].content == "Reply"
        assert await h.runtime.shutdown()

    async def lambda_change(change: RuntimeChange) -> None:
        pass

    asyncio.run(exercise())


def test_save_retry_surfaces_the_new_failure_and_recovery_requirement(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        h = Harness(tmp_path, (backend(),))
        chat = await h.chat()
        with patch.object(
            h.storage, "save_records", side_effect=OSError("first failure")
        ):
            job = await h.runtime.generate(chat.id, chat.revision)
            assert job
            await job.wait()
        error = CommitUncertainError("uncertain retry")
        with (
            patch.object(h.storage, "save_records", side_effect=error),
            pytest.raises(CommitUncertainError),
        ):
            await h.runtime.retry_save(chat.id)
        assert job.save_error is error and job.phase is JobPhase.SAVE_FAILED
        assert "recovery and restart" in str(h.registry.get(chat.id).latest_error)
        await h.runtime.retry_save(chat.id)
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_generation_does_not_race_a_model_unload_during_discovery(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        config = backend(native=True)
        adapter = ControlledBackend(config)
        adapter.loaded = True
        h = Harness(tmp_path, (config,), adapter_factory=lambda _: adapter)
        chat = await h.chat()
        adapter.discovery_release.clear()
        unloading = asyncio.create_task(
            h.runtime.unload_model(ModelRef(config.id, "model"))
        )
        await adapter.discovery_entered.wait()
        assert await h.runtime.generate(chat.id, chat.revision) is None
        assert "being unloaded" in str(h.registry.get(chat.id).latest_error)
        adapter.discovery_release.set()
        await unloading
        assert adapter.unloads == ["instance"] and not adapter.requests
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_idle_unload_rechecks_last_use_after_waiting_for_management(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        config = backend(native=True)
        assert config.model_management
        config = replace(
            config,
            model_management=replace(config.model_management, idle_unload_minutes=1),
        )
        adapter = ControlledBackend(config)
        h = Harness(tmp_path, (config,), adapter_factory=lambda _: adapter)
        with patch("jouzetsu.inference.time.monotonic", return_value=0):
            _ = await h.runtime.load_model(ModelRef(config.id, "model"))
        adapter.load_entered.clear()
        adapter.load_release.clear()
        with patch("jouzetsu.inference.time.monotonic", return_value=120):
            loading = asyncio.create_task(
                h.runtime.load_model(ModelRef(config.id, "model"))
            )
            await adapter.load_entered.wait()
            unloading = asyncio.create_task(h.runtime.unload_idle_models(now=120))
            await asyncio.sleep(0)
            adapter.load_release.set()
            _ = await loading
            await unloading
            assert not adapter.unloads
            await h.runtime.unload_idle_models(now=180)
            assert adapter.unloads == ["instance"]
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_concurrent_start_owns_one_monitor_and_one_initial_discovery(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        config = backend()
        adapter = ControlledBackend(config)
        h = Harness(tmp_path, (config,), adapter_factory=lambda _: adapter)
        adapter.discovery_release.clear()
        one = asyncio.create_task(h.runtime.start())
        await adapter.discovery_entered.wait()
        two = asyncio.create_task(h.runtime.start())
        await asyncio.sleep(0)
        adapter.discovery_release.set()
        await asyncio.gather(one, two)
        assert adapter.discoveries == 1 and adapter.probes == 0
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_throttled_preview_flushes_even_when_the_backend_pauses(tmp_path: Path) -> None:
    class BurstBackend(ControlledBackend):
        @override
        async def stream(
            self, request: GenerationRequest
        ) -> AsyncGenerator[ChatStreamEvent]:
            yield ModelReady("model", "Model", 4096)
            yield PredictionFragment("First", None, False)
            yield PredictionFragment(" tail", None, False)
            await asyncio.Event().wait()

    async def exercise() -> None:
        config = backend()
        adapter = BurstBackend(config)
        h = Harness(tmp_path, (config,), adapter_factory=lambda _: adapter)
        chat = await h.chat()
        sleeping, release, published = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def controlled_sleep(delay: float) -> None:
            sleeping.set()
            await release.wait()

        async def notify(change: RuntimeChange) -> None:
            preview = h.registry.get(chat.id).live_response
            if preview and preview.content == "First tail":
                published.set()

        with (
            patch("jouzetsu.inference._STREAM_UPDATE_SECONDS", 60),
            patch("jouzetsu.inference.asyncio.sleep", new=controlled_sleep),
            patch.object(h.runtime, "_notify_callback", new=notify),
        ):
            job = await h.runtime.generate(chat.id, chat.revision)
            assert job
            await sleeping.wait()
            preview = h.registry.get(chat.id).live_response
            assert preview and preview.content == "First"
            release.set()
            await published.wait()
            assert preview.content == "First tail"
            assert len(h.reload(chat).messages) == 1
            await h.runtime.cancel(chat.id)
        assert h.reload(chat).messages[-1].content == "First tail"
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_start_can_retry_after_a_cache_persistence_failure(tmp_path: Path) -> None:
    class UpdatedMetadataBackend(ControlledBackend):
        @override
        async def discover_models(self) -> ModelDiscovery:
            discovery = await super().discover_models()
            return replace(
                discovery,
                models=tuple(
                    replace(model, publisher="new metadata")
                    for model in discovery.models
                ),
            )

    async def exercise() -> None:
        config = backend()
        adapter = UpdatedMetadataBackend(config)
        h = Harness(tmp_path, (config,), adapter_factory=lambda _: adapter)
        with (
            patch.object(
                ConfigStore, "save", side_effect=OSError("cache storage full")
            ),
            pytest.raises(OSError),
        ):
            await h.runtime.start()
        await h.runtime.start()
        assert (
            h.runtime.configuration.backend(config.id)
            .model("model")
            .descriptor.publisher
            == "new metadata"
        )
        assert adapter.discoveries == 2
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_finalisation_is_protected_before_its_first_await(tmp_path: Path) -> None:
    async def exercise() -> None:
        h = Harness(tmp_path, (backend(),))
        chat = await h.chat()
        entered, release = asyncio.Event(), asyncio.Event()
        original = h.runtime._finish  # pyright: ignore[reportPrivateUsage]

        async def paused_finish(job: GenerationJob) -> None:
            entered.set()
            await release.wait()
            await original(job)

        with patch.object(h.runtime, "_finish", new=paused_finish):
            job = await h.runtime.generate(chat.id, chat.revision)
            assert job
            await entered.wait()
            assert job.phase is JobPhase.SAVING
            with pytest.raises(ChatBusy):
                await h.runtime.cancel(chat.id)
            release.set()
            await job.wait()
        assert job.phase is JobPhase.FINISHED
        assert h.reload(chat).messages[-1].content == "Reply"
        assert await h.runtime.shutdown()

    asyncio.run(exercise())
