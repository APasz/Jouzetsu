"""Focused persistence and access controllers used by :mod:`jouzetsu.state`."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass, field, replace
from logging import Logger
from typing import Protocol

from .access import (
    forget_pending_device,
    refresh_seen_device,
    register_seen_device,
    set_device_access,
    set_device_label,
)
from .async_workers import complete_before_cancelling, run_in_worker
from .config import AppConfig, ConfigStore
from .events import StateChangeKind
from .lmstudio import LMStudioClientProtocol
from .models import Chat, ChatJSON
from .runtime import ConnectionState, ModelDescriptor, RuntimePhase, RuntimeStatus
from .storage import ChatStorage

log: Logger = logging.getLogger(__name__)
_POLL_INTERVAL_SECONDS: float = 120.0
_OFFLINE_POLL_INITIAL_SECONDS: float = 5.0
_OFFLINE_POLL_MAX_SECONDS: float = 60.0
_INVENTORY_RETRY_DELAYS_SECONDS: tuple[float, ...] = (0.25, 1.0)
_PERSISTENCE_DEBOUNCE_SECONDS: float = 1.5
_PERSISTENCE_MAX_DELAY_SECONDS: float = 10.0
_ACTIVE_PHASES: frozenset[RuntimePhase] = frozenset[RuntimePhase](
    {
        RuntimePhase.STARTING,
        RuntimePhase.LOADING_MODEL,
        RuntimePhase.PROCESSING_PROMPT,
        RuntimePhase.REASONING,
        RuntimePhase.GENERATING,
        RuntimePhase.REVIEWING,
        RuntimePhase.STOPPING,
    }
)


class RuntimeTarget(Protocol):
    """The runtime state held for one chat by the application façade."""

    chat: Chat
    runtime_status: RuntimeStatus


@dataclass(frozen=True, slots=True)
class _PendingChatWrite:
    record: ChatJSON | None
    first_dirty_at: float
    last_dirty_at: float

    @property
    def deadline(self) -> float:
        return min(
            self.last_dirty_at + _PERSISTENCE_DEBOUNCE_SECONDS,
            self.first_dirty_at + _PERSISTENCE_MAX_DELAY_SECONDS,
        )


@dataclass(slots=True)
class PersistenceController:
    """Coalesce explicit chat upserts and deletions into durable file changes."""

    config: AppConfig
    config_store: ConfigStore
    storage: ChatStorage

    _pending_chat_writes: dict[str, _PendingChatWrite] = field(default_factory=dict)
    _chat_write_task: asyncio.Task[None] | None = None
    _chat_write_error: Exception | None = None
    _flush_chats_event: asyncio.Event = field(default_factory=asyncio.Event)
    _write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    on_chat_write_error: Callable[[Exception], Awaitable[None]] | None = None

    @property
    def chat_write_error(self) -> Exception | None:
        """Expose deferred draft failures to the runtime and browser status view."""
        return self._chat_write_error

    def queue_chat_changes(
        self,
        changed_chats: Iterable[Chat],
        *,
        deleted_chat_ids: Iterable[str] = (),
    ) -> None:
        """Queue the latest documents for changed chats and explicit removals only."""

        records: dict[str, ChatJSON] = {}
        for chat in changed_chats:
            if chat.id in records:
                raise ValueError(f"duplicate changed chat id: {chat.id}")
            records[chat.id] = chat.to_dict()
        deleted_ids: set[str] = set(deleted_chat_ids)
        overlap: set[str] = set(records).intersection(deleted_ids)
        if overlap:
            raise ValueError(
                f"cannot save and delete the same chat: {sorted(overlap)!r}"
            )
        if not records and not deleted_ids:
            return
        now = time.monotonic()
        for chat_id, record in records.items():
            self._queue_chat_write(chat_id, record, now)
        for chat_id in deleted_ids:
            self._queue_chat_write(chat_id, None, now)
        self._chat_write_error = None
        if self._chat_write_task is None or self._chat_write_task.done():
            self._chat_write_task = asyncio.create_task(self._write_pending_chats())

    def _queue_chat_write(
        self, chat_id: str, record: ChatJSON | None, now: float
    ) -> None:
        previous = self._pending_chat_writes.get(chat_id)
        self._pending_chat_writes[chat_id] = _PendingChatWrite(
            record,
            previous.first_dirty_at if previous else now,
            now,
        )

    async def flush_chats(self) -> None:
        """Wait until every queued chat change has reached durable storage."""
        task: asyncio.Task[None] | None = self._chat_write_task
        if task is None and self._pending_chat_writes:
            self._chat_write_error = None
            task = asyncio.create_task(self._write_pending_chats())
            self._chat_write_task = task
        if task is not None:
            self._flush_chats_event.set()
            await asyncio.shield(task)
        if self._chat_write_error is not None:
            raise self._chat_write_error

    async def save_chat_candidate(self, chat: Chat) -> None:
        """Save an unpublished edit without retaining rejected candidates for retry."""

        record = chat.to_dict()
        async with self._write_lock:
            await complete_before_cancelling(
                run_in_worker(self.storage.save_records, (record,))
            )
            self._discard_pending_chat(chat.id)

    async def delete_chat(self, chat_id: str) -> None:
        """Never put an uncommitted deletion in the draft retry queue."""

        async with self._write_lock:
            await complete_before_cancelling(
                run_in_worker(
                    self.storage.save_records, (), deleted_chat_ids=frozenset({chat_id})
                )
            )
            self._discard_pending_chat(chat_id)

    def _discard_pending_chat(self, chat_id: str) -> None:
        _ = self._pending_chat_writes.pop(chat_id, None)
        if not self._pending_chat_writes:
            self._chat_write_error = None

    async def save_config(self, candidate: AppConfig | None = None) -> None:
        """Write configuration at the explicit state-commit boundary."""

        snapshot = deepcopy(candidate if candidate is not None else self.config)
        async with self._write_lock:
            await complete_before_cancelling(
                run_in_worker(self.config_store.save, snapshot)
            )

    async def _write_pending_chats(self) -> None:
        """Coalesce each document independently, preserving newer writes on failure."""
        try:
            while self._pending_chat_writes:
                await self._wait_for_draft_pause()
                async with self._write_lock:
                    now = time.monotonic()
                    writes = {
                        chat_id: write
                        for chat_id, write in self._pending_chat_writes.items()
                        if self._flush_chats_event.is_set() or write.deadline <= now
                    }
                    if not writes:
                        continue
                    for chat_id in writes:
                        del self._pending_chat_writes[chat_id]
                    records = tuple(
                        write.record
                        for write in writes.values()
                        if write.record is not None
                    )
                    deleted_ids = frozenset(
                        chat_id
                        for chat_id, write in writes.items()
                        if write.record is None
                    )
                    try:
                        await complete_before_cancelling(
                            run_in_worker(
                                self.storage.save_records,
                                records,
                                deleted_chat_ids=deleted_ids,
                            )
                        )
                    except Exception as exc:
                        for chat_id, write in writes.items():
                            self._pending_chat_writes.setdefault(chat_id, write)
                        self._chat_write_error = exc
                        log.exception("chat persistence worker failed")
                        if self.on_chat_write_error is not None:
                            try:
                                await self.on_chat_write_error(exc)
                            except Exception as notification_error:  # noqa: BLE001
                                log.error(
                                    "persistence error notification failed error_type=%s",
                                    type(notification_error).__name__,
                                )
                        return
                    self._chat_write_error = None
        finally:
            self._chat_write_task = None
            self._flush_chats_event.clear()

    async def _wait_for_draft_pause(self) -> None:
        """Wake for the earliest quiet draft or continuous-typing deadline."""
        while self._pending_chat_writes and not self._flush_chats_event.is_set():
            delay = (
                min(write.deadline for write in self._pending_chat_writes.values())
                - time.monotonic()
            )
            if delay <= 0:
                return
            try:
                _ = await asyncio.wait_for(
                    self._flush_chats_event.wait(), timeout=delay
                )
            except TimeoutError:
                continue


@dataclass(slots=True)
class AccessController:
    """Persist access configuration changes through the owning application."""

    config: AppConfig
    persist_config: Callable[[AppConfig], Awaitable[None]]
    persist_activity: Callable[[AppConfig], Awaitable[None]]

    def _candidate(self) -> AppConfig:
        return replace(self.config, access=deepcopy(self.config.access))

    async def set_defaults(
        self,
        *,
        default_private: bool,
        allow_localhost_without_approval: bool,
        global_settings_for_approved: bool,
        allow_network_device_reassociation: bool,
    ) -> None:
        access = replace(
            self.config.access,
            default_private=default_private,
            allow_localhost_without_approval=allow_localhost_without_approval,
            global_settings_for_approved=global_settings_for_approved,
            allow_network_device_reassociation=allow_network_device_reassociation,
        )
        await self.persist_config(replace(self.config, access=access))

    async def set_device_access(self, device_id: str, access_allowed: bool) -> None:
        candidate = self._candidate()
        set_device_access(candidate, device_id=device_id, access_allowed=access_allowed)
        await self.persist_config(candidate)

    async def forget_pending_device(self, device_id: str) -> None:
        candidate = self._candidate()
        forget_pending_device(candidate, device_id=device_id)
        await self.persist_config(candidate)

    async def register_device(
        self,
        device_id: str,
        label: str,
        *,
        last_ip: str,
        hostname: str,
    ) -> bool:
        candidate = self._candidate()
        if not register_seen_device(
            candidate,
            device_id=device_id,
            label=label,
            last_ip=last_ip,
            hostname=hostname,
        ):
            return False
        await self.persist_config(candidate)
        return True

    async def refresh_device_activity(
        self,
        device_id: str,
        label: str,
        *,
        last_ip: str,
        hostname: str,
        hostname_observed: bool,
    ) -> None:
        """Persist an existing browser's activity without notifying every client."""

        candidate = self._candidate()
        if not refresh_seen_device(
            candidate,
            device_id=device_id,
            label=label,
            last_ip=last_ip,
            hostname=hostname,
            hostname_observed=hostname_observed,
        ):
            return
        await self.persist_activity(candidate)

    async def set_device_label(self, device_id: str, label: str) -> None:
        candidate = self._candidate()
        set_device_label(candidate, device_id=device_id, label=label)
        await self.persist_config(candidate)


class RuntimeController:
    """Own LM Studio reachability, inventory, and background monitoring."""

    def __init__(
        self,
        config: AppConfig,
        client: LMStudioClientProtocol,
        targets: Callable[[], Iterable[RuntimeTarget]],
        display_name: Callable[[str, str], str],
        notify: Callable[[StateChangeKind], Awaitable[None]],
    ) -> None:
        self._config: AppConfig = config
        self._client: LMStudioClientProtocol = client
        self._targets: Callable[[], Iterable[RuntimeTarget]] = targets
        self._display_name: Callable[[str, str], str] = display_name
        self._notify: Callable[[StateChangeKind], Awaitable[None]] = notify
        self._connection: ConnectionState = ConnectionState.CHECKING
        self._connection_error: str = ""
        self._inventory: tuple[ModelDescriptor, ...] = ()
        self._models_by_identifier: dict[str, ModelDescriptor] = {}
        self._refresh_lock: asyncio.Lock = asyncio.Lock()
        self._monitor_task: asyncio.Task[None] | None = None
        self._next_poll_interval_seconds: float = _POLL_INTERVAL_SECONDS

    @property
    def inventory(self) -> tuple[ModelDescriptor, ...]:
        return self._inventory

    @property
    def connection(self) -> ConnectionState:
        return self._connection

    def model_descriptor(self, key_or_instance_id: str) -> ModelDescriptor | None:
        return self._models_by_identifier.get(key_or_instance_id)

    def idle_status(self, chat: Chat) -> RuntimeStatus:
        if self._connection is ConnectionState.CHECKING:
            return RuntimeStatus(RuntimePhase.CHECKING)
        if self._connection is ConnectionState.OFFLINE:
            return RuntimeStatus(RuntimePhase.OFFLINE, detail=self._connection_error)

        selected_model: str = chat.model or self._config.server.default_model
        if not selected_model:
            loaded = next((model for model in self._inventory if model.is_loaded), None)
            if loaded is None:
                return RuntimeStatus(RuntimePhase.NO_MODEL_SELECTED)
            return RuntimeStatus(
                RuntimePhase.READY,
                model_key=loaded.key,
                model_name=self._display_name(loaded.key, loaded.display_name),
            )

        descriptor: ModelDescriptor | None = self.model_descriptor(selected_model)
        if descriptor is None:
            return RuntimeStatus(
                RuntimePhase.MODEL_UNLOADED,
                model_key=selected_model,
                model_name=self._display_name(selected_model, ""),
                detail="Model is not available on this server",
            )
        if not descriptor.is_loaded:
            return RuntimeStatus(
                RuntimePhase.MODEL_UNLOADED,
                model_key=descriptor.key,
                model_name=self._display_name(descriptor.key, descriptor.display_name),
            )
        return RuntimeStatus(
            RuntimePhase.READY,
            model_key=descriptor.key,
            model_name=self._display_name(descriptor.key, descriptor.display_name),
        )

    async def start(self) -> None:
        if self._monitor_task is not None and not self._monitor_task.done():
            return
        await self.refresh()
        self._monitor_task = asyncio.create_task(self._monitor())
        log.info(
            "started LM Studio runtime monitor poll_interval_seconds=%d",
            _POLL_INTERVAL_SECONDS,
        )

    async def refresh(self) -> bool:
        """Serialise inventory requests and report this refresh's outcome."""

        async with self._refresh_lock:
            previous_inventory: tuple[ModelDescriptor, ...] = self._inventory
            previous_connection: ConnectionState = self._connection
            refreshed: bool = await self._refresh_inventory()
            models_changed: bool = (
                previous_inventory != self._inventory
                or previous_connection is not self._connection
            )
            statuses_changed: bool = self._refresh_targets(self.idle_status)
        if models_changed or statuses_changed:
            await self._notify(
                StateChangeKind.FULL if models_changed else StateChangeKind.STATUS
            )
        return refreshed

    async def _refresh_inventory(self) -> bool:
        previous_connection: ConnectionState = self._connection
        try:
            inventory: tuple[
                ModelDescriptor, ...
            ] = await self._load_inventory_with_retries()
        except Exception as exc:  # noqa: BLE001
            self._connection = ConnectionState.OFFLINE
            if previous_connection is ConnectionState.OFFLINE:
                self._next_poll_interval_seconds = min(
                    self._next_poll_interval_seconds * 2,
                    _OFFLINE_POLL_MAX_SECONDS,
                )
            else:
                self._next_poll_interval_seconds = _OFFLINE_POLL_INITIAL_SECONDS
            if previous_connection is not ConnectionState.OFFLINE:
                log.warning("LM Studio became unavailable error=%s", exc)
            self._connection_error = str(exc)
            return False

        models_by_identifier: dict[str, ModelDescriptor] = {}
        for model in inventory:
            for identifier in (model.key, *model.loaded_instance_ids):
                _ = models_by_identifier.setdefault(identifier, model)
        self._inventory = inventory
        self._models_by_identifier = models_by_identifier
        self._connection = ConnectionState.ONLINE
        self._connection_error = ""
        self._next_poll_interval_seconds = _POLL_INTERVAL_SECONDS
        if previous_connection is not ConnectionState.ONLINE:
            loaded_instance_count: int = sum(
                len(model.loaded_instance_ids) for model in self._inventory
            )
            log.info(
                "LM Studio connection established model_count=%d loaded_instance_count=%d",
                len(self._inventory),
                loaded_instance_count,
            )
        return True

    async def unload_model(self, model_key: str) -> ModelDescriptor:
        descriptor: ModelDescriptor | None = self.model_descriptor(model_key)
        if descriptor is None:
            raise ValueError(f"unknown model: {model_key}")
        if not descriptor.loaded_instance_ids:
            raise ValueError(f"model is not loaded: {model_key}")
        log.info(
            "unloading model model_key=%s instance_count=%d",
            descriptor.key,
            len(descriptor.loaded_instance_ids),
        )
        for instance_id in descriptor.loaded_instance_ids:
            await self._client.unload_model_instance(instance_id)
        await self.refresh()
        return descriptor

    async def unload_model_instance(self, instance_id: str) -> ModelDescriptor:
        descriptor: ModelDescriptor | None = self.model_descriptor(instance_id)
        if descriptor is None or instance_id not in descriptor.loaded_instance_ids:
            raise ValueError(f"unknown loaded model instance: {instance_id}")
        log.info(
            "unloading model instance model_key=%s instance_id=%s",
            descriptor.key,
            instance_id,
        )
        await self._client.unload_model_instance(instance_id)
        await self.refresh()
        return descriptor

    async def shutdown(self) -> None:
        if self._monitor_task is None:
            return
        _ = self._monitor_task.cancel()
        try:
            await self._monitor_task
        except asyncio.CancelledError:
            pass
        self._monitor_task = None
        log.info("stopped LM Studio runtime monitor")

    async def _monitor(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._next_poll_interval_seconds)
                await self.refresh()
        except asyncio.CancelledError:
            return

    async def _load_inventory_with_retries(self) -> tuple[ModelDescriptor, ...]:
        """Retry short-lived inventory failures before changing runtime status."""
        for delay_seconds in (*_INVENTORY_RETRY_DELAYS_SECONDS, None):
            try:
                return await self._client.list_model_inventory()
            except Exception:
                if delay_seconds is None:
                    raise
                log.info(
                    "LM Studio inventory request will retry delay_seconds=%s",
                    delay_seconds,
                )
                await asyncio.sleep(delay_seconds)
        raise RuntimeError("unreachable inventory retry state")

    def _refresh_targets(self, status_for: Callable[[Chat], RuntimeStatus]) -> bool:
        changed: bool = False
        for target in self._targets():
            if target.runtime_status.phase in _ACTIVE_PHASES:
                continue
            status: RuntimeStatus = status_for(target.chat)
            if self._status_should_change(target.runtime_status, status):
                target.runtime_status = status
                changed = True
        return changed

    @staticmethod
    def _status_should_change(
        current: RuntimeStatus, next_status: RuntimeStatus
    ) -> bool:
        """Keep completed metrics visible until a model or chat state changes."""
        if current == next_status:
            return False
        return not (
            next_status.phase is RuntimePhase.READY
            and current.phase is RuntimePhase.READY
            and current.model_key == next_status.model_key
            and current.metrics is not None
        )
