"""Focused persistence and access controllers used by :mod:`jouzetsu.state`."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from logging import Logger
from typing import Protocol

from .access import register_seen_device, set_device_access, set_device_label
from .config import AppConfig
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
_PERSISTENCE_DEBOUNCE_SECONDS: float = 0.15
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


@dataclass(slots=True)
class PersistenceController:
    """Coalesce explicit chat upserts and deletions into durable file changes."""

    config: AppConfig
    storage: ChatStorage

    _pending_chat_records: dict[str, ChatJSON] = field(default_factory=dict)
    _pending_deleted_chat_ids: set[str] = field(default_factory=set)
    _chat_write_task: asyncio.Task[None] | None = None
    _chat_write_error: Exception | None = None
    _flush_chats_event: asyncio.Event = field(default_factory=asyncio.Event)

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
            raise ValueError(f"cannot save and delete the same chat: {sorted(overlap)!r}")
        if not records and not deleted_ids:
            return

        for chat_id, record in records.items():
            self._pending_chat_records[chat_id] = record
            self._pending_deleted_chat_ids.discard(chat_id)
        for chat_id in deleted_ids:
            _ = self._pending_chat_records.pop(chat_id, None)
            self._pending_deleted_chat_ids.add(chat_id)
        self._chat_write_error = None
        if self._chat_write_task is None or self._chat_write_task.done():
            self._chat_write_task = asyncio.create_task(self._write_pending_chats())

    async def flush_chats(self) -> None:
        """Wait until every queued chat change has reached durable storage."""
        task: asyncio.Task[None] | None = self._chat_write_task
        if task is None and (self._pending_chat_records or self._pending_deleted_chat_ids):
            self._chat_write_error = None
            task = asyncio.create_task(self._write_pending_chats())
            self._chat_write_task = task
        if task is not None:
            self._flush_chats_event.set()
            await task
        if self._chat_write_error is not None:
            raise self._chat_write_error

    async def save_config(self) -> None:
        """Write configuration at the explicit state-commit boundary."""

        self.config.save()

    async def _write_pending_chats(self) -> None:
        """Persist queued changes after a short debounce, retaining only the latest per chat."""
        try:
            try:
                _ = await asyncio.wait_for(self._flush_chats_event.wait(), timeout=_PERSISTENCE_DEBOUNCE_SECONDS)
            except TimeoutError:
                pass
            self._flush_chats_event.clear()
            while self._pending_chat_records or self._pending_deleted_chat_ids:
                chat_records: tuple[ChatJSON, ...] = tuple(self._pending_chat_records.values())
                deleted_chat_ids: frozenset[str] = frozenset(self._pending_deleted_chat_ids)
                try:
                    self.storage.save_records(chat_records, deleted_chat_ids=deleted_chat_ids)
                except Exception as exc:
                    self._chat_write_error = exc
                    log.exception("chat persistence worker failed")
                    return
                self._pending_chat_records.clear()
                self._pending_deleted_chat_ids.clear()
        finally:
            self._chat_write_task = None


@dataclass(slots=True)
class AccessController:
    """Persist access configuration changes through the owning application."""

    config: AppConfig
    persist_config: Callable[[], Awaitable[None]]

    async def set_defaults(
        self,
        *,
        default_private: bool,
        allow_localhost_without_approval: bool,
        global_settings_for_approved: bool,
        allow_network_device_reassociation: bool,
    ) -> None:
        access = self.config.access
        access.default_private = default_private
        access.allow_localhost_without_approval = allow_localhost_without_approval
        access.global_settings_for_approved = global_settings_for_approved
        access.allow_network_device_reassociation = allow_network_device_reassociation
        await self.persist_config()

    async def set_device_access(self, device_id: str, access_allowed: bool) -> None:
        set_device_access(self.config, device_id=device_id, access_allowed=access_allowed)
        await self.persist_config()

    async def register_device(
        self,
        device_id: str,
        label: str,
        *,
        last_ip: str,
        hostname: str,
    ) -> bool:
        if not register_seen_device(
            self.config,
            device_id=device_id,
            label=label,
            last_ip=last_ip,
            hostname=hostname,
        ):
            return False
        await self.persist_config()
        return True

    async def set_device_label(self, device_id: str, label: str) -> None:
        set_device_label(self.config, device_id=device_id, label=label)
        await self.persist_config()


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
        self._inventory: tuple[ModelDescriptor, ...] = ()
        self._monitor_task: asyncio.Task[None] | None = None
        self._next_poll_interval_seconds: float = _POLL_INTERVAL_SECONDS

    @property
    def inventory(self) -> tuple[ModelDescriptor, ...]:
        return self._inventory

    def model_descriptor(self, key_or_instance_id: str) -> ModelDescriptor | None:
        return next(
            (
                model
                for model in self._inventory
                if model.key == key_or_instance_id or key_or_instance_id in model.loaded_instance_ids
            ),
            None,
        )

    def idle_status(self, chat: Chat) -> RuntimeStatus:
        if self._connection is ConnectionState.CHECKING:
            return RuntimeStatus(RuntimePhase.CHECKING)
        if self._connection is ConnectionState.OFFLINE:
            return RuntimeStatus(RuntimePhase.OFFLINE)

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
        log.info("started LM Studio runtime monitor poll_interval_seconds=%d", _POLL_INTERVAL_SECONDS)

    async def refresh(self) -> None:
        previous_connection: ConnectionState = self._connection
        try:
            self._inventory = await self._load_inventory_with_retries()
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
            error_detail: str = str(exc)
            await self._refresh_targets(lambda _chat: RuntimeStatus(RuntimePhase.OFFLINE, detail=error_detail))
            return

        self._connection = ConnectionState.ONLINE
        self._next_poll_interval_seconds = _POLL_INTERVAL_SECONDS
        if previous_connection is not ConnectionState.ONLINE:
            loaded_instance_count: int = sum(len(model.loaded_instance_ids) for model in self._inventory)
            log.info(
                "LM Studio connection established model_count=%d loaded_instance_count=%d",
                len(self._inventory),
                loaded_instance_count,
            )
        await self._refresh_targets(self.idle_status)

    async def unload_model(self, model_key: str) -> ModelDescriptor:
        descriptor: ModelDescriptor | None = self.model_descriptor(model_key)
        if descriptor is None:
            raise ValueError(f"unknown model: {model_key}")
        if not descriptor.loaded_instance_ids:
            raise ValueError(f"model is not loaded: {model_key}")
        log.info("unloading model model_key=%s instance_count=%d", descriptor.key, len(descriptor.loaded_instance_ids))
        for instance_id in descriptor.loaded_instance_ids:
            await self._client.unload_model_instance(instance_id)
        await self.refresh()
        return descriptor

    async def unload_model_instance(self, instance_id: str) -> ModelDescriptor:
        descriptor: ModelDescriptor | None = self.model_descriptor(instance_id)
        if descriptor is None or instance_id not in descriptor.loaded_instance_ids:
            raise ValueError(f"unknown loaded model instance: {instance_id}")
        log.info("unloading model instance model_key=%s instance_id=%s", descriptor.key, instance_id)
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
                log.info("LM Studio inventory request will retry delay_seconds=%s", delay_seconds)
                await asyncio.sleep(delay_seconds)
        raise RuntimeError("unreachable inventory retry state")

    async def _refresh_targets(self, status_for: Callable[[Chat], RuntimeStatus]) -> None:
        changed: bool = False
        for target in self._targets():
            if target.runtime_status.phase in _ACTIVE_PHASES:
                continue
            status: RuntimeStatus = status_for(target.chat)
            if self._status_should_change(target.runtime_status, status):
                target.runtime_status = status
                changed = True
        if changed:
            await self._notify(StateChangeKind.STATUS)

    @staticmethod
    def _status_should_change(current: RuntimeStatus, next_status: RuntimeStatus) -> bool:
        """Keep completed metrics visible until a model or chat state changes."""
        if current == next_status:
            return False
        return not (
            next_status.phase is RuntimePhase.READY
            and current.phase is RuntimePhase.READY
            and current.model_key == next_status.model_key
            and current.metrics is not None
        )
