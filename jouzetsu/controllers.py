"""Ordered persistence used by the application runtime."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from logging import Logger

from .async_workers import complete_before_cancelling, run_in_worker
from .config import AppConfig, ConfigStore
from .models import Chat, ChatJSON
from .storage import ChatStorage

log: Logger = logging.getLogger(__name__)
_PERSISTENCE_DEBOUNCE_SECONDS: float = 1.5
_PERSISTENCE_MAX_DELAY_SECONDS: float = 10.0


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
