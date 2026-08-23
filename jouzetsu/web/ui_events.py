"""Server-sent UI state notifications."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Final

from ..events import StateChangeKind
from ..state import AppState

_HEARTBEAT_SECONDS: Final[float] = 15.0


@dataclass(frozen=True, slots=True)
class UiEvent:
    """One coalescable state update delivered to connected browser views."""

    revision: int
    kind: StateChangeKind
    characters_changed: bool = False


class UiEventBroker:
    """Non-blocking fan-out from ``AppState`` to per-browser SSE queues."""

    def __init__(self, state: AppState) -> None:
        self._state: AppState = state
        self._revision: int = 0
        self._queues: set[asyncio.Queue[UiEvent]] = set()
        self._remove_listener: Callable[[], None] | None = None
        self.start()

    def start(self) -> None:
        """Attach exactly one listener, including after a reused app lifespan."""

        if self._remove_listener is None:
            self._remove_listener = self._state.add_listener(self._on_state_change)

    async def _on_state_change(self, kind: StateChangeKind) -> None:
        self._revision += 1
        event: UiEvent = UiEvent(
            revision=self._revision,
            kind=kind,
            characters_changed=kind is StateChangeKind.CHARACTERS,
        )
        for queue in tuple(self._queues):
            queued_event: UiEvent = event
            if queue.full():
                try:
                    pending_event: UiEvent = queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                else:
                    queued_event = _coalesce(pending_event, event)
            try:
                queue.put_nowait(queued_event)
            except asyncio.QueueFull:
                continue

    def subscribe(self) -> asyncio.Queue[UiEvent]:
        """Create a bounded queue so a stalled browser cannot stall generation."""

        queue: asyncio.Queue[UiEvent] = asyncio.Queue(maxsize=1)
        self._queues.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[UiEvent]) -> None:
        """Release a browser queue after its streaming response disconnects."""

        self._queues.discard(queue)

    def close(self) -> None:
        """Detach listeners and browser queues during one lifespan shutdown."""

        if self._remove_listener is not None:
            self._remove_listener()
            self._remove_listener = None
        self._queues.clear()


async def stream_events(
    broker: UiEventBroker, queue: asyncio.Queue[UiEvent]
) -> AsyncIterator[str]:
    """Yield SSE frames and release the queue when its response disconnects."""

    try:
        yield _sse_frame(UiEvent(revision=0, kind=StateChangeKind.FULL))
        while True:
            try:
                event: UiEvent = await asyncio.wait_for(
                    queue.get(), timeout=_HEARTBEAT_SECONDS
                )
            except TimeoutError:
                yield ": ping\n\n"
                continue
            yield _sse_frame(event)
    finally:
        broker.unsubscribe(queue)


def _coalesce(previous: UiEvent, current: UiEvent) -> UiEvent:
    """Keep structural refreshes when a browser queue drops intermediate events."""

    kind: StateChangeKind = (
        StateChangeKind.FULL
        if previous.kind is StateChangeKind.FULL or current.kind is StateChangeKind.FULL
        else current.kind
    )
    return UiEvent(
        revision=current.revision,
        kind=kind,
        characters_changed=previous.characters_changed or current.characters_changed,
    )


def _sse_frame(event: UiEvent) -> str:
    """Encode a typed UI event as an SSE frame without unsafe interpolation."""

    payload: str = json.dumps(
        {
            "revision": event.revision,
            "kind": event.kind.value,
            "characters_changed": event.characters_changed,
        },
        separators=(",", ":"),
    )
    return f"event: state\ndata: {payload}\n\n"
