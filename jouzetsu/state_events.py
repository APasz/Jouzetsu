"""Safe fan-out for application-state change notifications."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from logging import Logger

from .events import StateChangeKind

log: Logger = logging.getLogger(__name__)
ChangeListener = Callable[[StateChangeKind], Awaitable[None]]
_DELETED_CLIENT_ERROR_TEXT = "The client this element belongs to has been deleted."


class StateNotifier:
    """Publish state changes without allowing one stale browser to break others."""

    def __init__(self) -> None:
        self._listeners: list[ChangeListener] = []

    def add(self, listener: ChangeListener) -> Callable[[], None]:
        self._listeners.append(listener)
        return lambda: self.remove(listener)

    def remove(self, listener: ChangeListener) -> None:
        try:
            self._listeners.remove(listener)
        except ValueError:
            return

    async def publish(self, kind: StateChangeKind = StateChangeKind.FULL) -> None:
        for listener in tuple(self._listeners):
            try:
                await listener(kind)
            except Exception as exc:  # noqa: BLE001
                if _DELETED_CLIENT_ERROR_TEXT in str(exc):
                    self.remove(listener)
                    log.debug("removed listener for deleted client")
                    continue
                log.warning("state listener raised: %s", exc)
