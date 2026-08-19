"""Cross-layer state change events."""

from __future__ import annotations

from enum import Enum


class StateChangeKind(Enum):
    """Scope of a state notification sent to UI listeners."""

    FULL = "full"
    STREAM = "stream"
    STATUS = "status"
    CHARACTERS = "characters"
