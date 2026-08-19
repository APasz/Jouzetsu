"""Request-scoped view data."""

from __future__ import annotations

from dataclasses import dataclass

from ...access import AccessDecision


@dataclass(frozen=True, slots=True)
class PageContext:
    """Request-scoped access and display information used by page renderers."""

    decision: AccessDecision
    client_label: str
    csrf_token: str
