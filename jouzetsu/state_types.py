"""Internal state data structures shared by the application controllers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .models import Message
from .runtime import RuntimePhase, RuntimeStatus

if TYPE_CHECKING:
    from .backend_protocol import BackendError
    from .models import Chat


@dataclass(slots=True)
class ChatSession:
    """Mutable runtime state associated with one persisted chat."""

    chat: Chat
    is_generating: bool = False
    generation_task: asyncio.Task[None] | None = None
    runtime_status: RuntimeStatus = field(
        default_factory=lambda: RuntimeStatus(RuntimePhase.CHECKING)
    )
    live_reasoning: str = ""
    empty_state_message: str = ""
    live_response: Message | None = None
    latest_error: BackendError | None = None
