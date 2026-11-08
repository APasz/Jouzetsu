"""Advance a controlled clock to verify pause and continuous-typing durability."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path
from unittest.mock import patch

import pytest
from inference_support import Harness, backend

from jouzetsu.models import ChatJSON


def test_draft_write_waits_for_pause_but_bounds_continuous_typing(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        h = Harness(tmp_path, (backend(),))
        chat = await h.chat()
        clock = 0.0
        waits: list[float] = []

        async def elapse(
            waiter: Coroutine[object, object, bool], timeout: float
        ) -> bool:
            nonlocal clock
            waiter.close()
            waits.append(timeout)
            clock += min(1.0, timeout)
            chat.draft = f"typed at {clock}"
            h.persistence.queue_chat_changes((chat,))
            raise TimeoutError

        with (
            patch("jouzetsu.controllers.time.monotonic", side_effect=lambda: clock),
            patch("jouzetsu.controllers.asyncio.wait_for", side_effect=elapse),
        ):
            h.persistence.queue_chat_changes((chat,))
            # The worker runs to the ten-second deadline without wall-clock waits.
            task = h.persistence._chat_write_task  # pyright: ignore[reportPrivateUsage]
            assert task is not None
            await task
        assert clock == 10.0
        assert waits[0] == 1.5
        assert h.reload(chat).draft == "typed at 10.0"
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_single_draft_update_is_saved_after_quiet_period_and_flush_bypasses_delay(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        h = Harness(tmp_path, (backend(),))
        chat = await h.chat()
        clock = 0.0
        waits: list[float] = []

        async def elapse(
            waiter: Coroutine[object, object, bool], timeout: float
        ) -> bool:
            nonlocal clock
            waiter.close()
            waits.append(timeout)
            clock += timeout
            raise TimeoutError

        with (
            patch("jouzetsu.controllers.time.monotonic", side_effect=lambda: clock),
            patch("jouzetsu.controllers.asyncio.wait_for", side_effect=elapse),
        ):
            _ = await h.runtime.update_draft(chat.id, 0, "Paused")
            task = h.persistence._chat_write_task  # pyright: ignore[reportPrivateUsage]
            assert task is not None
            await task
            assert clock == 1.5 and waits == [1.5]
            _ = await h.runtime.update_draft(chat.id, 1, "Flushed")
            await h.persistence.flush_chats()
            assert waits == [1.5]
        assert h.reload(chat).draft == "Flushed"
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_typing_in_one_chat_does_not_delay_a_quiet_chat(tmp_path: Path) -> None:
    async def exercise() -> None:
        h = Harness(tmp_path, (backend(),))
        quiet, typing = await h.chat(), await h.chat()
        clock = 0.0
        saved: list[tuple[float, tuple[str, ...]]] = []
        original = h.storage.save_records

        def record_write(
            records: tuple[ChatJSON, ...],
            *,
            deleted_chat_ids: frozenset[str] = frozenset(),
        ) -> None:
            saved.append((clock, tuple(item["id"] for item in records)))
            original(records, deleted_chat_ids=deleted_chat_ids)

        async def elapse(
            waiter: Coroutine[object, object, bool], timeout: float
        ) -> bool:
            nonlocal clock
            waiter.close()
            clock += min(0.5, timeout)
            if clock <= 2:
                typing.draft = f"typing at {clock}"
                h.persistence.queue_chat_changes((typing,))
            raise TimeoutError

        with (
            patch("jouzetsu.controllers.time.monotonic", side_effect=lambda: clock),
            patch("jouzetsu.controllers.asyncio.wait_for", side_effect=elapse),
            patch.object(h.storage, "save_records", side_effect=record_write),
        ):
            quiet.draft = "Quiet"
            h.persistence.queue_chat_changes((quiet, typing))
            task = h.persistence._chat_write_task  # pyright: ignore[reportPrivateUsage]
            assert task
            await task
        assert saved[0] == (1.5, (quiet.id,))
        assert saved[-1] == (3.5, (typing.id,))
        assert h.reload(quiet).draft == "Quiet"
        assert h.reload(typing).draft == "typing at 2.0"
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_error_notification_cannot_mask_the_original_draft_failure(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        h = Harness(tmp_path, (backend(),))
        chat = await h.chat()

        async def broken_notification(error: Exception) -> None:
            raise ValueError("notification failed")

        h.persistence.on_chat_write_error = broken_notification
        error = OSError("storage failed")
        with patch.object(h.storage, "save_records", side_effect=error):
            _ = await h.runtime.update_draft(chat.id, 0, "Recoverable")
            with pytest.raises(OSError) as caught:
                await h.persistence.flush_chats()
        assert caught.value is error
        assert h.persistence.chat_write_error is error
        assert await h.runtime.shutdown()
        assert h.reload(chat).draft == "Recoverable"

    asyncio.run(exercise())
