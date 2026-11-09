"""Shutdown drains open UI streams before flushing application state."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
from contextlib import ExitStack, aclosing
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import httpx
import pytest
from app_support import WebHarness
from browser_support import create_browser_server
from web_support import csrf_headers

from jouzetsu.events import StateChangeKind
from jouzetsu.web.ui_events import UiEvent, stream_events


@pytest.mark.parametrize("pending_event", [False, True])
def test_broker_close_wakes_idle_streams_and_discards_pending_ui_events(
    tmp_path: Path, pending_event: bool
) -> None:
    async def exercise() -> None:
        h = WebHarness(tmp_path)
        broker = h.web._event_broker  # pyright: ignore[reportPrivateUsage]
        queue = broker.subscribe()
        waiting = asyncio.Event()
        get_event = queue.get

        async def wait_for_event() -> UiEvent:
            waiting.set()
            return await get_event()

        try:
            async with aclosing(stream_events(broker, queue)) as frames:
                assert "event: state" in await anext(frames)
                with patch.object(queue, "get", side_effect=wait_for_event):
                    next_frame = asyncio.create_task(anext(frames))
                    await waiting.wait()
                    if pending_event:
                        await broker._on_state_change(StateChangeKind.FULL)  # pyright: ignore[reportPrivateUsage]
                    h.web.close_event_streams()
                    h.web.close_event_streams()
                    with pytest.raises(StopAsyncIteration):
                        await asyncio.wait_for(next_frame, timeout=1)
        finally:
            await h.close()

    asyncio.run(exercise())


def test_closed_broker_rejects_late_subscriptions_and_can_start_a_new_lifespan(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        h = WebHarness(tmp_path)
        broker = h.web._event_broker  # pyright: ignore[reportPrivateUsage]
        old_queue = broker.subscribe()
        try:
            h.web.close_event_streams()
            late_queue = broker.subscribe()
            for queue in (old_queue, late_queue):
                with pytest.raises(asyncio.QueueShutDown):
                    _ = queue.get_nowait()
            broker.start()
            new_queue = broker.subscribe()
            await h.state._notify(StateChangeKind.DRAFT)  # pyright: ignore[reportPrivateUsage]
            assert new_queue.get_nowait().kind is StateChangeKind.DRAFT
            with pytest.raises(asyncio.QueueShutDown):
                _ = await old_queue.get()
        finally:
            await h.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("loop", ["asyncio", "uvloop"])
@pytest.mark.parametrize("connections", [1, 2])
def test_server_shutdown_finishes_open_sse_requests_and_flushes_drafts(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    loop: Literal["asyncio", "uvloop"],
    connections: int,
) -> None:
    if loop == "uvloop" and importlib.util.find_spec("uvloop") is None:
        pytest.skip("Optional uvloop is unavailable")
    server = create_browser_server(tmp_path, loop=loop)
    chat = server.state.chats[0]
    caplog.set_level(logging.ERROR, logger="uvicorn.error")
    try:
        with (
            httpx.Client(base_url=server.base_url, timeout=3) as client,
            ExitStack() as stack,
        ):
            streams = [
                stack.enter_context(client.stream("GET", "/events"))
                for _ in range(connections)
            ]
            lines = [response.iter_lines() for response in streams]
            for response, frames in zip(streams, lines, strict=True):
                assert response.status_code == 200
                assert next(frames) == "event: state"
            page = client.get(f"/chats/{chat.id}")
            saved = client.post(
                "/chat/draft",
                headers=csrf_headers(page),
                data={
                    "chat_id": chat.id,
                    "draft_revision": "0",
                    "draft": "Flush this draft on shutdown",
                },
            )
            assert saved.status_code == 200
            server.server.should_exit = True
            for frames in lines:
                _ = tuple(frames)
        server.close()
        assert server.client.closed
        assert (
            server.state.storage.load_all()[0].draft == "Flush this draft on shutdown"
        )
        assert not [
            record for record in caplog.records if record.levelno >= logging.ERROR
        ]
    finally:
        server.close()
