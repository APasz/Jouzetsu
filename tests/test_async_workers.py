from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pytest

from jouzetsu import async_workers
from jouzetsu.async_workers import run_in_worker


@pytest.fixture
def isolated_worker_executor(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep unit-test event loops from sharing the process-wide worker pool."""

    executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(async_workers, "_WORKER_EXECUTOR", executor)
    yield
    executor.shutdown(wait=True, cancel_futures=True)


def test_run_in_worker_returns_the_blocking_result(
    isolated_worker_executor: None,
) -> None:
    def blocking_operation(value: int) -> int:
        return value * 2

    assert asyncio.run(run_in_worker(blocking_operation, 21)) == 42


def test_run_in_worker_propagates_worker_errors(
    isolated_worker_executor: None,
) -> None:
    def failing_operation() -> None:
        raise RuntimeError("worker failed")

    with pytest.raises(RuntimeError, match="worker failed"):
        _ = asyncio.run(run_in_worker(failing_operation))


def test_run_in_worker_propagates_cancellation(
    isolated_worker_executor: None,
) -> None:
    started = threading.Event()
    release = threading.Event()
    exited = threading.Event()

    def blocking_operation() -> None:
        started.set()
        _ = release.wait(timeout=1.0)
        exited.set()

    async def scenario() -> None:
        task: asyncio.Task[None] = asyncio.create_task(
            run_in_worker(blocking_operation)
        )
        await asyncio.sleep(0.01)
        assert started.is_set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()

    asyncio.run(scenario())
    assert exited.wait(timeout=1.0)
