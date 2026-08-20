"""Explicit worker execution for short blocking system operations."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial

_WORKER_COUNT: int = 4
_COMPLETION_POLL_INTERVAL_SECONDS: float = 0.01
_WORKER_EXECUTOR: ThreadPoolExecutor = ThreadPoolExecutor(
    max_workers=_WORKER_COUNT,
    thread_name_prefix="jouzetsu-worker",
)

async def run_in_worker[**P, T](
    function: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
) -> T:
    """Run a short blocking operation without relying on asyncio's default executor."""

    callback: Callable[[], T] = partial(function, *args, **kwargs)
    future: Future[T] = _WORKER_EXECUTOR.submit(callback)
    try:
        while not future.done():
            await asyncio.sleep(_COMPLETION_POLL_INTERVAL_SECONDS)
    except asyncio.CancelledError:
        _ = future.cancel()
        raise
    return future.result()
