"""Explicit worker execution for short blocking system operations."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial

_WORKER_COUNT: int = 4
_WORKER_EXECUTOR: ThreadPoolExecutor = ThreadPoolExecutor(
    max_workers=_WORKER_COUNT,
    thread_name_prefix="jouzetsu-worker",
)


async def run_in_worker[**P, T](
    function: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
) -> T:
    """Run a short blocking operation without relying on asyncio's default executor."""

    callback: Callable[[], T] = partial(function, *args, **kwargs)
    return await asyncio.get_running_loop().run_in_executor(_WORKER_EXECUTOR, callback)
