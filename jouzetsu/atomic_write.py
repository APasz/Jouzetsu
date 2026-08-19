"""Small durable file-write primitives shared by JSON persistence."""

from __future__ import annotations

import os
import tempfile
import logging
from logging import Logger
from pathlib import Path

log: Logger = logging.getLogger(__name__)


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Replace ``path`` with fully written text without exposing a partial file.

    The temporary file is created beside the destination so ``os.replace`` is
    atomic on the target filesystem. Best-effort fsync calls reduce the chance
    that an acknowledged write is lost during a sudden power failure.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f"{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding=encoding) as file:
            _ = file.write(text)
            file.flush()
            _sync_file(file.fileno())
        os.replace(temporary_path, path)
        _sync_directory(path.parent)
    except Exception:
        log.exception("atomic write failed path=%s", path)
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise


def move_path(source: Path, destination: Path) -> None:
    """Atomically move a file and durably record both affected directories."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)
    _sync_directory(destination.parent)
    if source.parent != destination.parent:
        _sync_directory(source.parent)


def _sync_file(descriptor: int) -> None:
    try:
        os.fsync(descriptor)
    except OSError:
        pass


def _sync_directory(directory: Path) -> None:
    try:
        descriptor: int = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        _sync_file(descriptor)
    finally:
        os.close(descriptor)
