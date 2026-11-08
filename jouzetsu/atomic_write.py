"""Small durable file-write primitives shared by JSON persistence."""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from logging import Logger
from pathlib import Path
from threading import RLock

log: Logger = logging.getLogger(__name__)


class CommitUncertainError(OSError):
    """A file changed, but its directory entry could not be durably confirmed."""


class DurableWriteGuard:
    """Prevent writes from silently continuing after an uncertain commit."""

    def __init__(self) -> None:
        self._failures: dict[Path, CommitUncertainError] = {}
        self._lock: RLock = RLock()

    @contextmanager
    def protect(self, *paths: Path) -> Generator[None]:
        """Block affected paths until the store is reopened after recovery."""

        resolved_paths = tuple(path.resolve() for path in paths)
        with self._lock:
            for path in resolved_paths:
                if path in self._failures:
                    raise CommitUncertainError(
                        f"storage requires recovery before another write: {path}"
                    ) from self._failures[path]
            try:
                yield
            except CommitUncertainError as exc:
                for path in resolved_paths:
                    self._failures[path] = exc
                raise


def ensure_directory(directory: Path) -> None:
    """Create directories and durably record each new parent entry."""

    if directory.is_dir():
        _sync_directory(directory.parent)
        return
    parent = directory.parent
    if parent != directory:
        ensure_directory(parent)
    directory.mkdir(exist_ok=True)
    _sync_directory(parent)


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Encode and durably replace one text document."""

    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Replace ``path`` with fully written bytes without exposing a partial file.

    The temporary file is created beside the destination so ``os.replace`` is
    atomic on the target filesystem. Both file and directory fsync must succeed
    before the write is acknowledged.
    """
    ensure_directory(path.parent)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f"{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as file:
            _ = file.write(data)
            file.flush()
            _sync_file(file.fileno())
        os.replace(temporary_path, path)
        _confirm_directories(path.parent)
    except Exception:
        log.exception("atomic write failed path=%s", path)
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise


def move_path(source: Path, destination: Path) -> None:
    """Atomically move a file and durably record both affected directories."""

    ensure_directory(destination.parent)
    os.replace(source, destination)
    _confirm_directories(destination.parent, source.parent)


def remove_path(path: Path) -> None:
    """Remove a file and durably confirm its directory entry changed."""

    try:
        path.unlink()
    except FileNotFoundError:
        return
    _confirm_directories(path.parent)


def _confirm_directories(*directories: Path) -> None:
    try:
        for directory in dict.fromkeys(directories):
            _sync_directory(directory)
    except OSError as exc:
        raise CommitUncertainError(
            "file changed but directory durability could not be confirmed"
        ) from exc


def _sync_file(descriptor: int) -> None:
    os.fsync(descriptor)


def _sync_directory(directory: Path) -> None:
    descriptor: int = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        _sync_file(descriptor)
    finally:
        os.close(descriptor)
