"""Linux advisory locking for Jouzetsu's single-writer data directory."""

from __future__ import annotations

import fcntl
import logging
import os
from datetime import UTC, datetime
from logging import Logger
from pathlib import Path
from typing import TextIO

LOCK_FILE_NAME: str = "jouzetsu.lock"
log: Logger = logging.getLogger(__name__)


class DataDirectoryInUseError(RuntimeError):
    """Raised when another Jouzetsu process already owns the data directory."""


class LinuxDataDirectoryLock:
    """An exclusive Linux advisory lock retained for one Jouzetsu process."""

    def __init__(self, path: Path, file: TextIO) -> None:
        self.path: Path = path
        self._file: TextIO | None = file

    @classmethod
    def acquire(cls, directory: Path) -> LinuxDataDirectoryLock:
        """Acquire the directory lock or raise with the current owner details."""
        directory.mkdir(parents=True, exist_ok=True)
        path: Path = directory / LOCK_FILE_NAME
        file: TextIO = path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            _ = file.seek(0)
            owner: str = file.read().strip()
            file.close()
            details: str = f" ({owner})" if owner else ""
            log.warning(
                "data directory lock unavailable directory=%s owner=%s",
                directory,
                owner or "unknown",
            )
            raise DataDirectoryInUseError(
                f"Jouzetsu is already using {directory}{details}"
            ) from exc

        _ = file.seek(0)
        _ = file.truncate()
        _ = file.write(
            f"pid={os.getpid()} started_at={datetime.now(UTC).isoformat()}\n"
        )
        file.flush()
        os.fsync(file.fileno())
        log.info("acquired data directory lock path=%s", path)
        return cls(path, file)

    def release(self) -> None:
        """Release the advisory lock without deleting its stable lock-file path."""
        if self._file is None:
            return
        fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        self._file.close()
        self._file = None
        log.info("released data directory lock path=%s", self.path)
