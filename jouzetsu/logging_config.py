"""Central logging setup for Jouzetsu."""

from __future__ import annotations

import logging
import os
import shutil
import sys
from collections.abc import Iterable
from logging import FileHandler, Formatter, Handler, Logger, LogRecord, StreamHandler
from pathlib import Path
from typing import Final, cast, override

from .config import AppConfig

CHAT_LOGGER_NAME: Final[str] = "jouzetsu.chat"
APPLICATION_LOGGER_NAME: Final[str] = "jouzetsu"
_LOG_FORMAT: Final[str] = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_MANAGED_HANDLER_ATTR: Final[str] = "_jouzetsu_managed"
_ROLLED_OVER_DIRECTORIES: set[Path] = set()
_LOG_LEVELS_BY_NAME: Final[dict[str, int]] = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


def chat_logger() -> Logger:
    """Return the dedicated chat-event logger."""
    return logging.getLogger(CHAT_LOGGER_NAME)


class _StartupLogBuffer(Handler):
    """Retain pre-configuration application records for later file replay."""

    def __init__(self, records: list[LogRecord]) -> None:
        super().__init__()
        self._records: list[LogRecord] = records

    @override
    def emit(self, record: LogRecord) -> None:
        self._records.append(record)


def configure_bootstrap_logging() -> list[LogRecord]:
    """Configure temporary console logging while the application config is read."""
    formatter: Formatter = Formatter(_LOG_FORMAT)
    application_logger: Logger = _application_logger()
    _remove_managed_handlers(application_logger)

    startup_records: list[LogRecord] = []
    application_logger.setLevel(logging.INFO)
    application_logger.propagate = False
    application_logger.addHandler(_managed_handler(StreamHandler(sys.stderr), level=logging.INFO, formatter=formatter))
    application_logger.addHandler(
        _managed_handler(_StartupLogBuffer(startup_records), level=logging.INFO, formatter=formatter)
    )
    return startup_records


def configure_logging(config: AppConfig, *, startup_records: Iterable[LogRecord] = ()) -> None:
    """Install console, system, error, and chat log handlers.

    The function is idempotent so tests and repeated app construction do not
    duplicate handlers.
    """
    level: int = _log_level_from_env()
    formatter: Formatter = Formatter(_LOG_FORMAT)
    root_logger: Logger = logging.getLogger()
    application_logger: Logger = _application_logger()
    chat_events_logger: Logger = chat_logger()

    _remove_managed_handlers(root_logger)
    _remove_managed_handlers(application_logger)
    _remove_managed_handlers(chat_events_logger)

    application_logger.setLevel(level)
    chat_events_logger.setLevel(level)
    application_logger.propagate = False
    chat_events_logger.propagate = False

    application_logger.addHandler(
        _managed_handler(
            StreamHandler(sys.stderr),
            level=level,
            formatter=formatter,
        )
    )

    if not config.logging.enabled:
        return

    _rollover_log_directory(config.log_directory)
    system_handler: Handler = _managed_handler(
        FileHandler(config.log_directory / "system.log", mode="a", encoding="utf-8"),
        level=level,
        formatter=formatter,
    )
    error_handler: Handler = _managed_handler(
        FileHandler(config.log_directory / "error.log", mode="a", encoding="utf-8"),
        level=logging.ERROR,
        formatter=formatter,
    )
    chat_handler: Handler = _managed_handler(
        FileHandler(config.log_directory / "chat.log", mode="a", encoding="utf-8"),
        level=logging.INFO,
        formatter=formatter,
    )
    application_logger.addHandler(system_handler)
    application_logger.addHandler(error_handler)
    chat_events_logger.addHandler(chat_handler)
    chat_events_logger.addHandler(error_handler)
    _replay_startup_records(startup_records, (system_handler, error_handler))


def _log_level_from_env() -> int:
    raw_level: str = os.environ.get("JOUZETSU_LOG_LEVEL", "INFO").upper()
    return _LOG_LEVELS_BY_NAME.get(raw_level, logging.INFO)


def _rollover_log_directory(directory: Path) -> None:
    """Move the prior session aside, retaining exactly one previous log set."""
    resolved_directory: Path = directory.resolve()
    if resolved_directory in _ROLLED_OVER_DIRECTORIES:
        resolved_directory.mkdir(parents=True, exist_ok=True)
        return
    if not resolved_directory.name:
        raise ValueError("logging directory must not be the filesystem root")

    previous_directory: Path = resolved_directory.with_name(f"{resolved_directory.name}.old")
    if previous_directory.is_symlink() or previous_directory.is_file():
        previous_directory.unlink()
    elif previous_directory.exists():
        shutil.rmtree(previous_directory)
    if resolved_directory.exists():
        _ = resolved_directory.replace(previous_directory)
    resolved_directory.mkdir(parents=True, exist_ok=True)
    _ROLLED_OVER_DIRECTORIES.add(resolved_directory)


def _managed_handler(handler: Handler, *, level: int, formatter: Formatter) -> Handler:
    handler.setLevel(level)
    handler.setFormatter(formatter)
    setattr(handler, _MANAGED_HANDLER_ATTR, True)
    return handler


def _application_logger() -> Logger:
    """Return the parent logger for all Jouzetsu application modules."""
    return logging.getLogger(APPLICATION_LOGGER_NAME)


def _replay_startup_records(records: Iterable[LogRecord], handlers: tuple[Handler, ...]) -> None:
    """Write bootstrap records to the newly configured Jouzetsu file handlers."""
    for record in records:
        for handler in handlers:
            if record.levelno >= handler.level:
                _ = handler.handle(record)


def _remove_managed_handlers(logger: Logger) -> None:
    for handler in list[Handler](logger.handlers):
        if not cast(bool, getattr(handler, _MANAGED_HANDLER_ATTR, False)):
            continue
        logger.removeHandler(handler)
        handler.close()
