"""Random, customisable copy for pristine chat screens."""

from __future__ import annotations

import logging
import secrets
from logging import Logger
from pathlib import Path
from typing import Final

log: Logger = logging.getLogger(__name__)
EMPTY_CHAT_MESSAGES_FILE_NAME: Final[str] = "empty-chat-messages.txt"
MAX_EMPTY_CHAT_MESSAGE_WORDS: Final[int] = 20
BUILTIN_EMPTY_CHAT_MESSAGES: Final[tuple[str, ...]] = (
    "Ready when you are",
    "Make it interesting",
    "A blank page with excellent posture",
    "Professional mode, mostly",
    "The plot awaits",
    "Your move, genius",
    "Ideas welcome; chaos negotiable",
    "A fresh start, lightly caffeinated",
    "Be bold, be kind, use punctuation",
    "No pressure. Tiny bit of pressure",
    "Ask nicely... Or mischievously",
    "Fresh chat. Fresh sheets. No judgement.",
    "Flirt with an idea, not a data breach",
    "Let's make trouble—professionally",
)


class EmptyChatMessageProvider:
    """Choose a short built-in or user-supplied message for a pristine chat."""

    def __init__(self, data_directory: Path) -> None:
        self.path: Path = Path(data_directory) / EMPTY_CHAT_MESSAGES_FILE_NAME
        self._file_signature: tuple[int, int] | None = None
        self._messages: tuple[str, ...] = BUILTIN_EMPTY_CHAT_MESSAGES
        self._refresh_if_changed()

    @property
    def available_messages(self) -> tuple[str, ...]:
        """Return the currently loaded built-in and valid custom messages."""

        self._refresh_if_changed()
        return self._messages

    def choose(self) -> str:
        """Choose one message, reloading custom additions when their file changes."""

        return secrets.choice(self.available_messages)

    def _refresh_if_changed(self) -> None:
        signature: tuple[int, int] | None
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            signature = None
        except OSError as exc:
            log.warning("could not inspect empty-chat message file path=%s error=%s", self.path, exc)
            return
        else:
            signature = (stat.st_mtime_ns, stat.st_size)

        if signature == self._file_signature:
            return
        if signature is None:
            self._messages = BUILTIN_EMPTY_CHAT_MESSAGES
            self._file_signature = None
            return

        try:
            text: str = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            log.warning("could not read empty-chat message file path=%s error=%s", self.path, exc)
            return

        self._messages = (*BUILTIN_EMPTY_CHAT_MESSAGES, *_parse_custom_messages(text, path=self.path))
        self._file_signature = signature


def _parse_custom_messages(text: str, *, path: Path) -> tuple[str, ...]:
    """Read one short message per non-comment line, skipping invalid additions."""

    messages: list[str] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        message: str = raw_line.strip()
        if not message or message.startswith("#"):
            continue
        word_count: int = len(message.split())
        if word_count <= MAX_EMPTY_CHAT_MESSAGE_WORDS:
            messages.append(message)
            continue
        log.warning(
            "ignored empty-chat message path=%s line=%d word_count=%d maximum=%d",
            path,
            line_number,
            word_count,
            MAX_EMPTY_CHAT_MESSAGE_WORDS,
        )
    return tuple(messages)
