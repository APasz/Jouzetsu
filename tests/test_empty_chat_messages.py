from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

from pytest import LogCaptureFixture

from jouzetsu.empty_chat_messages import (
    BUILTIN_EMPTY_CHAT_MESSAGES,
    EMPTY_CHAT_MESSAGES_FILE_NAME,
    MAX_EMPTY_CHAT_MESSAGE_WORDS,
    EmptyChatMessageProvider,
)


def test_builtin_empty_chat_messages_are_between_one_and_twenty_words() -> None:
    assert all(1 <= len(message.split()) <= MAX_EMPTY_CHAT_MESSAGE_WORDS for message in BUILTIN_EMPTY_CHAT_MESSAGES)


def test_empty_chat_message_file_adds_valid_lines_and_skips_comments(tmp_path: Path) -> None:
    custom_path: Path = tmp_path / EMPTY_CHAT_MESSAGES_FILE_NAME
    _ = custom_path.write_text("# A comment\n\nOne word\nA neatly custom message\n", encoding="utf-8")

    provider = EmptyChatMessageProvider(tmp_path)

    assert provider.available_messages == (*BUILTIN_EMPTY_CHAT_MESSAGES, "One word", "A neatly custom message")
    with patch("jouzetsu.empty_chat_messages.secrets.choice", side_effect=_select_last_message):
        assert provider.choose() == "A neatly custom message"


def test_empty_chat_message_file_reloads_for_later_new_chats(tmp_path: Path) -> None:
    provider = EmptyChatMessageProvider(tmp_path)
    custom_path: Path = tmp_path / EMPTY_CHAT_MESSAGES_FILE_NAME
    _ = custom_path.write_text("A late arrival\n", encoding="utf-8")

    assert provider.available_messages[-1] == "A late arrival"


def test_empty_chat_message_file_ignores_lines_over_twenty_words(tmp_path: Path, caplog: LogCaptureFixture) -> None:
    custom_path: Path = tmp_path / EMPTY_CHAT_MESSAGES_FILE_NAME
    _ = custom_path.write_text(" ".join("word" for _ in range(MAX_EMPTY_CHAT_MESSAGE_WORDS + 1)), encoding="utf-8")

    provider = EmptyChatMessageProvider(tmp_path)

    assert provider.available_messages == BUILTIN_EMPTY_CHAT_MESSAGES
    assert "ignored empty-chat message" in caplog.text


def _select_last_message(messages: Sequence[str]) -> str:
    return messages[-1]
