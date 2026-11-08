"""Individual chat JSON documents with recoverable deletion and corruption isolation."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from logging import Logger
from pathlib import Path
from typing import cast

from .atomic_write import (
    DurableWriteGuard,
    atomic_write_text,
    ensure_directory,
    move_path,
)
from .models import Chat, ChatJSON, FutureSchemaVersionError

log: Logger = logging.getLogger(__name__)
_CHAT_DIRECTORY_NAME: str = "chats"
_TRASH_DIRECTORY_NAME: str = "trash"


class ChatStorage:
    """Persist each chat independently; unreadable files remain untouched."""

    def __init__(self, path: Path) -> None:
        self.path: Path = Path(path)
        self.directory: Path = self.path.parent / _CHAT_DIRECTORY_NAME
        self.trash_directory: Path = self.directory / _TRASH_DIRECTORY_NAME
        self._isolated_ids: set[str] = set()
        self._write_guard: DurableWriteGuard = DurableWriteGuard()
        ensure_directory(self.path.parent)
        ensure_directory(self.directory)
        ensure_directory(self.trash_directory)

    def load_all(self) -> list[Chat]:
        """Load valid documents without moving or repairing unreadable files."""

        chats: list[Chat] = self._load_documents()
        log.info(
            "loaded chat store directory=%s chat_count=%d", self.directory, len(chats)
        )
        return chats

    @property
    def isolated_count(self) -> int:
        """Number of unreadable documents excluded from the loaded library."""
        return len(self._isolated_ids)

    def save_all(self, chats: Iterable[Chat]) -> None:
        """Synchronise a chat snapshot without deleting isolated documents."""

        chat_records: list[ChatJSON] = [chat.to_dict() for chat in chats]
        expected_ids: set[str] = _chat_record_ids(chat_records)
        expected_ids.update(self._isolated_ids)
        deleted_chat_ids: set[str] = {
            path.stem
            for path in self.directory.glob("*.json")
            if path.stem not in expected_ids
        }
        self.save_records(chat_records, deleted_chat_ids=deleted_chat_ids)

    def save_records(
        self,
        chat_records: Sequence[ChatJSON],
        *,
        deleted_chat_ids: Iterable[str] = (),
    ) -> None:
        """Persist changed documents and move explicitly deleted documents to trash."""

        documents: list[tuple[str, Path, str]] = []
        record_ids: set[str] = set()
        written_count: int = 0
        for chat_record in chat_records:
            chat_id: str = chat_record["id"]
            if chat_id in record_ids:
                raise ValueError(f"duplicate chat id: {chat_id}")
            record_ids.add(chat_id)
            data: str = json.dumps(chat_record, ensure_ascii=False, indent=4) + "\n"
            path: Path = self.path_for(chat_id)
            if chat_id in self._isolated_ids:
                raise ValueError("cannot overwrite an isolated chat document")
            documents.append((chat_id, path, data))

        deleted_ids: set[str] = set(deleted_chat_ids)
        overlap: set[str] = record_ids.intersection(deleted_ids)
        if overlap:
            raise ValueError(
                f"cannot save and delete the same chat: {sorted(overlap)!r}"
            )

        # Move deletions first. A sudden process failure can therefore leave a
        # prior snapshot in place, but cannot revive a successfully removed
        # chat from the active directory on the next startup.
        for chat_id in deleted_ids:
            self.delete(chat_id)

        for _chat_id, path, data in documents:
            with self._write_guard.protect(path):
                if self._document_matches(path, data):
                    continue
                atomic_write_text(path, data)
                written_count += 1
        log.info(
            "saved chat records directory=%s changed_count=%d deleted_count=%d written_count=%d",
            self.directory,
            len(chat_records),
            len(deleted_ids),
            written_count,
        )

    def delete(self, chat_id: str) -> None:
        """Move one persisted chat document to trash after state resolves it."""

        path: Path = self.path_for(chat_id)
        _ = self._move_document_to_trash(path)

    def _move_document_to_trash(self, path: Path) -> Path | None:
        """Move one active document into the recoverable chat trash directory."""

        with self._write_guard.protect(path):
            if not path.exists():
                return None
            trash_path: Path = self._trash_path_for(path)
            move_path(path, trash_path)
        log.info(
            "moved chat to trash path=%s trash_path=%s chat_id=%s",
            path,
            trash_path,
            path.stem,
        )
        return trash_path

    def _trash_path_for(self, path: Path) -> Path:
        """Return a non-conflicting trash path for one original chat document."""

        candidate: Path = self.trash_directory / path.name
        if not candidate.exists():
            return candidate
        timestamp: str = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        sequence: int = 1
        while candidate.exists():
            candidate = (
                self.trash_directory
                / f"{path.stem}.deleted-{timestamp}-{sequence}.json"
            )
            sequence += 1
        return candidate

    def path_for(self, chat_id: str) -> Path:
        """Return an in-directory path for one generated chat identifier."""

        if not chat_id or not chat_id.isalnum():
            raise ValueError("chat id must be alphanumeric")
        return self.directory / f"{chat_id}.json"

    def _load_documents(self) -> list[Chat]:
        """Load valid split documents without touching newer-schema documents."""

        self._isolated_ids.clear()
        chats: list[Chat] = []
        for path in sorted(self.directory.glob("*.json")):
            chat: Chat | None = self._load_one(path)
            if chat is not None:
                chats.append(chat)
        return chats

    def _load_one(self, path: Path) -> Chat | None:
        try:
            raw: object = cast(object, json.loads(path.read_text(encoding="utf-8")))
            if not isinstance(raw, dict):
                raise TypeError("root must be an object")
            mapping: dict[object, object] = cast(dict[object, object], raw)
            chat: Chat = Chat.from_dict(
                {key: value for key, value in mapping.items() if isinstance(key, str)}
            )
            if path.stem != chat.id:
                raise ValueError("document id does not match its file name")
            return chat
        except FutureSchemaVersionError as exc:
            self._isolated_ids.add(path.stem)
            log.warning("skipped future chat document path=%s reason=%s", path, exc)
            return None
        except Exception as exc:  # noqa: BLE001
            self._isolated_ids.add(path.stem)
            log.error(
                "isolated invalid chat document path=%s error_type=%s",
                path,
                type(exc).__name__,
            )
            return None

    @staticmethod
    def _document_matches(path: Path, data: str) -> bool:
        """Return whether the existing document already equals the next canonical JSON."""

        try:
            if path.stat().st_size != len(data.encode("utf-8")):
                return False
            return path.read_text(encoding="utf-8") == data
        except FileNotFoundError:
            return False


def _chat_record_ids(chat_records: Sequence[ChatJSON]) -> set[str]:
    """Return unique chat IDs from a complete snapshot or reject an invalid one."""

    chat_ids: set[str] = set()
    for chat_record in chat_records:
        chat_id: str = chat_record["id"]
        if chat_id in chat_ids:
            raise ValueError(f"duplicate chat id: {chat_id}")
        chat_ids.add(chat_id)
    return chat_ids
