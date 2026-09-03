"""Individual JSON-document persistence for chat transcripts.

The historical single-file store is imported once into ``chats/`` beside the
legacy file. Keeping every chat independent avoids rewriting every transcript
into one ever-growing document and confines corruption to one conversation.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from logging import Logger
from pathlib import Path
from typing import cast

from .atomic_write import atomic_write_text, move_path
from .models import Chat, ChatJSON, FutureSchemaVersionError

log: Logger = logging.getLogger(__name__)
_CORRUPT_FILE_SUFFIX: str = ".corrupt-"
_MIGRATED_FILE_SUFFIX: str = ".migrated-"
_CHAT_DIRECTORY_NAME: str = "chats"
_TRASH_DIRECTORY_NAME: str = "trash"


class ChatStorage:
    """Persist each chat independently, importing the legacy aggregate once."""

    def __init__(self, path: Path) -> None:
        self.path: Path = Path(path)
        self.directory: Path = self.path.parent / _CHAT_DIRECTORY_NAME
        self.trash_directory: Path = self.directory / _TRASH_DIRECTORY_NAME
        self._future_schema_document_ids: set[str] = set()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.trash_directory.mkdir(parents=True, exist_ok=True)

    def load_all(self) -> list[Chat]:
        """Load every chat document, migrating a legacy aggregate when present."""

        legacy_chats: list[Chat] | None = self._load_legacy()
        if legacy_chats is not None:
            return self._migrate_legacy(legacy_chats)

        chats: list[Chat] = self._load_documents()
        log.info(
            "loaded chat store directory=%s chat_count=%d", self.directory, len(chats)
        )
        return chats

    def save_all(self, chats: Iterable[Chat]) -> None:
        """Synchronise a chat snapshot without deleting newer-schema documents."""

        chat_records: list[ChatJSON] = [chat.to_dict() for chat in chats]
        expected_ids: set[str] = _chat_record_ids(chat_records)
        expected_ids.update(self._future_schema_document_ids)
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

    def _load_legacy(self) -> list[Chat] | None:
        """Read the old aggregate store, retaining it as migration source until archived."""

        if not self.path.exists():
            return None
        try:
            raw: object = cast(
                object, json.loads(self.path.read_text(encoding="utf-8"))
            )
            if not isinstance(raw, dict):
                raise TypeError("root must be an object")
            root: dict[object, object] = cast(dict[object, object], raw)
            chats_raw: object = root.get("chats", [])
            if not isinstance(chats_raw, list):
                raise TypeError("root must contain a chat list")
            chats: list[Chat] = []
            for item in cast(list[object], chats_raw):
                if not isinstance(item, dict):
                    raise TypeError("chat entry must be an object")
                chat_data: dict[str, object] = {
                    key: value
                    for key, value in cast(dict[object, object], item).items()
                    if isinstance(key, str)
                }
                chats.append(Chat.from_dict(chat_data))
        except FutureSchemaVersionError as exc:
            log.warning(
                "left future legacy chat store untouched path=%s reason=%s",
                self.path,
                exc,
            )
            return None
        except Exception as exc:  # noqa: BLE001
            self._quarantine_legacy(str(exc))
            return None
        log.info(
            "loaded legacy chat store path=%s chat_count=%d", self.path, len(chats)
        )
        return chats

    def _migrate_legacy(self, chats: list[Chat]) -> list[Chat]:
        """Import legacy records without discarding newer documents from an interrupted migration."""

        migrated_chats: list[Chat] = self._merge_migration_records(
            chats, self._load_documents()
        )
        # A document this build cannot read must win over an older aggregate
        # record with the same id, so migration never replaces it.
        migrated_chats = [
            chat
            for chat in migrated_chats
            if chat.id not in self._future_schema_document_ids
        ]

        try:
            self.save_all(migrated_chats)
        except Exception:
            log.exception(
                "legacy chat migration failed path=%s directory=%s",
                self.path,
                self.directory,
            )
            raise
        archive_path: Path = self._migrated_backup_path()
        try:
            _ = self.path.replace(archive_path)
        except OSError:
            log.exception("could not archive migrated chat store path=%s", self.path)
            # The aggregate remains authoritative until this rename succeeds.
            # Continuing to accept writes in the document directory would let a
            # later startup overwrite those newer writes from the old aggregate.
            raise
        log.info(
            "migrated legacy chat store path=%s archive=%s directory=%s chat_count=%d",
            self.path,
            archive_path,
            self.directory,
            len(migrated_chats),
        )
        return migrated_chats

    def _load_documents(self) -> list[Chat]:
        """Load valid split documents without touching newer-schema documents."""

        self._future_schema_document_ids.clear()
        chats: list[Chat] = []
        for path in sorted(self.directory.glob("*.json")):
            chat: Chat | None = self._load_one(path)
            if chat is not None:
                chats.append(chat)
        return chats

    @staticmethod
    def _merge_migration_records(
        legacy_chats: list[Chat], document_chats: list[Chat]
    ) -> list[Chat]:
        """Prefer the newest version of each chat while retaining chats unique to either source."""

        merged_by_id: dict[str, Chat] = {chat.id: chat for chat in legacy_chats}
        ordered_ids: list[str] = [chat.id for chat in legacy_chats]
        for document_chat in document_chats:
            legacy_chat: Chat | None = merged_by_id.get(document_chat.id)
            if legacy_chat is None:
                ordered_ids.append(document_chat.id)
                merged_by_id[document_chat.id] = document_chat
            elif document_chat.updated_at > legacy_chat.updated_at:
                merged_by_id[document_chat.id] = document_chat
        return [merged_by_id[chat_id] for chat_id in ordered_ids]

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
            self._future_schema_document_ids.add(path.stem)
            log.warning("skipped future chat document path=%s reason=%s", path, exc)
            return None
        except Exception as exc:  # noqa: BLE001
            self._quarantine_document(path, str(exc))
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

    def _quarantine_legacy(self, reason: str) -> None:
        backup_path: Path = self._corrupt_backup_path(self.path)
        try:
            _ = self.path.replace(backup_path)
        except OSError:
            log.exception(
                "could not quarantine corrupt legacy chat store path=%s", self.path
            )
            return
        log.warning(
            "quarantined corrupt legacy chat store reason=%s backup=%s",
            reason,
            backup_path,
        )

    def _quarantine_document(self, path: Path, reason: str) -> None:
        backup_path: Path = self._corrupt_backup_path(path)
        try:
            _ = path.replace(backup_path)
        except OSError:
            log.exception("could not quarantine corrupt chat path=%s", path)
            return
        log.warning(
            "quarantined corrupt chat document reason=%s backup=%s", reason, backup_path
        )

    @staticmethod
    def _corrupt_backup_path(path: Path) -> Path:
        return ChatStorage._timestamped_backup_path(path, _CORRUPT_FILE_SUFFIX)

    def _migrated_backup_path(self) -> Path:
        return self._timestamped_backup_path(self.path, _MIGRATED_FILE_SUFFIX)

    @staticmethod
    def _timestamped_backup_path(path: Path, suffix: str) -> Path:
        timestamp: str = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        candidate: Path = path.with_name(f"{path.name}{suffix}{timestamp}")
        sequence: int = 1
        while candidate.exists():
            candidate = path.with_name(f"{path.name}{suffix}{timestamp}-{sequence}")
            sequence += 1
        return candidate


def _chat_record_ids(chat_records: Sequence[ChatJSON]) -> set[str]:
    """Return unique chat IDs from a complete snapshot or reject an invalid one."""

    chat_ids: set[str] = set()
    for chat_record in chat_records:
        chat_id: str = chat_record["id"]
        if chat_id in chat_ids:
            raise ValueError(f"duplicate chat id: {chat_id}")
        chat_ids.add(chat_id)
    return chat_ids
