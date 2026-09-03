from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

from jouzetsu.models import Chat, Message
from jouzetsu.storage import ChatStorage


def _legacy_chat_document(chat: Chat) -> dict[str, object]:
    """Return the historical unversioned representation of one chat."""

    document: dict[str, object] = dict(chat.to_dict())
    _ = document.pop("schema_version")
    return document


class ChatStorageTests(unittest.TestCase):
    def test_save_persists_each_chat_in_its_own_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path: Path = Path(tmp_dir) / "chats.json"
            storage = ChatStorage(path)
            first = Chat()
            second = Chat()

            with self.assertLogs("jouzetsu.storage", level="INFO") as logs:
                storage.save_all([first, second])
                chats = storage.load_all()

            self.assertEqual({chat.id for chat in chats}, {first.id, second.id})
            self.assertTrue(storage.path_for(first.id).is_file())
            self.assertTrue(storage.path_for(second.id).is_file())
            self.assertFalse(path.exists())
            saved = cast(
                dict[str, object],
                json.loads(storage.path_for(first.id).read_text(encoding="utf-8")),
            )
            self.assertEqual(saved["schema_version"], 1)
            log_output: str = "\n".join(logs.output)
            self.assertIn(
                f"saved chat records directory={storage.directory} changed_count=2",
                log_output,
            )
            self.assertIn(
                f"loaded chat store directory={storage.directory} chat_count=2",
                log_output,
            )

    def test_message_updated_at_is_persisted_in_its_chat_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path: Path = Path(tmp_dir) / "chats.json"
            storage = ChatStorage(path)
            message = Message(content="Revised", created_at=123.5, updated_at=456.0)
            chat = Chat(messages=[message])

            storage.save_all([chat])

            saved = cast(
                dict[str, object],
                json.loads(storage.path_for(chat.id).read_text(encoding="utf-8")),
            )
            messages = cast(list[dict[str, object]], saved["messages"])
            self.assertEqual(messages[0]["updated_at"], 456.0)

    def test_legacy_aggregate_is_migrated_and_archived_after_every_document_is_written(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path: Path = Path(tmp_dir) / "chats.json"
            first = Chat()
            second = Chat()
            legacy_content: str = json.dumps(
                {"chats": [_legacy_chat_document(first), _legacy_chat_document(second)]}
            )
            _ = path.write_text(legacy_content, encoding="utf-8")
            storage = ChatStorage(path)

            chats = storage.load_all()

            self.assertEqual({chat.id for chat in chats}, {first.id, second.id})
            self.assertTrue(storage.path_for(first.id).is_file())
            self.assertTrue(storage.path_for(second.id).is_file())
            self.assertFalse(path.exists())
            archives: list[Path] = list(path.parent.glob("chats.json.migrated-*"))
            self.assertEqual(len(archives), 1)
            self.assertEqual(archives[0].read_text(encoding="utf-8"), legacy_content)

    def test_legacy_migration_fails_if_the_aggregate_cannot_be_archived(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path: Path = Path(tmp_dir) / "chats.json"
            chat = Chat()
            legacy_content: str = json.dumps({"chats": [_legacy_chat_document(chat)]})
            _ = path.write_text(legacy_content, encoding="utf-8")
            storage = ChatStorage(path)
            unwritable_archive: Path = path.parent / "missing" / "chats.json.migrated"

            with (
                patch.object(
                    storage, "_migrated_backup_path", return_value=unwritable_archive
                ),
                self.assertRaises(OSError),
            ):
                _ = storage.load_all()

            self.assertEqual(path.read_text(encoding="utf-8"), legacy_content)
            self.assertTrue(storage.path_for(chat.id).is_file())

    def test_legacy_migration_preserves_newer_documents_from_an_interrupted_prior_migration(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path: Path = Path(tmp_dir) / "chats.json"
            legacy_chat = Chat(id="chat123", title="Legacy", updated_at=100.0)
            newer_document = Chat(
                id=legacy_chat.id, title="Newer document", updated_at=200.0
            )
            legacy_content: str = json.dumps(
                {"chats": [_legacy_chat_document(legacy_chat)]}
            )
            _ = path.write_text(legacy_content, encoding="utf-8")
            storage = ChatStorage(path)
            storage.save_all([newer_document])

            chats = storage.load_all()

            self.assertEqual([chat.title for chat in chats], ["Newer document"])
            self.assertFalse(path.exists())
            self.assertEqual(ChatStorage(path).load_all()[0].title, "Newer document")

    def test_save_moves_documents_for_deleted_chats_to_trash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = ChatStorage(Path(tmp_dir) / "chats.json")
            retained = Chat()
            removed = Chat()
            storage.save_all([retained, removed])

            storage.save_all([retained])

            self.assertTrue(storage.path_for(retained.id).exists())
            self.assertFalse(storage.path_for(removed.id).exists())
            trash_path: Path = storage.trash_directory / f"{removed.id}.json"
            self.assertTrue(trash_path.exists())
            self.assertEqual(
                cast(
                    dict[str, object],
                    json.loads(trash_path.read_text(encoding="utf-8")),
                )["id"],
                removed.id,
            )
            self.assertEqual([chat.id for chat in storage.load_all()], [retained.id])

    def test_save_moves_deleted_documents_before_writing_retained_chats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = ChatStorage(Path(tmp_dir) / "chats.json")
            retained = Chat(title="Original")
            removed = Chat()
            storage.save_all([retained, removed])
            retained.title = "Changed"

            with (
                patch(
                    "jouzetsu.storage.atomic_write_text",
                    side_effect=OSError("disk failure"),
                ),
                self.assertRaisesRegex(OSError, "disk failure"),
            ):
                storage.save_all([retained])

            self.assertFalse(storage.path_for(removed.id).exists())
            self.assertTrue((storage.trash_directory / f"{removed.id}.json").exists())

    def test_incremental_save_writes_only_changed_documents_without_scanning_the_chat_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = ChatStorage(Path(tmp_dir) / "chats.json")
            chats: list[Chat] = [Chat() for _ in range(128)]
            storage.save_all(chats)
            changed: Chat = chats[-1]
            unchanged: Chat = chats[0]
            changed.set_manual_title("Changed")

            with patch(
                "jouzetsu.storage.Path.glob",
                side_effect=AssertionError("incremental save scanned chat directory"),
            ):
                storage.save_records([changed.to_dict()])

            reloaded_by_id: dict[str, Chat] = {
                chat.id: chat for chat in ChatStorage(storage.path).load_all()
            }
            self.assertEqual(reloaded_by_id[changed.id].title, "Changed")
            self.assertTrue(storage.path_for(unchanged.id).is_file())

    def test_save_skips_documents_that_already_match_the_current_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = ChatStorage(Path(tmp_dir) / "chats.json")
            chat = Chat()
            storage.save_all([chat])

            with patch("jouzetsu.storage.atomic_write_text") as write:
                storage.save_all([chat])

            write.assert_not_called()

    def test_malformed_legacy_store_is_quarantined_before_returning_empty_store(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path: Path = Path(tmp_dir) / "chats.json"
            malformed_content: str = '{"chats": ['
            _ = path.write_text(malformed_content, encoding="utf-8")

            chats = ChatStorage(path).load_all()

            self.assertEqual(chats, [])
            self.assertFalse(path.exists())
            backups: list[Path] = list(path.parent.glob("chats.json.corrupt-*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(encoding="utf-8"), malformed_content)

    def test_invalid_chat_document_is_quarantined_without_losing_other_chats(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = ChatStorage(Path(tmp_dir) / "chats.json")
            valid = Chat()
            storage.save_all([valid])
            damaged: Path = storage.directory / "bad.json"
            _ = damaged.write_text("{", encoding="utf-8")

            chats = storage.load_all()

            self.assertEqual([chat.id for chat in chats], [valid.id])
            self.assertFalse(damaged.exists())
            self.assertEqual(len(list(storage.directory.glob("bad.json.corrupt-*"))), 1)

    def test_unversioned_split_document_round_trips_as_v1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = ChatStorage(Path(tmp_dir) / "chats.json")
            chat = Chat(title="Legacy")
            _ = storage.path_for(chat.id).write_text(
                json.dumps(_legacy_chat_document(chat)), encoding="utf-8"
            )

            loaded = storage.load_all()
            storage.save_all(loaded)

            saved = cast(
                dict[str, object],
                json.loads(storage.path_for(chat.id).read_text(encoding="utf-8")),
            )
            self.assertEqual(loaded, [chat])
            self.assertEqual(saved["schema_version"], 1)

    def test_future_schema_document_is_left_intact_during_snapshot_saves(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = ChatStorage(Path(tmp_dir) / "chats.json")
            chat = Chat()
            document: dict[str, object] = dict(chat.to_dict())
            document["schema_version"] = 2
            path = storage.path_for(chat.id)
            _ = path.write_text(json.dumps(document), encoding="utf-8")

            loaded = storage.load_all()
            storage.save_all(loaded)

            self.assertEqual(loaded, [])
            self.assertEqual(
                cast(dict[str, object], json.loads(path.read_text(encoding="utf-8"))),
                document,
            )
            self.assertFalse((storage.trash_directory / path.name).exists())
            self.assertEqual(
                list(storage.directory.glob(f"{path.name}.corrupt-*")), []
            )

    def test_future_schema_legacy_store_is_left_intact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path: Path = Path(tmp_dir) / "chats.json"
            chat = Chat()
            document: dict[str, object] = dict(chat.to_dict())
            document["schema_version"] = 2
            future_content: str = json.dumps({"chats": [document]})
            _ = path.write_text(future_content, encoding="utf-8")

            self.assertEqual(ChatStorage(path).load_all(), [])

            self.assertEqual(path.read_text(encoding="utf-8"), future_content)
            self.assertEqual(list(path.parent.glob("chats.json.corrupt-*")), [])

    def test_legacy_migration_does_not_replace_a_future_document_with_the_same_id(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path: Path = Path(tmp_dir) / "chats.json"
            legacy_chat = Chat(id="chat123", title="Legacy")
            future_document: dict[str, object] = dict(
                Chat(id=legacy_chat.id, title="Future").to_dict()
            )
            future_document["schema_version"] = 2
            legacy_content: str = json.dumps(
                {"chats": [_legacy_chat_document(legacy_chat)]}
            )
            _ = path.write_text(legacy_content, encoding="utf-8")
            storage = ChatStorage(path)
            future_path = storage.path_for(legacy_chat.id)
            _ = future_path.write_text(json.dumps(future_document), encoding="utf-8")

            self.assertEqual(storage.load_all(), [])

            self.assertEqual(
                cast(
                    dict[str, object],
                    json.loads(future_path.read_text(encoding="utf-8")),
                ),
                future_document,
            )
