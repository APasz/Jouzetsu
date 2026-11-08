from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

from jouzetsu.models import CHAT_SCHEMA_VERSION, Chat, Message
from jouzetsu.storage import ChatStorage


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
            self.assertEqual(saved["schema_version"], CHAT_SCHEMA_VERSION)
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

    def test_invalid_chat_document_is_isolated_without_losing_other_chats(
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
            self.assertEqual(damaged.read_text(), "{")
            self.assertEqual(list(storage.directory.glob("bad.json.corrupt-*")), [])

    def test_future_schema_document_is_left_intact_during_snapshot_saves(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = ChatStorage(Path(tmp_dir) / "chats.json")
            chat = Chat()
            document: dict[str, object] = dict(chat.to_dict())
            document["schema_version"] = CHAT_SCHEMA_VERSION + 1
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
            self.assertEqual(list(storage.directory.glob(f"{path.name}.corrupt-*")), [])
