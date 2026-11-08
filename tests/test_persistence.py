"""Real-file failure injection through the sole Gate 3 mutation owner."""

from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Event
from unittest.mock import patch

import pytest
from app_support import WebHarness
from inference_support import backend

from jouzetsu.config import ConfigStore
from jouzetsu.events import StateChangeKind
from jouzetsu.inference import RevisionConflict
from jouzetsu.models import CharacterName


@pytest.mark.parametrize(
    "operation", ("create", "fork", "delete", "last_delete", "edit", "send")
)
def test_rejected_committed_command_never_publishes_or_replays(
    tmp_path: Path, operation: str
) -> None:
    async def scenario() -> None:
        h = WebHarness(tmp_path)
        chat = await h.state.new_chat()
        if operation != "last_delete":
            await h.state.new_chat()
        await h.state.inference.send_message(
            chat.id, 0, "original", "Original", generate=False
        )
        snapshot = chat.to_dict()
        ids = {item.id for item in h.state.chats}
        events: list[StateChangeKind] = []

        async def changed(kind: StateChangeKind) -> None:
            events.append(kind)

        h.state.add_listener(changed)

        async def command() -> None:
            runtime = h.state.inference
            if operation == "create":
                await runtime.create_chat()
            elif operation == "fork":
                await runtime.fork_chat(chat.id, chat.revision, chat.messages[0].id)
            elif operation in {"delete", "last_delete"}:
                await runtime.delete_chat(chat.id, chat.revision)
            elif operation == "edit":
                await runtime.edit_chat(
                    chat.id,
                    chat.revision,
                    lambda candidate: candidate.set_manual_title("Rejected"),
                )
            else:
                await runtime.send_message(
                    chat.id, chat.revision, "rejected", "Rejected", generate=False
                )

        method = "delete" if operation in {"delete", "last_delete"} else "save_records"
        with (
            patch.object(
                h.state.storage, method, side_effect=OSError("injected failure")
            ),
            pytest.raises(OSError),
        ):
            await command()
        assert not events
        assert chat.to_dict() == snapshot
        assert {item.id for item in h.state.chats} == ids
        assert {item.id for item in h.state.storage.load_all()} == ids
        await h.state.inference.edit_chat(
            chat.id,
            chat.revision,
            lambda candidate: candidate.set_manual_title("Valid"),
        )
        await h.state.persistence.flush_chats()
        reloaded = next(
            item for item in h.state.storage.load_all() if item.id == chat.id
        )
        assert reloaded.title == "Valid"
        assert [message.content for message in reloaded.messages] == ["Original"]
        await h.close()

    asyncio.run(scenario())


def test_chat_creation_and_deletion_do_not_write_configuration(tmp_path: Path) -> None:
    async def scenario() -> None:
        h = WebHarness(tmp_path)
        with patch.object(
            ConfigStore, "save", side_effect=OSError("config unavailable")
        ):
            chat = await h.state.new_chat()
            await h.state.inference.delete_chat(chat.id, chat.revision)
        assert not h.state.chats
        assert not h.state.storage.load_all()
        assert list(h.state.storage.trash_directory.glob("*.json"))
        await h.close()

    asyncio.run(scenario())


def test_pending_draft_cannot_overwrite_a_commit_or_replay_failed_edit(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        h = WebHarness(tmp_path)
        chat = await h.state.new_chat()
        runtime = h.state.inference
        await runtime.update_draft(chat.id, 0, "Draft")
        with (
            patch.object(
                h.state.storage, "save_records", side_effect=OSError("failed")
            ),
            pytest.raises(OSError),
        ):
            await runtime.edit_chat(
                chat.id, 0, lambda candidate: candidate.set_manual_title("Rejected")
            )
        await runtime.edit_chat(
            chat.id, 0, lambda candidate: candidate.set_manual_title("Accepted")
        )
        await h.state.persistence.flush_chats()
        assert h.state.storage.load_all()[0].to_dict() == chat.to_dict()
        assert chat.title == "Accepted" and chat.draft == "Draft"
        await h.close()

    asyncio.run(scenario())


def test_slow_write_keeps_event_loop_responsive_and_stale_edit_fails(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        h = WebHarness(tmp_path)
        chat = await h.state.new_chat()
        entered, release = Event(), Event()
        save = h.state.storage.save_records

        def delayed(records: object, **kwargs: object) -> None:
            entered.set()
            assert release.wait(3)
            save(records, **kwargs)  # pyright: ignore[reportArgumentType]

        with patch.object(h.state.storage, "save_records", delayed):
            first = asyncio.create_task(
                h.state.inference.edit_chat(
                    chat.id, 0, lambda candidate: candidate.set_manual_title("First")
                )
            )
            assert await asyncio.to_thread(entered.wait, 3)
            second = asyncio.create_task(
                h.state.inference.edit_chat(
                    chat.id, 0, lambda candidate: candidate.set_manual_title("Stale")
                )
            )
            await asyncio.sleep(0)
            assert chat.title == "New chat"
            release.set()
            await first
            with pytest.raises(RevisionConflict):
                await second
        assert h.state.storage.load_all()[0].title == "First"
        await h.close()

    asyncio.run(scenario())


def test_config_candidates_share_one_lock_and_preserve_backend_changes(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        h = WebHarness(tmp_path)
        added = backend()
        await asyncio.gather(
            h.state.set_global_system_prompt("Changed"),
            h.state.inference.update_backend(added),
        )
        saved = ConfigStore.for_home(tmp_path).load()
        assert saved.generation.system_prompt == "Changed"
        assert saved.backends.backend(added.id) == added
        with (
            patch.object(ConfigStore, "save", side_effect=OSError("failed")),
            pytest.raises(OSError),
        ):
            await h.state.set_global_system_prompt("Rejected")
        assert h.state.config.generation.system_prompt == "Changed"
        await h.state.set_approval_phrase("accepted")
        assert (
            ConfigStore.for_home(tmp_path).load().generation.system_prompt == "Changed"
        )
        await h.close()

    asyncio.run(scenario())


def test_character_failure_preserves_library_and_emits_no_success(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        h = WebHarness(tmp_path)
        original = await h.state.new_character()
        events: list[StateChangeKind] = []

        async def changed(kind: StateChangeKind) -> None:
            events.append(kind)

        h.state.add_listener(changed)
        with (
            patch.object(
                h.state.character_storage, "save", side_effect=OSError("failed")
            ),
            pytest.raises(OSError),
        ):
            await h.state.update_character(
                original.id,
                expected_revision=original.revision,
                name=CharacterName("Rejected"),
                fields=[],
            )
        assert h.state.character(original.id) is original
        assert not events
        assert h.state.character_storage.load_all()[0].to_dict() == original.to_dict()
        await h.close()

    asyncio.run(scenario())
