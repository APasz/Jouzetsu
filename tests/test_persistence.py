"""Failure-path coverage for durable publication of committed edits."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Generator, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Literal, assert_never
from unittest.mock import patch

import pytest
from test_state import FakeLMStudioClient

from jouzetsu.atomic_write import CommitUncertainError
from jouzetsu.character_storage import CharacterStorage
from jouzetsu.config import (
    AppConfig,
    ConfigStore,
    DeviceAccessSettings,
    ThemeColorway,
    encode_config,
)
from jouzetsu.events import StateChangeKind
from jouzetsu.models import (
    Character,
    CharacterName,
    Chat,
    ChatJSON,
    ChatSamplingOverrides,
)
from jouzetsu.state import AppState
from jouzetsu.storage import ChatStorage

type ChatEdit = Literal[
    "rename",
    "model",
    "sampling",
    "prompt",
    "spelling",
    "reasoning",
    "edit",
    "delete",
    "delete_following",
    "merge",
    "truncate",
    "apply_review",
    "discard_review",
]
type ConfigEdit = Literal[
    "generation",
    "server",
    "appearance",
    "colourway",
    "alias",
    "access_defaults",
    "approve_device",
    "label_device",
    "forget_device",
    "register_device",
    "activity",
]


@pytest.fixture
def state(tmp_path: Path) -> Generator[AppState]:
    store = ConfigStore.for_home(tmp_path)
    config = store.load()
    config.access.devices["dvc_existing123"] = DeviceAccessSettings(label="Existing")
    store.save(config)
    state = AppState(
        config, ChatStorage(config.chats_file), FakeLMStudioClient(), config_store=store
    )
    _ = state.active_chat.add_message("user", "First")
    _ = state.active_chat.add_message("user", "Second")
    reply = state.active_chat.add_message("assistant", "Original answer")
    reply.set_continuity_rewrite("Reviewed answer")
    asyncio.run(state.persist_chats())
    try:
        yield state
    finally:
        _ = asyncio.run(state.shutdown())


async def _edit_chat(state: AppState, operation: ChatEdit) -> None:
    chat = state.active_chat
    first, second, reply = chat.messages
    match operation:
        case "rename":
            await state.rename_chat(chat.id, "Changed title")
        case "model":
            await state.set_active_chat_model("different-model")
        case "sampling":
            await state.set_active_chat_sampling_overrides(
                ChatSamplingOverrides(temperature=0.2)
            )
        case "prompt":
            await state.set_active_chat_system_prompt("Changed prompt")
        case "spelling":
            await state.set_active_chat_postprocess_british_spellings(False)
        case "reasoning":
            await state.set_active_chat_save_reasoning(False)
        case "edit":
            await state.edit_message(first.id, "Changed message")
        case "delete":
            await state.delete_message(first.id)
        case "delete_following":
            _ = await state.delete_message_and_following_with_undo(second.id)
        case "merge":
            _ = await state.merge_message_with_previous(second.id)
        case "truncate":
            await state.truncate_chat_to_message(first.id)
        case "apply_review":
            await state.apply_continuity_rewrite(reply.id)
        case "discard_review":
            await state.discard_continuity_rewrite(reply.id)
        case _:
            assert_never(operation)


@pytest.mark.parametrize(
    "operation",
    (
        "rename",
        "model",
        "sampling",
        "prompt",
        "spelling",
        "reasoning",
        "edit",
        "delete",
        "delete_following",
        "merge",
        "truncate",
        "apply_review",
        "discard_review",
    ),
)
def test_rejected_chat_edit_never_changes_or_later_replays_state(
    state: AppState, operation: ChatEdit
) -> None:
    before = state.active_chat.to_dict()
    path = state.storage.path_for(state.active_chat.id)
    saved = path.read_bytes()
    events: list[StateChangeKind] = []

    async def listener(kind: StateChangeKind) -> None:
        events.append(kind)

    _ = state.add_listener(listener)

    def fail_save(
        _records: Sequence[ChatJSON], *, deleted_chat_ids: Iterable[str] = ()
    ) -> None:
        assert state.active_chat.to_dict() == before
        assert not tuple(deleted_chat_ids)
        raise OSError("disk unavailable")

    async def scenario() -> None:
        with (
            patch.object(state.storage, "save_records", side_effect=fail_save),
            pytest.raises(OSError, match="disk unavailable"),
        ):
            await _edit_chat(state, operation)
        assert state.active_chat.to_dict() == before
        assert events == []
        await state.persist_chats()

    asyncio.run(scenario())
    assert state.active_chat.to_dict() == before
    assert path.read_bytes() == saved


async def _edit_config(state: AppState, operation: ConfigEdit) -> None:
    config = state.config
    match operation:
        case "generation":
            await state.set_generation_defaults(
                replace(config.generation, temperature=0.2)
            )
        case "server":
            await state.set_global_settings(
                replace(config.generation, temperature=0.2),
                replace(config.server, base_url="http://different-server:1234/v1"),
            )
        case "appearance":
            await state.set_appearance_settings(
                config.theme,
                config.ui.icon_colors,
                dark_mode=not config.ui.dark_mode,
                message_action_style=config.ui.message_action_style,
                activity_start_color="#112233",
                activity_end_color="#445566",
            )
        case "colourway":
            await state.reset_theme_colorway(ThemeColorway.SYSTEM)
        case "alias":
            await state.set_model_alias("demo-model", "Changed alias")
        case "access_defaults":
            await state.set_access_defaults(
                default_private=True,
                allow_localhost_without_approval=True,
                global_settings_for_approved=True,
                allow_network_device_reassociation=False,
            )
        case "approve_device":
            await state.set_device_access_allowed("dvc_existing123", True)
        case "label_device":
            await state.set_access_device_label("dvc_existing123", "Changed label")
        case "forget_device":
            await state.forget_pending_access_device("dvc_existing123")
        case "register_device":
            await state.register_access_device("dvc_another123", "Another browser")
        case "activity":
            await state.refresh_access_device_activity(
                "dvc_existing123", "Existing", last_ip="192.0.2.5"
            )
        case _:
            assert_never(operation)


@pytest.mark.parametrize(
    "operation",
    (
        "generation",
        "server",
        "appearance",
        "colourway",
        "alias",
        "access_defaults",
        "approve_device",
        "label_device",
        "forget_device",
        "register_device",
        "activity",
    ),
)
def test_rejected_settings_never_publish_or_later_replay(
    state: AppState, operation: ConfigEdit
) -> None:
    # Alias changes accept registered selections, including an offline model.
    state.config.server.default_model = "demo-model"
    asyncio.run(state._persistence.save_config())  # pyright: ignore[reportPrivateUsage]
    before = encode_config(state.config)
    saved = state.config.config_file.read_bytes()
    events: list[StateChangeKind] = []

    async def listener(kind: StateChangeKind) -> None:
        events.append(kind)

    _ = state.add_listener(listener)

    def fail_save(_store: ConfigStore, _candidate: AppConfig) -> None:
        assert encode_config(state.config) == before
        raise OSError("disk unavailable")

    async def scenario() -> None:
        with (
            patch.object(ConfigStore, "save", autospec=True, side_effect=fail_save),
            pytest.raises(OSError, match="disk unavailable"),
        ):
            await _edit_config(state, operation)
        assert encode_config(state.config) == before
        assert events == []
        await state._persistence.save_config()  # pyright: ignore[reportPrivateUsage]

    asyncio.run(scenario())
    assert state.config.config_file.read_bytes() == saved


def test_success_is_published_only_after_the_saved_candidate(state: AppState) -> None:
    previous_title = state.active_chat.title
    save = state.storage.save_records
    events: list[StateChangeKind] = []

    def inspect_save(
        records: Sequence[ChatJSON], *, deleted_chat_ids: Iterable[str] = ()
    ) -> None:
        assert state.active_chat.title == previous_title
        assert events == []
        save(records, deleted_chat_ids=deleted_chat_ids)

    async def listener(kind: StateChangeKind) -> None:
        assert state.active_chat.title == "Saved title"
        assert state.storage.load_all()[0].title == "Saved title"
        events.append(kind)

    _ = state.add_listener(listener)
    with patch.object(state.storage, "save_records", side_effect=inspect_save):
        asyncio.run(state.rename_chat(state.active_chat.id, "Saved title"))
    assert events == [StateChangeKind.FULL]


def test_failed_committed_edit_does_not_replace_a_pending_draft(
    state: AppState,
) -> None:
    previous_title = state.active_chat.title

    async def scenario() -> None:
        await state.set_chat_draft(state.active_chat.id, "Accepted draft")
        with (
            patch.object(
                state.storage, "save_records", side_effect=OSError("disk unavailable")
            ),
            pytest.raises(OSError),
        ):
            await state.rename_chat(state.active_chat.id, "Rejected title")
        await state._persistence.flush_chats()  # pyright: ignore[reportPrivateUsage]

    asyncio.run(scenario())
    saved = state.storage.load_all()[0]
    assert saved.title == previous_title
    assert saved.draft == "Accepted draft"


def test_uncertain_commit_keeps_memory_and_blocks_following_writes(
    state: AppState,
) -> None:
    before = state.active_chat.to_dict()
    with (
        patch(
            "jouzetsu.atomic_write._confirm_directories",
            side_effect=CommitUncertainError("sync failed"),
        ),
        pytest.raises(CommitUncertainError),
    ):
        asyncio.run(state.rename_chat(state.active_chat.id, "Unconfirmed title"))
    assert state.active_chat.to_dict() == before
    with pytest.raises(CommitUncertainError, match="requires recovery"):
        asyncio.run(state.rename_chat(state.active_chat.id, "Another title"))
    # The guard also prevents the old snapshot overwriting the uncertain new file.
    with pytest.raises(CommitUncertainError, match="requires recovery"):
        asyncio.run(state.persist_chats())
    assert state.active_chat.to_dict() == before


def test_shutdown_flushes_pending_drafts_for_already_saved_chats(
    state: AppState,
) -> None:
    async def scenario() -> None:
        await state.set_chat_draft(state.active_chat.id, "Draft before shutdown")
        assert await state.shutdown()

    asyncio.run(scenario())
    assert state.storage.load_all()[0].draft == "Draft before shutdown"


def test_committed_save_recovers_a_failed_pending_draft(state: AppState) -> None:
    async def scenario() -> None:
        await state.set_chat_draft(state.active_chat.id, "Accepted draft")
        with (
            patch.object(
                state.storage, "save_records", side_effect=OSError("disk unavailable")
            ),
            pytest.raises(OSError),
        ):
            await state._persistence.flush_chats()  # pyright: ignore[reportPrivateUsage]
        await state.rename_chat(state.active_chat.id, "Saved after recovery")
        assert await state.shutdown()

    asyncio.run(scenario())
    saved = state.storage.load_all()[0]
    assert saved.title == "Saved after recovery"
    assert saved.draft == "Accepted draft"


def test_config_store_snapshot_does_not_alias_mutable_runtime_settings(
    tmp_path: Path,
) -> None:
    store = ConfigStore.for_home(tmp_path)
    config = store.load()
    saved = config.config_file.read_bytes()
    before = encode_config(config)
    config.generation.temperature = 0.1
    config.access.devices["dvc_another123"] = DeviceAccessSettings(label="Uncommitted")
    with (
        patch(
            "jouzetsu.config.store.atomic_write_text",
            side_effect=OSError("disk unavailable"),
        ),
        pytest.raises(OSError),
    ):
        store.save(config)
    assert store._persisted is not None  # pyright: ignore[reportPrivateUsage]
    assert encode_config(store._persisted) == before  # pyright: ignore[reportPrivateUsage]
    assert config.config_file.read_bytes() == saved


def test_uncertain_config_commit_prevents_saving_an_older_snapshot(
    state: AppState,
) -> None:
    before = encode_config(state.config)
    with (
        patch(
            "jouzetsu.atomic_write._confirm_directories",
            side_effect=CommitUncertainError("sync failed"),
        ),
        pytest.raises(CommitUncertainError),
    ):
        asyncio.run(
            state.set_generation_defaults(
                replace(state.config.generation, temperature=0.1)
            )
        )
    assert encode_config(state.config) == before
    with pytest.raises(CommitUncertainError, match="requires recovery"):
        asyncio.run(state._persistence.save_config())  # pyright: ignore[reportPrivateUsage]


type Creation = Literal["new", "fork", "character", "cast"]


async def named_characters(state: AppState) -> tuple[Character, ...]:
    characters: list[Character] = []
    for name in ("First", "Second"):
        draft = await state.new_character()
        characters.append(
            await state.update_character(
                draft.id,
                expected_revision=draft.revision,
                name=CharacterName(name),
                fields=[],
            )
        )
    return tuple(characters)


async def create_chat(
    state: AppState, kind: Creation, characters: tuple[Character, ...]
) -> Chat:
    match kind:
        case "new":
            return await state.new_chat()
        case "fork":
            return await state.fork_chat_at_message(state.active_chat.messages[0].id)
        case "character":
            return await state.start_chat_from_character(characters[0].id)
        case "cast":
            return await state.start_chat_from_characters(
                tuple(character.id for character in characters)
            )


def listen(state: AppState) -> list[StateChangeKind]:
    events: list[StateChangeKind] = []

    async def listener(kind: StateChangeKind) -> None:
        events.append(kind)

    _ = state.add_listener(listener)
    return events


@pytest.mark.parametrize("kind", ("new", "fork", "character", "cast"))
def test_creation_failure_never_publishes_or_queues_a_rejected_chat(
    state: AppState, kind: Creation
) -> None:
    async def scenario() -> None:
        characters = await named_characters(state)
        original = state.active_chat
        before = original.to_dict()
        config_before = encode_config(state.config)
        saved_config = state.config.config_file.read_bytes()
        saved_chat = state.storage.path_for(original.id).read_bytes()
        saved_characters = [character.to_dict() for character in characters]
        events = listen(state)
        with (
            patch.object(
                state.storage, "save_records", side_effect=OSError("creation failed")
            ),
            pytest.raises(OSError, match="creation failed"),
        ):
            _ = await create_chat(state, kind, characters)
        assert state.chats == [original]
        assert state.active_chat is original
        assert original.to_dict() == before
        assert encode_config(state.config) == config_before
        assert state.config.config_file.read_bytes() == saved_config
        assert state.storage.path_for(original.id).read_bytes() == saved_chat
        assert [
            state.character(character.id).to_dict() for character in characters
        ] == saved_characters
        assert events == []
        await state.rename_chat(original.id, "Later valid edit")
        await state.persist_chats()
        assert [chat.id for chat in state.storage.load_all()] == [original.id]

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ("new", "fork", "character", "cast"))
def test_creation_config_failure_leaves_only_recoverable_unpublished_json(
    state: AppState, kind: Creation
) -> None:
    async def scenario() -> None:
        characters = await named_characters(state)
        original = state.active_chat
        before = original.to_dict()
        config_bytes = state.config.config_file.read_bytes()
        events = listen(state)
        with (
            patch.object(ConfigStore, "save", side_effect=OSError("config failed")),
            pytest.raises(OSError, match="config failed"),
        ):
            _ = await create_chat(state, kind, characters)
        recovered = state.storage.load_all()
        assert len(recovered) == 2
        assert state.chats == [original]
        assert state.active_chat is original
        assert original.to_dict() == before
        assert state.config.ui.active_chat_id == original.id
        assert state.config.config_file.read_bytes() == config_bytes
        assert events == []
        await state.rename_chat(original.id, "Valid after failed creation")
        assert state.active_chat is original
        assert len(state.chats) == 1
        # Reopening explicitly recovers the extra document; the old saved selection remains valid.
        store = ConfigStore.for_paths(state.config.paths)
        reopened = AppState(
            store.load(),
            ChatStorage(state.config.chats_file),
            FakeLMStudioClient(),
            config_store=store,
        )
        assert len(reopened.chats) == 2
        assert reopened.active_chat.id == original.id
        assert await reopened.shutdown()

    asyncio.run(scenario())


def test_failed_selection_preserves_registry_config_and_notifications(
    state: AppState,
) -> None:
    async def scenario() -> None:
        old = state.active_chat
        selected = await state.new_chat()
        before = state.config.config_file.read_bytes()
        events = listen(state)
        with (
            patch.object(ConfigStore, "save", side_effect=OSError("config failed")),
            pytest.raises(OSError),
        ):
            await state.select_chat(old.id)
        assert state.active_chat is selected
        assert state.config.ui.active_chat_id == selected.id
        assert state.config.config_file.read_bytes() == before
        assert events == []
        await state.rename_chat(selected.id, "Still selected")
        assert (
            ConfigStore.for_paths(state.config.paths).load().ui.active_chat_id
            == selected.id
        )

    asyncio.run(scenario())


@pytest.mark.parametrize("last", (False, True))
@pytest.mark.parametrize("stage", ("save_records", "move"))
def test_failed_deletion_keeps_live_chats_and_never_retries_the_deletion(
    state: AppState, last: bool, stage: str
) -> None:
    async def scenario() -> None:
        original = state.active_chat
        if not last:
            _ = await state.new_chat()
            await state.select_chat(original.id)
        before = [chat.to_dict() for chat in state.chats]
        events = listen(state)
        target = (
            "jouzetsu.storage.move_path"
            if stage == "move"
            else "jouzetsu.storage.ChatStorage.save_records"
        )
        with (
            patch(target, side_effect=OSError("delete failed")),
            pytest.raises(OSError),
        ):
            await state.delete_chat(original.id)
        assert state.active_chat is original
        assert [chat.to_dict() for chat in state.chats] == before
        assert state.config.ui.active_chat_id == original.id
        assert events == []
        assert state.storage.path_for(original.id).exists()
        await state.persist_chats()
        await state.rename_chat(original.id, "Still exists")
        assert (
            next(
                chat for chat in state.storage.load_all() if chat.id == original.id
            ).title
            == "Still exists"
        )
        assert not list(state.storage.trash_directory.glob("*.json"))

    asyncio.run(scenario())


def test_last_chat_config_failure_preserves_original_and_saved_replacement(
    state: AppState,
) -> None:
    original = state.active_chat
    saved = original.to_dict()
    config_bytes = state.config.config_file.read_bytes()
    events = listen(state)
    with (
        patch.object(ConfigStore, "save", side_effect=OSError("config failed")),
        pytest.raises(OSError),
    ):
        asyncio.run(state.delete_chat(original.id))
    assert state.active_chat is original
    assert state.chats == [original]
    assert original.to_dict() == saved
    assert events == []
    assert state.config.config_file.read_bytes() == config_bytes
    assert len(state.storage.load_all()) == 2
    assert not list(state.storage.trash_directory.glob("*.json"))


def test_last_chat_deletion_orders_replacement_selection_then_trash(
    state: AppState,
) -> None:
    original = state.active_chat
    order: list[str] = []
    save_records = state.storage.save_records
    save_config = ConfigStore.save

    def record_chats(
        records: Sequence[ChatJSON], *, deleted_chat_ids: Iterable[str] = ()
    ) -> None:
        assert state.active_chat is original
        order.append("trash" if tuple(deleted_chat_ids) else "replacement")
        save_records(records, deleted_chat_ids=deleted_chat_ids)

    def record_config(store: ConfigStore, config: AppConfig) -> None:
        assert state.active_chat is original
        assert state.storage.path_for(config.ui.active_chat_id).exists()
        order.append("selection")
        save_config(store, config)

    events = listen(state)
    with (
        patch.object(state.storage, "save_records", side_effect=record_chats),
        patch.object(ConfigStore, "save", autospec=True, side_effect=record_config),
    ):
        asyncio.run(state.delete_chat(original.id))
    assert order == ["replacement", "selection", "trash"]
    assert events == [StateChangeKind.FULL]
    assert state.active_chat.id != original.id
    assert [chat.id for chat in state.storage.load_all()] == [state.active_chat.id]
    assert list(state.storage.trash_directory.glob("*.json"))


def test_uncertain_deletion_retains_memory_and_recoverable_trash_and_blocks_resurrection(
    state: AppState,
) -> None:
    original = state.active_chat
    saved = state.storage.path_for(original.id).read_bytes()
    events = listen(state)
    from jouzetsu import atomic_write

    sync = atomic_write._sync_directory  # pyright: ignore[reportPrivateUsage]

    def fail_trash_sync(directory: Path) -> None:
        if directory == state.storage.trash_directory:
            raise OSError("trash sync failed")
        sync(directory)

    with (
        patch("jouzetsu.atomic_write._sync_directory", side_effect=fail_trash_sync),
        pytest.raises(CommitUncertainError),
    ):
        asyncio.run(state.delete_chat(original.id))
    assert state.active_chat is original
    assert state.chats == [original]
    assert events == []
    assert not state.storage.path_for(original.id).exists()
    trash = list(state.storage.trash_directory.glob("*.json"))
    assert len(trash) == 1 and trash[0].read_bytes() == saved
    disk_config = ConfigStore.for_paths(state.config.paths).load()
    assert state.storage.path_for(disk_config.ui.active_chat_id).exists()
    with pytest.raises(CommitUncertainError, match="requires recovery"):
        asyncio.run(state.persist_chats())
    assert not state.storage.path_for(original.id).exists()


@contextmanager
def blocked_chat_write(
    state: AppState, *, fail: bool = False
) -> Generator[tuple[asyncio.Event, threading.Event]]:
    """Pause the first real write with explicit cross-thread/async handshakes."""

    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    save = state.storage.save_records
    first = True

    def blocked(
        records: Sequence[ChatJSON], *, deleted_chat_ids: Iterable[str] = ()
    ) -> None:
        nonlocal first
        if first:
            first = False
            assert threading.current_thread() is not threading.main_thread()
            loop.call_soon_threadsafe(started.set)
            if not release.wait(timeout=5):
                raise TimeoutError("test did not release persistence worker")
            if fail:
                raise OSError("controlled failure")
        save(records, deleted_chat_ids=deleted_chat_ids)

    with patch.object(state.storage, "save_records", side_effect=blocked):
        try:
            yield started, release
        finally:
            release.set()


def test_slow_committed_save_keeps_loop_responsive_and_preserves_concurrent_edits(
    state: AppState,
) -> None:
    async def scenario() -> None:
        original = state.active_chat
        with blocked_chat_write(state) as (started, release):
            rename = asyncio.create_task(state.rename_chat(original.id, "Saved title"))
            await asyncio.wait_for(started.wait(), timeout=5)
            assert original.title != "Saved title"
            model = asyncio.create_task(state.set_active_chat_model("new-model"))
            draft = asyncio.create_task(state.set_chat_draft(original.id, "New draft"))
            # A separate asynchronous service can finish while the disk worker is paused.
            _ = await state.refresh_runtime_status()
            assert not rename.done()
            release.set()
            await asyncio.gather(rename, model, draft)
        await state.persist_chats()
        saved = state.storage.load_all()[0]
        assert state.active_chat is original
        assert saved.title == original.title == "Saved title"
        assert saved.model == original.model == "new-model"
        assert saved.draft == original.draft == "New draft"

    asyncio.run(scenario())


def test_draft_arriving_during_a_write_is_retained_and_cannot_overwrite_a_commit(
    state: AppState,
) -> None:
    async def scenario() -> None:
        chat = state.active_chat
        await state.set_chat_draft(chat.id, "First draft")
        with blocked_chat_write(state) as (started, release):
            flush = asyncio.create_task(state._persistence.flush_chats())  # pyright: ignore[reportPrivateUsage]
            await asyncio.wait_for(started.wait(), timeout=5)
            await state.set_chat_draft(chat.id, "Latest draft")
            rename = asyncio.create_task(state.rename_chat(chat.id, "Committed title"))
            release.set()
            await asyncio.gather(flush, rename)
        await state._persistence.flush_chats()  # pyright: ignore[reportPrivateUsage]
        saved = state.storage.load_all()[0]
        assert saved.title == "Committed title"
        assert saved.draft == "Latest draft"

    asyncio.run(scenario())


def test_cancelled_caller_cannot_release_publication_while_its_thread_is_writing(
    state: AppState,
) -> None:
    async def scenario() -> None:
        chat = state.active_chat
        with blocked_chat_write(state) as (started, release):
            rename = asyncio.create_task(
                state.rename_chat(chat.id, "Completed despite disconnect")
            )
            await asyncio.wait_for(started.wait(), timeout=5)
            _ = rename.cancel()
            model = asyncio.create_task(state.set_active_chat_model("later-model"))
            checkpoint = asyncio.Event()
            asyncio.get_running_loop().call_soon(checkpoint.set)
            await checkpoint.wait()
            assert not rename.done()
            assert not model.done()
            assert chat.title != "Completed despite disconnect"
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await rename
            await model
        await state.persist_chats()
        saved = state.storage.load_all()[0]
        assert saved.title == chat.title == "Completed despite disconnect"
        assert saved.model == chat.model == "later-model"

    asyncio.run(scenario())


def test_concurrent_valid_command_does_not_replay_a_failed_candidate(
    state: AppState,
) -> None:
    async def scenario() -> None:
        chat = state.active_chat
        title = chat.title
        events = listen(state)
        with blocked_chat_write(state, fail=True) as (started, release):
            rename = asyncio.create_task(state.rename_chat(chat.id, "Rejected title"))
            await asyncio.wait_for(started.wait(), timeout=5)
            model = asyncio.create_task(state.set_active_chat_model("valid-model"))
            release.set()
            with pytest.raises(OSError, match="controlled failure"):
                await rename
            await model
        await state.persist_chats()
        saved = state.storage.load_all()[0]
        assert saved.title == chat.title == title
        assert saved.model == chat.model == "valid-model"
        assert events == [StateChangeKind.FULL]

    asyncio.run(scenario())


def test_configuration_candidates_are_prepared_and_saved_in_order(
    state: AppState,
) -> None:
    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()
        save = ConfigStore.save
        first = True

        def blocked(store: ConfigStore, candidate: AppConfig) -> None:
            nonlocal first
            assert threading.current_thread() is not threading.main_thread()
            if first:
                first = False
                loop.call_soon_threadsafe(started.set)
                if not release.wait(timeout=5):
                    raise TimeoutError("test did not release config worker")
            save(store, candidate)

        with patch.object(ConfigStore, "save", autospec=True, side_effect=blocked):
            settings = asyncio.create_task(
                state.set_generation_defaults(
                    replace(state.config.generation, temperature=0.2)
                )
            )
            await asyncio.wait_for(started.wait(), timeout=5)
            prompt = asyncio.create_task(state.set_global_system_prompt("Later prompt"))
            label = asyncio.create_task(
                state.set_access_device_label("dvc_existing123", "Later label")
            )
            release.set()
            await asyncio.gather(settings, prompt, label)
        reloaded = ConfigStore.for_paths(state.config.paths).load()
        assert (
            reloaded.generation.temperature
            == state.config.generation.temperature
            == 0.2
        )
        assert reloaded.generation.system_prompt == "Later prompt"
        assert reloaded.access.devices["dvc_existing123"].label == "Later label"
        assert encode_config(reloaded) == encode_config(state.config)

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ("send", "regenerate", "continue"))
def test_generation_acceptance_failure_preserves_the_transcript_and_starts_no_stream(
    state: AppState, operation: Literal["send", "regenerate", "continue"]
) -> None:
    original = state.active_chat
    before = original.to_dict()
    messages = tuple(original.messages)
    events = listen(state)

    async def scenario() -> None:
        with (
            patch.object(
                state.storage, "save_records", side_effect=OSError("save failed")
            ),
            pytest.raises(OSError),
        ):
            match operation:
                case "send":
                    _ = await state.send_user_message("Rejected message")
                case "regenerate":
                    await state.regenerate_last()
                case "continue":
                    await state.continue_last_response()
        assert not state.is_generating()
        assert original.to_dict() == before
        assert all(
            current is previous
            for current, previous in zip(original.messages, messages, strict=True)
        )
        assert events == []
        await state.rename_chat(original.id, "Later valid title")
        assert isinstance(state.client, FakeLMStudioClient)
        assert not state.client.generations

    asyncio.run(scenario())


def test_character_failure_preserves_the_library_and_emits_no_success(
    state: AppState,
) -> None:
    async def scenario() -> None:
        original = await state.new_character()
        saved = original.to_dict()
        events = listen(state)
        with (
            patch.object(
                state.character_storage, "delete", side_effect=OSError("delete failed")
            ),
            pytest.raises(OSError),
        ):
            await state.delete_character(original.id)
        assert state.character(original.id) is original
        assert state.character(original.id).to_dict() == saved
        with (
            patch.object(
                state.character_storage, "save", side_effect=OSError("save failed")
            ),
            pytest.raises(OSError),
        ):
            _ = await state.new_character()
        with (
            patch.object(
                state.character_storage, "save", side_effect=OSError("save failed")
            ),
            pytest.raises(OSError),
        ):
            _ = await state.update_character(
                original.id,
                expected_revision=original.revision,
                name=CharacterName("Rejected"),
                fields=[],
            )
        assert state.character(original.id) is original
        assert state.characters == [original]
        assert [
            character.to_dict()
            for character in CharacterStorage(
                state.config.characters_directory
            ).load_all()
        ] == [saved]
        assert events == []

    asyncio.run(scenario())


def test_empty_store_creation_failure_does_not_publish_an_application(
    tmp_path: Path,
) -> None:
    store = ConfigStore.for_home(tmp_path)
    config = store.load()
    storage = ChatStorage(config.chats_file)
    before = encode_config(config)
    with (
        patch.object(
            storage, "save_records", side_effect=OSError("initial save failed")
        ),
        pytest.raises(OSError),
    ):
        _ = AppState(config, storage, FakeLMStudioClient(), config_store=store)
    assert storage.load_all() == []
    assert encode_config(config) == before


def test_initial_config_failure_leaves_a_recoverable_chat_without_changing_config(
    tmp_path: Path,
) -> None:
    store = ConfigStore.for_home(tmp_path)
    config = store.load()
    storage = ChatStorage(config.chats_file)
    saved = config.config_file.read_bytes()
    with (
        patch.object(ConfigStore, "save", side_effect=OSError("initial config failed")),
        pytest.raises(OSError),
    ):
        _ = AppState(config, storage, FakeLMStudioClient(), config_store=store)
    assert config.ui.active_chat_id == ""
    assert config.config_file.read_bytes() == saved
    assert len(storage.load_all()) == 1
    reopened = AppState(store.load(), storage, FakeLMStudioClient(), config_store=store)
    assert len(reopened.chats) == 1
    assert asyncio.run(reopened.shutdown())
