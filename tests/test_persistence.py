"""Failure-path coverage for durable publication of committed edits."""

from __future__ import annotations

import asyncio
from collections.abc import Generator, Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Literal, assert_never
from unittest.mock import patch

import pytest
from test_state import FakeLMStudioClient

from jouzetsu.atomic_write import CommitUncertainError
from jouzetsu.config import (
    AppConfig,
    ConfigStore,
    DeviceAccessSettings,
    ThemeColorway,
    encode_config,
)
from jouzetsu.events import StateChangeKind
from jouzetsu.models import ChatJSON, ChatSamplingOverrides
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
