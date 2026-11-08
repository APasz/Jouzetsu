"""Gate 2 lifecycle tests use controlled streams and real JSON persistence."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from inference_support import Harness, Plan, backend

from jouzetsu.backend_models import (
    GenerationProfile,
    ModelRef,
)
from jouzetsu.backend_protocol import (
    BackendError,
    BackendErrorCategory,
    BackendReachability,
)
from jouzetsu.config import AppConfig, ConfigStore
from jouzetsu.inference import (
    ChangeKind,
    ChatBusy,
    GenerationMode,
    JobPhase,
    RevisionConflict,
    RuntimeChange,
)
from jouzetsu.models import Chat, ChatJSON, ImageAttachment, ImageMediaType


def test_backend_queues_are_fifo_independent_and_lock_chats(tmp_path: Path) -> None:
    async def exercise() -> None:
        a, b = backend(), backend("Other")
        h = Harness(tmp_path, (a, b))
        chats = [await h.chat() for _ in range(3)]
        await h.runtime.select_backend(chats[2].id, chats[2].revision, b.id)
        plan = Plan(hold=True)
        h.adapters[a.id].plans.extend((plan, Plan(text="Second")))
        first = await h.runtime.generate(chats[0].id, chats[0].revision)
        assert first is not None
        await plan.produced.wait()
        queued = await h.runtime.generate(chats[1].id, chats[1].revision)
        other = await h.runtime.generate(chats[2].id, chats[2].revision)
        assert queued and other
        assert queued.phase is JobPhase.QUEUED
        with pytest.raises(ChatBusy):
            await h.runtime.edit_chat(
                chats[1].id,
                chats[1].revision,
                lambda chat: chat.set_manual_title("Rejected"),
            )
        await other.wait()
        assert not first.done.is_set()
        assert len(h.adapters[a.id].requests) == 1
        plan.release.set()
        await first.wait()
        await queued.wait()
        assert [chat.messages[-1].content for chat in chats] == [
            "Reply",
            "Second",
            "Reply",
        ]
        assert h.registry.active_chat_id == ""
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_per_backend_concurrency_and_one_job_per_chat(tmp_path: Path) -> None:
    async def exercise() -> None:
        config = backend(limit=2)
        h = Harness(tmp_path, (config,))
        chats = [await h.chat() for _ in range(3)]
        plans = [Plan(hold=True), Plan(hold=True), Plan()]
        h.adapters[config.id].plans.extend(plans)
        jobs = [await h.runtime.generate(chat.id, chat.revision) for chat in chats]
        await plans[0].produced.wait()
        await plans[1].produced.wait()
        assert len(h.adapters[config.id].requests) == 2
        with pytest.raises(ChatBusy):
            await h.runtime.generate(chats[0].id, chats[0].revision)
        for plan in plans:
            plan.release.set()
        for job in jobs:
            assert job
            await job.wait()
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("queued", [False, True])
def test_cancel_before_task_start_or_while_queued(tmp_path: Path, queued: bool) -> None:
    async def exercise() -> None:
        config = backend()
        h = Harness(tmp_path, (config,))
        if queued:
            blocker = await h.chat()
            hold = Plan(hold=True)
            h.adapters[config.id].plans.append(hold)
            _ = await h.runtime.generate(blocker.id, blocker.revision)
            await hold.produced.wait()
        chat = await h.chat()
        job = await h.runtime.generate(chat.id, chat.revision)
        assert job
        await h.runtime.cancel(chat.id)
        assert job.phase is JobPhase.CANCELLED
        assert len(chat.messages) == 1
        assert not h.registry.get(chat.id).is_generating
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "text,reasoning", [("", ""), ("", "private reasoning"), ("partial", "trace")]
)
def test_failure_preserves_only_actual_text_and_error_is_ephemeral(
    tmp_path: Path, text: str, reasoning: str
) -> None:
    async def exercise() -> None:
        config = backend()
        h = Harness(tmp_path, (config,))
        chat = await h.chat()
        h.adapters[config.id].plans.append(
            Plan(
                text,
                reasoning,
                BackendError(BackendErrorCategory.TIMEOUT, "Stream timed out."),
            )
        )
        job = await h.runtime.generate(chat.id, chat.revision)
        assert job
        await job.wait()
        assert job.phase is JobPhase.FAILED
        assert len(chat.messages) == (2 if text else 1)
        assert h.registry.get(chat.id).latest_error is job.error
        saved = h.reload(chat)
        assert saved.to_dict() == chat.to_dict()
        assert "Stream timed out" not in str(saved.to_dict())
        if text:
            assert saved.messages[-1].content == text
            assert saved.messages[-1].reasoning == reasoning
            attribution = saved.messages[-1].attributions[0]
            assert attribution.model == ModelRef(config.id, "model")
            assert attribution.backend_name == "Local"
            assert attribution.model_name == "Alias"
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("text", ["", "new partial"])
def test_regeneration_keeps_original_until_real_text_then_saves_replacement(
    tmp_path: Path, text: str
) -> None:
    async def exercise() -> None:
        config = backend()
        h = Harness(tmp_path, (config,))
        chat = await h.chat()
        original_job = await h.runtime.generate(chat.id, chat.revision)
        assert original_job
        await original_job.wait()
        original = chat.messages[-1]
        plan = Plan(text=text, reasoning="reasoning alone", hold=True)
        h.adapters[config.id].plans.append(plan)
        job = await h.runtime.generate(
            chat.id, chat.revision, GenerationMode.REGENERATE
        )
        assert job
        await plan.produced.wait()
        assert chat.messages[-1] is original
        assert original.content == "Reply"
        preview = h.registry.get(chat.id).live_response
        assert (preview is not None) == bool(text)
        assert h.reload(chat).messages[-1].content == "Reply"
        await h.runtime.cancel(chat.id)
        assert chat.messages[-1] is original
        assert original.content == (text or "Reply")
        assert len(chat.messages) == 2
        assert h.reload(chat).messages[-1].content == original.content
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_caller_cancellation_does_not_cancel_generation(tmp_path: Path) -> None:
    async def exercise() -> None:
        config = backend()
        h = Harness(tmp_path, (config,))
        chat = await h.chat()
        plan = Plan(hold=True)
        h.adapters[config.id].plans.append(plan)
        job = await h.runtime.generate(chat.id, chat.revision)
        assert job
        await plan.produced.wait()
        waiter = asyncio.create_task(job.wait())
        _ = waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert h.registry.get(chat.id).is_generating
        plan.release.set()
        await job.wait()
        assert chat.messages[-1].content == "Reply"
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_draft_updates_never_persist_streamed_tokens_and_do_not_change_request(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        config = backend()
        h = Harness(tmp_path, (config,))
        chat = await h.chat()
        plan = Plan(hold=True)
        h.adapters[config.id].plans.append(plan)
        job = await h.runtime.generate(chat.id, chat.revision)
        assert job
        await plan.produced.wait()
        _ = await h.runtime.update_draft(chat.id, 0, "Next prompt")
        with pytest.raises(RevisionConflict):
            await h.runtime.update_draft(chat.id, 0, "Stale")
        await h.persistence.flush_chats()
        disk = h.reload(chat)
        assert disk.draft == "Next prompt"
        assert len(disk.messages) == 1
        assert "Next prompt" not in str(job.request)
        await h.runtime.cancel(chat.id)
        assert h.reload(chat).draft == "Next prompt"
        assert h.reload(chat).messages[-1].content == "Reply"
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_save_failure_retains_output_for_save_retry_and_keeps_drafts_safe(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        config = backend()
        h = Harness(tmp_path, (config,))
        chat = await h.chat()
        with patch.object(
            h.persistence.storage, "save_records", side_effect=OSError("disk full")
        ):
            job = await h.runtime.generate(chat.id, chat.revision)
            assert job
            await job.wait()
        assert job.phase is JobPhase.SAVE_FAILED
        assert isinstance(job.save_error, OSError)
        assert len(chat.messages) == 1
        assert h.registry.get(chat.id).live_response is not None
        with pytest.raises(ChatBusy):
            await h.runtime.send_message(chat.id, chat.revision, "next", "Blocked")
        _ = await h.runtime.update_draft(chat.id, 0, "Still composing")
        await h.persistence.flush_chats()
        assert len(h.reload(chat).messages) == 1
        await h.runtime.retry_save(chat.id)
        assert h.reload(chat).messages[-1].content == "Reply"
        assert h.reload(chat).draft == "Still composing"
        assert len(h.adapters[config.id].requests) == 1
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_offline_send_persists_user_and_duplicate_survives_restart_and_edits(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        config = backend()
        h = Harness(tmp_path, (config,))
        h.adapters[config.id].online = False
        with pytest.raises(BackendError):
            await h.runtime.refresh_backend(config.id)
        chat = await h.runtime.create_chat()
        sent = await h.runtime.send_message(chat.id, 0, "receipt", "Saved offline")
        assert sent.job is None and not sent.duplicate
        assert h.registry.get(chat.id).latest_error is not None
        await h.runtime.edit_chat(
            chat.id, chat.revision, lambda candidate: candidate.messages.clear()
        )
        assert await h.runtime.shutdown()
        reloaded = h.reload(chat)
        h2 = Harness(tmp_path, (config,))
        _ = h2.registry.add(
            reloaded,
            runtime_status=h.registry.get(chat.id).runtime_status,
            persist=False,
            activate=False,
        )
        duplicate = await h2.runtime.send_message(
            chat.id, 0, "receipt", "Saved offline"
        )
        assert duplicate.message_id == sent.message_id
        assert duplicate.duplicate and duplicate.job is None
        assert not reloaded.messages
        with pytest.raises(ValueError, match="different content"):
            await h2.runtime.send_message(chat.id, 0, "receipt", "Changed")
        assert await h2.runtime.shutdown()

    asyncio.run(exercise())


def test_concurrent_revision_edits_reject_stale_write_and_preserve_identity(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        h = Harness(tmp_path, (backend(),))
        chat = await h.chat()
        original = chat.messages[0]
        revision = chat.revision
        results = await asyncio.gather(
            h.runtime.edit_chat(
                chat.id,
                revision,
                lambda candidate: candidate.set_manual_title("Winner"),
            ),
            h.runtime.edit_chat(
                chat.id, revision, lambda candidate: candidate.set_manual_title("Loser")
            ),
            return_exceptions=True,
        )
        assert results[0] is None and isinstance(results[1], RevisionConflict)
        assert chat.title == "Winner"
        assert chat.messages[0] is original
        assert h.reload(chat).title == "Winner"
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_failed_send_has_no_success_event_or_idempotency_receipt(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        config = backend()
        h = Harness(tmp_path, (config,))
        chat = await h.runtime.create_chat()
        h.changes.clear()
        with (
            patch.object(h.storage, "save_records", side_effect=OSError("full")),
            pytest.raises(OSError),
        ):
            await h.runtime.send_message(chat.id, 0, "retry", "Rejected")
        assert not h.changes and not chat.messages and not chat.submissions
        result = await h.runtime.send_message(
            chat.id, 0, "retry", "Different valid text", generate=False
        )
        assert not result.duplicate
        assert h.reload(chat).messages[0].content == "Different valid text"
        assert not h.adapters[config.id].requests
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_slow_commit_does_not_block_other_chat_and_cancel_still_publishes(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        h = Harness(tmp_path, (backend(),))
        a, b = await h.chat(), await h.chat()
        entered, release = threading.Event(), threading.Event()
        original = h.storage.save_records

        def slow(
            records: tuple[ChatJSON, ...],
            *,
            deleted_chat_ids: frozenset[str] = frozenset(),
        ) -> None:
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test worker not released")
            original(records, deleted_chat_ids=deleted_chat_ids)

        with patch.object(h.storage, "save_records", side_effect=slow):
            edit = asyncio.create_task(
                h.runtime.edit_chat(
                    a.id, a.revision, lambda chat: chat.set_manual_title("Durable")
                )
            )
            while not entered.is_set():
                await asyncio.sleep(0)
            _ = edit.cancel()
            _ = await h.runtime.update_draft(b.id, 0, "Unrelated async work")
            assert a.title != "Durable"
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await edit
        assert a.title == "Durable"
        assert h.reload(a).title == "Durable"
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_disruptive_backend_edits_blocked_but_rename_snapshots_attribution(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        config = backend()
        h = Harness(tmp_path, (config,))
        chat = await h.chat()
        plan = Plan(hold=True)
        h.adapters[config.id].plans.append(plan)
        job = await h.runtime.generate(chat.id, chat.revision)
        assert job
        await plan.produced.wait()
        with pytest.raises(ChatBusy):
            await h.runtime.update_backend(replace(config, enabled=False))
        with pytest.raises(ChatBusy):
            await h.runtime.remove_backend(config.id)
        await h.runtime.update_backend(replace(config, name="Renamed"))
        plan.release.set()
        await job.wait()
        assert chat.messages[-1].attributions[0].backend_name == "Local"
        assert (
            ConfigStore.for_home(tmp_path).load().backends.backend(config.id).name
            == "Renamed"
        )
        await h.runtime.set_default_backend(None)
        await h.runtime.remove_backend(config.id)
        assert chat.inference.selected_backend_id == config.id
        assert config.id in h.reload(chat).inference.profiles
        assert await h.runtime.generate(chat.id, chat.revision) is None
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_backend_config_failure_leaves_memory_and_adapter_unchanged(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        config = backend()
        h = Harness(tmp_path, (config,))
        adapter = h.adapters[config.id]
        with (
            patch.object(ConfigStore, "save", side_effect=OSError("full")),
            pytest.raises(OSError),
        ):
            await h.runtime.update_backend(replace(config, base_url="http://new/v1"))
        assert h.runtime.configuration.backend(config.id) == config
        assert not adapter.closed
        assert not any(change.kind is ChangeKind.CONFIG for change in h.changes)
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_unknown_vision_blocks_without_stripping_history(tmp_path: Path) -> None:
    async def exercise() -> None:
        config = backend()
        h = Harness(tmp_path, (config,))
        chat = await h.runtime.create_chat()
        image = ImageAttachment("a" * 64, "image.png", ImageMediaType.PNG, 1, 1, 1)
        sent = await h.runtime.send_message(
            chat.id, 0, "image", "Look", images=(image,)
        )
        assert sent.job is None
        assert chat.messages[0].images == (image,)
        assert not h.adapters[config.id].requests
        assert h.registry.get(chat.id).latest_error is not None
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_prompt_order_and_consecutive_roles_exclude_draft_reasoning_and_errors(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        config = backend()
        h = Harness(tmp_path, (config,))
        chat = await h.chat("First")
        _ = await h.runtime.send_message(
            chat.id, chat.revision, "second", "Second", generate=False
        )
        _ = await h.runtime.update_draft(chat.id, 0, "Not in prompt")
        job = await h.runtime.generate(chat.id, chat.revision)
        assert job
        await job.wait()
        user = [item.content for item in job.request.messages if item.role == "user"]
        assert user == ["First", "Second"]
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_native_loading_unload_protection_and_idle_scope(tmp_path: Path) -> None:
    async def exercise() -> None:
        config = backend(native=True)
        assert config.model_management
        config = replace(
            config,
            model_management=replace(config.model_management, idle_unload_minutes=1),
        )
        h = Harness(tmp_path, (config,))
        chat = await h.chat()
        plan = Plan(hold=True)
        adapter = h.adapters[config.id]
        adapter.plans.append(plan)
        job = await h.runtime.generate(chat.id, chat.revision)
        assert job
        await plan.produced.wait()
        with pytest.raises(ChatBusy):
            await h.runtime.unload_model(ModelRef(config.id, "model"))
        assert adapter.loads == ["model"]
        plan.release.set()
        await job.wait()
        assert chat.messages[-1].attributions[0].instance_id == "instance"
        await h.runtime.unload_idle_models(now=float("inf"))
        assert adapter.unloads == ["instance"]
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("automatic", [False, True])
def test_native_load_failures_create_no_assistant(
    tmp_path: Path, automatic: bool
) -> None:
    async def exercise() -> None:
        config = backend(native=True)
        assert config.model_management
        config = replace(
            config,
            model_management=replace(
                config.model_management, automatic_loading=automatic
            ),
        )
        h = Harness(tmp_path, (config,))
        adapter = h.adapters[config.id]
        adapter.load_error = BackendError(
            BackendErrorCategory.LOAD_FAILURE, "Load failed."
        )
        chat = await h.chat()
        job = await h.runtime.generate(chat.id, chat.revision)
        assert job
        await job.wait()
        assert job.error and job.error.category is BackendErrorCategory.LOAD_FAILURE
        assert len(chat.messages) == 1 and not adapter.requests
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_offline_check_drains_queue_without_automatic_retry(tmp_path: Path) -> None:
    async def exercise() -> None:
        config = backend()
        h = Harness(tmp_path, (config,))
        chats = [await h.chat() for _ in range(3)]
        plan = Plan(hold=True)
        adapter = h.adapters[config.id]
        adapter.plans.append(plan)
        jobs = [await h.runtime.generate(chat.id, chat.revision) for chat in chats]
        await plan.produced.wait()
        adapter.online = False
        with pytest.raises(BackendError):
            await h.runtime.refresh_backend(config.id)
        assert all(job and job.phase is JobPhase.FAILED for job in jobs[1:])
        assert len(adapter.requests) == 1
        adapter.online = True
        await h.runtime.refresh_backend(config.id)
        assert len(adapter.requests) == 1
        plan.release.set()
        assert jobs[0]
        await jobs[0].wait()
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_shutdown_cancels_running_and_queued_and_persists_partial(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        config = backend()
        h = Harness(tmp_path, (config,))
        a, b = await h.chat(), await h.chat()
        plan = Plan(hold=True)
        h.adapters[config.id].plans.append(plan)
        _ = await h.runtime.generate(a.id, a.revision)
        await plan.produced.wait()
        _ = await h.runtime.generate(b.id, b.revision)
        assert await h.runtime.shutdown()
        assert len(h.reload(a).messages) == 2
        assert len(h.reload(b).messages) == 1
        assert h.adapters[config.id].closed

    asyncio.run(exercise())


def test_start_checks_once_then_suspends_idle_polling(tmp_path: Path) -> None:
    async def exercise() -> None:
        config = backend()
        h = Harness(tmp_path, (config,))
        await h.runtime.start()
        await asyncio.sleep(0)
        assert h.adapters[config.id].discoveries == 1
        assert h.runtime.health(config.id).state is BackendReachability.STALE
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_draft_failure_notifies_and_flush_surfaces_original_error(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        h = Harness(tmp_path, (backend(),))
        chat = await h.chat()
        error = OSError("draft storage unavailable")
        with patch.object(h.storage, "save_records", side_effect=error):
            _ = await h.runtime.update_draft(chat.id, 0, "Unsaved draft")
            h.changes.clear()
            with pytest.raises(OSError) as caught:
                await h.persistence.flush_chats()
            assert caught.value is error
            assert h.persistence.chat_write_error is error
            assert RuntimeChange(ChangeKind.STATUS) in h.changes
        assert chat.draft == "Unsaved draft"
        assert h.reload(chat).draft == ""
        assert await h.runtime.shutdown()
        assert h.reload(chat).draft == "Unsaved draft"

    asyncio.run(exercise())


def test_no_backend_defaults_are_retroactive_and_profiles_remain_independent(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        h = Harness(tmp_path, ())
        old = await h.runtime.create_chat()
        config = backend()
        await h.runtime.update_backend(config)
        await h.runtime.set_default_backend(config.id)
        new = await h.runtime.create_chat()
        assert old.inference.selected_backend_id is None
        assert new.inference.selected_profile == config.defaults
        changed = replace(config, defaults=GenerationProfile("other"))
        await h.runtime.update_backend(changed)
        assert new.inference.selected_profile == config.defaults
        await h.runtime.reset_profile(new.id, new.revision)
        assert new.inference.selected_profile == changed.defaults
        assert h.reload(old).inference.selected_backend_id is None
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_continue_preserves_original_on_empty_failure_then_appends_actual_text(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        config = backend()
        h = Harness(tmp_path, (config,))
        chat = await h.chat()
        first = await h.runtime.generate(chat.id, chat.revision)
        assert first
        await first.wait()
        original = chat.messages[-1]
        h.adapters[config.id].plans.append(
            Plan(text="", error=BackendError(BackendErrorCategory.PROTOCOL, "Rejected"))
        )
        failed = await h.runtime.generate(
            chat.id, chat.revision, GenerationMode.CONTINUE
        )
        assert failed
        await failed.wait()
        assert original.content == "Reply"
        h.adapters[config.id].plans.append(Plan(text=" continued"))
        continued = await h.runtime.generate(
            chat.id, chat.revision, GenerationMode.CONTINUE
        )
        assert continued
        await continued.wait()
        assert (
            chat.messages[-1] is original and original.content == "Reply\n\n continued"
        )
        assert len(chat.messages) == 2
        assert h.reload(chat).messages[-1].content == "Reply\n\n continued"
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("operation", ["create", "fork", "delete"])
def test_lifecycle_storage_rejection_has_no_publication(
    tmp_path: Path, operation: str
) -> None:
    async def exercise() -> None:
        h = Harness(tmp_path, (backend(),))
        chat = await h.chat()
        before = chat.to_dict()
        h.changes.clear()
        with (
            patch.object(h.storage, "save_records", side_effect=OSError("full")),
            pytest.raises(OSError),
        ):
            if operation == "create":
                _ = await h.runtime.create_chat()
            elif operation == "fork":
                _ = await h.runtime.fork_chat(
                    chat.id, chat.revision, chat.messages[0].id
                )
            else:
                await h.runtime.delete_chat(chat.id, chat.revision)
        assert len(h.registry) == 1 and not h.changes
        assert chat.to_dict() == before == h.reload(chat).to_dict()
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_backend_commit_cancellation_finishes_adapter_publication(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        config = backend()
        h = Harness(tmp_path, (config,))
        previous = h.adapters[config.id]
        entered, release = threading.Event(), threading.Event()
        original = ConfigStore.save

        def slow(store: ConfigStore, candidate: AppConfig) -> None:
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test worker not released")
            original(store, candidate)

        changed = replace(config, base_url="http://changed/v1")
        with patch.object(ConfigStore, "save", new=slow):
            update = asyncio.create_task(h.runtime.update_backend(changed))
            while not entered.is_set():
                await asyncio.sleep(0)
            _ = update.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await update
        assert h.runtime.configuration.backend(config.id) == changed
        assert h.adapters[config.id] is not previous and previous.closed
        assert h.store.load().backends.backend(config.id) == changed
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_manual_loading_with_auto_disabled_and_generic_management_rejection(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        native, generic = backend(native=True), backend("Generic")
        assert native.model_management
        native = replace(
            native,
            model_management=replace(native.model_management, automatic_loading=False),
        )
        h = Harness(tmp_path, (native, generic))
        ready = await h.runtime.load_model(ModelRef(native.id, "model"))
        assert ready.instance_id == "instance"
        chat = await h.chat()
        job = await h.runtime.generate(chat.id, chat.revision)
        assert job
        await job.wait()
        assert job.phase is JobPhase.FINISHED
        with pytest.raises(ValueError, match="does not support"):
            _ = await h.runtime.load_model(ModelRef(generic.id, "model"))
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_queued_request_retains_accepted_parameters_and_labels(tmp_path: Path) -> None:
    async def exercise() -> None:
        config = backend()
        h = Harness(tmp_path, (config,))
        a, b = await h.chat(), await h.chat()
        plan = Plan(hold=True)
        h.adapters[config.id].plans.append(plan)
        first = await h.runtime.generate(a.id, a.revision)
        assert first
        await plan.produced.wait()
        queued = await h.runtime.generate(b.id, b.revision)
        assert queued
        accepted = queued.request
        await h.runtime.update_backend(
            replace(config, name="New name", defaults=GenerationProfile("new-default"))
        )
        with pytest.raises(ChatBusy):
            await h.runtime.set_profile(b.id, b.revision, GenerationProfile("other"))
        plan.release.set()
        await first.wait()
        await queued.wait()
        assert queued.request is accepted
        assert h.adapters[config.id].requests[-1] is accepted
        assert b.messages[-1].attributions[0].backend_name == "Local"
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", True),
        ("draft_revision", -1),
        ("submissions", {}),
        ("submissions", [{"id": "a", "message_id": "b", "fingerprint": "bad"}]),
    ],
)
def test_invalid_persisted_runtime_fields_fail_loudly(
    field: str, value: object
) -> None:
    document: dict[str, object] = dict(Chat().to_dict())
    document[field] = value
    with pytest.raises((ValueError, TypeError)):
        _ = Chat.from_dict(document)


def test_config_notifications_observe_complete_backend_publication(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        h = Harness(tmp_path, ())
        observed: list[bool] = []

        async def notify(change: RuntimeChange) -> None:
            if change.kind is ChangeKind.CONFIG:
                observed.append(
                    all(
                        h.runtime.health(item.id)
                        for item in h.runtime.configuration.backends
                    )
                )

        h.runtime._notify_callback = notify  # pyright: ignore[reportPrivateUsage]
        config = backend()
        await h.runtime.update_backend(config)
        assert observed == [True]
        assert await h.runtime.shutdown()

    asyncio.run(exercise())
