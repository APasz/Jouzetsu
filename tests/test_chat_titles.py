"""Model-generated titles share scheduling and persistence with chat inference."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import pytest
from inference_support import Harness, Plan, backend

from jouzetsu.backend_models import ModelRef
from jouzetsu.backend_protocol import BackendError, BackendErrorCategory, GenerationRequest
from jouzetsu.inference import ChangeKind, GenerationMode, JobPhase
from jouzetsu.models import Chat, ChatTitleSource, Message
from jouzetsu.runtime import ChatStreamEvent, GenerationMetrics, PredictionComplete, PredictionFragment


@pytest.mark.parametrize("native", [False, True])
def test_first_completed_reply_gets_a_title_without_transcript_output(tmp_path: Path, native: bool) -> None:
    async def exercise() -> None:
        config = backend(native=native)
        h = Harness(tmp_path, (config,))
        chat = await h.runtime.create_chat()
        reply = Plan(text="An explanation of the northern lights.", hold=True)
        title = Plan(text='"Northern Lights Explained"', reasoning="Private title reasoning", hold=True)
        adapter = h.adapters[config.id]
        adapter.plans.extend((reply, title))
        sent = await h.runtime.send_message(chat.id, chat.revision, "first", "What causes auroras?")
        assert sent.job
        await reply.produced.wait()
        assert chat.title == "New chat" and len(adapter.requests) == 1
        reply.release.set()
        await sent.job.wait()
        await title.produced.wait()
        naming = h.runtime.job(chat.id)
        assert naming and naming.mode is GenerationMode.AUTO_TITLE
        assert naming.request.model == sent.job.request.model
        assert [message.content for message in naming.request.messages[1:]] == [reply.text]
        assert h.registry.get(chat.id).live_response is None
        assert not h.registry.get(chat.id).live_reasoning
        assert h.reload(chat).messages[-1].content == reply.text
        transcript = [message.to_dict() for message in chat.messages]
        title.release.set()
        await naming.wait()

        assert chat.title == "Northern Lights Explained"
        assert chat.title_source is ChatTitleSource.GENERATED
        assert h.reload(chat).title == chat.title
        assert [message.to_dict() for message in chat.messages] == transcript
        later = await h.runtime.send_message(chat.id, chat.revision, "second", "Tell me more")
        assert later.job
        await later.job.wait()
        assert len(adapter.requests) == 3
        assert chat.title == "Northern Lights Explained"
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("manual_title", ["New chat", "Mira & Ren"])
def test_first_reply_preserves_user_and_character_titles(tmp_path: Path, manual_title: str) -> None:
    async def exercise() -> None:
        config = backend()
        h = Harness(tmp_path, (config,))
        template = Chat()
        template.set_manual_title(manual_title)
        chat = await h.runtime.create_chat(template)
        sent = await h.runtime.send_message(chat.id, chat.revision, "first", "Hello")
        assert sent.job
        await sent.job.wait()
        assert chat.title == manual_title
        assert len(h.adapters[config.id].requests) == 1
        assert not h.runtime.job(chat.id)
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("outcome", ["cancelled", "failed", "empty"])
def test_incomplete_first_reply_does_not_request_a_title(
    tmp_path: Path, outcome: Literal["cancelled", "failed", "empty"]
) -> None:
    async def exercise() -> None:
        config = backend()
        h = Harness(tmp_path, (config,))
        chat = await h.runtime.create_chat()
        plan = Plan(
            text="" if outcome == "empty" else "Partial reply",
            hold=outcome == "cancelled",
            error=BackendError(BackendErrorCategory.TIMEOUT, "Interrupted") if outcome == "failed" else None,
        )
        h.adapters[config.id].plans.append(plan)
        sent = await h.runtime.send_message(chat.id, chat.revision, "first", "Hello")
        assert sent.job
        await plan.produced.wait()
        if outcome == "cancelled":
            await h.runtime.cancel(chat.id)
        await sent.job.wait()
        assert chat.title == "New chat"
        assert len(h.adapters[config.id].requests) == 1
        assert not h.runtime.job(chat.id)
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_regenerated_title_uses_selected_model_and_all_visible_messages(tmp_path: Path) -> None:
    async def exercise() -> None:
        original, selected = backend("Original"), backend("Selected")
        h = Harness(tmp_path, (original, selected))
        chat = await h.runtime.create_chat(
            Chat(
                title="My title",
                title_source=ChatTitleSource.MANUAL,
                system_prompt="Continue roleplaying as a pirate.",
                messages=[
                    Message(role="system", content="Hidden instructions"),
                    Message(role="user", content="First question"),
                    Message(role="assistant", content="First reply", reasoning="Hidden reasoning"),
                    Message(role="user", content="Later topic"),
                    Message(role="assistant", content="Latest reply"),
                ],
                draft="Unsent draft",
            )
        )
        await h.runtime.select_backend(chat.id, chat.revision, selected.id)
        transcript = [message.to_dict() for message in chat.messages]
        h.adapters[selected.id].plans.append(Plan(text="A New Conversation Title"))
        job = await h.runtime.generate(chat.id, chat.revision, GenerationMode.TITLE)
        assert job
        await job.wait()
        assert job.request.model == ModelRef(selected.id, "model")
        assert [message.content for message in job.request.messages[1:-1]] == [
            message.content for message in chat.messages if message.role != "system"
        ]
        assert "pirate" not in job.request.messages[0].content
        assert chat.title == "A New Conversation Title"
        assert chat.title_source is ChatTitleSource.GENERATED
        assert [message.to_dict() for message in chat.messages] == transcript
        assert chat.draft == "Unsent draft"
        assert not h.adapters[original.id].requests
        assert h.reload(chat).title == chat.title
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("outcome", ["cancelled", "failed", "empty", "invalid"])
def test_failed_title_retains_existing_name_and_unlocks_chat(
    tmp_path: Path, outcome: Literal["cancelled", "failed", "empty", "invalid"]
) -> None:
    async def exercise() -> None:
        config = backend()
        h = Harness(tmp_path, (config,))
        chat = await h.chat()
        saved = chat.to_dict()
        plan = Plan(
            text="" if outcome == "empty" else '""' if outcome == "invalid" else "Partial title",
            hold=outcome == "cancelled",
            error=BackendError(BackendErrorCategory.TIMEOUT, "Interrupted") if outcome == "failed" else None,
        )
        h.adapters[config.id].plans.append(plan)
        job = await h.runtime.generate(chat.id, chat.revision, GenerationMode.TITLE)
        assert job
        await plan.produced.wait()
        assert h.registry.get(chat.id).live_response is None
        if outcome == "cancelled":
            await h.runtime.cancel(chat.id)
        await job.wait()
        assert chat.to_dict() == saved == h.reload(chat).to_dict()
        assert not h.registry.get(chat.id).is_generating
        assert not h.runtime.job(chat.id)
        if outcome != "cancelled":
            assert job.error
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_title_generation_waits_for_reply_save_recovery(tmp_path: Path) -> None:
    async def exercise() -> None:
        config = backend()
        h = Harness(tmp_path, (config,))
        chat = await h.runtime.create_chat()
        reply, title = Plan(hold=True), Plan(text="Recovered Response Title", hold=True)
        h.adapters[config.id].plans.extend((reply, title))
        sent = await h.runtime.send_message(chat.id, chat.revision, "first", "Hello")
        assert sent.job
        await reply.produced.wait()
        with patch.object(type(h.persistence), "save_chat_candidate", side_effect=OSError("disk full")):
            reply.release.set()
            await sent.job.wait()
        assert sent.job.phase is JobPhase.SAVE_FAILED
        assert len(h.adapters[config.id].requests) == 1
        await h.runtime.retry_save(chat.id)
        await title.produced.wait()
        naming = h.runtime.job(chat.id)
        assert naming and naming.mode is GenerationMode.AUTO_TITLE
        title.release.set()
        await naming.wait()
        assert h.reload(chat).title == "Recovered Response Title"
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_automatic_title_respects_backend_queue_order(tmp_path: Path) -> None:
    async def exercise() -> None:
        config = backend()
        h = Harness(tmp_path, (config,))
        first, second = await h.runtime.create_chat(), await h.chat()
        reply, queued_reply = Plan(hold=True), Plan(text="Second chat reply", hold=True)
        adapter = h.adapters[config.id]
        adapter.plans.extend((reply, queued_reply, Plan(text="First Chat Title")))
        sent = await h.runtime.send_message(first.id, first.revision, "first", "Hello")
        assert sent.job
        await reply.produced.wait()
        queued = await h.runtime.generate(second.id, second.revision)
        assert queued
        reply.release.set()
        await sent.job.wait()
        await queued_reply.produced.wait()
        naming = h.runtime.job(first.id)
        assert naming and naming.mode is GenerationMode.AUTO_TITLE
        assert naming.phase is JobPhase.QUEUED
        assert len(adapter.requests) == 2
        queued_reply.release.set()
        await queued.wait()
        await naming.wait()
        assert adapter.requests == [sent.job.request, queued.request, naming.request]
        assert first.title == "First Chat Title"
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_title_save_failure_can_be_retried_without_transcript_changes(tmp_path: Path) -> None:
    async def exercise() -> None:
        config = backend()
        h = Harness(tmp_path, (config,))
        chat = await h.chat()
        saved = chat.to_dict()
        plan = Plan(text="Recovered Title", hold=True)
        h.adapters[config.id].plans.append(plan)
        job = await h.runtime.generate(chat.id, chat.revision, GenerationMode.TITLE)
        assert job
        await plan.produced.wait()
        with patch.object(type(h.persistence), "save_chat_candidate", side_effect=OSError("disk full")):
            plan.release.set()
            await job.wait()
        assert job.phase is JobPhase.SAVE_FAILED
        assert chat.to_dict() == saved
        assert h.registry.get(chat.id).live_response is None
        await h.runtime.retry_save(chat.id)
        assert chat.title == h.reload(chat).title == "Recovered Title"
        assert [message.to_dict() for message in chat.messages] == saved["messages"]
        assert not h.runtime.job(chat.id)
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_first_successful_reply_after_a_cancelled_reply_still_gets_a_title(tmp_path: Path) -> None:
    async def exercise() -> None:
        config = backend()
        h = Harness(tmp_path, (config,))
        chat = await h.runtime.create_chat()
        partial = Plan(text="Incomplete response", hold=True)
        title = Plan(text="The Completed Answer", hold=True)
        h.adapters[config.id].plans.extend((partial, Plan(text="A completed response"), title))
        first = await h.runtime.send_message(chat.id, chat.revision, "first", "Hello")
        assert first.job
        await partial.produced.wait()
        await h.runtime.cancel(chat.id)
        second = await h.runtime.send_message(chat.id, chat.revision, "second", "Try again")
        assert second.job
        await second.job.wait()
        naming = h.runtime.job(chat.id)
        assert naming and naming.mode is GenerationMode.AUTO_TITLE
        await title.produced.wait()
        assert naming.request.messages[-1].content == "A completed response"
        title.release.set()
        await naming.wait()
        assert chat.title == "The Completed Answer"
        assert await h.runtime.shutdown()

    asyncio.run(exercise())


def test_title_tokens_only_notify_when_the_generation_phase_changes(tmp_path: Path) -> None:
    async def exercise() -> None:
        config = backend()
        h = Harness(tmp_path, (config,))
        chat = await h.chat()

        async def stream(_request: GenerationRequest) -> AsyncGenerator[ChatStreamEvent]:
            for character in "Long private reasoning that should not refresh the transcript for every token.":
                yield PredictionFragment(character, None, True)
            for character in "Concise Chat Title":
                yield PredictionFragment(character, None, False)
            yield PredictionComplete(GenerationMetrics())

        h.changes.clear()
        with patch.object(h.adapters[config.id], "stream", stream):
            job = await h.runtime.generate(chat.id, chat.revision, GenerationMode.TITLE)
            assert job
            await job.wait()
        assert chat.title == "Concise Chat Title"
        assert sum(change.kind is ChangeKind.STATUS for change in h.changes) < 10
        assert await h.runtime.shutdown()

    asyncio.run(exercise())
