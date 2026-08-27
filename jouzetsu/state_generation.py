"""Streaming-generation orchestration independent of the application façade."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import replace
from logging import Logger

from .config import CONTINUE_PROMPT_CONTENT, GenerationSettings, SpellingReplacement
from .continuity import (
    ContinuityReviewDecision,
    build_continuity_review_chat,
    parse_continuity_review,
)
from .events import StateChangeKind
from .lmstudio import LMStudioClientProtocol, LMStudioError
from .logging_config import chat_logger
from .models import Chat, Message
from .runtime import (
    FirstToken,
    GenerationMetrics,
    ModelDescriptor,
    ModelLoadProgress,
    ModelReady,
    PredictionComplete,
    PredictionFragment,
    PromptProcessingProgress,
    RuntimePhase,
    RuntimeStatus,
)
from .state_types import ChatSession
from .text_processing import apply_case_preserving_word_replacements

log: Logger = logging.getLogger(__name__)
chat_log: Logger = chat_logger()
_STREAM_UPDATE_INTERVAL_SECONDS: float = 0.01
_CANCELLATION_GRACE_SECONDS: float = 2.0
_FORCED_CANCELLATION_GRACE_SECONDS: float = 0.5

CommitChat = Callable[[Chat], Awaitable[None]]
DefaultGeneration = Callable[[], GenerationSettings]
DefaultModel = Callable[[], str]
IdleStatus = Callable[[Chat], RuntimeStatus]
ModelDescriptorFor = Callable[[str], ModelDescriptor | None]
ModelDisplayName = Callable[[str, str], str]
Notify = Callable[[StateChangeKind], Awaitable[None]]


def chat_with_continuation_prompt(chat: Chat) -> Chat:
    """Build a transient request that asks the model for one more turn."""

    return replace(
        chat,
        messages=[
            *chat.messages,
            Message(role="user", content=CONTINUE_PROMPT_CONTENT),
        ],
    )


def generation_request_snapshot(chat: Chat) -> Chat:
    """Copy a transcript before its generated assistant message mutates in place."""

    return replace(
        chat,
        messages=[
            Message(
                id=message.id,
                role=message.role,
                content=message.content,
                model=message.model,
                continuity_rewrite=message.continuity_rewrite,
                reasoning=message.reasoning,
                created_at=message.created_at,
                updated_at=message.updated_at,
            )
            for message in chat.messages
        ],
    )


class GenerationController:
    """Own streaming tasks, status transitions, cancellation, and review passes."""

    def __init__(
        self,
        *,
        client: LMStudioClientProtocol,
        default_generation: DefaultGeneration,
        default_model: DefaultModel,
        model_descriptor: ModelDescriptorFor,
        model_display_name: ModelDisplayName,
        idle_status: IdleStatus,
        commit_chat: CommitChat,
        notify: Notify,
    ) -> None:
        self._client: LMStudioClientProtocol = client
        self._default_generation: DefaultGeneration = default_generation
        self._default_model: DefaultModel = default_model
        self._model_descriptor: ModelDescriptorFor = model_descriptor
        self._model_display_name: ModelDisplayName = model_display_name
        self._idle_status: IdleStatus = idle_status
        self._commit_chat: CommitChat = commit_chat
        self._notify: Notify = notify

    async def start(
        self,
        session: ChatSession,
        *,
        assistant: Message | None = None,
        request_chat: Chat | None = None,
    ) -> None:
        """Persist and launch one response stream for an idle chat session."""

        if session.is_generating:
            return
        chat: Chat = session.chat
        generation: GenerationSettings = chat.sampling_overrides.resolve(
            self._default_generation()
        )
        model: str = chat.model or self._default_model()
        request: Chat = generation_request_snapshot(request_chat or chat)
        initial_content: str
        is_continuation: bool = assistant is not None

        if assistant is None:
            assistant = chat.add_message("assistant", "")
        else:
            chat.touch()
        initial_content = assistant.content
        session.is_generating = True
        session.cancellation_requested = asyncio.Event()
        session.live_reasoning = ""
        session.runtime_status = self._starting_status(model)
        british_spelling_replacements: tuple[SpellingReplacement, ...] = (
            tuple(self._default_generation().british_spelling_replacements)
            if chat.postprocess_british_spellings
            else ()
        )
        session.generation_task = asyncio.create_task(
            self._stream(
                session,
                assistant,
                request=request,
                model=model,
                generation=generation,
                initial_content=initial_content,
                is_continuation=is_continuation,
                british_spelling_replacements=british_spelling_replacements,
            )
        )
        await self._commit_chat(chat)
        chat_log.info(
            "generation_started chat_id=%s message_id=%s model=%s continuation=%s",
            chat.id,
            assistant.id,
            model,
            is_continuation,
        )

    async def cancel(self, session: ChatSession) -> bool:
        """Request cancellation and, if needed, force-stop a stalled stream task."""

        task: asyncio.Task[None] | None = session.generation_task
        if task is None:
            if not session.is_generating:
                return True
            log.error("generation is active without a task chat_id=%s", session.chat.id)
            await self._recover_orphaned_generation(session)
            return False

        if task.done():
            session.generation_task = None
            if session.is_generating:
                log.warning(
                    "generation task completed before final cleanup chat_id=%s",
                    session.chat.id,
                )
                await self._recover_orphaned_generation(session)
            return True

        session.cancellation_requested.set()
        forced_cancellation: bool = False
        try:
            await asyncio.wait_for(task, timeout=_CANCELLATION_GRACE_SECONDS)
        except TimeoutError:
            forced_cancellation = True
            log.warning(
                "generation task did not stop cleanly chat_id=%s", session.chat.id
            )
            _ = task.cancel()
            try:
                await asyncio.wait_for(task, timeout=_FORCED_CANCELLATION_GRACE_SECONDS)
            except asyncio.CancelledError:
                pass
            except TimeoutError:
                log.error(
                    "generation task could not be cancelled chat_id=%s", session.chat.id
                )
                return False
        session.generation_task = None
        if session.runtime_status.phase is RuntimePhase.STOPPING:
            session.runtime_status = self._idle_status(session.chat)
            await self._notify(StateChangeKind.STATUS)
        return not forced_cancellation

    async def cancel_all(self, sessions: Iterable[ChatSession]) -> bool:
        """Stop all active streams concurrently and report whether every stop was clean."""

        cancellations: tuple[asyncio.Task[bool], ...] = tuple(
            asyncio.create_task(self.cancel(session))
            for session in sessions
            if session.is_generating or session.generation_task is not None
        )
        if not cancellations:
            return True
        results: tuple[bool | BaseException, ...] = tuple(
            await asyncio.gather(*cancellations, return_exceptions=True)
        )
        for result in results:
            if isinstance(result, BaseException):
                log.error(
                    "generation cancellation crashed during shutdown error=%r", result
                )
                return False
            if not result:
                return False
        return True

    async def _recover_orphaned_generation(self, session: ChatSession) -> None:
        """Restore a usable idle session when a task ended before its finalizer ran."""

        session.is_generating = False
        session.live_reasoning = ""
        if session.runtime_status.phase is not RuntimePhase.ERROR:
            session.runtime_status = self._idle_status(session.chat)
        await self._notify(StateChangeKind.STATUS)

    def request_stop(self, session: ChatSession) -> bool:
        """Publish the immediate stopping phase before awaiting task cancellation."""

        if not session.is_generating:
            return False
        current: RuntimeStatus = session.runtime_status
        session.runtime_status = self.next_runtime_status(
            current,
            RuntimePhase.STOPPING,
            model_key=current.model_key,
            model_name=current.model_name,
            output_tokens=current.output_tokens,
        )
        return True

    def _starting_status(self, model: str) -> RuntimeStatus:
        now: float = asyncio.get_running_loop().time()
        descriptor: ModelDescriptor | None = self._model_descriptor(model)
        phase: RuntimePhase = (
            RuntimePhase.LOADING_MODEL
            if descriptor is not None and not descriptor.is_loaded
            else RuntimePhase.STARTING
        )
        return RuntimeStatus(
            phase,
            model_key=model,
            model_name=self._model_name(model),
            started_at=now,
            phase_started_at=now,
        )

    async def _stream(
        self,
        session: ChatSession,
        assistant: Message,
        *,
        request: Chat,
        model: str,
        generation: GenerationSettings,
        initial_content: str,
        is_continuation: bool,
        british_spelling_replacements: tuple[SpellingReplacement, ...],
    ) -> None:
        chat: Chat = session.chat
        buffer: list[str] = self._initial_buffer(
            initial_content, is_continuation=is_continuation
        )
        output_tokens: int = 0
        last_stream_update_at: float = 0.0
        completed_metrics: GenerationMetrics | None = None
        try:
            async for event in self._client.stream_chat(
                chat=request, model=model, generation=generation
            ):
                if session.cancellation_requested.is_set():
                    break
                if isinstance(event, ModelLoadProgress):
                    session.runtime_status = self._status_for_load_progress(
                        session.runtime_status, event
                    )
                    await self._notify(StateChangeKind.STATUS)
                    continue
                if isinstance(event, ModelReady):
                    self._apply_model_ready(session, assistant, event)
                    await self._notify(StateChangeKind.STATUS)
                    continue
                if isinstance(event, PromptProcessingProgress):
                    if session.runtime_status.phase is RuntimePhase.LOADING_MODEL:
                        continue
                    session.runtime_status = self.next_runtime_status(
                        session.runtime_status,
                        RuntimePhase.PROCESSING_PROMPT,
                        model_key=session.runtime_status.model_key,
                        model_name=session.runtime_status.model_name,
                        progress=event.progress,
                    )
                    await self._notify(StateChangeKind.STATUS)
                    continue
                if isinstance(event, FirstToken):
                    session.runtime_status = self.next_runtime_status(
                        session.runtime_status,
                        RuntimePhase.GENERATING,
                        model_key=session.runtime_status.model_key,
                        model_name=session.runtime_status.model_name,
                        output_tokens=output_tokens,
                    )
                    await self._notify(StateChangeKind.STATUS)
                    continue
                if isinstance(event, PredictionComplete):
                    completed_metrics = event.metrics
                    continue

                output_tokens += event.token_count or 0
                if event.is_reasoning:
                    session.live_reasoning += event.content
                else:
                    buffer.append(event.content)
                    assistant.update_content(
                        apply_case_preserving_word_replacements(
                            "".join(buffer), british_spelling_replacements
                        )
                    )
                session.runtime_status = self.next_runtime_status(
                    session.runtime_status,
                    RuntimePhase.REASONING
                    if event.is_reasoning
                    else RuntimePhase.GENERATING,
                    model_key=session.runtime_status.model_key,
                    model_name=session.runtime_status.model_name,
                    output_tokens=output_tokens,
                )
                now: float = asyncio.get_running_loop().time()
                if now - last_stream_update_at >= _STREAM_UPDATE_INTERVAL_SECONDS:
                    await self._notify(StateChangeKind.STREAM)
                    last_stream_update_at = now
            await self._finish_stream(
                session,
                assistant,
                model=model,
                generation=generation,
                initial_content=initial_content,
                british_spelling_replacements=british_spelling_replacements,
                output_tokens=output_tokens,
                completed_metrics=completed_metrics,
            )
        except LMStudioError as exc:
            self._record_error(session, assistant, initial_content, exc)
        except Exception as exc:  # noqa: BLE001
            self._record_unexpected_error(session, assistant, initial_content, exc)
        finally:
            saved_reasoning: str = session.live_reasoning if chat.save_reasoning else ""
            if assistant.reasoning != saved_reasoning:
                assistant.set_reasoning(saved_reasoning)
                chat.touch()
            session.is_generating = False
            session.live_reasoning = ""
            await self._commit_chat(chat)
            if session.generation_task is asyncio.current_task():
                session.generation_task = None

    async def _finish_stream(
        self,
        session: ChatSession,
        assistant: Message,
        *,
        model: str,
        generation: GenerationSettings,
        initial_content: str,
        british_spelling_replacements: tuple[SpellingReplacement, ...],
        output_tokens: int,
        completed_metrics: GenerationMetrics | None,
    ) -> None:
        chat: Chat = session.chat
        has_response: bool = bool(assistant.content.strip())
        if not has_response and not session.cancellation_requested.is_set():
            assistant.update_content("(no response)")
        if session.cancellation_requested.is_set():
            session.runtime_status = self._idle_status(chat)
            chat_log.info(
                "generation_cancelled chat_id=%s message_id=%s output_tokens=%d",
                chat.id,
                assistant.id,
                output_tokens,
            )
            return
        if generation.continuity_review and has_response:
            await self._review_continuity(
                session,
                assistant,
                model=model,
                generation=generation,
                british_spelling_replacements=british_spelling_replacements,
                output_tokens=output_tokens,
            )
        final_output_tokens: int = (
            completed_metrics.output_tokens
            if completed_metrics is not None
            and completed_metrics.output_tokens is not None
            else output_tokens
        )
        session.runtime_status = RuntimeStatus(
            RuntimePhase.READY,
            model_key=session.runtime_status.model_key,
            model_name=session.runtime_status.model_name,
            output_tokens=final_output_tokens,
            metrics=completed_metrics,
        )
        chat_log.info(
            "generation_completed chat_id=%s message_id=%s output_tokens=%d content_length=%d",
            chat.id,
            assistant.id,
            final_output_tokens,
            len(assistant.content),
        )

    def _status_for_load_progress(
        self, current: RuntimeStatus, event: ModelLoadProgress
    ) -> RuntimeStatus:
        return self.next_runtime_status(
            current,
            RuntimePhase.LOADING_MODEL,
            model_key=current.model_key,
            model_name=current.model_name,
            progress=event.progress,
        )

    def _apply_model_ready(
        self, session: ChatSession, assistant: Message, event: ModelReady
    ) -> None:
        chat: Chat = session.chat
        if assistant.model != event.model_key:
            assistant.set_model(event.model_key)
            chat.touch()
        phase: RuntimePhase = (
            RuntimePhase.LOADING_MODEL
            if session.runtime_status.phase is RuntimePhase.LOADING_MODEL
            else RuntimePhase.STARTING
        )
        session.runtime_status = self.next_runtime_status(
            session.runtime_status,
            phase,
            model_key=event.model_key,
            model_name=self._model_display_name(event.model_key, event.model_name),
            detail=f"{event.context_length:,} token context",
        )

    async def _review_continuity(
        self,
        session: ChatSession,
        assistant: Message,
        *,
        model: str,
        generation: GenerationSettings,
        british_spelling_replacements: tuple[SpellingReplacement, ...],
        output_tokens: int,
    ) -> None:
        """Use a constrained second pass to propose only explicitly flawed rewrites."""

        chat: Chat = session.chat
        current: RuntimeStatus = session.runtime_status
        session.runtime_status = self.next_runtime_status(
            current,
            RuntimePhase.REVIEWING,
            model_key=current.model_key,
            model_name=current.model_name,
            output_tokens=output_tokens,
        )
        await self._notify(StateChangeKind.STATUS)

        fragments: list[str] = []
        review_generation: GenerationSettings = replace(generation, temperature=0.0)
        try:
            async for event in self._client.stream_chat(
                chat=build_continuity_review_chat(chat),
                model=model,
                generation=review_generation,
            ):
                if session.cancellation_requested.is_set():
                    chat_log.info(
                        "continuity_review_cancelled chat_id=%s message_id=%s",
                        chat.id,
                        assistant.id,
                    )
                    return
                if isinstance(event, PredictionFragment) and not event.is_reasoning:
                    fragments.append(event.content)
        except LMStudioError as exc:
            log.warning("continuity review failed chat_id=%s error=%s", chat.id, exc)
            chat_log.warning(
                "continuity_review_failed chat_id=%s message_id=%s",
                chat.id,
                assistant.id,
            )
            return
        except Exception:
            log.exception("continuity review crashed chat_id=%s", chat.id)
            chat_log.error(
                "continuity_review_crashed chat_id=%s message_id=%s",
                chat.id,
                assistant.id,
            )
            return

        review = parse_continuity_review("".join(fragments))
        if review.decision is ContinuityReviewDecision.REPLACE:
            rewrite: str = apply_case_preserving_word_replacements(
                review.replacement,
                british_spelling_replacements,
            )
            if rewrite != assistant.content:
                assistant.set_continuity_rewrite(rewrite)
                chat.touch()
                await self._notify(StateChangeKind.STREAM)
        chat_log.info(
            "continuity_review_completed chat_id=%s message_id=%s decision=%s",
            chat.id,
            assistant.id,
            review.decision.value,
        )

    def _record_error(
        self,
        session: ChatSession,
        assistant: Message,
        initial_content: str,
        error: LMStudioError,
    ) -> None:
        error_text: str = f"⚠ {error}"
        assistant.update_content(
            f"{initial_content}\n\n{error_text}" if initial_content else error_text
        )
        session.runtime_status = RuntimeStatus(
            RuntimePhase.ERROR,
            model_key=session.runtime_status.model_key,
            model_name=session.runtime_status.model_name,
            detail=str(error),
        )
        log.error("generation error: %s", error)
        chat_log.error(
            "generation_error chat_id=%s message_id=%s error=%r",
            session.chat.id,
            assistant.id,
            str(error),
        )

    def _record_unexpected_error(
        self,
        session: ChatSession,
        assistant: Message,
        initial_content: str,
        error: Exception,
    ) -> None:
        error_text: str = f"⚠ Unexpected error: {error}"
        assistant.update_content(
            f"{initial_content}\n\n{error_text}" if initial_content else error_text
        )
        session.runtime_status = RuntimeStatus(
            RuntimePhase.ERROR,
            model_key=session.runtime_status.model_key,
            model_name=session.runtime_status.model_name,
            detail=f"Unexpected error: {error}",
        )
        log.exception("generation crashed")
        chat_log.error(
            "generation_crashed chat_id=%s message_id=%s error=%r",
            session.chat.id,
            assistant.id,
            str(error),
        )

    def _model_name(self, model_key: str) -> str:
        descriptor: ModelDescriptor | None = self._model_descriptor(model_key)
        fallback: str = descriptor.display_name if descriptor is not None else model_key
        return self._model_display_name(model_key, fallback)

    @staticmethod
    def _initial_buffer(initial_content: str, *, is_continuation: bool) -> list[str]:
        if not initial_content:
            return []
        if not is_continuation or initial_content.endswith("\n"):
            return [initial_content, "\n"]
        return [initial_content, "\n\n"]

    @staticmethod
    def next_runtime_status(
        previous: RuntimeStatus,
        phase: RuntimePhase,
        *,
        model_key: str,
        model_name: str,
        progress: float | None = None,
        output_tokens: int = 0,
        detail: str = "",
    ) -> RuntimeStatus:
        """Advance one generation phase while retaining total elapsed time."""

        now: float = asyncio.get_running_loop().time()
        started_at: float = (
            previous.started_at if previous.started_at is not None else now
        )
        phase_started_at: float = (
            previous.phase_started_at
            if previous.phase is phase and previous.phase_started_at is not None
            else now
        )
        return RuntimeStatus(
            phase,
            model_key=model_key,
            model_name=model_name,
            progress=progress,
            started_at=started_at,
            phase_started_at=phase_started_at,
            output_tokens=output_tokens,
            detail=detail,
        )
