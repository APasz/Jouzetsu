"""Server-owned inference jobs, explicit chat commands and per-backend FIFO queues.

This service is the sole owner of chat and configuration mutations.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine
from contextlib import aclosing, asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from functools import wraps
from typing import Concatenate
from uuid import uuid4

from .async_workers import complete_before_cancelling
from .atomic_write import CommitUncertainError
from .backend_codec import backend_fingerprint
from .backend_models import (
    BackendConfig,
    BackendConfiguration,
    BackendId,
    BackendKind,
    ChatInferenceSettings,
    GenerationAttribution,
    GenerationProfile,
    ModelRef,
)
from .backend_protocol import (
    BackendError,
    BackendErrorCategory,
    BackendHealth,
    BackendReachability,
    GenerationRequest,
    InferenceBackend,
    InferenceMessage,
)
from .backends import create_backend
from .config import CONTINUE_PROMPT_CONTENT, AppConfig, SpellingReplacement
from .controllers import PersistenceController
from .image_storage import ImageStorage
from .lmstudio import BRITISH_ENGLISH_INSTRUCTION
from .models import Chat, ImageAttachment, Message
from .runtime import (
    ModelDescriptor,
    ModelLoadProgress,
    ModelReady,
    PredictionComplete,
    PredictionFragment,
    ReasoningOption,
    RuntimePhase,
    RuntimeStatus,
)
from .state_registry import ChatRegistry
from .submissions import Submission, submission_fingerprint
from .text_processing import apply_case_preserving_word_replacements

log = logging.getLogger(__name__)
_POLL_SECONDS = 120.0
_SHUTDOWN_SECONDS = 10.0
_STREAM_UPDATE_SECONDS = 0.05
_CANCELLATION_GRACE_SECONDS = 2.0


class RevisionConflict(ValueError):
    """An edit was based on an older committed document or shared draft."""


class ChatBusy(ValueError):
    """Committed state is protected by a queued, active or unsaved job."""


class GenerationMode(str, Enum):
    REPLY = "reply"
    REGENERATE = "regenerate"
    CONTINUE = "continue"


class JobPhase(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SAVING = "saving"
    SAVE_FAILED = "save_failed"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ChangeKind(str, Enum):
    CHAT = "chat"
    DRAFT = "draft"
    STATUS = "status"
    CONFIG = "config"


@dataclass(frozen=True, slots=True)
class RuntimeChange:
    kind: ChangeKind
    chat_id: str | None = None


@dataclass(slots=True)
class GenerationJob:
    id: str
    chat_id: str
    request: GenerationRequest
    backend_name: str
    model_name: str
    mode: GenerationMode
    save_reasoning: bool
    replacements: tuple[SpellingReplacement, ...]
    replacement_id: str | None = None
    phase: JobPhase = JobPhase.QUEUED
    task: asyncio.Task[None] | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)
    ready: ModelReady | None = None
    text_fragments: list[str] = field(default_factory=list)
    reasoning_fragments: list[str] = field(default_factory=list)
    has_text: bool = False
    preview_at: float = 0.0
    preview_dirty: bool = False
    preview_task: asyncio.Task[None] | None = None
    preview: Message = field(default_factory=lambda: Message(role="assistant"))
    error: BackendError | None = None
    save_error: Exception | None = None
    cancelled: bool = False
    manually_registered: bool = False
    reported_reasoning: ReasoningOption | None = None

    @property
    def text(self) -> str:
        return "".join(self.text_fragments)

    @property
    def reasoning(self) -> str:
        return "".join(self.reasoning_fragments)

    async def wait(self) -> None:
        """Caller cancellation never cancels the server-owned job."""
        await self.done.wait()


@dataclass(frozen=True, slots=True)
class SendResult:
    message_id: str
    duplicate: bool
    job: GenerationJob | None = None
    draft_revision: int | None = None
    draft_cleared: bool = False


@dataclass(slots=True)
class _BackendRuntime:
    adapter: InferenceBackend
    health: BackendHealth = field(default_factory=BackendHealth)
    inventory: tuple[ModelDescriptor, ...] = ()
    queue: deque[GenerationJob] = field(default_factory=deque)
    active: dict[str, GenerationJob] = field(default_factory=dict)
    management_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    maintenance: int = 0
    last_used: dict[str, float] = field(default_factory=dict)
    unload_retry_at: float = 0.0
    unloading_model_id: str | None = None
    discovery_error: BackendError | None = None

    @property
    def busy(self) -> bool:
        return bool(self.queue or self.active or self.maintenance)


async def _ignore_change(change: RuntimeChange) -> None:
    pass


def _command[**P, T](
    method: Callable[Concatenate[InferenceRuntime, P], Awaitable[T]],
) -> Callable[Concatenate[InferenceRuntime, P], Coroutine[object, object, T]]:
    """Drain accepted commands before shutdown closes their resources."""

    @wraps(method)
    async def run(self: InferenceRuntime, *args: P.args, **kwargs: P.kwargs) -> T:
        return await self._run_command(method, *args, **kwargs)  # pyright: ignore[reportPrivateUsage]

    return run


class InferenceRuntime:
    """One owner for shared documents; no operation consults active selection."""

    def __init__(
        self,
        registry: ChatRegistry,
        persistence: PersistenceController,
        *,
        backend_factory: Callable[[BackendConfig], InferenceBackend] | None = None,
        notify: Callable[[RuntimeChange], Awaitable[None]] = _ignore_change,
    ) -> None:
        self.registry = registry
        self.persistence = persistence
        images = ImageStorage(persistence.config.paths.images_directory)
        self._factory: Callable[[BackendConfig], InferenceBackend] = (
            backend_factory or (lambda config: create_backend(config, images=images))
        )
        self._notify_callback = notify
        self._backends: dict[BackendId, _BackendRuntime] = {
            config.id: _BackendRuntime(self._factory(config))
            for config in self.configuration.backends
        }
        self._chat_locks: dict[str, asyncio.Lock] = {}
        self._config_lock = asyncio.Lock()
        self._jobs: dict[str, GenerationJob] = {}
        self._closing = False
        self._commands_in_flight = 0
        self._commands_idle = asyncio.Event()
        self._commands_idle.set()
        self._unclosed_adapters: list[InferenceBackend] = []
        self._start_task: asyncio.Task[None] | None = None
        self._observers = 0
        self._refresh_requested = False
        self._monitor_task: asyncio.Task[None] | None = None
        self._activity = asyncio.Event()
        self.persistence.on_chat_write_error = self._draft_write_failed

    async def _run_command[**P, T](
        self,
        method: Callable[Concatenate[InferenceRuntime, P], Awaitable[T]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        if self._closing:
            raise RuntimeError("runtime is shutting down")
        self._commands_in_flight += 1
        self._commands_idle.clear()
        try:
            return await method(self, *args, **kwargs)
        finally:
            self._commands_in_flight -= 1
            if not self._commands_in_flight:
                self._commands_idle.set()

    async def _draft_write_failed(self, error: Exception) -> None:
        await self._notify(ChangeKind.STATUS)

    @property
    def configuration(self) -> BackendConfiguration:
        return self.persistence.config.backends

    def health(self, backend_id: BackendId) -> BackendHealth:
        return self._backends[backend_id].health

    def inventory(self, backend_id: BackendId) -> tuple[ModelDescriptor, ...]:
        """Observed model instances; use health to distinguish current and cached data."""
        return self._backends[backend_id].inventory

    def discovery_error(self, backend_id: BackendId) -> BackendError | None:
        """Inventory errors remain visible independently of inference availability."""
        return self._backends[backend_id].discovery_error

    def job(self, chat_id: str) -> GenerationJob | None:
        return self._jobs.get(chat_id)

    def _reject(self, chat_id: str, error: BackendError) -> None:
        session = self.registry.get(chat_id)
        session.latest_error = error
        session.runtime_status = replace(
            session.runtime_status,
            phase=RuntimePhase.OFFLINE
            if error.category is BackendErrorCategory.OFFLINE
            else RuntimePhase.ERROR,
        )

    def _lock(self, chat_id: str) -> asyncio.Lock:
        return self._chat_locks.setdefault(chat_id, asyncio.Lock())

    def _editable(self, chat_id: str, expected_revision: int) -> Chat:
        if self._closing:
            raise RuntimeError("runtime is shutting down")
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("revision must be a nonnegative integer")
        session = self.registry.get(chat_id)
        if session.is_generating:
            raise ChatBusy("chat has queued, active or unsaved generation")
        if session.chat.revision != expected_revision:
            raise RevisionConflict("committed chat revision changed")
        return session.chat

    async def _notify(self, kind: ChangeKind, chat_id: str | None = None) -> None:
        try:
            await self._notify_callback(RuntimeChange(kind, chat_id))
        except Exception as exc:  # noqa: BLE001
            # Notification failures cannot turn a durable commit into a rejection.
            log.error("runtime notification failed error_type=%s", type(exc).__name__)

    async def _commit(self, chat: Chat, candidate: Chat) -> None:
        async def publish() -> None:
            await self.persistence.save_chat_candidate(candidate)
            chat.apply(candidate)
            await self._notify(ChangeKind.CHAT, chat.id)

        await complete_before_cancelling(publish())

    @_command
    async def create_chat(self, template: Chat | None = None) -> Chat:
        if self._closing:
            raise RuntimeError("runtime is shutting down")
        candidate = template.copy() if template is not None else Chat()
        candidate.inference = ChatInferenceSettings.new_chat(self.configuration)

        async def publish() -> None:
            await self.persistence.save_chat_candidate(candidate)
            _ = self.registry.add(
                candidate,
                runtime_status=RuntimeStatus(RuntimePhase.READY),
            )
            await self._notify(ChangeKind.CHAT, candidate.id)

        async with self._lock(candidate.id):
            if candidate.id in self.registry:
                raise ValueError("chat already exists")
            await complete_before_cancelling(publish())
        return candidate

    @_command
    async def edit_chat(
        self, chat_id: str, expected_revision: int, edit: Callable[[Chat], None]
    ) -> None:
        """Apply an explicit committed edit to an isolated candidate under its lock.

        Route handlers supply existing domain methods for editing, merging,
        truncation and chat-wide settings; drafts use their separate command.
        """
        async with self._lock(chat_id):
            chat = self._editable(chat_id, expected_revision)
            candidate = chat.copy()
            edit(candidate)
            if (
                candidate.id != chat.id
                or candidate.draft != chat.draft
                or candidate.draft_images != chat.draft_images
                or candidate.draft_revision != chat.draft_revision
                or candidate.submissions != chat.submissions
            ):
                raise ValueError("committed edits cannot change chat identity or draft")
            candidate.revision = chat.revision + 1
            candidate.touch()
            candidate.__post_init__()
            for message in candidate.messages:
                message.__post_init__()
            await self._commit(chat, candidate)

    async def select_backend(
        self, chat_id: str, expected_revision: int, backend_id: BackendId
    ) -> None:
        def edit(chat: Chat) -> None:
            chat.inference = chat.inference.select_backend(
                self.configuration, backend_id
            )

        await self.edit_chat(chat_id, expected_revision, edit)

    async def set_profile(
        self, chat_id: str, expected_revision: int, profile: GenerationProfile
    ) -> None:
        def edit(chat: Chat) -> None:
            chat.inference = chat.inference.set_profile(profile)

        await self.edit_chat(chat_id, expected_revision, edit)

    async def reset_profile(self, chat_id: str, expected_revision: int) -> None:
        def edit(chat: Chat) -> None:
            chat.inference = chat.inference.reset_to_defaults(self.configuration)

        await self.edit_chat(chat_id, expected_revision, edit)

    @_command
    async def fork_chat(
        self, chat_id: str, expected_revision: int, message_id: str
    ) -> Chat:
        async with self._lock(chat_id):
            chat = self._editable(chat_id, expected_revision)
            candidate = chat.fork_through(message_id)

            async def publish() -> None:
                await self.persistence.save_chat_candidate(candidate)
                _ = self.registry.add(
                    candidate,
                    runtime_status=RuntimeStatus(RuntimePhase.READY),
                )
                await self._notify(ChangeKind.CHAT, candidate.id)

            await complete_before_cancelling(publish())
            return candidate

    @_command
    async def delete_chat(self, chat_id: str, expected_revision: int) -> None:
        async with self._lock(chat_id):
            _ = self._editable(chat_id, expected_revision)

            async def publish() -> None:
                await self.persistence.delete_chat(chat_id)
                _ = self.registry.remove(chat_id)
                await self._notify(ChangeKind.CHAT, chat_id)

            await complete_before_cancelling(publish())

    @_command
    async def update_draft(
        self,
        chat_id: str,
        expected_revision: int,
        content: str,
        images: tuple[ImageAttachment, ...] = (),
    ) -> int:
        async with self._lock(chat_id):
            if self._closing:
                raise RuntimeError("runtime is shutting down")
            chat = self.registry.get(chat_id).chat
            if type(expected_revision) is not int or expected_revision < 0:
                raise ValueError("draft revision must be a nonnegative integer")
            if chat.draft_revision != expected_revision:
                raise RevisionConflict("shared draft revision changed")
            candidate = replace(
                chat,
                draft=content,
                draft_images=images,
                draft_revision=chat.draft_revision + 1,
            )
            # Drafts deliberately publish before their coalesced disk write.
            chat.apply(candidate)
            self.persistence.queue_chat_changes((chat,))
            await self._notify(ChangeKind.DRAFT, chat_id)
            return chat.draft_revision

    @_command
    async def flush_drafts(self) -> None:
        await self.persistence.flush_chats()
        await self._notify(ChangeKind.STATUS)

    @_command
    async def send_message(
        self,
        chat_id: str,
        expected_revision: int,
        submission_id: str,
        content: str,
        *,
        images: tuple[ImageAttachment, ...] = (),
        generate: bool = True,
        expected_draft_revision: int | None = None,
    ) -> SendResult:
        fingerprint = submission_fingerprint(
            content, tuple(image.to_dict() for image in images)
        )
        async with self._lock(chat_id):
            chat = self.registry.get(chat_id).chat
            receipt = next(
                (item for item in chat.submissions if item.id == submission_id), None
            )
            if receipt is not None:
                if receipt.fingerprint != fingerprint:
                    raise ValueError(
                        "submission identifier reused with different content"
                    )
                return SendResult(receipt.message_id, True)
            chat = self._editable(chat_id, expected_revision)
            if not content.strip() and not images:
                raise ValueError("message cannot be empty")
            candidate = chat.copy()
            message = Message(role="user", content=content, images=images)
            candidate.messages.append(message)
            candidate.submissions += (
                Submission(submission_id, message.id, fingerprint),
            )
            draft_cleared = (
                expected_draft_revision is not None
                and chat.draft_revision == expected_draft_revision
            )
            if draft_cleared:
                candidate.draft = ""
                candidate.draft_images = ()
                candidate.draft_revision += 1
            candidate.revision += 1
            if settings := self._reasoning_fallback(candidate):
                candidate.inference = settings
            if len(candidate.messages) == 1:
                candidate.set_auto_title_from_first_message(content)
            candidate.touch()

            async def commit_submission() -> SendResult:
                await self._commit(chat, candidate)
                async with self._config_lock:
                    job = self._accept(chat, GenerationMode.REPLY) if generate else None
                await self._notify(ChangeKind.STATUS, chat_id)
                return SendResult(
                    message.id, False, job, candidate.draft_revision, draft_cleared
                )

            # Caller cancellation must not split durable submission from job acceptance.
            return await complete_before_cancelling(commit_submission())

    @_command
    async def generate(
        self,
        chat_id: str,
        expected_revision: int,
        mode: GenerationMode = GenerationMode.REPLY,
    ) -> GenerationJob | None:
        async with self._lock(chat_id):
            chat = self._editable(chat_id, expected_revision)
            async with self._config_lock:
                if settings := self._reasoning_fallback(chat):
                    candidate = chat.copy()
                    candidate.inference = settings
                    candidate.revision += 1
                    candidate.touch()
                    await self._commit(chat, candidate)
                job = self._accept(chat, mode)
            await self._notify(ChangeKind.STATUS, chat_id)
            return job

    def _reasoning_fallback(self, chat: Chat) -> ChatInferenceSettings | None:
        """Replace an unavailable selection with Automatic when generating."""
        profile = chat.inference.selected_profile
        ref = chat.inference.selected_model
        if profile is None or ref is None or profile.reasoning == "automatic":
            return None
        try:
            backend = self.configuration.backend(ref.backend_id)
            if backend.kind is not BackendKind.LM_STUDIO:
                return None
            model = backend.model(ref.model_id)
        except KeyError:
            return None
        if (
            not model.reasoning_options
            and model.descriptor.capabilities.reasoning is not False
        ):
            return None
        _, available = model.resolve_reasoning(profile.reasoning)
        if available:
            return None
        return chat.inference.set_profile(replace(profile, reasoning="automatic"))

    def _accept(self, chat: Chat, mode: GenerationMode) -> GenerationJob | None:
        if self._closing:
            return self._reject(
                chat.id,
                BackendError(
                    BackendErrorCategory.OTHER,
                    "Generation did not start because the application is shutting down.",
                ),
            )
        session = self.registry.get(chat.id)
        try:
            model = chat.inference.selected_model
            if model is None:
                raise BackendError(
                    BackendErrorCategory.INCOMPATIBLE,
                    "Select a backend and model before generating.",
                )
            try:
                config = self.configuration.backend(model.backend_id)
            except KeyError as exc:
                raise BackendError(
                    BackendErrorCategory.INCOMPATIBLE,
                    "The selected backend was removed.",
                ) from exc
            runtime = self._backends[model.backend_id]
            if (
                not config.enabled
                or runtime.health.state is BackendReachability.OFFLINE
            ):
                raise BackendError(
                    BackendErrorCategory.OFFLINE,
                    "The selected backend is offline or disabled.",
                )
            if runtime.unloading_model_id == model.model_id:
                raise BackendError(
                    BackendErrorCategory.LOAD_FAILURE,
                    "The selected model is being unloaded. Try again after unloading finishes.",
                )
            registered = config.model(model.model_id)
            profile = chat.inference.selected_profile
            if profile is None:
                raise ValueError("selected backend has no profile")
            messages = list(chat.messages)
            replacement: Message | None = None
            if mode in {GenerationMode.REGENERATE, GenerationMode.CONTINUE}:
                if not messages or messages[-1].role != "assistant":
                    raise ValueError(
                        "only the final assistant response can be regenerated or continued"
                    )
                if mode is GenerationMode.REGENERATE:
                    replacement = messages.pop()
            generation = self.persistence.config.generation
            prompt = chat.system_prompt.strip() or generation.system_prompt.strip()
            if generation.british_english:
                prompt = "\n\n".join(
                    item for item in (prompt, BRITISH_ENGLISH_INSTRUCTION) if item
                )
            prepared = [InferenceMessage("system", prompt)] if prompt else []
            prepared.extend(
                InferenceMessage(message.role, message.content, message.images)
                for message in messages
                if message.content or message.images
            )
            if mode is GenerationMode.CONTINUE or (
                mode is GenerationMode.REGENERATE
                and messages
                and messages[-1].role == "assistant"
            ):
                prepared.append(InferenceMessage("user", CONTINUE_PROMPT_CONTENT))
            request = GenerationRequest(
                model,
                tuple(prepared),
                registered.descriptor.capabilities,
                config.parameter_support,
                profile.parameters,
                registered.resolve_reasoning(profile.reasoning)[0]
                if config.kind is BackendKind.LM_STUDIO
                else None,
            )
            job = GenerationJob(
                uuid4().hex,
                chat.id,
                request,
                config.name,
                registered.alias
                or registered.descriptor.display_name
                or model.model_id,
                mode,
                chat.save_reasoning,
                tuple(generation.british_spelling_replacements)
                if chat.postprocess_british_spellings
                else (),
                replacement.id if replacement else None,
                manually_registered=registered.manually_registered,
            )
        except BackendError as exc:
            return self._reject(chat.id, exc)
        except KeyError:
            return self._reject(
                chat.id,
                BackendError(
                    BackendErrorCategory.INCOMPATIBLE,
                    "The selected model is not registered on this backend.",
                ),
            )
        session.latest_error = None
        session.live_response = None
        session.is_generating = True
        session.runtime_status = RuntimeStatus(
            RuntimePhase.QUEUED, model.model_id, job.model_name
        )
        self._jobs[chat.id] = job
        runtime.queue.append(job)
        self._activity.set()
        self._pump(model.backend_id)
        return job

    def _pump(self, backend_id: BackendId) -> None:
        if self._closing:
            return
        runtime = self._backends[backend_id]
        limit = self.configuration.backend(backend_id).concurrency_limit
        while runtime.queue and len(runtime.active) < limit:
            job = runtime.queue.popleft()
            runtime.active[job.id] = job
            job.task = asyncio.create_task(self._run(job))
            self.registry.get(job.chat_id).generation_task = job.task

    async def _run(self, job: GenerationJob) -> None:
        runtime = self._backends[job.request.model.backend_id]
        session = self.registry.get(job.chat_id)
        job.phase = JobPhase.RUNNING
        session.runtime_status = RuntimeStatus(
            RuntimePhase.STARTING, job.request.model.model_id, job.model_name
        )
        try:
            if job.cancelled:
                raise asyncio.CancelledError
            if runtime.health.state is not BackendReachability.ONLINE:
                try:
                    await self.refresh_backend(runtime.adapter.id)
                except BackendError as exc:
                    if (
                        not job.manually_registered
                        or runtime.adapter.model_management is not None
                        or exc.category is BackendErrorCategory.OFFLINE
                    ):
                        raise
            if runtime.health.state is BackendReachability.OFFLINE:
                raise BackendError(
                    BackendErrorCategory.OFFLINE, "The selected backend is offline."
                )
            config = self.configuration.backend(runtime.adapter.id)
            management = config.model_management
            if management is not None and runtime.adapter.model_management is not None:
                async with runtime.management_lock:
                    if management.automatic_loading:
                        session.runtime_status = replace(
                            session.runtime_status, phase=RuntimePhase.LOADING_MODEL
                        )
                        async with aclosing(
                            runtime.adapter.model_management.load_model(
                                job.request.model.model_id
                            )
                        ) as events:
                            async for event in events:
                                if job.cancelled:
                                    raise asyncio.CancelledError
                                if isinstance(event, ModelReady):
                                    job.ready = event
                                else:
                                    session.runtime_status = replace(
                                        session.runtime_status, progress=event.progress
                                    )
                                await self._notify(ChangeKind.STATUS, job.chat_id)
                    else:
                        discovery = await runtime.adapter.discover_models()
                        descriptor = next(
                            (
                                item
                                for item in discovery.models
                                if item.key == job.request.model.model_id
                            ),
                            None,
                        )
                        if descriptor is None or not descriptor.is_loaded:
                            raise BackendError(
                                BackendErrorCategory.LOAD_FAILURE,
                                "The selected model is not loaded and automatic loading is disabled.",
                            )
            session.runtime_status = replace(
                session.runtime_status,
                phase=RuntimePhase.PROCESSING_PROMPT,
                progress=None,
            )
            completed = False
            if job.cancelled:
                raise asyncio.CancelledError
            async with aclosing(runtime.adapter.stream(job.request)) as events:
                async for event in events:
                    if job.cancelled:
                        raise asyncio.CancelledError
                    if completed:
                        raise BackendError(
                            BackendErrorCategory.PROTOCOL,
                            "The backend returned events after completion.",
                        )
                    if isinstance(event, ModelReady):
                        job.ready = event
                        runtime.health = BackendHealth(
                            BackendReachability.ONLINE, datetime.now(UTC)
                        )
                    elif isinstance(event, PredictionFragment):
                        fragments = (
                            job.reasoning_fragments
                            if event.is_reasoning
                            else job.text_fragments
                        )
                        fragments.append(event.content)
                        if not event.is_reasoning:
                            job.has_text = job.has_text or bool(event.content.strip())
                        phase = (
                            RuntimePhase.REASONING
                            if event.is_reasoning
                            else RuntimePhase.GENERATING
                        )
                        if session.runtime_status.phase is not phase:
                            session.runtime_status = replace(
                                session.runtime_status, phase=phase
                            )
                        job.preview_dirty = True
                        now = time.monotonic()
                        first_text = job.has_text and session.live_response is None
                        if (
                            not first_text
                            and now - job.preview_at < _STREAM_UPDATE_SECONDS
                        ):
                            if job.preview_task is None:
                                job.preview_task = asyncio.create_task(
                                    self._publish_delayed_preview(job)
                                )
                            continue
                        self._update_preview(job)
                    elif isinstance(event, PredictionComplete):
                        completed = True
                        job.reported_reasoning = event.metrics.reasoning_level
                        session.runtime_status = replace(
                            session.runtime_status, metrics=event.metrics
                        )
                    elif isinstance(event, ModelLoadProgress):
                        session.runtime_status = replace(
                            session.runtime_status,
                            phase=RuntimePhase.LOADING_MODEL,
                            progress=event.progress,
                        )
                    await self._notify(ChangeKind.STATUS, job.chat_id)
            if job.cancelled:
                raise asyncio.CancelledError
            if not completed:
                raise BackendError(
                    BackendErrorCategory.PROTOCOL,
                    "The backend disconnected before confirming completion.",
                )
            if not job.has_text:
                raise BackendError(
                    BackendErrorCategory.PROTOCOL,
                    "The backend completed without assistant text.",
                )
        except asyncio.CancelledError:
            job.cancelled = True
        except BackendError as exc:
            job.error = exc
            if exc.category is BackendErrorCategory.OFFLINE:
                runtime.health = BackendHealth(
                    BackendReachability.OFFLINE, datetime.now(UTC)
                )
                await self._fail_queued(runtime, exc)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "generation failed backend_id=%s job_id=%s error_type=%s",
                runtime.adapter.id,
                job.id,
                type(exc).__name__,
            )
            job.error = BackendError(
                BackendErrorCategory.OTHER, "Generation failed unexpectedly."
            )
        finally:
            job.phase = JobPhase.SAVING
            # Persistence finishes before cancellation is delivered or locks release.
            try:
                if job.preview_task is not None:
                    preview_task = job.preview_task
                    _ = preview_task.cancel()
                    try:
                        await preview_task
                    except asyncio.CancelledError:
                        pass
                await complete_before_cancelling(self._finish(job))
            finally:
                _ = runtime.active.pop(job.id, None)
                if job.ready is not None:
                    runtime.last_used[job.ready.model_key] = time.monotonic()
                self._pump(runtime.adapter.id)
                self._activity.set()
                job.done.set()

    async def _finish(self, job: GenerationJob) -> None:
        session = self.registry.get(job.chat_id)
        session.latest_error = job.error
        if job.has_text:
            self._update_preview(job)
            try:
                await self._save_output(job)
            except Exception as exc:  # noqa: BLE001
                await self._retain_unsaved_output(job, exc)
                return
        self._release(job)
        await self._notify(ChangeKind.STATUS, job.chat_id)

    async def _retain_unsaved_output(
        self, job: GenerationJob, error: Exception
    ) -> None:
        job.save_error = error
        job.phase = JobPhase.SAVE_FAILED
        session = self.registry.get(job.chat_id)
        session.latest_error = BackendError(
            BackendErrorCategory.OTHER,
            "The response save could not be confirmed. Storage recovery and restart are required; the response remains in memory."
            if isinstance(error, CommitUncertainError)
            else "The response could not be saved. Retry saving before editing this chat.",
        )
        session.runtime_status = replace(
            session.runtime_status, phase=RuntimePhase.ERROR
        )
        await self._notify(ChangeKind.STATUS, job.chat_id)

    async def _publish_delayed_preview(self, job: GenerationJob) -> None:
        try:
            while job.preview_dirty:
                delay = max(
                    0.0, job.preview_at + _STREAM_UPDATE_SECONDS - time.monotonic()
                )
                await asyncio.sleep(delay)
                if job.preview_dirty:
                    self._update_preview(job)
                    await self._notify(ChangeKind.STATUS, job.chat_id)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "preview publication failed job_id=%s error_type=%s",
                job.id,
                type(exc).__name__,
            )
        finally:
            job.preview_task = None

    def _update_preview(self, job: GenerationJob) -> None:
        job.preview_dirty = False
        job.preview_at = time.monotonic()
        session = self.registry.get(job.chat_id)
        session.live_reasoning = job.reasoning
        if job.has_text:
            job.preview.content = apply_case_preserving_word_replacements(
                job.text, job.replacements
            )
            session.live_response = job.preview

    async def _save_output(self, job: GenerationJob) -> None:
        async with self._lock(job.chat_id):
            chat = self.registry.get(job.chat_id).chat
            candidate = chat.copy()
            ready = job.ready
            attribution = GenerationAttribution(
                ModelRef(
                    job.request.model.backend_id,
                    ready.model_key if ready else job.request.model.model_id,
                ),
                job.backend_name,
                job.model_name
                if ready is None or ready.model_key == job.request.model.model_id
                else ready.model_name,
                ready.instance_id if ready else None,
                job.reported_reasoning or job.request.reasoning or "automatic",
            )
            response = replace(
                job.preview,
                updated_at=time.time(),
                reasoning=job.reasoning if job.save_reasoning else "",
                attributions=(attribution,),
            )
            if job.replacement_id is None:
                candidate.messages.append(response)
            else:
                original = candidate.find_message(job.replacement_id)
                if original is None or candidate.messages[-1].id != original.id:
                    raise RuntimeError(
                        "generation replacement target changed while locked"
                    )
                response.id = original.id
                response.created_at = original.created_at
                original.apply(response)
            candidate.revision += 1
            candidate.touch()
            await self._commit(chat, candidate)

    def _release(self, job: GenerationJob) -> None:
        session = self.registry.get(job.chat_id)
        session.is_generating = False
        session.live_response = None
        session.live_reasoning = ""
        session.generation_task = None
        session.runtime_status = replace(
            session.runtime_status,
            phase=RuntimePhase.ERROR if job.error else RuntimePhase.READY,
            progress=None,
        )
        job.phase = (
            JobPhase.CANCELLED
            if job.cancelled
            else JobPhase.FAILED
            if job.error
            else JobPhase.FINISHED
        )
        _ = self._jobs.pop(job.chat_id, None)
        if job.task is None:
            job.done.set()

    @_command
    async def retry_save(self, chat_id: str) -> None:
        job = self._jobs[chat_id]
        if job.phase is not JobPhase.SAVE_FAILED:
            raise ValueError("chat has no failed response save")
        # Reserve the retry synchronously before its first await.
        job.phase = JobPhase.SAVING
        try:
            await complete_before_cancelling(self._retry_save(job))
        except Exception as exc:
            await self._retain_unsaved_output(job, exc)
            raise

    async def _retry_save(self, job: GenerationJob) -> None:
        await self._save_output(job)
        job.save_error = None
        self.registry.get(job.chat_id).latest_error = job.error
        self._release(job)
        await self._notify(ChangeKind.STATUS, job.chat_id)

    @_command
    async def cancel(self, chat_id: str) -> None:
        job = self._jobs.get(chat_id)
        if job is None:
            return
        if job.phase in {JobPhase.SAVING, JobPhase.SAVE_FAILED}:
            raise ChatBusy("response must finish saving")
        runtime = self._backends[job.request.model.backend_id]
        if job in runtime.queue:
            runtime.queue.remove(job)
            job.cancelled = True
            self._release(job)
            await self._notify(ChangeKind.STATUS, chat_id)
        elif job.task is not None:
            self._request_cancel(job)
            session = self.registry.get(chat_id)
            session.runtime_status = replace(
                session.runtime_status, phase=RuntimePhase.STOPPING
            )
            await self._notify(ChangeKind.STATUS, chat_id)
            _, pending = await asyncio.wait(
                {job.task}, timeout=_CANCELLATION_GRACE_SECONDS
            )
            if pending:
                message = "Generation is still stopping; the chat remains locked until generation and saving finish."
                if job.phase is JobPhase.RUNNING:
                    session.latest_error = BackendError(
                        BackendErrorCategory.TIMEOUT, message
                    )
                await self._notify(ChangeKind.STATUS, chat_id)
                raise TimeoutError(message)
            await asyncio.shield(job.task)

    @staticmethod
    def _request_cancel(job: GenerationJob) -> None:
        if not job.cancelled:
            job.cancelled = True
            if job.task is not None and job.phase is JobPhase.RUNNING:
                _ = job.task.cancel()

    @_command
    async def dismiss_error(self, chat_id: str) -> None:
        session = self.registry.get(chat_id)
        session.latest_error = None
        if not session.is_generating and session.runtime_status.phase in {
            RuntimePhase.ERROR,
            RuntimePhase.OFFLINE,
        }:
            session.runtime_status = replace(
                session.runtime_status, phase=RuntimePhase.READY
            )
        await self._notify(ChangeKind.CHAT, chat_id)

    async def _fail_queued(self, runtime: _BackendRuntime, error: BackendError) -> None:
        while runtime.queue:
            job = runtime.queue.popleft()
            job.error = error
            self.registry.get(job.chat_id).latest_error = error
            self._release(job)
            await self._notify(ChangeKind.STATUS, job.chat_id)

    @_command
    async def update_backend(
        self,
        candidate: BackendConfig,
        *,
        expected: BackendConfig | None = None,
        expected_fingerprint: str | None = None,
    ) -> None:
        await complete_before_cancelling(
            self._update_backend(candidate, expected, expected_fingerprint)
        )

    async def _update_backend(
        self,
        candidate: BackendConfig,
        expected: BackendConfig | None,
        expected_fingerprint: str | None,
    ) -> None:
        async with self._config_lock:
            previous = next(
                (
                    item
                    for item in self.configuration.backends
                    if item.id == candidate.id
                ),
                None,
            )
            self.validate_backend_snapshot(previous, expected_fingerprint)
            if expected is not None and previous != expected:
                raise RevisionConflict("backend configuration changed while editing")
            disruptive = previous is not None and (
                previous.kind,
                previous.base_url,
                previous.enabled,
                previous.concurrency_limit,
                previous.model_management,
            ) != (
                candidate.kind,
                candidate.base_url,
                candidate.enabled,
                candidate.concurrency_limit,
                candidate.model_management,
            )
            runtime = self._backends.get(candidate.id)
            if disruptive and runtime is not None and runtime.busy:
                raise ChatBusy("backend has queued or active work")
            if previous is not None and previous.kind is not candidate.kind:
                raise ValueError("backend kind cannot change; create a new identity")
            replacement = (
                _BackendRuntime(self._factory(candidate))
                if runtime is None or disruptive
                else runtime
            )
            try:
                await self._save_configuration(
                    self.configuration.with_backend(candidate), notify=False
                )
            except BaseException:
                if replacement is not runtime:
                    await self._close_adapter(replacement.adapter)
                raise
            self._backends[candidate.id] = replacement
            if disruptive and runtime is not None:
                await self._close_adapter(runtime.adapter)
            await self._notify(ChangeKind.CONFIG)

    @_command
    async def remove_backend(
        self, backend_id: BackendId, *, expected_fingerprint: str | None = None
    ) -> None:
        await complete_before_cancelling(
            self._remove_backend(backend_id, expected_fingerprint)
        )

    async def _remove_backend(
        self, backend_id: BackendId, expected_fingerprint: str | None
    ) -> None:
        async with self._config_lock:
            self.validate_backend_snapshot(
                next(
                    (
                        backend
                        for backend in self.configuration.backends
                        if backend.id == backend_id
                    ),
                    None,
                ),
                expected_fingerprint,
            )
            runtime = self._backends[backend_id]
            if runtime.busy:
                raise ChatBusy("backend has queued or active work")
            candidate = self.configuration.remove_backend(backend_id)
            await self._save_configuration(candidate, notify=False)
            del self._backends[backend_id]
            await self._close_adapter(runtime.adapter)
            await self._notify(ChangeKind.CONFIG)

    @_command
    async def set_default_backend(
        self, backend_id: BackendId | None, *, expected_fingerprint: str | None = None
    ) -> None:
        await complete_before_cancelling(
            self._set_default_backend(backend_id, expected_fingerprint)
        )

    async def _set_default_backend(
        self, backend_id: BackendId | None, expected_fingerprint: str | None
    ) -> None:
        async with self._config_lock:
            self.validate_backend_snapshot(self.configuration, expected_fingerprint)
            await self._save_configuration(
                replace(self.configuration, default_backend_id=backend_id)
            )

    @staticmethod
    def validate_backend_snapshot(
        current: BackendConfig | BackendConfiguration | None, expected: str | None
    ) -> None:
        """Reject stale form state before validation and again under the write lock."""
        if expected is not None and (
            current is None or backend_fingerprint(current) != expected
        ):
            raise RevisionConflict(
                "Backend settings changed elsewhere. Your edits are preserved; copy them before reloading."
            )

    @_command
    async def update_settings(
        self, edit: Callable[[AppConfig], None], *, notify: bool = True
    ) -> None:
        """Serialize non-backend settings with backend discovery and configuration writes."""
        async with self._config_lock:
            candidate = deepcopy(self.persistence.config)
            edit(candidate)
            if (
                candidate.paths != self.persistence.config.paths
                or candidate.backends != self.configuration
            ):
                raise ValueError(
                    "backend and path changes require their explicit commands"
                )
            candidate.validate()
            if candidate == self.persistence.config:
                return

            async def publish() -> None:
                await self.persistence.save_config(candidate)
                self.persistence.config.apply(candidate)
                if notify:
                    await self._notify(ChangeKind.CONFIG)

            await complete_before_cancelling(publish())

    async def _save_configuration(
        self, backends: BackendConfiguration, *, notify: bool = True
    ) -> None:
        if self._closing:
            raise RuntimeError("runtime is shutting down")

        candidate = replace(self.persistence.config, backends=backends)
        await self.persistence.save_config(candidate)
        self.persistence.config.apply(candidate)
        if notify:
            await self._notify(ChangeKind.CONFIG)

    @asynccontextmanager
    async def _maintenance(
        self, backend_id: BackendId
    ) -> AsyncGenerator[_BackendRuntime]:
        async with self._config_lock:
            if self._closing:
                raise RuntimeError("runtime is shutting down")
            runtime = self._backends[backend_id]
            runtime.maintenance += 1
        try:
            async with runtime.management_lock:
                yield runtime
        finally:
            runtime.maintenance -= 1

    @_command
    async def refresh_backend(self, backend_id: BackendId) -> None:
        async with self._maintenance(backend_id) as runtime:
            if not self.configuration.backend(backend_id).enabled:
                return
            try:
                discovery = await runtime.adapter.discover_models()
                runtime.inventory = discovery.models
                async with self._config_lock:
                    current = self.configuration.backend(backend_id)
                    merged = current.merge_discovery(discovery.models)
                    if merged != current:
                        await self._save_configuration(
                            self.configuration.with_backend(merged)
                        )
                runtime.health = BackendHealth(
                    BackendReachability.ONLINE, datetime.now(UTC)
                )
                runtime.discovery_error = None
            except BackendError as exc:
                runtime.discovery_error = exc
                log.warning(
                    "model discovery failed backend_id=%s category=%s",
                    backend_id,
                    exc.category.value,
                )
                runtime.health = BackendHealth(
                    BackendReachability.OFFLINE
                    if exc.category is BackendErrorCategory.OFFLINE
                    else BackendReachability.STALE,
                    datetime.now(UTC),
                )
                if runtime.health.state is BackendReachability.OFFLINE:
                    await self._fail_queued(runtime, exc)
                await self._notify(ChangeKind.STATUS)
                raise
        if runtime.health.state is BackendReachability.OFFLINE:
            await self._fail_queued(
                runtime,
                BackendError(
                    BackendErrorCategory.OFFLINE,
                    "The backend went offline while this chat was queued.",
                ),
            )
        await self._notify(ChangeKind.STATUS)

    @_command
    async def unload_model(self, model: ModelRef) -> None:
        await self._unload_model(model)

    async def _unload_model(
        self, model: ModelRef, *, idle_before: float | None = None
    ) -> None:
        async with self._maintenance(model.backend_id) as runtime:
            if not self.configuration.backend(model.backend_id).enabled:
                raise ValueError("backend is disabled")
            if any(
                job.request.model == model
                for job in (*runtime.queue, *runtime.active.values())
            ):
                raise ChatBusy("model has queued or active work")
            if (
                idle_before is not None
                and runtime.last_used.get(model.model_id, float("inf")) > idle_before
            ):
                return
            management = runtime.adapter.model_management
            if management is None:
                raise ValueError("backend does not support model management")
            # Reserve before discovery yields; later submissions must not race an unload.
            runtime.unloading_model_id = model.model_id
            try:
                discovery = await runtime.adapter.discover_models()
                descriptor = next(
                    (item for item in discovery.models if item.key == model.model_id),
                    None,
                )
                if descriptor is None and idle_before is None:
                    raise ValueError("model is not available")
                if descriptor is not None:
                    for instance_id in descriptor.loaded_instance_ids:
                        await management.unload_model(instance_id)
                _ = runtime.last_used.pop(model.model_id, None)
            finally:
                runtime.unloading_model_id = None

    @_command
    async def load_model(self, model: ModelRef) -> ModelReady:
        """Explicit native loading also works with automatic loading disabled."""
        async with self._maintenance(model.backend_id) as runtime:
            if not self.configuration.backend(model.backend_id).enabled:
                raise ValueError("backend is disabled")
            management = runtime.adapter.model_management
            if management is None:
                raise ValueError("backend does not support model management")
            ready: ModelReady | None = None
            async with aclosing(management.load_model(model.model_id)) as events:
                async for event in events:
                    if isinstance(event, ModelReady):
                        ready = event
                    await self._notify(ChangeKind.STATUS)
            if ready is None:
                raise BackendError(
                    BackendErrorCategory.LOAD_FAILURE,
                    "The backend did not confirm model loading.",
                )
            runtime.last_used[model.model_id] = time.monotonic()
            self._activity.set()
            return ready

    @_command
    async def unload_idle_models(self, *, now: float | None = None) -> None:
        current_time = time.monotonic() if now is None else now
        for config in self.configuration.backends:
            management = config.model_management
            if (
                not config.enabled
                or management is None
                or management.idle_unload_minutes is None
            ):
                continue
            runtime = self._backends[config.id]
            if (
                runtime.health.state is BackendReachability.OFFLINE
                or current_time < runtime.unload_retry_at
            ):
                continue
            for model_id, last_used in tuple(runtime.last_used.items()):
                if current_time - last_used < management.idle_unload_minutes * 60:
                    continue
                model = ModelRef(config.id, model_id)
                try:
                    await self._unload_model(
                        model,
                        idle_before=current_time - management.idle_unload_minutes * 60,
                    )
                except ChatBusy:
                    continue
                except BackendError:
                    runtime.unload_retry_at = current_time + _POLL_SECONDS
                    raise

    def set_observers(self, count: int) -> None:
        if type(count) is not int or count < 0:
            raise ValueError("observer count must be a nonnegative integer")
        if count and not self._observers:
            self._refresh_requested = True
        self._observers = count
        self._activity.set()

    @_command
    async def start(self) -> None:
        if self._start_task is None:
            self._start_task = asyncio.create_task(self._start())
        task = self._start_task
        try:
            await complete_before_cancelling(task)
        except Exception:
            if self._start_task is task:
                self._start_task = None
            raise

    async def _start(self) -> None:
        for config in self.configuration.backends:
            try:
                await self.refresh_backend(config.id)
            except BackendError:
                pass
        if not self._closing:
            self._monitor_task = asyncio.create_task(self._monitor())

    async def _monitor(self) -> None:
        while not self._closing:
            self._activity.clear()
            active = self._observers > 0 or any(
                runtime.busy for runtime in self._backends.values()
            )
            if active and self._refresh_requested:
                self._refresh_requested = False
                await self._refresh_monitored_backends()
                continue
            if not active:
                for runtime in self._backends.values():
                    if runtime.health.state is BackendReachability.ONLINE:
                        runtime.health = replace(
                            runtime.health, state=BackendReachability.STALE
                        )
                # A local idle-unload deadline may still need servicing.
                deadlines = [
                    max(
                        last + config.model_management.idle_unload_minutes * 60,
                        self._backends[config.id].unload_retry_at,
                    )
                    for config in self.configuration.backends
                    if config.enabled
                    and config.model_management is not None
                    and config.model_management.idle_unload_minutes is not None
                    and self._backends[config.id].health.state
                    is not BackendReachability.OFFLINE
                    for last in self._backends[config.id].last_used.values()
                ]
                if not deadlines:
                    await self._activity.wait()
                    continue
                delay = max(0.1, min(deadlines) - time.monotonic())
            else:
                delay = _POLL_SECONDS
            try:
                await asyncio.wait_for(self._activity.wait(), delay)
            except TimeoutError:
                if active:
                    await self._refresh_monitored_backends()
                try:
                    await self.unload_idle_models()
                except Exception as exc:  # noqa: BLE001
                    log.error("idle unload failed error_type=%s", type(exc).__name__)

    async def _refresh_monitored_backends(self) -> None:
        for config in self.configuration.backends:
            try:
                await self.refresh_backend(config.id)
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "backend refresh failed backend_id=%s error_type=%s",
                    config.id,
                    type(exc).__name__,
                )

    async def shutdown(self) -> bool:
        self._closing = True
        if self._monitor_task is not None:
            _ = self._monitor_task.cancel()
        for runtime in self._backends.values():
            while runtime.queue:
                job = runtime.queue.popleft()
                job.cancelled = True
                self._release(job)
            for job in runtime.active.values():
                self._request_cancel(job)
        tasks = [
            job.task
            for runtime in self._backends.values()
            for job in runtime.active.values()
            if job.task is not None
        ]
        if self._monitor_task is not None:
            tasks.append(self._monitor_task)
        try:
            async with asyncio.timeout(_SHUTDOWN_SECONDS):
                await self._commands_idle.wait()
                if tasks:
                    _ = await asyncio.wait(tasks)
                await self.persistence.flush_chats()
                for runtime in self._backends.values():
                    if not any(
                        adapter is runtime.adapter
                        for adapter in self._unclosed_adapters
                    ):
                        self._unclosed_adapters.append(runtime.adapter)
                for adapter in tuple(self._unclosed_adapters):
                    await self._close_adapter(adapter)
        except TimeoutError:
            return False
        except Exception as exc:  # noqa: BLE001
            log.error("runtime shutdown failed error_type=%s", type(exc).__name__)
            return False
        return not self._jobs and not self._unclosed_adapters

    async def _close_adapter(self, adapter: InferenceBackend) -> None:
        if not any(item is adapter for item in self._unclosed_adapters):
            self._unclosed_adapters.append(adapter)
        try:
            await adapter.close()
        except Exception as exc:  # noqa: BLE001
            log.error(
                "backend cleanup failed backend_id=%s error_type=%s",
                adapter.id,
                type(exc).__name__,
            )
        else:
            self._unclosed_adapters = [
                item for item in self._unclosed_adapters if item is not adapter
            ]
