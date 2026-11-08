"""Shared, strict streaming Chat Completions transport.

Only this module interprets the wire stream. Prompts, scheduling, persistence,
and model loading belong to the application and optional native adapter.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Sequence
from contextlib import aclosing
from dataclasses import replace
from typing import Final, Literal, TypedDict, cast

import aiohttp

from .async_workers import run_in_worker
from .backend_protocol import (
    BackendError,
    BackendErrorCategory,
    GenerationRequest,
    InferenceMessage,
)
from .image_storage import ImageStorage
from .models import Role
from .runtime import (
    ChatStreamEvent,
    FirstToken,
    GenerationMetrics,
    ModelReady,
    PredictionComplete,
    PredictionFragment,
)

CONNECT_TIMEOUT: Final[float] = 10.0
DISCOVERY_TIMEOUT: Final[float] = 15.0
LOAD_TIMEOUT: Final[float] = 600.0
UNLOAD_TIMEOUT: Final[float] = 60.0
STREAM_IDLE_TIMEOUT: Final[float] = 120.0
_MAX_EVENT_BYTES: Final[int] = 1024 * 1024


class TextPart(TypedDict):
    type: Literal["text"]
    text: str


class ImageURL(TypedDict):
    url: str


class ImagePart(TypedDict):
    type: Literal["image_url"]
    image_url: ImageURL


class WireMessage(TypedDict):
    role: Role
    content: str | list[TextPart | ImagePart]


def protocol_error(
    message: str = "The backend returned an invalid streaming response.",
) -> BackendError:
    return BackendError(BackendErrorCategory.PROTOCOL, message)


def json_object(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict) or any(
        not isinstance(key, str) for key in cast(dict[object, object], raw)
    ):
        raise protocol_error()
    return cast(dict[str, object], raw)


async def iter_sse_data(chunks: AsyncIterable[bytes]) -> AsyncIterator[str]:
    """Frame bounded SSE events across UTF-8, CR, LF and CRLF boundaries."""
    buffer = b""
    lines: list[bytes] = []
    event_size = 0
    first_line = True
    skip_lf = False
    async for chunk in chunks:
        if not chunk:
            continue
        if skip_lf:
            chunk = chunk.removeprefix(b"\n")
            skip_lf = False
        buffer += chunk
        while match := re.search(rb"[\r\n]", buffer):
            offset = match.start()
            line, delimiter, buffer = (
                buffer[:offset],
                buffer[offset : offset + 1],
                buffer[offset + 1 :],
            )
            if delimiter == b"\r":
                if buffer:
                    buffer = buffer.removeprefix(b"\n")
                else:
                    skip_lf = True
            if first_line:
                line = line.removeprefix(b"\xef\xbb\xbf")
                first_line = False
            event_size += len(line) + 1
            if event_size > _MAX_EVENT_BYTES:
                raise protocol_error(
                    "A backend streaming event exceeded the size limit."
                )
            if not line:
                if lines:
                    try:
                        yield b"\n".join(lines).decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise protocol_error() from exc
                lines.clear()
                event_size = 0
            else:
                field, _, value = line.partition(b":")
                if field == b"data":
                    lines.append(value.removeprefix(b" "))
        if len(buffer) + event_size > _MAX_EVENT_BYTES:
            raise protocol_error("A backend streaming event exceeded the size limit.")
    if buffer or lines:
        raise protocol_error("The backend disconnected during a streaming event.")


def build_messages(
    messages: Sequence[InferenceMessage], images: ImageStorage | None
) -> list[WireMessage]:
    """Keep every role and image in order; never merge a backend's input turns."""

    result: list[WireMessage] = []
    image_urls: dict[str, str] = {}
    for message in messages:
        if not message.images:
            result.append({"role": message.role, "content": message.content})
            continue
        if images is None:
            raise BackendError(
                BackendErrorCategory.INCOMPATIBLE, "Image storage is unavailable."
            )
        parts: list[TextPart | ImagePart] = []
        if message.content:
            parts.append(TextPart(type="text", text=message.content))
        for attachment in message.images:
            url = image_urls.get(attachment.id)
            if url is None:
                try:
                    url = images.data_url(attachment)
                except (OSError, ValueError) as exc:
                    raise BackendError(
                        BackendErrorCategory.INCOMPATIBLE,
                        "An image original is missing or damaged.",
                    ) from exc
                image_urls[attachment.id] = url
            parts.append(ImagePart(type="image_url", image_url={"url": url}))
        result.append({"role": message.role, "content": parts})
    return result


def _token_count(usage: dict[str, object], name: str) -> int | None:
    value = usage.get(name)
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise protocol_error()
    return value


async def completion_events(
    response: aiohttp.ClientResponse,
    *,
    ready: ModelReady,
    started_at: float,
    wire_model: str,
    native: bool = False,
) -> AsyncGenerator[ChatStreamEvent]:
    first_token_at: float | None = None
    output_tokens: int | None = None
    prompt_tokens: int | None = None
    stop_reason: str | None = None
    completed = False
    ready_emitted = False
    reported_model: str | None = None
    async for data in iter_sse_data(response.content.iter_chunked(4096)):
        if data == "[DONE]":
            completed = True
            break
        try:
            chunk = json_object(cast(object, json.loads(data)))
        except (ValueError, TypeError) as exc:
            raise protocol_error() from exc
        if "error" in chunk:
            raise protocol_error("The backend reported a streaming error.")
        model = chunk.get("model")
        if model is not None:
            if (
                not isinstance(model, str)
                or not model.strip()
                or model != model.strip()
            ):
                raise protocol_error()
            if reported_model is not None and reported_model != model:
                raise protocol_error(
                    "The backend changed model identity during a response."
                )
            if native and model not in {wire_model, ready.model_key}:
                raise protocol_error(
                    "The backend responded with an unexpected model instance."
                )
            reported_model = model
        if not ready_emitted:
            if not native and reported_model is not None:
                ready = replace(
                    ready, model_key=reported_model, model_name=reported_model
                )
            yield ready
            ready_emitted = True
        elif (
            not native
            and reported_model is not None
            and reported_model != ready.model_key
        ):
            raise protocol_error(
                "The backend reported model identity after starting its response."
            )
        raw_usage = chunk.get("usage")
        if raw_usage is not None:
            usage = json_object(raw_usage)
            output_tokens = _token_count(usage, "completion_tokens")
            prompt_tokens = _token_count(usage, "prompt_tokens")
        raw_choices = chunk.get("choices")
        if not isinstance(raw_choices, list):
            raise protocol_error()
        choices = cast(list[object], raw_choices)
        if len(choices) > 1:
            raise protocol_error()
        for raw_choice in choices:
            if stop_reason is not None:
                raise protocol_error(
                    "The backend returned a choice after it had finished."
                )
            choice = json_object(raw_choice)
            index = choice.get("index", 0)
            if type(index) is not int or index != 0:
                raise protocol_error("Multiple completion choices are unsupported.")
            finish = choice.get("finish_reason")
            if finish is not None:
                if not isinstance(finish, str) or not finish.strip():
                    raise protocol_error()
                if finish in {"tool_calls", "function_call"}:
                    raise BackendError(
                        BackendErrorCategory.INCOMPATIBLE,
                        "The backend requested unsupported tool execution.",
                    )
                stop_reason = finish
            delta = json_object(choice.get("delta"))
            if delta.get("role") is not None and delta.get("role") != "assistant":
                raise protocol_error("The backend returned an invalid completion role.")
            if any(
                delta.get(key) is not None
                for key in ("tool_calls", "function_call", "refusal")
            ):
                raise BackendError(
                    BackendErrorCategory.INCOMPATIBLE,
                    "The backend returned unsupported non-text output.",
                )
            reasoning = delta.get("reasoning_content")
            if reasoning is None:
                reasoning = delta.get("reasoning")
            for content, is_reasoning in (
                (reasoning, True),
                (delta.get("content"), False),
            ):
                if content is None:
                    continue
                if not isinstance(content, str):
                    raise protocol_error()
                if content:
                    if first_token_at is None:
                        first_token_at = asyncio.get_running_loop().time()
                        yield FirstToken()
                    yield PredictionFragment(content, None, is_reasoning)
    if not completed and stop_reason is None:
        raise protocol_error("The backend disconnected before completing its response.")
    now = asyncio.get_running_loop().time()
    duration = now - first_token_at if first_token_at is not None else None
    yield PredictionComplete(
        GenerationMetrics(
            output_tokens=output_tokens,
            prompt_tokens=prompt_tokens,
            tokens_per_second=output_tokens / duration
            if output_tokens is not None and duration is not None and duration > 0
            else None,
            time_to_first_token_seconds=first_token_at - started_at
            if first_token_at is not None
            else None,
            stop_reason=stop_reason,
        )
    )


def _encode_request(
    request: GenerationRequest, ready: ModelReady, images: ImageStorage | None
) -> bytes:
    """Prepare potentially large prompts and image payloads off the event loop."""
    payload: dict[str, object] = {
        "model": ready.instance_id or request.model.model_id,
        "messages": build_messages(request.messages, images),
        **request.wire_parameters,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    try:
        return json.dumps(
            payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BackendError(
            BackendErrorCategory.INCOMPATIBLE,
            "The prepared request could not be encoded.",
        ) from exc


class ChatCompletionsTransport:
    """One owned HTTP session per configured backend, with safe error boundaries."""

    def __init__(self, base_url: str, images: ImageStorage | None = None) -> None:
        self.base_url: str = base_url.rstrip("/")
        self.images: ImageStorage | None = images
        self._session: aiohttp.ClientSession | None = None
        self._closed: bool = False

    async def session(self) -> aiohttp.ClientSession:
        if self._closed:
            raise RuntimeError("backend transport is closed")
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def json_request(
        self,
        method: str,
        url: str,
        *,
        payload: dict[str, object] | None = None,
        timeout: float = DISCOVERY_TIMEOUT,
        unsupported: frozenset[int] = frozenset(),
        failure: BackendErrorCategory = BackendErrorCategory.PROTOCOL,
    ) -> dict[str, object] | None:
        try:
            session = await self.session()
            async with session.request(
                method,
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    total=timeout, sock_connect=CONNECT_TIMEOUT
                ),
                allow_redirects=False,
            ) as response:
                if response.status in unsupported:
                    return None
                if response.status != 200:
                    raise BackendError(
                        failure, f"The backend returned HTTP {response.status}."
                    )
                return json_object(cast(object, await response.json(content_type=None)))
        except TimeoutError as exc:
            raise BackendError(
                BackendErrorCategory.TIMEOUT, "The backend request timed out."
            ) from exc
        except aiohttp.ClientConnectionError as exc:
            raise BackendError(
                BackendErrorCategory.OFFLINE, "The backend is unreachable."
            ) from exc
        except (aiohttp.ClientError, ValueError, TypeError) as exc:
            raise protocol_error() from exc

    async def stream(
        self, request: GenerationRequest, ready: ModelReady, *, native: bool = False
    ) -> AsyncGenerator[ChatStreamEvent]:
        try:
            payload = await run_in_worker(_encode_request, request, ready, self.images)
            wire_model = ready.instance_id or request.model.model_id
            session = await self.session()
            started = asyncio.get_running_loop().time()
            async with session.post(
                f"{self.base_url}/chat/completions",
                data=payload,
                headers={
                    "Accept": "text/event-stream",
                    "Content-Type": "application/json",
                },
                allow_redirects=False,
                timeout=aiohttp.ClientTimeout(
                    total=None,
                    sock_connect=CONNECT_TIMEOUT,
                    sock_read=STREAM_IDLE_TIMEOUT,
                ),
            ) as response:
                if response.status != 200:
                    raise BackendError(
                        BackendErrorCategory.INCOMPATIBLE
                        if response.status in {400, 404, 422}
                        else BackendErrorCategory.PROTOCOL,
                        f"The backend rejected generation (HTTP {response.status}).",
                    )
                if response.content_type != "text/event-stream":
                    raise protocol_error(
                        "The backend did not return a streaming response."
                    )
                async with aclosing(
                    completion_events(
                        response,
                        ready=ready,
                        started_at=started,
                        wire_model=wire_model,
                        native=native,
                    )
                ) as events:
                    async for event in events:
                        yield event
        except TimeoutError as exc:
            raise BackendError(
                BackendErrorCategory.TIMEOUT, "The backend stopped responding."
            ) from exc
        except aiohttp.ClientConnectionError as exc:
            raise BackendError(
                BackendErrorCategory.OFFLINE, "The backend connection was lost."
            ) from exc
        except aiohttp.ClientError as exc:
            raise protocol_error() from exc

    async def close(self) -> None:
        self._closed = True
        if self._session is not None:
            await self._session.close()
