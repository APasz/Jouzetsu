"""LM Studio Responses request encoding and streaming event decoding."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Literal, TypedDict, cast

import aiohttp

from .backend_protocol import (
    BackendError,
    BackendErrorCategory,
    GenerationRequest,
    UnsafeContextError,
)
from .chat_completions import (
    build_messages,
    encode_payload,
    iter_sse_data,
    json_object,
    protocol_error,
    reasoning_effort,
    token_count,
)
from .image_storage import ImageStorage
from .models import Role
from .runtime import (
    REASONING_OPTIONS,
    ChatStreamEvent,
    FirstToken,
    GenerationMetrics,
    ModelReady,
    PredictionComplete,
    PredictionFragment,
    ReasoningOption,
)


class InputText(TypedDict):
    type: Literal["input_text"]
    text: str


class InputImage(TypedDict):
    type: Literal["input_image"]
    image_url: str


class InputMessage(TypedDict):
    role: Role
    content: str | list[InputText | InputImage]


def encode_request(
    request: GenerationRequest, ready: ModelReady, images: ImageStorage | None
) -> bytes:
    """Send ordered history and originals without storing a server-side conversation."""
    messages: list[InputMessage] = []
    for message in build_messages(request.messages, images):
        content = message["content"]
        if isinstance(content, str):
            messages.append({"role": message["role"], "content": content})
            continue
        parts: list[InputText | InputImage] = []
        for part in content:
            if part["type"] == "text":
                parts.append({"type": "input_text", "text": part["text"]})
            else:
                parts.append(
                    {"type": "input_image", "image_url": part["image_url"]["url"]}
                )
        messages.append({"role": message["role"], "content": parts})
    parameters: dict[str, float | int] = {
        name: value for name, value in request.wire_parameters.items()
    }
    max_tokens = parameters.pop("max_tokens", None)
    payload: dict[str, object] = {
        "model": ready.instance_id or request.model.model_id,
        "input": messages,
        **parameters,
        "stream": True,
        "store": False,
        "truncation": "disabled",
    }
    if max_tokens is not None:
        payload["max_output_tokens"] = max_tokens
    if request.reasoning is not None:
        payload["reasoning"] = {"effort": reasoning_effort(request.reasoning)}
    return encode_payload(payload)


def _validate_output_item(raw: object) -> None:
    item = json_object(raw)
    kind = item.get("type")
    if kind not in ("message", "reasoning"):
        raise BackendError(
            BackendErrorCategory.INCOMPATIBLE,
            "The backend returned unsupported non-text output.",
        )
    if kind == "message":
        if item.get("role") not in (None, "assistant"):
            raise protocol_error("The backend returned an invalid completion role.")
        content = item.get("content")
        if content is not None:
            if not isinstance(content, list):
                raise protocol_error()
            for raw_part in cast(list[object], content):
                if json_object(raw_part).get("type") != "output_text":
                    raise BackendError(
                        BackendErrorCategory.INCOMPATIBLE,
                        "The backend returned unsupported non-text output.",
                    )


async def response_events(
    response: aiohttp.ClientResponse,
    *,
    ready: ModelReady,
    started_at: float,
    wire_model: str,
    requested_reasoning: ReasoningOption | None,
) -> AsyncGenerator[ChatStreamEvent]:
    """Translate Responses events without treating a reasoning-only stream as text."""
    created = False
    response_id = ""
    first_token_at: float | None = None
    async for data in iter_sse_data(response.content.iter_chunked(4096)):
        if data == "[DONE]":
            break
        try:
            event = json_object(cast(object, json.loads(data)))
        except (TypeError, ValueError) as exc:
            raise protocol_error() from exc
        kind = event.get("type")
        if not isinstance(kind, str):
            raise protocol_error()
        if kind in {"error", "response.failed"}:
            raise protocol_error("The backend reported a streaming error.")
        if kind == "response.created":
            if created:
                raise protocol_error()
            result = json_object(event.get("response"))
            if result.get("model") not in (wire_model, ready.model_key):
                raise protocol_error(
                    "The backend responded with an unexpected model instance."
                )
            identifier = result.get("id")
            if not isinstance(identifier, str) or not identifier.strip():
                raise protocol_error()
            if result.get("truncation") != "disabled":
                raise UnsafeContextError()
            response_id = identifier
            created = True
            yield ready
        elif not created:
            raise protocol_error("The backend did not start a response.")
        elif kind in {"response.output_text.delta", "response.reasoning_text.delta"}:
            delta = event.get("delta")
            if not isinstance(delta, str):
                raise protocol_error()
            if delta:
                if first_token_at is None:
                    first_token_at = asyncio.get_running_loop().time()
                    yield FirstToken()
                yield PredictionFragment(
                    delta, None, kind == "response.reasoning_text.delta"
                )
        elif kind in {"response.output_item.added", "response.output_item.done"}:
            _validate_output_item(event.get("item"))
        elif kind.startswith("response.refusal") or "function_call" in kind:
            raise BackendError(
                BackendErrorCategory.INCOMPATIBLE,
                "The backend returned unsupported non-text output.",
            )
        elif kind in {"response.completed", "response.incomplete"}:
            result = json_object(event.get("response"))
            if result.get("id") != response_id:
                raise protocol_error("The backend changed response identity.")
            if result.get("model") not in (wire_model, ready.model_key):
                raise protocol_error(
                    "The backend responded with an unexpected model instance."
                )
            if result.get("truncation") != "disabled":
                raise UnsafeContextError()
            expected_status = (
                "completed" if kind == "response.completed" else "incomplete"
            )
            if result.get("status") != expected_status:
                raise protocol_error()
            output = result.get("output")
            if not isinstance(output, list):
                raise protocol_error()
            for raw_item in cast(list[object], output):
                _validate_output_item(raw_item)
            usage = (
                json_object(result["usage"]) if result.get("usage") is not None else {}
            )
            output_tokens = token_count(usage, "output_tokens")
            input_tokens = token_count(usage, "input_tokens")
            details = (
                json_object(result["incomplete_details"])
                if result.get("incomplete_details") is not None
                else {}
            )
            reason = details.get("reason")
            if reason is not None and not isinstance(reason, str):
                raise protocol_error()
            reasoning = (
                json_object(result["reasoning"])
                if result.get("reasoning") is not None
                else {}
            )
            effort = reasoning.get("effort")
            reported: ReasoningOption | None = (
                "off"
                if effort == "none"
                else "on"
                if effort == "medium" and requested_reasoning == "on"
                else effort
                if effort in REASONING_OPTIONS
                else None
            )
            now = asyncio.get_running_loop().time()
            duration = now - first_token_at if first_token_at is not None else None
            yield PredictionComplete(
                GenerationMetrics(
                    output_tokens=output_tokens,
                    prompt_tokens=input_tokens,
                    tokens_per_second=output_tokens / duration
                    if output_tokens is not None
                    and duration is not None
                    and duration > 0
                    else None,
                    time_to_first_token_seconds=first_token_at - started_at
                    if first_token_at is not None
                    else None,
                    stop_reason=reason,
                    reasoning_level=reported,
                )
            )
            return
        elif kind not in (
            "response.in_progress",
            "response.content_part.added",
            "response.content_part.done",
            "response.output_text.done",
            "response.reasoning_text.done",
        ):
            raise protocol_error("The backend returned an unsupported streaming event.")
    raise protocol_error("The backend disconnected before completing its response.")
