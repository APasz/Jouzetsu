"""Typed helpers for the optional post-generation continuity review."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from .models import Chat, Message

CONTINUITY_REVIEW_PROMPT: str = """Review the immediately preceding assistant response for continuity with this conversation.

Check only for concrete conversational problems: contradictions with earlier messages, failure to answer the latest user request, ignored explicit constraints, incorrect self-references, or an abrupt incomplete ending. Do not fact-check, add new information, or change the response merely to make it different.

If no correction is needed, respond with exactly:
KEEP

If a correction is needed, respond with exactly `REPLACE` on the first line, followed by the complete replacement response. Preserve useful content, use the original response's language and style, and do not mention this review or explain your decision."""


class ContinuityReviewDecision(Enum):
    """The limited, safe set of actions a continuity review may request."""

    KEEP = "keep"
    REPLACE = "replace"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class ContinuityReviewResult:
    """Parsed review output, with a replacement only for an explicit request."""

    decision: ContinuityReviewDecision
    replacement: str = ""


def build_continuity_review_chat(chat: Chat) -> Chat:
    """Create an ephemeral reviewer turn without mutating the stored transcript."""

    return replace(
        chat,
        messages=[
            *chat.messages,
            Message(role="user", content=CONTINUITY_REVIEW_PROMPT),
        ],
    )


def parse_continuity_review(content: str) -> ContinuityReviewResult:
    """Accept only the strict review protocol, preserving the draft otherwise."""

    normalized: str = content.strip().replace("\r\n", "\n")
    if normalized == "KEEP":
        return ContinuityReviewResult(ContinuityReviewDecision.KEEP)
    prefix: str = "REPLACE\n"
    if normalized.startswith(prefix):
        replacement: str = normalized.removeprefix(prefix).strip()
        if replacement:
            return ContinuityReviewResult(ContinuityReviewDecision.REPLACE, replacement)
    return ContinuityReviewResult(ContinuityReviewDecision.INVALID)
