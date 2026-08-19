from __future__ import annotations

from jouzetsu.continuity import (
    CONTINUITY_REVIEW_PROMPT,
    ContinuityReviewDecision,
    build_continuity_review_chat,
    parse_continuity_review,
)
from jouzetsu.models import Chat


def test_continuity_review_request_is_ephemeral_and_appends_the_protocol_prompt() -> None:
    chat = Chat()
    _ = chat.add_message("user", "Original request")

    review_chat = build_continuity_review_chat(chat)

    assert review_chat is not chat
    assert len(chat.messages) == 1
    assert review_chat.messages[-1].role == "user"
    assert review_chat.messages[-1].content == CONTINUITY_REVIEW_PROMPT


def test_continuity_review_accepts_only_the_strict_keep_or_complete_replace_protocol() -> None:
    keep = parse_continuity_review("  KEEP\n")
    replacement = parse_continuity_review("REPLACE\r\nRevised answer")
    invalid = parse_continuity_review("REPLACE\n")

    assert keep.decision is ContinuityReviewDecision.KEEP
    assert replacement.decision is ContinuityReviewDecision.REPLACE
    assert replacement.replacement == "Revised answer"
    assert invalid.decision is ContinuityReviewDecision.INVALID
