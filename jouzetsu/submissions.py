"""Durable submission receipts independent of later transcript edits."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict

from .json_codec import JsonDecoder


class SubmissionJSON(TypedDict):
    id: str
    message_id: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class Submission:
    id: str
    message_id: str
    fingerprint: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", self.id):
            raise ValueError("invalid submission identifier")
        if not self.message_id:
            raise ValueError("submission requires a message identifier")
        if not re.fullmatch(r"[0-9a-f]{64}", self.fingerprint):
            raise ValueError("invalid submission fingerprint")

    def to_dict(self) -> SubmissionJSON:
        return {
            "id": self.id,
            "message_id": self.message_id,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, raw: object) -> Submission:
        decoder = JsonDecoder()
        values = decoder.object(
            raw,
            path="submission",
            required=frozenset({"id", "message_id", "fingerprint"}),
        )
        fields = tuple(
            decoder.string(values.get(key), path=f"submission.{key}", default="")
            for key in ("id", "message_id", "fingerprint")
        )
        return decoder.construct(lambda: cls(*fields), path="submission")


def submission_fingerprint(
    content: str, images: tuple[Mapping[str, object], ...]
) -> str:
    payload = json.dumps(
        [content, images], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
