"""Strict JSON boundary validation shared by persisted document codecs."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast


@dataclass(frozen=True, slots=True)
class JsonIssue:
    """One invalid JSON location and its explanation."""

    path: str
    message: str


class JsonValidationError(ValueError):
    """Raised when a persisted document cannot be decoded safely."""

    def __init__(self, issues: list[JsonIssue]) -> None:
        self.issues: tuple[JsonIssue, ...] = tuple(issues)
        super().__init__(
            "; ".join(f"{issue.path}: {issue.message}" for issue in self.issues)
        )


class JsonDecoder:
    """Collect strict type and shape errors while decoding one JSON document."""

    def __init__(self) -> None:
        self.issues: list[JsonIssue] = []

    def issue(self, path: str, message: str) -> None:
        self.issues.append(JsonIssue(path, message))

    def object(
        self,
        raw: object,
        *,
        path: str,
        required: frozenset[str],
        optional: frozenset[str] = frozenset(),
    ) -> dict[str, object]:
        string_values: dict[str, object] = self.mapping(raw, path=path)
        accepted: frozenset[str] = required | optional
        for key in sorted(set(string_values) - accepted):
            self.issue(f"{path}.{key}", "is not a recognised setting")
        for key in sorted(required - set(string_values)):
            self.issue(f"{path}.{key}", "is required")
        return string_values

    def mapping(self, raw: object, *, path: str) -> dict[str, object]:
        """Decode a string-keyed object whose keys are application data."""

        if not isinstance(raw, dict):
            self.issue(path, "must be an object")
            return {}
        values: dict[object, object] = cast(dict[object, object], raw)
        if any(not isinstance(key, str) for key in values):
            self.issue(path, "must use string keys")
        string_values: dict[str, object] = {
            key: value for key, value in values.items() if isinstance(key, str)
        }
        return string_values

    def string(self, raw: object, *, path: str, default: str) -> str:
        if isinstance(raw, str):
            return raw
        self.issue(path, "must be a string")
        return default

    def boolean(self, raw: object, *, path: str, default: bool) -> bool:
        if isinstance(raw, bool):
            return raw
        self.issue(path, "must be a boolean")
        return default

    def optional_boolean(self, raw: object, *, path: str) -> bool | None:
        return None if raw is None else self.boolean(raw, path=path, default=False)

    def optional_string(self, raw: object, *, path: str) -> str | None:
        return None if raw is None else self.string(raw, path=path, default="")

    def choice[T](self, raw: object, *, path: str, choices: tuple[T, ...]) -> T:
        for choice in choices:
            if type(raw) is str and raw == choice:
                return choice
        self.issue(
            path, f"must be one of {', '.join(str(choice) for choice in choices)}"
        )
        return choices[0]

    def construct[T](self, factory: Callable[[], T], *, path: str) -> T:
        """Reject boundary errors before invoking a domain model's validation."""

        if self.issues:
            raise JsonValidationError(self.issues)
        try:
            return factory()
        except (TypeError, ValueError) as exc:
            self.issue(path, str(exc))
            raise JsonValidationError(self.issues) from exc

    def integer(self, raw: object, *, path: str, default: int) -> int:
        if isinstance(raw, int) and not isinstance(raw, bool):
            return raw
        self.issue(path, "must be an integer")
        return default

    def optional_integer(
        self, raw: object, *, path: str, default: int | None
    ) -> int | None:
        if raw is None:
            return None
        return self.integer(
            raw, path=path, default=default if default is not None else 1
        )

    def number(self, raw: object, *, path: str, default: float) -> float:
        if isinstance(raw, int | float) and not isinstance(raw, bool):
            try:
                value: float = float(raw)
            except OverflowError:
                pass
            else:
                if math.isfinite(value):
                    return value
        self.issue(path, "must be a finite number")
        return default

    def list(self, raw: object, *, path: str) -> list[object]:
        if isinstance(raw, list):
            return cast(list[object], raw)
        self.issue(path, "must be a list")
        return []
