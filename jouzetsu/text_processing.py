"""Text post-processing helpers."""

from __future__ import annotations

import re
from collections.abc import Sequence

from .config import SpellingReplacement


def apply_case_preserving_word_replacements(
    text: str,
    replacements: Sequence[SpellingReplacement],
) -> str:
    """Apply case-insensitive whole-word replacements while preserving casing."""
    if not text or not replacements:
        return text

    replacement_by_source: dict[str, str] = {
        replacement.source.casefold(): replacement.replacement
        for replacement in replacements
    }
    alternatives: list[str] = sorted(
        (re.escape(item.source) for item in replacements), key=len, reverse=True
    )
    pattern: re.Pattern[str] = re.compile(
        r"(?<![A-Za-z0-9_])(" + "|".join(alternatives) + r")(?![A-Za-z0-9_])",
        re.IGNORECASE,
    )

    def replace_match(match: re.Match[str]) -> str:
        source_text: str = match.group(1)
        replacement: str = replacement_by_source[source_text.casefold()]
        return _apply_source_case(source_text, replacement)

    return pattern.sub(replace_match, text)


def _apply_source_case(source_text: str, replacement: str) -> str:
    if source_text.isupper():
        return replacement.upper()
    if source_text.islower():
        return replacement.lower()
    if source_text[:1].isupper() and source_text[1:].islower():
        return replacement[:1].upper() + replacement[1:].lower()
    return _apply_character_case_pattern(source_text, replacement)


def _apply_character_case_pattern(source_text: str, replacement: str) -> str:
    chars: list[str] = []
    for index, char in enumerate(replacement):
        if index < len(source_text) and source_text[index].isupper():
            chars.append(char.upper())
        else:
            chars.append(char.lower())
    return "".join(chars)
