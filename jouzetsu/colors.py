"""Shared validation and deterministic colour calculations."""

from __future__ import annotations

import re
from typing import Final

HEX_COLOR_PATTERN: Final[re.Pattern[str]] = re.compile(r"#[0-9a-fA-F]{6}")
_TEXT_CONTRAST: Final[float] = 4.5
BLACK: Final[str] = "#000000"
WHITE: Final[str] = "#ffffff"


def mix_colors(start: str, end: str, ratio: float) -> str:
    """Mix validated sRGB colours, where zero keeps start and one selects end."""

    if not 0 <= ratio <= 1:
        raise ValueError("colour mix ratio must be between zero and one")
    channels = (
        round(first + (second - first) * ratio)
        for first, second in zip(_hex_rgb(start), _hex_rgb(end), strict=True)
    )
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def contrast_ratio(first: str, second: str) -> float:
    """Return relative-luminance contrast between two validated colours."""

    low, high = sorted((_luminance(first), _luminance(second)))
    return (high + 0.05) / (low + 0.05)


def contrasting_text(background: str) -> str:
    """Select the neutral text colour with the highest contrast."""

    return max((BLACK, WHITE), key=lambda color: contrast_ratio(color, background))


def readable_accent(accent: str, background: str) -> str:
    """Keep an accent hue while increasing its text contrast when necessary."""

    if contrast_ratio(accent, background) >= _TEXT_CONTRAST:
        return accent
    target = contrasting_text(background)
    low, high = 0.0, 1.0
    for _ in range(12):
        midpoint = (low + high) / 2
        if (
            contrast_ratio(mix_colors(accent, target, midpoint), background)
            >= _TEXT_CONTRAST
        ):
            high = midpoint
        else:
            low = midpoint
    return mix_colors(accent, target, high)


def _hex_rgb(color: str) -> tuple[int, int, int]:
    if not HEX_COLOR_PATTERN.fullmatch(color):
        raise ValueError("colour must be a six-digit hexadecimal value")
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def _luminance(color: str) -> float:
    def linear(channel: int) -> float:
        value = channel / 255
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    red, green, blue = _hex_rgb(color)
    return 0.2126 * linear(red) + 0.7152 * linear(green) + 0.0722 * linear(blue)
