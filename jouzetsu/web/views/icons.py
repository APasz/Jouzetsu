"""The small, local icon set used throughout the interface."""

from __future__ import annotations

from typing import Final, Literal

from fastcore.xml import (  # pyright: ignore[reportMissingTypeStubs]
    FT,
    ft,  # pyright: ignore[reportUnknownVariableType]
)

type HTML = FT
type IconName = Literal[
    "activity",
    "box",
    "review",
    "close",
    "fork",
    "menu",
    "merge",
    "more",
    "pencil",
    "play",
    "plus",
    "refresh",
    "save",
    "scissors",
    "send",
    "settings",
    "shield",
    "stop",
    "trash",
]

_PATHS: Final[dict[IconName, tuple[str, ...]]] = {
    "activity": ("M3 12h4l3-8 4 16 3-8h4",),
    "box": ("m21 16-9 5-9-5V8l9-5 9 5Z", "M3.3 7.8 12 13l8.7-5.2", "M12 22V13"),
    "review": (
        "M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14Z",
        "m16 16 4 4",
        "M8 11h6",
        "M11 8v6",
    ),
    "close": ("m18 6-12 12", "m6 6 12 12"),
    "fork": (
        "M6 3v12",
        "M18 6a3 3 0 0 0-3 3v1a3 3 0 0 1-3 3H6",
        "M18 18a3 3 0 1 0 0 6 3 3 0 0 0 0-6",
    ),
    "menu": ("M4 6h16", "M4 12h16", "M4 18h16"),
    "merge": (
        "M6 3v6a3 3 0 0 0 3 3h6a3 3 0 0 1 3 3v6",
        "m14 18 4 3 4-3",
        "M18 3v3a3 3 0 0 1-3 3H9a3 3 0 0 0-3 3v6",
        "m10 6-4 3-4-3",
    ),
    "more": ("M5 12h.01", "M12 12h.01", "M19 12h.01"),
    "pencil": ("M12 20h9", "M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"),
    "play": ("m5 3 14 9-14 9Z",),
    "plus": ("M12 5v14", "M5 12h14"),
    "refresh": (
        "M21 12a9 9 0 0 1-15.5 6.2L3 16",
        "M3 21v-5h5",
        "M3 12a9 9 0 0 1 15.5-6.2L21 8",
        "M21 3v5h-5",
    ),
    "save": (
        "M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2Z",
        "M17 21v-8H7v8",
        "M7 3v5h8",
    ),
    "scissors": (
        "M6 8a3 3 0 1 0 0 6 3 3 0 0 0 0-6",
        "M6 14a3 3 0 1 0 0 6 3 3 0 0 0 0-6",
        "m8.6 8.6 12.8 12.8",
        "m8.6 15.4 12.8-12.8",
    ),
    "send": ("m22 2-7 20-4-9-9-4Z", "M22 2 11 13"),
    "settings": (
        "M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7",
        "M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1-2 2-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.5v.2h-2.8v-.2a1.7 1.7 0 0 0-1-1.5 1.7 1.7 0 0 0-1.9.3l-.1.1-2-2 .1-.1a1.7 1.7 0 0 0 .3-1.9 1.7 1.7 0 0 0-1.5-1H5.7v-2.8h.2a1.7 1.7 0 0 0 1.5-1 1.7 1.7 0 0 0-.3-1.9L7 8.2l2-2 .1.1a1.7 1.7 0 0 0 1.9.3 1.7 1.7 0 0 0 1 1.5v.2h2.8v.2a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.9-.3l.1-.1 2 2-.1.1a1.7 1.7 0 0 0 .3-1.9 1.7 1.7 0 0 0 1.5-1H5.7v-2.8h.2a1.7 1.7 0 0 0 1.5-1Z",
    ),
    "shield": ("M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10",),
    "stop": ("M6 6h12v12H6Z",),
    "trash": ("M3 6h18", "M8 6V4h8v2", "M19 6l-1 14H6L5 6", "M10 11v6", "M14 11v6"),
}


def render_icon(name: IconName, *, classes: str = "jouzetsu-svg-icon") -> HTML:
    """Render a compact current-color line icon from the shared local set."""

    return ft(
        "svg",
        *(ft("path", d=path) for path in _PATHS[name]),
        viewBox="0 0 24 24",
        fill="none",
        stroke="currentColor",
        stroke_width="2",
        stroke_linecap="round",
        stroke_linejoin="round",
        aria_hidden="true",
        cls=classes,
    )
