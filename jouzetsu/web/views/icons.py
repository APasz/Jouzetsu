"""Shared, locally bundled Phosphor icons for the interface."""

from __future__ import annotations

from functools import cache
from hashlib import sha256
from pathlib import Path
from typing import Final, Literal

from fastcore.xml import (  # pyright: ignore[reportMissingTypeStubs]
    FT,
    ft,  # pyright: ignore[reportUnknownVariableType]
)

type HTML = FT
type IconName = Literal[
    "activity",
    "arrow-right",
    "box",
    "review",
    "close",
    "fork",
    "image",
    "lock",
    "menu",
    "merge",
    "minus",
    "more",
    "pencil",
    "play",
    "play-bold",
    "plus",
    "refresh",
    "save",
    "scissors",
    "send",
    "settings",
    "shield",
    "stop",
    "trash",
    "warning",
]

ICON_VIEW_BOX: Final[str] = "0 0 256 256"


@cache
def icon_sprite_url() -> str:
    """Version the shared sprite by content so updated icons bypass browser caches."""

    sprite: Path = Path(__file__).resolve().parents[1] / "static/icons/phosphor.svg"
    version: str = sha256(sprite.read_bytes()).hexdigest()[:12]
    return f"/static/icons/phosphor.svg?v={version}"


def render_icon(
    name: IconName,
    *,
    classes: str = "jouzetsu-svg-icon",
    armed_name: IconName | None = None,
) -> HTML:
    """Render a decorative Phosphor icon, optionally with an armed gesture variant."""

    return ft(
        "svg",
        ft(
            "use",
            href=f"{icon_sprite_url()}#{name}",
            cls="jouzetsu-icon-regular" if armed_name else None,
        ),
        ft(
            "use",
            href=f"{icon_sprite_url()}#{armed_name}",
            cls="jouzetsu-icon-armed",
        )
        if armed_name
        else None,
        viewBox=ICON_VIEW_BOX,
        fill="currentColor",
        stroke="none",
        aria_hidden="true",
        focusable="false",
        cls=classes,
    )
