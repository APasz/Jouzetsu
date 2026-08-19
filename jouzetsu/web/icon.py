"""Helpers for serving the app icon with its configured colour palette."""

from __future__ import annotations

from pathlib import Path
from typing import Final
from urllib.parse import urlencode

from ..config import IconColorSettings

ICON_FILENAME: Final[str] = "icon.svg"
FALLBACK_ICON_FILENAME: Final[str] = "fallback-icon.svg"
_LINEWORK_MARKER: Final[str] = "{{jouzetsu-icon-linework}}"
_ACCENT_MARKER: Final[str] = "{{jouzetsu-icon-accent}}"
_SURFACE_MARKER: Final[str] = "{{jouzetsu-icon-surface}}"


def icon_artwork_path(asset_directory: Path) -> Path:
    """Return the private icon when installed, otherwise the release-safe artwork."""

    private_icon: Path = asset_directory / ICON_FILENAME
    if private_icon.is_file():
        return private_icon
    return asset_directory / FALLBACK_ICON_FILENAME


def app_icon_url(colors: IconColorSettings) -> str:
    """Return a palette-specific URL so browsers refresh the icon after a change."""

    colors.validate()
    query: str = urlencode(
        {
            "linework": colors.linework_color,
            "accent": colors.accent_color,
            "surface": colors.surface_color,
        }
    )
    return f"/icon.svg?{query}"


def configured_icon_svg(path: Path, colors: IconColorSettings) -> str:
    """Apply validated colour settings to the canonical SVG artwork."""

    colors.validate()
    source: str = path.read_text(encoding="utf-8")
    return (
        source.replace(_LINEWORK_MARKER, colors.linework_color)
        .replace(_ACCENT_MARKER, colors.accent_color)
        .replace(_SURFACE_MARKER, colors.surface_color)
    )
