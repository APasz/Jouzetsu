"""Render the configured semantic palette as a compact CSS variable sheet."""

from __future__ import annotations

import re
from typing import Final

from ..config import AppConfig, ThemeSettings
from .host_stats_styles import host_stat_meter_rules

_HEX_COLOR_PATTERN: Final[re.Pattern[str]] = re.compile(r"#[0-9a-fA-F]{6}")
_PALETTE_VARIABLES: Final[tuple[tuple[str, str], ...]] = (
    ("--jouzetsu-bg", "canvas"),
    ("--jouzetsu-surface", "surface"),
    ("--jouzetsu-surface-2", "surface_raised"),
    ("--jouzetsu-border", "border"),
    ("--jouzetsu-border-strong", "border_strong"),
    ("--jouzetsu-text", "text"),
    ("--jouzetsu-text-dim", "text_muted"),
    ("--jouzetsu-text-inverse", "text_inverse"),
    ("--jouzetsu-primary", "primary"),
    ("--jouzetsu-primary-hover", "primary_hover"),
    ("--jouzetsu-primary-dim", "primary_muted"),
    ("--jouzetsu-primary-soft", "primary_subtle"),
    ("--jouzetsu-on-accent", "on_accent"),
    ("--jouzetsu-secondary", "secondary"),
    ("--jouzetsu-secondary-dim", "secondary_muted"),
    ("--jouzetsu-secondary-soft", "secondary_subtle"),
    ("--jouzetsu-edit", "edit"),
    ("--jouzetsu-streaming-highlight", "streaming_highlight"),
    ("--jouzetsu-action-delete", "action_delete"),
    ("--jouzetsu-action-regenerate", "action_regenerate"),
    ("--jouzetsu-action-delete-muted", "action_delete_muted"),
    ("--jouzetsu-action-regenerate-muted", "action_regenerate_muted"),
    ("--jouzetsu-action-merge-muted", "action_merge_muted"),
    ("--jouzetsu-action-edit-muted", "action_edit_muted"),
    ("--jouzetsu-action-continue-muted", "action_continue_muted"),
)
_DERIVED_VARIABLES: Final[tuple[tuple[str, str], ...]] = (
    ("--jouzetsu-edit-glow", "rgb(from var(--jouzetsu-edit) r g b / 24%)"),
    ("--jouzetsu-edit-focus", "rgb(from var(--jouzetsu-edit) r g b / 16%)"),
    ("--jouzetsu-secondary-streaming-glow", "rgb(from var(--jouzetsu-secondary) r g b / 30%)"),
    ("--jouzetsu-backdrop", "rgb(from var(--jouzetsu-bg) r g b / 88%)"),
    ("--jouzetsu-shadow-soft", "rgb(from var(--jouzetsu-bg) r g b / 32%)"),
    ("--jouzetsu-shadow-panel", "rgb(from var(--jouzetsu-bg) r g b / 70%)"),
    ("--jouzetsu-shadow-meter", "rgb(from var(--jouzetsu-bg) r g b / 65%)"),
    ("--jouzetsu-shadow-notice", "rgb(from var(--jouzetsu-bg) r g b / 60%)"),
)


def render_theme_css(config: AppConfig) -> str:
    """Return the complete runtime stylesheet derived from validated configuration."""

    palette: ThemeSettings = config.theme
    palette.validate()
    palette_values: dict[str, str] = palette.values()
    start: str = _safe_hex_color(config.host_stats.activity_start_color, palette.canvas)
    end: str = _safe_hex_color(config.host_stats.activity_end_color, palette.primary)
    declarations: list[str] = [
        *(f"    {variable}: {palette_values[field]};" for variable, field in _PALETTE_VARIABLES),
        *(f"    {variable}: {value};" for variable, value in _DERIVED_VARIABLES),
        f"    --jouzetsu-stat-start: {start};",
        f"    --jouzetsu-stat-end: {end};",
    ]
    return ":root {\n" + "\n".join(declarations) + "\n}\n" + host_stat_meter_rules(start, end) + "\n"


def _safe_hex_color(value: str, fallback: str) -> str:
    """Return a CSS-safe configured colour, or the configured palette fallback."""

    return value if _HEX_COLOR_PATTERN.fullmatch(value) else fallback
