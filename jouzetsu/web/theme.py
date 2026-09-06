"""Derive neutral app surfaces and four independently owned colourways."""

from __future__ import annotations

from typing import Final

from ..colors import BLACK, contrasting_text, mix_colors, readable_accent
from ..config import (
    AppColorwaySettings,
    AppConfig,
    ColorwaySettings,
    MessageAction,
    ThemeColorway,
)
from ..config.defaults import builtin_message_action_color, builtin_theme_values
from .host_stats_styles import host_stat_meter_rules

_MESSAGE_SHADES: Final[tuple[str, ...]] = (
    "accent",
    "muted",
    "subtle",
    "hover",
    "foreground",
    "icon",
    "glow",
    "focus",
    "highlight",
    "body",
)
_CONTROL_SHADES: Final[tuple[str, ...]] = (
    "accent",
    "muted",
    "subtle",
    "hover",
    "foreground",
    "on-accent",
    "on-hover",
)
_SHADOW_OPACITIES: Final[tuple[tuple[str, int], ...]] = (
    ("backdrop", 88),
    ("shadow-soft", 32),
    ("shadow-panel", 70),
    ("shadow-meter", 65),
    ("shadow-notice", 60),
)


def render_theme_css(config: AppConfig) -> str:
    """Generate base surfaces, role shades, and message-local colour assignments."""

    config.host_stats.validate()
    surfaces, colourways = _resolved_theme_colours(config)
    variables: dict[str, str] = {
        f"--jouzetsu-app-{name.replace('_', '-')}": value
        for name, value in surfaces.items()
    }
    variables.update(
        (
            f"--jouzetsu-app-{action.color_field.replace('_', '-')}",
            _message_action_color(config.theme.app, action, surface=surfaces["canvas"]),
        )
        for action in MessageAction
    )
    for owner, shades in colourways.items():
        variables.update(
            (f"--jouzetsu-{owner.value}-{name}", value)
            for name, value in shades.items()
        )
    variables.update(
        (
            f"--jouzetsu-app-{name}",
            f"rgb(from var(--jouzetsu-app-canvas) r g b / {opacity}%)",
        )
        for name, opacity in _SHADOW_OPACITIES
    )
    start = config.host_stats.activity_start_color
    end = config.host_stats.activity_end_color
    variables["--jouzetsu-stat-start"] = start
    variables["--jouzetsu-stat-end"] = end
    declarations = [f"    {name}: {value};" for name, value in variables.items()]
    declarations.append(
        f"    color-scheme: {'dark' if config.ui.dark_mode else 'light'};"
    )
    rules = [":root {\n" + "\n".join(declarations) + "\n}"]
    rules.extend(
        (
            _scope_colorway(":root", "control", ThemeColorway.APP, _CONTROL_SHADES),
            _scope_colorway(
                ".jouzetsu-composer", "control", ThemeColorway.USER, _CONTROL_SHADES
            ),
        )
    )
    for owner in ThemeColorway:
        if owner is ThemeColorway.APP:
            continue
        rules.append(
            _scope_colorway(
                f".jouzetsu-message.is-{owner.value}", "message", owner, _MESSAGE_SHADES
            )
        )
    rules.append(host_stat_meter_rules(start, end))
    return "\n".join(rules) + "\n"


def resolved_theme_colours(config: AppConfig) -> dict[ThemeColorway, dict[str, str]]:
    """Return the effective colour behind each configurable theme setting."""

    surfaces, colourways = _resolved_theme_colours(config)
    resolved: dict[ThemeColorway, dict[str, str]] = {
        owner: dict(shades) for owner, shades in colourways.items()
    }
    resolved[ThemeColorway.APP].update(surfaces)
    return resolved


def _resolved_theme_colours(
    config: AppConfig,
) -> tuple[dict[str, str], dict[ThemeColorway, dict[str, str]]]:
    """Resolve App surfaces and colourway shades from one validated configuration."""

    config.theme.validate()
    surfaces = _app_surfaces(config.theme.app, dark_mode=config.ui.dark_mode)
    colourways = {
        owner: _colorway_shades(
            settings,
            background=(
                surfaces["surface"]
                if owner is ThemeColorway.APP
                else surfaces["canvas"]
            ),
            text=surfaces["text"],
        )
        for owner, settings in config.theme.colorways()
    }
    return surfaces, colourways


def _scope_colorway(
    selector: str, scope: str, owner: ThemeColorway, shades: tuple[str, ...]
) -> str:
    aliases = "\n".join(
        f"    --jouzetsu-{scope}-{shade}: var(--jouzetsu-{owner.value}-{shade});"
        for shade in shades
    )
    return f"{selector} {{\n{aliases}\n}}"


def _app_surfaces(settings: AppColorwaySettings, *, dark_mode: bool) -> dict[str, str]:
    defaults = builtin_theme_values()
    mode = "dark" if dark_mode else "light"
    canvas = settings.canvas or defaults[f"{mode}_canvas"]
    surface = settings.surface or canvas
    text = settings.text or readable_accent(defaults[f"{mode}_text"], surface)
    key_visual = settings.key_visual or settings.accent
    message_actions = settings.message_actions or key_visual
    return {
        "canvas": canvas,
        "surface": surface,
        "surface_raised": settings.surface_raised or mix_colors(surface, text, 0.06),
        "border": settings.border or mix_colors(surface, text, 0.2),
        "border_strong": settings.border_strong or mix_colors(surface, text, 0.35),
        "text": text,
        "text_muted": settings.text_muted or text,
        "key_visual": key_visual,
        "message_actions": readable_accent(message_actions, canvas),
    }


def _message_action_color(
    settings: AppColorwaySettings, action: MessageAction, *, surface: str
) -> str:
    """Resolve one semantic action colour with accessible contrast on its surface."""

    configured: str | None = getattr(settings, action.color_field)
    return readable_accent(configured or builtin_message_action_color(action), surface)


def _colorway_shades(
    settings: ColorwaySettings, *, background: str, text: str
) -> dict[str, str]:
    accent = settings.accent
    neutral = contrasting_text(background)
    is_app = isinstance(settings, AppColorwaySettings)
    hover = settings.hover or mix_colors(accent, BLACK, 0.12)
    subtle = settings.subtle or mix_colors(
        background, text if is_app else accent, 0.06 if is_app else 0.09
    )
    muted = settings.muted or mix_colors(
        background, text if is_app else accent, 0.2 if is_app else 0.5
    )
    foreground = readable_accent(text, subtle)
    return {
        "accent": accent,
        "muted": muted,
        "subtle": subtle,
        "hover": hover,
        "foreground": foreground,
        "body": foreground,
        "on-accent": contrasting_text(accent),
        "on-hover": contrasting_text(hover),
        "icon": accent,
        "highlight": mix_colors(accent, neutral, 0.7),
        "glow": f"rgb(from {accent} r g b / 24%)",
        "focus": f"rgb(from {accent} r g b / 16%)",
    }
