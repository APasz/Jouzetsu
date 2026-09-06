"""Validated packaged defaults and complete default-configuration factories."""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import cache
from importlib.resources import files
from types import MappingProxyType
from typing import Final, cast

from ..colors import HEX_COLOR_PATTERN as _HEX_COLOR_PATTERN
from .models import (
    AccessSettings,
    AppConfig,
    GenerationSettings,
    HostStatsSettings,
    LoggingSettings,
    MessageAction,
    ServerSettings,
    SpellingReplacement,
    ThemeColorway,
    ThemeSettings,
    UiSettings,
)
from .paths import AppPaths

_DEFAULT_DATA_PACKAGE: Final[str] = "jouzetsu.config"
_LEGACY_THEME_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "canvas",
        "surface",
        "surface_raised",
        "border",
        "border_strong",
        "text",
        "text_muted",
        "text_inverse",
        "primary",
        "primary_hover",
        "primary_muted",
        "primary_subtle",
        "on_accent",
        "secondary",
        "secondary_muted",
        "secondary_subtle",
        "edit",
        "streaming_highlight",
        "action_delete",
        "action_regenerate",
        "action_delete_muted",
        "action_regenerate_muted",
        "action_merge_muted",
        "action_edit_muted",
        "action_continue_muted",
    }
)


class DefaultDataError(RuntimeError):
    """Raised when a packaged default resource is corrupt or incomplete."""


def _load_json_resource(file_name: str) -> object:
    resource = files(_DEFAULT_DATA_PACKAGE).joinpath("default_data", file_name)
    try:
        return cast(object, json.loads(resource.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise DefaultDataError(
            f"could not load packaged default data: {resource}"
        ) from exc


@cache
def builtin_spelling_replacements() -> tuple[SpellingReplacement, ...]:
    """Return the immutable, validated British-spelling replacement set."""

    raw: object = _load_json_resource("british_spelling_replacements.json")
    if not isinstance(raw, list):
        raise DefaultDataError("British-spelling defaults must be a JSON list")

    replacements: list[SpellingReplacement] = []
    seen_sources: set[str] = set()
    for index, item in enumerate(cast(list[object], raw)):
        if not isinstance(item, dict):
            raise DefaultDataError(
                f"British-spelling default {index} must be an object"
            )
        values: dict[object, object] = cast(dict[object, object], item)
        if set(values) != {"source", "replacement"}:
            raise DefaultDataError(
                f"British-spelling default {index} must contain source and replacement"
            )
        source: object = values["source"]
        replacement: object = values["replacement"]
        if not isinstance(source, str) or not isinstance(replacement, str):
            raise DefaultDataError(
                f"British-spelling default {index} source and replacement must be strings"
            )
        item_value: SpellingReplacement = SpellingReplacement(source, replacement)
        try:
            item_value.validate()
        except ValueError as exc:
            raise DefaultDataError(
                f"invalid British-spelling default {index}: {exc}"
            ) from exc
        source_key: str = source.casefold()
        if source_key in seen_sources:
            raise DefaultDataError(f"duplicate British-spelling source: {source}")
        seen_sources.add(source_key)
        replacements.append(item_value)
    return tuple(replacements)


@cache
def builtin_theme_values() -> Mapping[str, str]:
    """Return default accents, semantic actions, and shared base colours."""

    expected: frozenset[str] = frozenset(
        {
            *(owner.value for owner in ThemeColorway),
            *(action.color_field for action in MessageAction),
            "dark_canvas",
            "dark_text",
            "light_canvas",
            "light_text",
        }
    )
    return _theme_resource_values("theme.json", expected_fields=expected)


def builtin_message_action_color(action: MessageAction) -> str:
    """Return the packaged semantic colour for one message action."""

    return builtin_theme_values()[action.color_field]


@cache
def legacy_theme_values() -> Mapping[str, str]:
    """Return the historical palette only for decoding existing configurations."""

    return _theme_resource_values(
        "legacy-theme.json", expected_fields=_LEGACY_THEME_FIELDS
    )


def _theme_resource_values(
    file_name: str, *, expected_fields: frozenset[str] | None = None
) -> Mapping[str, str]:
    raw: object = _load_json_resource(file_name)
    if not isinstance(raw, dict):
        raise DefaultDataError(f"{file_name} must be a JSON object")
    values = cast(dict[object, object], raw)
    if expected_fields is not None and frozenset(values) != expected_fields:
        raise DefaultDataError(
            f"{file_name} must define every default colour exactly once"
        )
    normalised: dict[str, str] = {}
    for name, value in values.items():
        if (
            not isinstance(name, str)
            or not isinstance(value, str)
            or not _HEX_COLOR_PATTERN.fullmatch(value)
        ):
            raise DefaultDataError(
                f"{file_name} colours must be six-digit hexadecimal values"
            )
        normalised[name] = value.lower()
    return MappingProxyType(normalised)


@cache
def builtin_theme() -> ThemeSettings:
    """Return the immutable packaged default colourways."""

    return ThemeSettings()


def default_config(paths: AppPaths) -> AppConfig:
    """Create a fully valid persisted configuration for one path set."""

    config: AppConfig = AppConfig(
        paths=paths,
        server=ServerSettings(),
        generation=GenerationSettings(
            british_spelling_replacements=list(builtin_spelling_replacements())
        ),
        ui=UiSettings(),
        theme=builtin_theme(),
        logging=LoggingSettings(directory=paths.log_directory),
        host_stats=HostStatsSettings(),
        access=AccessSettings(),
    )
    config.validate()
    return config
