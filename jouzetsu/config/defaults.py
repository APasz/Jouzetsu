"""Validated packaged defaults and complete default-configuration factories."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import fields
from functools import cache
from importlib.resources import files
from types import MappingProxyType
from typing import Final, cast

from .models import (
    AccessSettings,
    AppConfig,
    GenerationSettings,
    HostStatsSettings,
    LoggingSettings,
    ServerSettings,
    SpellingReplacement,
    ThemeSettings,
    UiSettings,
)
from .paths import AppPaths

_DEFAULT_DATA_PACKAGE: Final[str] = "jouzetsu.config"
_HEX_COLOR_PATTERN: re.Pattern[str] = re.compile(r"#[0-9a-fA-F]{6}")


class DefaultDataError(RuntimeError):
    """Raised when a packaged default resource is corrupt or incomplete."""


def _load_json_resource(file_name: str) -> object:
    resource = files(_DEFAULT_DATA_PACKAGE).joinpath("default_data", file_name)
    try:
        return cast(object, json.loads(resource.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise DefaultDataError(f"could not load packaged default data: {resource}") from exc


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
            raise DefaultDataError(f"British-spelling default {index} must be an object")
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
            raise DefaultDataError(f"invalid British-spelling default {index}: {exc}") from exc
        source_key: str = source.casefold()
        if source_key in seen_sources:
            raise DefaultDataError(f"duplicate British-spelling source: {source}")
        seen_sources.add(source_key)
        replacements.append(item_value)
    return tuple(replacements)


@cache
def builtin_theme_values() -> Mapping[str, str]:
    """Return the immutable, validated semantic theme-token mapping."""

    raw: object = _load_json_resource("theme.json")
    if not isinstance(raw, dict):
        raise DefaultDataError("theme defaults must be a JSON object")
    values: dict[object, object] = cast(dict[object, object], raw)
    expected_fields: frozenset[str] = frozenset(field.name for field in fields(ThemeSettings))
    configured_fields: frozenset[str] = frozenset(
        key for key in values if isinstance(key, str)
    )
    if len(configured_fields) != len(values) or configured_fields != expected_fields:
        raise DefaultDataError("theme defaults must define every theme token exactly once")

    normalised: dict[str, str] = {}
    for field_name in expected_fields:
        value: object = values[field_name]
        if not isinstance(value, str) or not _HEX_COLOR_PATTERN.fullmatch(value):
            raise DefaultDataError(
                f"theme default {field_name} must be a six-digit hexadecimal color"
            )
        normalised[field_name] = value.lower()
    return MappingProxyType(normalised)


@cache
def builtin_theme() -> ThemeSettings:
    """Return the immutable packaged default theme."""

    return ThemeSettings(**dict(builtin_theme_values()))


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
