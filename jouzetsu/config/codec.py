"""Strict, versioned JSON codec for persisted application configuration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Final, cast

from ..backend_codec import decode_backends, encode_backends
from ..backend_models import BackendConfiguration
from ..colors import HEX_COLOR_PATTERN
from ..json_codec import (
    JsonDecoder as _Decoder,
    JsonIssue as ConfigIssue,
    JsonValidationError as ConfigValidationError,
)
from .defaults import (
    builtin_spelling_replacements,
    builtin_theme,
    default_config,
    legacy_theme_values,
)
from .models import (
    AccessSettings,
    AppColorwaySettings,
    AppConfig,
    ColorwaySettings,
    DeviceAccessSettings,
    GenerationSettings,
    IconColorSettings,
    LoggingSettings,
    MessageAction,
    MessageActionStyle,
    SpellingReplacement,
    StarterPrompt,
    ThemeSettings,
    UiSettings,
)
from .paths import AppPaths

__all__ = [
    "CONFIG_VERSION",
    "ConfigIssue",
    "ConfigValidationError",
    "decode_config",
    "encode_config",
]

CONFIG_VERSION: int = 2
_LEGACY_UI_FIELDS: Final[frozenset[str]] = frozenset({"message_action_icon_style"})
_OPTIONAL_UI_FIELDS: Final[frozenset[str]] = frozenset({"message_action_style"})
_LEGACY_MESSAGE_ACTION_STYLES: Final[dict[str, MessageActionStyle]] = {
    "monochrome": MessageActionStyle.UNIFORM,
    "muted_color": MessageActionStyle.SEMANTIC,
}
_LEGACY_PRIMARY_ACTION_FIELDS: Final[dict[MessageAction, str]] = {
    MessageAction.DELETE: "action_delete",
    MessageAction.REGENERATE: "action_regenerate",
}
type ConfigDocument = dict[str, object]


def decode_config(raw: object, *, paths: AppPaths) -> AppConfig:
    """Decode a complete v2 JSON document or raise every detected validation issue."""

    decoder: _Decoder = _Decoder()
    defaults: AppConfig = default_config(paths)
    root: dict[str, object] = decoder.object(
        raw,
        path="config",
        required=frozenset(
            {"version", "generation", "ui", "logging", "access", "backends"}
        ),
        optional=frozenset({"theme"}),
    )
    version: int = decoder.integer(
        root.get("version"), path="config.version", default=CONFIG_VERSION
    )
    if version != CONFIG_VERSION:
        decoder.issue("config.version", f"must be {CONFIG_VERSION}")

    config: AppConfig = AppConfig(
        paths=paths,
        generation=_decode_generation(
            decoder, root.get("generation"), defaults.generation
        ),
        ui=_decode_ui(decoder, root.get("ui"), defaults.ui),
        theme=_decode_theme(decoder, root.get("theme"), defaults.theme),
        logging=_decode_logging(decoder, root.get("logging"), defaults.logging, paths),
        access=_decode_access(decoder, root.get("access"), defaults.access),
        backends=decode_backends(root["backends"])
        if "backends" in root
        else BackendConfiguration(),
    )
    try:
        config.validate()
    except (TypeError, ValueError) as exc:
        decoder.issue("config", str(exc))
    if decoder.issues:
        raise ConfigValidationError(decoder.issues)
    return config


def encode_config(config: AppConfig) -> ConfigDocument:
    """Encode one validated persisted configuration as the canonical v2 document."""

    config.validate()
    document: ConfigDocument = {
        "version": CONFIG_VERSION,
        "backends": encode_backends(config.backends),
        "generation": _encode_generation(config.generation),
        "ui": {
            "host": config.ui.host,
            "port": config.ui.port,
            "dark_mode": config.ui.dark_mode,
            "auto_open_browser": config.ui.auto_open_browser,
            "message_action_style": config.ui.message_action_style.value,
            "icon_colors": {
                "linework_color": config.ui.icon_colors.linework_color,
                "accent_color": config.ui.icon_colors.accent_color,
                "surface_color": config.ui.icon_colors.surface_color,
            },
            "starter_prompts": [
                {"label": prompt.label, "content": prompt.content}
                for prompt in config.ui.starter_prompts
            ],
        },
        "logging": {
            "enabled": config.logging.enabled,
            "directory": _path_to_config_string(
                config.log_directory, paths=config.paths
            ),
        },
        "access": {
            "approval_phrase": config.access.approval_phrase,
            "devices": {
                device_id: {
                    "access_allowed": device.access_allowed,
                    "label": device.label,
                    "last_ip": device.last_ip,
                    "hostname": device.hostname,
                    "first_seen_at": device.first_seen_at,
                    "last_seen_at": device.last_seen_at,
                }
                for device_id, device in config.access.devices.items()
            },
        },
    }
    if config.theme != builtin_theme():
        document["theme"] = config.theme.values()
    return document


def _decode_generation(
    decoder: _Decoder, raw: object, defaults: GenerationSettings
) -> GenerationSettings:
    values: dict[str, object] = decoder.object(
        raw,
        path="config.generation",
        required=frozenset(
            {
                "temperature",
                "top_p",
                "max_tokens",
                "system_prompt",
                "british_english",
            }
        ),
        optional=frozenset({"british_spelling_replacements"}),
    )
    replacements: list[SpellingReplacement] = (
        list(builtin_spelling_replacements())
        if "british_spelling_replacements" not in values
        else _decode_replacements(
            decoder,
            values["british_spelling_replacements"],
            path="config.generation.british_spelling_replacements",
        )
    )
    return GenerationSettings(
        temperature=decoder.number(
            values.get("temperature"),
            path="config.generation.temperature",
            default=defaults.temperature,
        ),
        top_p=decoder.number(
            values.get("top_p"), path="config.generation.top_p", default=defaults.top_p
        ),
        max_tokens=decoder.integer(
            values.get("max_tokens"),
            path="config.generation.max_tokens",
            default=defaults.max_tokens,
        ),
        system_prompt=decoder.string(
            values.get("system_prompt"),
            path="config.generation.system_prompt",
            default=defaults.system_prompt,
        ),
        british_english=decoder.boolean(
            values.get("british_english"),
            path="config.generation.british_english",
            default=defaults.british_english,
        ),
        british_spelling_replacements=replacements,
    )


def _decode_ui(decoder: _Decoder, raw: object, defaults: UiSettings) -> UiSettings:
    values: dict[str, object] = decoder.object(
        raw,
        path="config.ui",
        required=frozenset(
            {
                "host",
                "port",
                "dark_mode",
                "auto_open_browser",
                "icon_colors",
                "starter_prompts",
            }
        ),
        optional=_OPTIONAL_UI_FIELDS | _LEGACY_UI_FIELDS,
    )
    icon_values: dict[str, object] = decoder.object(
        values.get("icon_colors"),
        path="config.ui.icon_colors",
        required=frozenset({"linework_color", "accent_color", "surface_color"}),
    )
    icon_defaults: IconColorSettings = defaults.icon_colors
    return UiSettings(
        host=decoder.string(
            values.get("host"), path="config.ui.host", default=defaults.host
        ),
        port=decoder.integer(
            values.get("port"), path="config.ui.port", default=defaults.port
        ),
        dark_mode=decoder.boolean(
            values.get("dark_mode"),
            path="config.ui.dark_mode",
            default=defaults.dark_mode,
        ),
        auto_open_browser=decoder.boolean(
            values.get("auto_open_browser"),
            path="config.ui.auto_open_browser",
            default=defaults.auto_open_browser,
        ),
        message_action_style=_decode_message_action_style(
            decoder, values, defaults.message_action_style
        ),
        icon_colors=IconColorSettings(
            linework_color=_decode_hex(
                decoder,
                icon_values.get("linework_color"),
                path="config.ui.icon_colors.linework_color",
                default=icon_defaults.linework_color,
            ),
            accent_color=_decode_hex(
                decoder,
                icon_values.get("accent_color"),
                path="config.ui.icon_colors.accent_color",
                default=icon_defaults.accent_color,
            ),
            surface_color=_decode_hex(
                decoder,
                icon_values.get("surface_color"),
                path="config.ui.icon_colors.surface_color",
                default=icon_defaults.surface_color,
            ),
        ),
        starter_prompts=_decode_starter_prompts(
            decoder, values.get("starter_prompts"), defaults=defaults.starter_prompts
        ),
    )


def _decode_message_action_style(
    decoder: _Decoder,
    values: dict[str, object],
    default: MessageActionStyle,
) -> MessageActionStyle:
    """Decode the current setting or migrate its former icon-style equivalent."""

    if "message_action_style" in values:
        raw: object = values["message_action_style"]
        if not isinstance(raw, str):
            decoder.issue("config.ui.message_action_style", "must be a string")
            return default
        try:
            return MessageActionStyle(raw)
        except ValueError:
            decoder.issue("config.ui.message_action_style", "is invalid")
            return default
    if "message_action_icon_style" not in values:
        return default

    legacy_raw: object = values["message_action_icon_style"]
    if not isinstance(legacy_raw, str):
        decoder.issue("config.ui.message_action_icon_style", "must be a string")
        return default
    try:
        return _LEGACY_MESSAGE_ACTION_STYLES[legacy_raw]
    except KeyError:
        decoder.issue("config.ui.message_action_icon_style", "is invalid")
        return default


def _decode_theme(
    decoder: _Decoder, raw: object, defaults: ThemeSettings
) -> ThemeSettings:
    if raw is None:
        return defaults
    if isinstance(raw, dict) and "primary" in raw:
        return _decode_legacy_theme(decoder, cast(object, raw), defaults)
    values = decoder.object(
        cast(object, raw), path="config.theme", required=frozenset(defaults.values())
    )
    return ThemeSettings(
        app=_decode_colorway(decoder, values.get("app"), defaults.app, "app"),
        user=_decode_colorway(decoder, values.get("user"), defaults.user, "user"),
        assistant=_decode_colorway(
            decoder, values.get("assistant"), defaults.assistant, "assistant"
        ),
        system=_decode_colorway(
            decoder, values.get("system"), defaults.system, "system"
        ),
    )


def _decode_colorway[C: ColorwaySettings](
    decoder: _Decoder, raw: object, defaults: C, name: str
) -> C:
    path = f"config.theme.{name}"
    values = decoder.object(
        raw,
        path=path,
        required=frozenset({"accent"}),
        optional=frozenset(defaults.values()) - {"accent"},
    )
    decoded: dict[str, str | None] = {}
    for field_name, default in defaults.values().items():
        value: object = values.get(field_name, default)
        decoded[field_name] = (
            None
            if value is None and field_name != "accent"
            else _decode_hex(
                decoder,
                value,
                path=f"{path}.{field_name}",
                default=default or defaults.accent,
            )
        )
    return replace(defaults, **decoded)


def _decode_legacy_theme(
    decoder: _Decoder, raw: object, defaults: ThemeSettings
) -> ThemeSettings:
    """Retain old accents and deliberate overrides without pinning automatic shades."""

    legacy = legacy_theme_values()
    values = decoder.object(raw, path="config.theme", required=frozenset(legacy))
    decoded = {
        name: _decode_hex(
            decoder, values.get(name), path=f"config.theme.{name}", default=default
        )
        for name, default in legacy.items()
    }

    def override(name: str) -> str | None:
        return decoded[name] if decoded[name].lower() != legacy[name] else None

    def colorway(source: str) -> ColorwaySettings:
        return ColorwaySettings(
            accent=decoded[source],
            muted=override(f"{source}_muted"),
            subtle=override(f"{source}_subtle"),
            hover=override("primary_hover") if source == "primary" else None,
        )

    def action_override(action: MessageAction) -> str | None:
        muted_override: str | None = override(f"action_{action.value}_muted")
        primary_field: str | None = _LEGACY_PRIMARY_ACTION_FIELDS.get(action)
        return muted_override or (
            override(primary_field) if primary_field is not None else None
        )

    return ThemeSettings(
        app=AppColorwaySettings(
            accent=defaults.app.accent,
            **{action.color_field: action_override(action) for action in MessageAction},
            canvas=override("canvas"),
            surface=override("surface"),
            surface_raised=override("surface_raised"),
            border=override("border"),
            border_strong=override("border_strong"),
            text=override("text"),
            text_muted=override("text_muted"),
        ),
        user=colorway("primary"),
        assistant=colorway("secondary"),
        system=defaults.system,
    )


def _decode_logging(
    decoder: _Decoder, raw: object, defaults: LoggingSettings, paths: AppPaths
) -> LoggingSettings:
    values: dict[str, object] = decoder.object(
        raw,
        path="config.logging",
        required=frozenset({"enabled", "directory"}),
    )
    directory_raw: str = decoder.string(
        values.get("directory"),
        path="config.logging.directory",
        default=str(paths.log_directory),
    )
    directory: Path = Path(directory_raw).expanduser()
    if not directory.is_absolute():
        directory = paths.home / directory
    return LoggingSettings(
        enabled=decoder.boolean(
            values.get("enabled"),
            path="config.logging.enabled",
            default=defaults.enabled,
        ),
        directory=directory,
    )


def _decode_access(
    decoder: _Decoder, raw: object, defaults: AccessSettings
) -> AccessSettings:
    values: dict[str, object] = decoder.object(
        raw,
        path="config.access",
        required=frozenset(
            {
                "approval_phrase",
                "devices",
            }
        ),
    )
    devices_raw: dict[str, object] = decoder.mapping(
        values.get("devices"), path="config.access.devices"
    )
    devices: dict[str, DeviceAccessSettings] = {}
    required_device_fields: frozenset[str] = frozenset(
        {
            "access_allowed",
            "label",
            "last_ip",
            "hostname",
            "first_seen_at",
            "last_seen_at",
        }
    )
    for device_id, device_raw in devices_raw.items():
        device_values: dict[str, object] = decoder.object(
            device_raw,
            path=f"config.access.devices.{device_id}",
            required=required_device_fields,
        )
        devices[device_id] = DeviceAccessSettings(
            access_allowed=decoder.boolean(
                device_values.get("access_allowed"),
                path=f"config.access.devices.{device_id}.access_allowed",
                default=False,
            ),
            label=decoder.string(
                device_values.get("label"),
                path=f"config.access.devices.{device_id}.label",
                default="",
            ),
            last_ip=decoder.string(
                device_values.get("last_ip"),
                path=f"config.access.devices.{device_id}.last_ip",
                default="",
            ),
            hostname=decoder.string(
                device_values.get("hostname"),
                path=f"config.access.devices.{device_id}.hostname",
                default="",
            ),
            first_seen_at=decoder.string(
                device_values.get("first_seen_at"),
                path=f"config.access.devices.{device_id}.first_seen_at",
                default="",
            ),
            last_seen_at=decoder.string(
                device_values.get("last_seen_at"),
                path=f"config.access.devices.{device_id}.last_seen_at",
                default="",
            ),
        )
    return AccessSettings(
        approval_phrase=decoder.string(
            values.get("approval_phrase"),
            path="config.access.approval_phrase",
            default=defaults.approval_phrase,
        ),
        devices=devices,
    )


def _decode_replacements(
    decoder: _Decoder, raw: object, *, path: str
) -> list[SpellingReplacement]:
    replacements: list[SpellingReplacement] = []
    for index, item in enumerate(decoder.list(raw, path=path)):
        values: dict[str, object] = decoder.object(
            item,
            path=f"{path}[{index}]",
            required=frozenset({"source", "replacement"}),
        )
        replacements.append(
            SpellingReplacement(
                source=decoder.string(
                    values.get("source"), path=f"{path}[{index}].source", default=""
                ),
                replacement=decoder.string(
                    values.get("replacement"),
                    path=f"{path}[{index}].replacement",
                    default="",
                ),
            )
        )
    return replacements


def _decode_starter_prompts(
    decoder: _Decoder, raw: object, *, defaults: list[StarterPrompt]
) -> list[StarterPrompt]:
    prompts: list[StarterPrompt] = []
    for index, item in enumerate(decoder.list(raw, path="config.ui.starter_prompts")):
        values: dict[str, object] = decoder.object(
            item,
            path=f"config.ui.starter_prompts[{index}]",
            required=frozenset({"label", "content"}),
        )
        prompts.append(
            StarterPrompt(
                label=decoder.string(
                    values.get("label"),
                    path=f"config.ui.starter_prompts[{index}].label",
                    default="",
                ),
                content=decoder.string(
                    values.get("content"),
                    path=f"config.ui.starter_prompts[{index}].content",
                    default="",
                ),
            )
        )
    return prompts if raw is not None else list(defaults)


def _decode_hex(decoder: _Decoder, raw: object, *, path: str, default: str) -> str:
    value: str = decoder.string(raw, path=path, default=default)
    if not HEX_COLOR_PATTERN.fullmatch(value):
        decoder.issue(path, "must be a six-digit hexadecimal color")
        return default
    return value.lower()


def _encode_generation(settings: GenerationSettings) -> dict[str, object]:
    values: dict[str, object] = {
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "max_tokens": settings.max_tokens,
        "system_prompt": settings.system_prompt,
        "british_english": settings.british_english,
    }
    if tuple(settings.british_spelling_replacements) != builtin_spelling_replacements():
        values["british_spelling_replacements"] = [
            {"source": item.source, "replacement": item.replacement}
            for item in settings.british_spelling_replacements
        ]
    return values


def _path_to_config_string(path: Path, *, paths: AppPaths) -> str:
    try:
        return str(path.relative_to(paths.home))
    except ValueError:
        return str(path)
