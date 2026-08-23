"""Strict, versioned JSON codec for persisted application configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .defaults import builtin_spelling_replacements, builtin_theme, default_config
from .models import (
    AccessSettings,
    AppConfig,
    DeviceAccessSettings,
    GenerationSettings,
    HostStatsDeviceSettings,
    HostStatsSettings,
    IconColorSettings,
    LoggingSettings,
    MessageActionIconStyle,
    ServerSettings,
    SpellingReplacement,
    StarterPrompt,
    ThemeSettings,
    UiSettings,
)
from .paths import AppPaths

CONFIG_VERSION: int = 1
type ConfigDocument = dict[str, object]


@dataclass(frozen=True, slots=True)
class ConfigIssue:
    """One invalid JSON location and its explanation."""

    path: str
    message: str


class ConfigValidationError(ValueError):
    """Raised when a persisted configuration cannot be decoded safely."""

    def __init__(self, issues: list[ConfigIssue]) -> None:
        self.issues: tuple[ConfigIssue, ...] = tuple(issues)
        super().__init__(
            "; ".join(f"{issue.path}: {issue.message}" for issue in self.issues)
        )


class _Decoder:
    """Collect strict type and shape errors while decoding one JSON document."""

    def __init__(self) -> None:
        self.issues: list[ConfigIssue] = []

    def issue(self, path: str, message: str) -> None:
        self.issues.append(ConfigIssue(path, message))

    def object(
        self,
        raw: object,
        *,
        path: str,
        required: frozenset[str],
        optional: frozenset[str] = frozenset(),
    ) -> dict[str, object]:
        string_values: dict[str, object] = self.mapping(raw, path=path)
        accepted: frozenset[str] = required | optional
        for key in sorted(set(string_values) - accepted):
            self.issue(f"{path}.{key}", "is not a recognised setting")
        for key in sorted(required - set(string_values)):
            self.issue(f"{path}.{key}", "is required")
        return string_values

    def mapping(self, raw: object, *, path: str) -> dict[str, object]:
        """Decode a string-keyed object whose keys are application data."""

        if not isinstance(raw, dict):
            self.issue(path, "must be an object")
            return {}
        values: dict[object, object] = cast(dict[object, object], raw)
        if any(not isinstance(key, str) for key in values):
            self.issue(path, "must use string keys")
        string_values: dict[str, object] = {
            key: value for key, value in values.items() if isinstance(key, str)
        }
        return string_values

    def string(self, raw: object, *, path: str, default: str) -> str:
        if isinstance(raw, str):
            return raw
        self.issue(path, "must be a string")
        return default

    def boolean(self, raw: object, *, path: str, default: bool) -> bool:
        if isinstance(raw, bool):
            return raw
        self.issue(path, "must be a boolean")
        return default

    def integer(self, raw: object, *, path: str, default: int) -> int:
        if isinstance(raw, int) and not isinstance(raw, bool):
            return raw
        self.issue(path, "must be an integer")
        return default

    def optional_integer(
        self, raw: object, *, path: str, default: int | None
    ) -> int | None:
        if raw is None:
            return None
        return self.integer(
            raw, path=path, default=default if default is not None else 1
        )

    def number(self, raw: object, *, path: str, default: float) -> float:
        if isinstance(raw, int | float) and not isinstance(raw, bool):
            value: float = float(raw)
            if math.isfinite(value):
                return value
        self.issue(path, "must be a finite number")
        return default

    def list(self, raw: object, *, path: str) -> list[object]:
        if isinstance(raw, list):
            return cast(list[object], raw)
        self.issue(path, "must be a list")
        return []


def decode_config(raw: object, *, paths: AppPaths) -> AppConfig:
    """Decode a complete v1 JSON document or raise every detected validation issue."""

    decoder: _Decoder = _Decoder()
    defaults: AppConfig = default_config(paths)
    root: dict[str, object] = decoder.object(
        raw,
        path="config",
        required=frozenset(
            {"version", "server", "generation", "ui", "logging", "host_stats", "access"}
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
        server=_decode_server(decoder, root.get("server"), defaults.server),
        generation=_decode_generation(
            decoder, root.get("generation"), defaults.generation
        ),
        ui=_decode_ui(decoder, root.get("ui"), defaults.ui),
        theme=_decode_theme(decoder, root.get("theme"), defaults.theme),
        logging=_decode_logging(decoder, root.get("logging"), defaults.logging, paths),
        host_stats=_decode_host_stats(
            decoder, root.get("host_stats"), defaults.host_stats
        ),
        access=_decode_access(decoder, root.get("access"), defaults.access),
    )
    try:
        config.validate()
    except ValueError as exc:
        decoder.issue("config", str(exc))
    if decoder.issues:
        raise ConfigValidationError(decoder.issues)
    return config


def encode_config(config: AppConfig) -> ConfigDocument:
    """Encode one validated persisted configuration as the canonical v1 document."""

    config.validate()
    document: ConfigDocument = {
        "version": CONFIG_VERSION,
        "server": {
            "base_url": config.server.base_url,
            "api_key": config.server.api_key,
            "default_model": config.server.default_model,
            "model_aliases": dict(config.server.model_aliases),
            "auto_unload_minutes": config.server.auto_unload_minutes,
        },
        "generation": _encode_generation(config.generation),
        "ui": {
            "host": config.ui.host,
            "port": config.ui.port,
            "dark_mode": config.ui.dark_mode,
            "auto_open_browser": config.ui.auto_open_browser,
            "active_chat_id": config.ui.active_chat_id,
            "message_action_icon_style": config.ui.message_action_icon_style,
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
        "host_stats": {
            "system": config.host_stats.system,
            "cpu": config.host_stats.cpu,
            "gpu": config.host_stats.gpu,
            "network": config.host_stats.network,
            "gpus": _encode_host_devices(config.host_stats.gpus),
            "interfaces": _encode_host_devices(config.host_stats.interfaces),
            "activity_start_color": config.host_stats.activity_start_color,
            "activity_end_color": config.host_stats.activity_end_color,
        },
        "access": {
            "default_private": config.access.default_private,
            "allow_localhost_without_approval": config.access.allow_localhost_without_approval,
            "global_settings_for_approved": config.access.global_settings_for_approved,
            "allow_network_device_reassociation": config.access.allow_network_device_reassociation,
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


def _decode_server(
    decoder: _Decoder, raw: object, defaults: ServerSettings
) -> ServerSettings:
    values: dict[str, object] = decoder.object(
        raw,
        path="config.server",
        required=frozenset(
            {
                "base_url",
                "api_key",
                "default_model",
                "model_aliases",
                "auto_unload_minutes",
            }
        ),
    )
    return ServerSettings(
        base_url=decoder.string(
            values.get("base_url"),
            path="config.server.base_url",
            default=defaults.base_url,
        ),
        api_key=decoder.string(
            values.get("api_key"),
            path="config.server.api_key",
            default=defaults.api_key,
        ),
        default_model=decoder.string(
            values.get("default_model"),
            path="config.server.default_model",
            default=defaults.default_model,
        ),
        model_aliases=_decode_string_mapping(
            decoder, values.get("model_aliases"), path="config.server.model_aliases"
        ),
        auto_unload_minutes=decoder.optional_integer(
            values.get("auto_unload_minutes"),
            path="config.server.auto_unload_minutes",
            default=defaults.auto_unload_minutes,
        ),
    )


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
                "continuity_review",
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
        continuity_review=decoder.boolean(
            values.get("continuity_review"),
            path="config.generation.continuity_review",
            default=defaults.continuity_review,
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
                "active_chat_id",
                "message_action_icon_style",
                "icon_colors",
                "starter_prompts",
            }
        ),
    )
    icon_values: dict[str, object] = decoder.object(
        values.get("icon_colors"),
        path="config.ui.icon_colors",
        required=frozenset({"linework_color", "accent_color", "surface_color"}),
    )
    icon_defaults: IconColorSettings = defaults.icon_colors
    style_raw: str = decoder.string(
        values.get("message_action_icon_style"),
        path="config.ui.message_action_icon_style",
        default=defaults.message_action_icon_style,
    )
    style: MessageActionIconStyle = cast(MessageActionIconStyle, style_raw)
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
        active_chat_id=decoder.string(
            values.get("active_chat_id"),
            path="config.ui.active_chat_id",
            default=defaults.active_chat_id,
        ),
        message_action_icon_style=style,
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


def _decode_theme(
    decoder: _Decoder, raw: object, defaults: ThemeSettings
) -> ThemeSettings:
    if raw is None:
        return defaults
    default_values: dict[str, str] = defaults.values()
    values: dict[str, object] = decoder.object(
        raw,
        path="config.theme",
        required=frozenset(default_values),
    )
    decoded_values: dict[str, str] = {
        field_name: _decode_hex(
            decoder,
            values.get(field_name),
            path=f"config.theme.{field_name}",
            default=default,
        )
        for field_name, default in default_values.items()
    }
    return ThemeSettings(**decoded_values)


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


def _decode_host_stats(
    decoder: _Decoder, raw: object, defaults: HostStatsSettings
) -> HostStatsSettings:
    values: dict[str, object] = decoder.object(
        raw,
        path="config.host_stats",
        required=frozenset(
            {
                "system",
                "cpu",
                "gpu",
                "network",
                "gpus",
                "interfaces",
                "activity_start_color",
                "activity_end_color",
            }
        ),
    )
    return HostStatsSettings(
        system=decoder.boolean(
            values.get("system"),
            path="config.host_stats.system",
            default=defaults.system,
        ),
        cpu=decoder.boolean(
            values.get("cpu"), path="config.host_stats.cpu", default=defaults.cpu
        ),
        gpu=decoder.boolean(
            values.get("gpu"), path="config.host_stats.gpu", default=defaults.gpu
        ),
        network=decoder.boolean(
            values.get("network"),
            path="config.host_stats.network",
            default=defaults.network,
        ),
        gpus=_decode_host_devices(
            decoder, values.get("gpus"), path="config.host_stats.gpus"
        ),
        interfaces=_decode_host_devices(
            decoder, values.get("interfaces"), path="config.host_stats.interfaces"
        ),
        activity_start_color=_decode_hex(
            decoder,
            values.get("activity_start_color"),
            path="config.host_stats.activity_start_color",
            default=defaults.activity_start_color,
        ),
        activity_end_color=_decode_hex(
            decoder,
            values.get("activity_end_color"),
            path="config.host_stats.activity_end_color",
            default=defaults.activity_end_color,
        ),
    )


def _decode_access(
    decoder: _Decoder, raw: object, defaults: AccessSettings
) -> AccessSettings:
    values: dict[str, object] = decoder.object(
        raw,
        path="config.access",
        required=frozenset(
            {
                "default_private",
                "allow_localhost_without_approval",
                "global_settings_for_approved",
                "allow_network_device_reassociation",
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
        default_private=decoder.boolean(
            values.get("default_private"),
            path="config.access.default_private",
            default=defaults.default_private,
        ),
        allow_localhost_without_approval=decoder.boolean(
            values.get("allow_localhost_without_approval"),
            path="config.access.allow_localhost_without_approval",
            default=defaults.allow_localhost_without_approval,
        ),
        global_settings_for_approved=decoder.boolean(
            values.get("global_settings_for_approved"),
            path="config.access.global_settings_for_approved",
            default=defaults.global_settings_for_approved,
        ),
        allow_network_device_reassociation=decoder.boolean(
            values.get("allow_network_device_reassociation"),
            path="config.access.allow_network_device_reassociation",
            default=defaults.allow_network_device_reassociation,
        ),
        approval_phrase=decoder.string(
            values.get("approval_phrase"),
            path="config.access.approval_phrase",
            default=defaults.approval_phrase,
        ),
        devices=devices,
    )


def _decode_string_mapping(
    decoder: _Decoder, raw: object, *, path: str
) -> dict[str, str]:
    values: dict[str, object] = decoder.mapping(raw, path=path)
    return {
        key: decoder.string(value, path=f"{path}.{key}", default="")
        for key, value in values.items()
    }


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


def _decode_host_devices(
    decoder: _Decoder, raw: object, *, path: str
) -> dict[str, HostStatsDeviceSettings]:
    raw_values: dict[str, object] = decoder.mapping(raw, path=path)
    devices: dict[str, HostStatsDeviceSettings] = {}
    for device_id, device_raw in raw_values.items():
        values: dict[str, object] = decoder.object(
            device_raw,
            path=f"{path}.{device_id}",
            required=frozenset({"visible", "label"}),
        )
        devices[device_id] = HostStatsDeviceSettings(
            visible=decoder.boolean(
                values.get("visible"), path=f"{path}.{device_id}.visible", default=True
            ),
            label=decoder.string(
                values.get("label"), path=f"{path}.{device_id}.label", default=""
            ),
        )
    return devices


def _decode_hex(decoder: _Decoder, raw: object, *, path: str, default: str) -> str:
    value: str = decoder.string(raw, path=path, default=default)
    if (
        not value.startswith("#")
        or len(value) != 7
        or any(character not in "0123456789abcdefABCDEF" for character in value[1:])
    ):
        decoder.issue(path, "must be a six-digit hexadecimal color")
        return default
    return value.lower()


def _encode_generation(settings: GenerationSettings) -> dict[str, object]:
    values: dict[str, object] = {
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "max_tokens": settings.max_tokens,
        "system_prompt": settings.system_prompt,
        "continuity_review": settings.continuity_review,
        "british_english": settings.british_english,
    }
    if tuple(settings.british_spelling_replacements) != builtin_spelling_replacements():
        values["british_spelling_replacements"] = [
            {"source": item.source, "replacement": item.replacement}
            for item in settings.british_spelling_replacements
        ]
    return values


def _encode_host_devices(
    devices: dict[str, HostStatsDeviceSettings],
) -> dict[str, dict[str, object]]:
    return {
        device_id: {"visible": settings.visible, "label": settings.label}
        for device_id, settings in devices.items()
    }


def _path_to_config_string(path: Path, *, paths: AppPaths) -> str:
    try:
        return str(path.relative_to(paths.home))
    except ValueError:
        return str(path)
