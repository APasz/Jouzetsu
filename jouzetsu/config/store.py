"""Filesystem persistence and runtime-only environment overlays."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

from ..atomic_write import atomic_write_text
from .codec import ConfigIssue, ConfigValidationError, decode_config, encode_config
from .defaults import default_config
from .models import AppConfig, UiSettings
from .paths import AppPaths, resolve_app_home


def _is_non_blank_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_valid_port(value: object) -> bool:
    return (
        isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 65535
    )


@dataclass(frozen=True, slots=True)
class EnvironmentOverrides:
    """Process-local settings that must never be persisted to config.json."""

    host: str | None = None
    port: int | None = None

    def __post_init__(self) -> None:
        if self.host is not None and not _is_non_blank_string(self.host):
            raise ValueError("JOUZETSU_HOST cannot be blank")
        if self.port is not None and not _is_valid_port(self.port):
            raise ValueError("JOUZETSU_PORT must be between 1 and 65535")

    @classmethod
    def from_environment(cls) -> EnvironmentOverrides:
        """Read and validate supported process-local environment overrides."""

        host: str | None = os.environ.get("JOUZETSU_HOST")
        raw_port: str | None = os.environ.get("JOUZETSU_PORT")
        if raw_port is None:
            return cls(host=host)
        try:
            port: int = int(raw_port)
        except ValueError as exc:
            raise ValueError("JOUZETSU_PORT must be an integer") from exc
        return cls(host=host, port=port)

    def apply(self, config: AppConfig) -> AppConfig:
        """Return the effective runtime configuration after applying this overlay."""

        if self.host is None and self.port is None:
            return config
        ui: UiSettings = replace(
            config.ui,
            host=config.ui.host if self.host is None else self.host,
            port=config.ui.port if self.port is None else self.port,
        )
        return replace(config, ui=ui)

    def remove(self, config: AppConfig, *, persisted: AppConfig | None) -> AppConfig:
        """Remove overridden runtime values before an effective config is written."""

        if persisted is None or (self.host is None and self.port is None):
            return config
        ui: UiSettings = replace(
            config.ui,
            host=persisted.ui.host if self.host is not None else config.ui.host,
            port=persisted.ui.port if self.port is not None else config.ui.port,
        )
        return replace(config, ui=ui)


@dataclass(slots=True)
class ConfigStore:
    """The only component allowed to read or write application configuration."""

    paths: AppPaths
    overrides: EnvironmentOverrides
    _persisted: AppConfig | None = None

    @classmethod
    def for_home(cls, app_home: Path | None = None) -> ConfigStore:
        """Construct one store for an explicit or environment-selected application home."""

        paths: AppPaths = AppPaths.for_home(resolve_app_home(app_home))
        return cls(paths=paths, overrides=EnvironmentOverrides.from_environment())

    @classmethod
    def for_config_file(cls, config_file: Path) -> ConfigStore:
        """Construct one store around a specific configuration file."""

        return cls(
            paths=AppPaths.for_config_file(config_file),
            overrides=EnvironmentOverrides.from_environment(),
        )

    @classmethod
    def for_paths(
        cls,
        paths: AppPaths,
        *,
        overrides: EnvironmentOverrides | None = None,
    ) -> ConfigStore:
        """Construct a store for known paths and explicitly selected overlays."""

        return cls(paths=paths, overrides=overrides or EnvironmentOverrides())

    def load(self) -> AppConfig:
        """Read one strict configuration document, creating a default document if absent."""

        if not self.paths.config_file.exists():
            default: AppConfig = default_config(self.paths)
            self.save(default)
            return self.overrides.apply(default)

        persisted: AppConfig = self._read_persisted()
        self._persisted = persisted
        self._ensure_directories(persisted)
        return self.overrides.apply(persisted)

    def save(self, effective: AppConfig) -> None:
        """Atomically write persistent settings, never materialising environment overrides."""

        if effective.paths != self.paths:
            raise ValueError("cannot save configuration using a different path set")
        if self._persisted is None:
            self._persisted = (
                self._read_persisted()
                if self.paths.config_file.exists()
                else default_config(self.paths)
            )
        persisted: AppConfig = self.overrides.remove(
            effective, persisted=self._persisted
        )
        encoded: dict[str, object] = encode_config(persisted)
        self.paths.config_file.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.paths.config_file, json.dumps(encoded, indent=4) + "\n")
        self._persisted = persisted
        self._ensure_directories(persisted)

    def _ensure_directories(self, config: AppConfig) -> None:
        self.paths.ensure_runtime_directories(log_directory=config.log_directory)

    def _read_persisted(self) -> AppConfig:
        """Decode the existing document without applying runtime-only overrides."""

        try:
            raw: object = json.loads(self.paths.config_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigValidationError(
                [ConfigIssue("config", f"invalid JSON: {exc.msg}")]
            ) from exc
        except OSError as exc:
            raise ConfigValidationError(
                [ConfigIssue("config", f"cannot read file: {exc}")]
            ) from exc
        return decode_config(raw, paths=self.paths)
