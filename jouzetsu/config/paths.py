"""Application-home resolution and derived mutable paths."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

APP_HOME_ENVIRONMENT_VARIABLE: Final[str] = "JOUZETSU_HOME"
CHARACTER_NAME_SUGGESTIONS_FILE_NAME: Final[str] = "character-names.json"
_SOURCE_CHECKOUT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]


def normalise_app_home(app_home: Path) -> Path:
    """Return one absolute application-home directory without creating it."""

    resolved_home: Path = app_home.expanduser().resolve()
    if resolved_home.exists() and not resolved_home.is_dir():
        raise ValueError(f"application home must be a directory: {resolved_home}")
    return resolved_home


def resolve_app_home(app_home: Path | None = None) -> Path:
    """Resolve an explicit, environment-selected, or development application home."""

    if app_home is not None:
        return normalise_app_home(app_home)
    configured_home: str | None = os.environ.get(APP_HOME_ENVIRONMENT_VARIABLE)
    if configured_home is not None:
        if not configured_home.strip():
            raise ValueError(f"{APP_HOME_ENVIRONMENT_VARIABLE} must name a directory")
        return normalise_app_home(Path(configured_home))
    if (_SOURCE_CHECKOUT_ROOT / "pyproject.toml").is_file():
        return _SOURCE_CHECKOUT_ROOT
    return Path.cwd().resolve()


@dataclass(frozen=True, slots=True)
class AppPaths:
    """All mutable paths derived from one self-contained application home."""

    home: Path
    config_file: Path
    data_dir: Path
    chats_file: Path
    characters_directory: Path
    character_presets_directory: Path
    character_name_suggestions_file: Path
    log_directory: Path

    @classmethod
    def for_home(cls, app_home: Path) -> AppPaths:
        """Build the complete path set for an application home."""

        home: Path = normalise_app_home(app_home)
        data_dir: Path = home / "data"
        return cls(
            home=home,
            config_file=home / "config.json",
            data_dir=data_dir,
            chats_file=data_dir / "chats.json",
            characters_directory=data_dir / "characters",
            character_presets_directory=data_dir / "character-presets",
            character_name_suggestions_file=(
                data_dir / CHARACTER_NAME_SUGGESTIONS_FILE_NAME
            ),
            log_directory=data_dir / "logs",
        )

    @classmethod
    def for_config_file(cls, config_file: Path) -> AppPaths:
        """Derive application paths from one explicit configuration file."""

        resolved_config_file: Path = config_file.expanduser().resolve()
        return replace(
            cls.for_home(resolved_config_file.parent),
            config_file=resolved_config_file,
        )

    def ensure_runtime_directories(self, *, log_directory: Path) -> None:
        """Create the mutable directories needed by one configured application."""

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chats_file.parent.mkdir(parents=True, exist_ok=True)
        self.characters_directory.mkdir(parents=True, exist_ok=True)
        self.character_presets_directory.mkdir(parents=True, exist_ok=True)
        log_directory.mkdir(parents=True, exist_ok=True)


def default_paths() -> AppPaths:
    """Resolve fresh default paths without freezing environment state at import time."""

    return AppPaths.for_home(resolve_app_home())
