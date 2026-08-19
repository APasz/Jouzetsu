"""Individual JSON-document persistence for reusable character profiles."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from logging import Logger
from pathlib import Path
from typing import cast

from .atomic_write import atomic_write_text
from .models import Character


log: Logger = logging.getLogger(__name__)
_CORRUPT_FILE_SUFFIX: str = ".corrupt-"


class CharacterStorage:
    """Persist each character independently so one damaged profile cannot affect others."""

    def __init__(self, directory: Path) -> None:
        self.directory: Path = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def load_all(self) -> list[Character]:
        """Load valid profile documents, quarantining only malformed files."""

        characters: list[Character] = []
        for path in sorted(self.directory.glob("*.json")):
            character: Character | None = self._load_one(path)
            if character is not None:
                characters.append(character)
        log.info("loaded character store directory=%s character_count=%d", self.directory, len(characters))
        return characters

    def save(self, character: Character) -> None:
        """Atomically replace the one document for ``character``."""

        data: str = json.dumps(character.to_dict(), ensure_ascii=False, indent=4) + "\n"
        path: Path = self.path_for(character.id)
        atomic_write_text(path, data)
        log.info("saved character path=%s character_id=%s revision=%d", path, character.id, character.revision)

    def delete(self, character_id: str) -> None:
        """Remove the known profile document after its id was resolved in application state."""

        path: Path = self.path_for(character_id)
        path.unlink(missing_ok=True)
        log.info("deleted character path=%s character_id=%s", path, character_id)

    def path_for(self, character_id: str) -> Path:
        """Return an in-directory path for one generated document identifier."""

        if not character_id or not character_id.isalnum():
            raise ValueError("character id must be alphanumeric")
        return self.directory / f"{character_id}.json"

    def _load_one(self, path: Path) -> Character | None:
        try:
            raw: object = cast(object, json.loads(path.read_text(encoding="utf-8")))
            if not isinstance(raw, dict):
                raise ValueError("root must be an object")
            mapping: dict[object, object] = cast(dict[object, object], raw)
            character: Character = Character.from_dict(
                {key: value for key, value in mapping.items() if isinstance(key, str)}
            )
            if path.stem != character.id:
                raise ValueError("document id does not match its file name")
            return character
        except Exception as exc:  # noqa: BLE001
            self._quarantine_corrupt_file(path, str(exc))
            return None

    def _quarantine_corrupt_file(self, path: Path, reason: str) -> None:
        """Move just one unreadable profile aside instead of losing the library."""

        backup_path: Path = self._corrupt_backup_path(path)
        try:
            _ = path.replace(backup_path)
        except OSError:
            log.exception("could not quarantine corrupt character file path=%s", path)
            return
        log.warning("quarantined corrupt character file reason=%s backup=%s", reason, backup_path)

    @staticmethod
    def _corrupt_backup_path(path: Path) -> Path:
        timestamp: str = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        candidate: Path = path.with_name(f"{path.name}{_CORRUPT_FILE_SUFFIX}{timestamp}")
        sequence: int = 1
        while candidate.exists():
            candidate = path.with_name(f"{path.name}{_CORRUPT_FILE_SUFFIX}{timestamp}-{sequence}")
            sequence += 1
        return candidate
